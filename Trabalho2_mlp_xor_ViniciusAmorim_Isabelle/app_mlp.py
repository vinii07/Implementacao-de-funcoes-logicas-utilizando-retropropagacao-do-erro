import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import os

# --- Lógica da Rede Neural ---
class MLP:
    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.2, tolerance=0.001, activation='binaria'):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.activation_type = activation
        
        # Inicialização de pesos e bias (entre -1 e 1)
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))
        
    def _activate(self, x):
        if self.activation_type == 'binaria':
            return 1 / (1 + np.exp(-x))
        else: # bipolar
            return (2 / (1 + np.exp(-x))) - 1
            
    def _derivative(self, x):
        if self.activation_type == 'binaria':
            return x * (1 - x)
        else: # bipolar
            return 0.5 * (1 - x**2)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self._activate(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self._activate(self.final_input)
        return self.final_output
    
    def backward(self, X, y, output):
        error_output = y - output
        delta_output = error_output * self._derivative(output)
        
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self._derivative(self.hidden_output)
        
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * self.learning_rate
        self.bias_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        
        self.weights_input_hidden += X.T.dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate


# --- Interface Gráfica (GUI) ---
class MLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP - Implementação de Funções Lógicas")
        self.root.geometry("900x650")
        
        self.mlp = None
        
        # Base de dados para todas as portas lógicas exigidas e opcionais
        self.training_data = {
            "XOR":  (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([[0], [1], [1], [0]])),
            "AND":  (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([[0], [0], [0], [1]])),
            "OR":   (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([[0], [1], [1], [1]])),
            "NAND": (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([[1], [1], [1], [0]])),
            "NOR":  (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([[1], [0], [0], [0]]))
        }
        
        self._generate_activation_plot()
        self._setup_ui()
        
    def _generate_activation_plot(self):
        """Gera e salva o gráfico das três funções de ativação ao iniciar o app"""
        x = np.linspace(-10, 10, 100)
        
        # Fórmulas Matemáticas
        sig_binaria = 1 / (1 + np.exp(-x))
        sig_bipolar = (2 / (1 + np.exp(-x))) - 1
        tang_hiperbolica = np.tanh(x)
        
        # Criando a figura com 3 gráficos (1 linha, 3 colunas)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # 1. Sigmóide Binária
        axes[0].plot(x, sig_binaria, color='blue', linewidth=2)
        axes[0].set_title('Sigmóide Binária')
        axes[0].grid(True, linestyle='--')
        
        # 2. Sigmóide Bipolar
        axes[1].plot(x, sig_bipolar, color='red', linewidth=2)
        axes[1].set_title('Sigmóide Bipolar')
        axes[1].grid(True, linestyle='--')
        
        # 3. Tangente Hiperbólica
        axes[2].plot(x, tang_hiperbolica, color='green', linewidth=2)
        axes[2].set_title('Tangente Hiperbólica')
        axes[2].grid(True, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('funcoes_ativacao.png')
        plt.close(fig)

    def _setup_ui(self):
        # Frame de Configuração (Esquerda)
        config_frame = ttk.LabelFrame(self.root, text="Configuração do Treinamento", padding=10)
        config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Label(config_frame, text="Função Lógica:").pack(anchor=tk.W)
        self.func_var = tk.StringVar(value="XOR")
        ttk.Combobox(config_frame, textvariable=self.func_var, values=["XOR", "AND", "OR", "NAND", "NOR"], state="readonly").pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="Neurônios Camada Oculta:").pack(anchor=tk.W)
        self.hidden_var = tk.IntVar(value=4)
        ttk.Entry(config_frame, textvariable=self.hidden_var).pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="Taxa de Aprendizado:").pack(anchor=tk.W)
        self.lr_var = tk.DoubleVar(value=0.2)
        ttk.Entry(config_frame, textvariable=self.lr_var).pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="Tolerância de Erro:").pack(anchor=tk.W)
        self.tol_var = tk.DoubleVar(value=0.001)
        ttk.Entry(config_frame, textvariable=self.tol_var).pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="Máximo de Épocas:").pack(anchor=tk.W)
        self.epochs_var = tk.IntVar(value=10000)
        ttk.Entry(config_frame, textvariable=self.epochs_var).pack(fill=tk.X, pady=5)
        
        ttk.Label(config_frame, text="Função de Ativação:").pack(anchor=tk.W)
        self.activation_var = tk.StringVar(value="bipolar")
        ttk.Radiobutton(config_frame, text="Sigmóide Binária", variable=self.activation_var, value="binaria").pack(anchor=tk.W)
        ttk.Radiobutton(config_frame, text="Sigmóide Bipolar", variable=self.activation_var, value="bipolar").pack(anchor=tk.W)
        
        self.train_btn = ttk.Button(config_frame, text="Treinar Rede", command=self.train_network)
        self.train_btn.pack(fill=tk.X, pady=20)
        
        # --- FRAME: TABELA VERDADE ---
        result_frame = ttk.LabelFrame(config_frame, text="Tabela Verdade (Resultados)", padding=10)
        result_frame.pack(fill=tk.X, pady=10)
        
        columns = ("x1", "x2", "esperado", "bruto", "resultado")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=4)
        self.tree.pack(fill=tk.X)
        
        self.tree.heading("x1", text="X1")
        self.tree.heading("x2", text="X2")
        self.tree.heading("esperado", text="Alvo")
        self.tree.heading("bruto", text="Bruto")
        self.tree.heading("resultado", text="Rede")
        
        self.tree.column("x1", width=30, anchor=tk.CENTER)
        self.tree.column("x2", width=30, anchor=tk.CENTER)
        self.tree.column("esperado", width=40, anchor=tk.CENTER)
        self.tree.column("bruto", width=60, anchor=tk.CENTER)
        self.tree.column("resultado", width=40, anchor=tk.CENTER)
        
        # Frame de Visualização (Direita)
        viz_frame = ttk.Frame(self.root, padding=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.set_title("Gráfico de Erro")
        self.ax.set_xlabel("Época")
        self.ax.set_ylabel("MSE")
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_label = ttk.Label(viz_frame, text="Status: Aguardando treinamento...")
        self.status_label.pack(pady=5)

    def train_network(self):
        try:
            func_name = self.func_var.get()
            X, y = self.training_data[func_name]
            
            # Ajustar y se for bipolar
            if self.activation_var.get() == 'bipolar':
                y_train = np.where(y == 0, -1, 1)
            else:
                y_train = y
                
            np.random.seed(42) # Semente fixa para reprodutibilidade no trabalho
            self.mlp = MLP(
                hidden_size=self.hidden_var.get(),
                learning_rate=self.lr_var.get(),
                tolerance=self.tol_var.get(),
                activation=self.activation_var.get()
            )
            
            errors = []
            max_epochs = self.epochs_var.get()
            
            self.status_label.config(text="Status: Treinando...")
            self.root.update()
            
            for epoch in range(max_epochs):
                output = self.mlp.forward(X)
                self.mlp.backward(X, y_train, output)
                
                mse = np.mean(np.square(y_train - output))
                errors.append(mse)
                
                if mse <= self.mlp.tolerance:
                    break
            
            # Atualizar Gráfico na Interface
            self.ax.clear()
            self.ax.plot(errors, color='blue', linewidth=2)
            self.ax.set_title(f"Treinamento - Função {func_name}")
            self.ax.set_xlabel("Época")
            self.ax.set_ylabel("Erro Quadrático Médio (MSE)")
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.canvas.draw()
            
            # --- SALVAR GRÁFICO NA PASTA ---
            filename = f"erro_{func_name.lower()}.png"
            self.fig.savefig(filename)
            
            # --- PREENCHER A TABELA VERDADE AUTOMATICAMENTE ---
            # Limpar tabela anterior
            for item in self.tree.get_children():
                self.tree.delete(item)
                
            # Calcular e inserir os novos resultados
            for i in range(len(X)):
                in1, in2 = X[i]
                alvo = y[i][0]
                
                # Previsão da rede
                saida_bruta = self.mlp.forward(np.array([[in1, in2]]))[0][0]
                
                # Aplicação do Threshold (Limiar) para interpretação do resultado
                if self.activation_var.get() == 'bipolar':
                    saida_binaria = 1 if saida_bruta > 0 else 0
                else:
                    saida_binaria = 1 if saida_bruta > 0.5 else 0
                    
                self.tree.insert("", tk.END, values=(in1, in2, alvo, f"{saida_bruta:.4f}", saida_binaria))
            
            self.status_label.config(text=f"Status: Convergiu em {len(errors)} épocas. Erro: {mse:.6f}")
            messagebox.showinfo("Sucesso", f"Treinamento concluído!\nÉpocas: {len(errors)}\nErro Final: {mse:.6f}\n\nGráfico salvo como: {filename}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha no treinamento: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLPApp(root)
    root.mainloop()
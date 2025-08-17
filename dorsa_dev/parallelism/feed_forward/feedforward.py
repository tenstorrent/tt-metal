import torch
import ttnn

"""
FFN(x) = f(xW_1 + b_1)(W_2) + b_2

- x: token embedding (after attention + residual/layer norm)
- W_1: weight matrix for the first linear layer (expands the embedding dimension)
- f: activation function (apply nonlinearity). ex. ReLU, GELU, Swish
- W_2: compress back down to original embed dim
- b_1, b_2: bias terms

[token vector] -> expand -> nonlinearity -> compress back
"""


class FeedForward(nn.Module):
    def __init__(self, embed_dim, expand_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, expand_dim)
        self.fc2 = nn.Linear(expand_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

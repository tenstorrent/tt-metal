import math

import torch
import torch.nn as nn


class PatchTSMixerGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.
    Args:
        d_model: dimension of the model i.e hidden features vector size.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attn_layer = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        w = self.softmax(self.attn_layer(x))
        return x * w


class PatchBatchNorm(nn.Module):
    """
    Module that applies batch normalization over the sequence length (time) dimension
    Args:
        d_model: dimension of the model.
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(d_model, eps)

    def forward(self, x):
        return self.batchnorm(x.transpose(1, 2)).transpose(1, 2)


class PatchTSMixerPositionalEncoding(nn.Module):
    """
    Class for positional encoding.
    """

    def __init__(self, num_patches, d_model, use_pe=True, pe_type="sincos"):
        super().__init__()
        if not use_pe:
            self.pe = nn.Parameter(torch.zeros(num_patches, d_model))
        else:
            if pe_type == "random":
                self.pe = nn.Parameter(torch.randn(num_patches, d_model))
            elif pe_type == "sincos":
                pos = torch.arange(0, num_patches).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)))
                pe = torch.zeros(num_patches, d_model)
                div_term = torch.zeros(num_patches, d_model)
                pe[:, 0::2] = torch.sin(pos * div_term)
                pe[:, 1::2] = torch.cos(pos * div_term)
                pe = (pe - pe.mean()) / (pe.std() * 10)
                self.register_buffer("pe", pe)  # non-learnable buffer
            else:
                raise ValueError("pe_type must be 'random' or 'sincos'")

    def forward(self, x):
        # x: (B, C, N_p, D)
        return x + self.pe


class PatchTSMixerNormLayer(nn.Module):
    """
    Normalization block.
    """

    def __init__(self, d_model, norm_type="LayerNorm", eps=1e-5):
        super().__init()
        self.norm_type = norm_type.lower()
        if "batch" in self.norm_type:
            self.norm = nn.BatchNorm1d(d_model, eps=eps)
        else:
            self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        # x: (B, C, N_p, D)
        if "batch" in self.norm_type:
            B, C, N_p, D = x.shape
            x_reshaped = x.view(B * C, N_p, D)
            x_reshaped = self.norm(x_reshaped.transpose(1, 2).transpose(1, 2))
            return x_reshaped.view(B, C, N_p, D)
        else:
            # LayerNorm over last dim
            return self.norm(x)


class PatchTSMixerMLP(nn.Module):
    def __init__(self, in_features, out_features, expansion=2, dropout=0.1):
        super().__init__()
        hidden = in_features * expansion
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(torch.nn.functional.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

"""
PyTorch reference implementation of Shared Expert module.

This module implements a simplified shared expert MLP following DeepSeek architecture:
- Three weight matrices: gate_proj, up_proj, down_proj
- No biases
- Forward pass: output = (silu(x @ gate_proj) * (x @ up_proj)) @ down_proj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchSharedExpert(nn.Module):
    """
    PyTorch reference implementation of Shared Expert MLP.

    Architecture:
        Input: [batch, seq_len, emb_dim]
        gate_out = x @ gate_proj → [batch, seq_len, hidden_dim]
        up_out = x @ up_proj → [batch, seq_len, hidden_dim]
        activated = silu(gate_out) * up_out → [batch, seq_len, hidden_dim]
        output = activated @ down_proj → [batch, seq_len, emb_dim]
    """

    def __init__(self, emb_dim: int = 7 * 1024, hidden_dim: int = 2 * 1024):
        """
        Initialize Shared Expert module.

        Args:
            emb_dim: Embedding dimension (default: 7168)
            hidden_dim: Hidden dimension (default: 2048)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Initialize weights (no biases, following DeepSeek)
        self.gate_proj = nn.Parameter(torch.randn(emb_dim, hidden_dim, dtype=torch.float32))
        self.up_proj = nn.Parameter(torch.randn(emb_dim, hidden_dim, dtype=torch.float32))
        self.down_proj = nn.Parameter(torch.randn(hidden_dim, emb_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without multi-chip logic.

        Args:
            x: Input tensor [batch, seq_len, emb_dim]

        Returns:
            Output tensor [batch, seq_len, emb_dim]
        """
        # Gate projection
        gate_out = x @ self.gate_proj  # [batch, seq_len, hidden_dim]

        # Up projection
        up_out = x @ self.up_proj  # [batch, seq_len, hidden_dim]

        # SiLU activation and element-wise multiplication
        activated = F.silu(gate_out) * up_out  # [batch, seq_len, hidden_dim]

        # Down projection
        output = activated @ self.down_proj  # [batch, seq_len, emb_dim]

        return output

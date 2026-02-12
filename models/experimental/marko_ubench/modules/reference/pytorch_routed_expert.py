# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class PytorchRoutedExpert(nn.Module):
    """PyTorch reference implementation of a routed expert with SwiGLU-style gated activation.

    This implements the computation: output = silu(x @ gate_proj) * (x @ up_proj) @ down_proj
    Similar to DeepSeek's routed expert architecture.

    Args:
        emb_dim: Input embedding dimension (default: 7*1024)
        hidden_dim: Hidden dimension for intermediate layer (default: 2*1024)
    """

    def __init__(self, emb_dim: int = 7 * 1024, hidden_dim: int = 2 * 1024):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Three weight matrices: gate_proj, up_proj, down_proj
        # No biases based on DeepSeek architecture
        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: output = silu(x @ gate_proj) * (x @ up_proj) @ down_proj

        Args:
            x: Input tensor of shape (..., emb_dim)

        Returns:
            Output tensor of shape (..., emb_dim)
        """
        # Verify input shape
        assert x.shape[-1] == self.emb_dim, f"Input last dim {x.shape[-1]} != emb_dim {self.emb_dim}"

        # Gate and up projections
        gate = self.gate_proj(x)  # (..., hidden_dim)
        up = self.up_proj(x)  # (..., hidden_dim)

        # Apply SiLU activation to gate and multiply with up
        activated = F.silu(gate) * up  # (..., hidden_dim)

        # Down projection
        output = self.down_proj(activated)  # (..., emb_dim)

        return output

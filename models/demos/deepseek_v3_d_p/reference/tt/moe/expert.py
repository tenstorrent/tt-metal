# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Torch reference implementation of Expert FFN module.

The Expert FFN follows the SwiGLU architecture:
    gate_out = x @ gate_proj.T
    up_out = x @ up_proj.T
    activated = silu(gate_out) * up_out
    output = activated @ down_proj.T
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchExpert(nn.Module):
    """
    Expert FFN with configurable initialization.

    Architecture (SwiGLU):
        gate_out = x @ gate_proj.T
        up_out = x @ up_proj.T
        activated = silu(gate_out) * up_out
        output = activated @ down_proj.T

    Can be initialized with:
    - Real weights from HuggingFace checkpoint (torch_weights)
    - Identity matrices for flow testing (use_identity=True)
    - Random normal weights (default)
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        torch_weights: dict = None,
        use_identity: bool = False,
    ):
        """
        Initialize Expert module.

        Args:
            emb_dim: Embedding/input dimension
            hidden_dim: Hidden/intermediate dimension (output of gate/up proj)
            torch_weights: Optional dict with gate_proj, up_proj, down_proj tensors
                          from HuggingFace checkpoint. If provided, uses these weights.
                          Shape convention: (out_features, in_features) per HF format.
            use_identity: If True and torch_weights is None, initialize with identity
                         matrices (requires emb_dim == hidden_dim). Useful for flow testing.
                         If False and torch_weights is None, uses random normal init.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        if torch_weights is not None:
            # Load from provided weights - shapes come from checkpoint
            # HF format: weight shape is (out_features, in_features)
            self.gate_proj = nn.Parameter(torch_weights["gate_proj"].float())
            self.up_proj = nn.Parameter(torch_weights["up_proj"].float())
            self.down_proj = nn.Parameter(torch_weights["down_proj"].float())
        elif use_identity:
            # Identity initialization for flow testing (requires square matrices)
            if emb_dim != hidden_dim:
                raise ValueError(
                    f"Identity initialization requires emb_dim == hidden_dim, "
                    f"got emb_dim={emb_dim}, hidden_dim={hidden_dim}"
                )
            self.gate_proj = nn.Parameter(torch.eye(emb_dim, dtype=torch.float32))
            self.up_proj = nn.Parameter(torch.eye(emb_dim, dtype=torch.float32))
            self.down_proj = nn.Parameter(torch.eye(emb_dim, dtype=torch.float32))
        else:
            # Random normal initialization
            # HF format: (out_features, in_features)
            self.gate_proj = nn.Parameter(torch.randn(hidden_dim, emb_dim) * 0.02)
            self.up_proj = nn.Parameter(torch.randn(hidden_dim, emb_dim) * 0.02)
            self.down_proj = nn.Parameter(torch.randn(emb_dim, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., emb_dim]

        Returns:
            Output tensor [..., emb_dim]
        """
        # Gate projection: x @ gate_proj.T (HF format: weight is out_features x in_features)
        gate_out = F.linear(x, self.gate_proj)

        # Up projection
        up_out = F.linear(x, self.up_proj)

        # SiLU activation and element-wise multiplication
        activated = F.silu(gate_out) * up_out

        # Down projection
        output = F.linear(activated, self.down_proj)

        return output

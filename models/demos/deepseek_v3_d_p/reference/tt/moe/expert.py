# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Torch reference implementation of Expert FFN module.

The Expert FFN follows the SwiGLU architecture:
    gate_out = x @ gate_proj.T
    up_out = x @ up_proj.T
    activated = silu(gate_out) * up_out
    output = activated @ down_proj.T

Kimi K3 replaces the SiLU gate with SiTU-GLU (``activation="situ_glu"``), which
tanh-caps both halves before the product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Kimi K3 SiTU-GLU betas. Must match SituGluConfigKimi in situ_glu_sfpu.h, which the
# fused kernel bakes in; a mismatch here silently loosens the PCC comparison.
SITU_BETA_GATE = 4.0
SITU_BETA_UP = 25.0


def situ_glu(gate: torch.Tensor, up: torch.Tensor, beta_gate=SITU_BETA_GATE, beta_up=SITU_BETA_UP) -> torch.Tensor:
    """SiTU-GLU (Kimi K3): (beta_gate*tanh(gate/beta_gate)*sigmoid(gate)) * (beta_up*tanh(up/beta_up))."""
    gate_half = beta_gate * torch.tanh(gate / beta_gate) * torch.sigmoid(gate)
    up_half = beta_up * torch.tanh(up / beta_up)
    return gate_half * up_half


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
        activation: str = "silu",
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
            activation: "silu" (default, DeepSeek) or "situ_glu" (Kimi K3).
        """
        super().__init__()
        if activation not in ("silu", "situ_glu"):
            raise ValueError(f"unsupported activation {activation!r}, expected 'silu' or 'situ_glu'")
        self.activation = activation
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

        if self.activation == "situ_glu":
            activated = situ_glu(gate_out, up_out)
        else:
            # SiLU activation and element-wise multiplication
            activated = F.silu(gate_out) * up_out

        # Down projection
        output = F.linear(activated, self.down_proj)

        return output

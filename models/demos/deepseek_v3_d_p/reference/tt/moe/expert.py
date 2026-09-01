# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Torch reference implementation of Expert FFN module.

The Expert FFN follows the SwiGLU architecture:
    gate_out = x @ gate_proj.T
    up_out = x @ up_proj.T
    activated = silu(gate_out) * up_out
    output = activated @ down_proj.T

Kimi-K3 substitutes SiTU-GLU for SiLU; see ``apply_glu_activation``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Activation selectors accepted by TorchExpert / apply_glu_activation.
ACTIVATION_SILU = "silu"
ACTIVATION_SITU = "situ"


def apply_glu_activation(
    gate_out: torch.Tensor,
    up_out: torch.Tensor,
    activation: str = ACTIVATION_SILU,
    situ_beta: float = 1.0,
    situ_linear_beta: float | None = None,
) -> torch.Tensor:
    """Combine a GLU pair into one activated tensor.

    ``silu``: ``silu(gate) * up`` -- the DeepSeek / Kimi-K2.6 SwiGLU.

    ``situ``: Kimi-K3's SiTU-GLU, ``beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)``.
    Mathematically identical to upstream ``SituAndMul`` (``modeling_kimi_linear.py:64``), which takes
    a single concatenated ``[gate | up]`` tensor and splits it in half; taking the two halves as
    separate arguments avoids a concat the device path would not do either. Computed in fp32 to match
    upstream, which casts both halves up before the tanh/sigmoid.

    NOTE: every K3 FFN site runs SiTU on device -- the routed experts through the fused kernel
    (``ttnn.RoutedExpertActivation.SituGlu``), the shared expert and the layer-0 dense FFN through
    the composed path in ``TtSharedExpert`` / ``TtFfn``. All three are Blackhole-only, and K3 names
    SiTU unconditionally, so there is no Wormhole configuration of K3 to compare against; ``silu``
    here serves the other models.
    """
    if activation == ACTIVATION_SILU:
        return F.silu(gate_out) * up_out
    if activation == ACTIVATION_SITU:
        gate = gate_out.float()
        up = up_out.float()
        situ_a = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
        if situ_linear_beta is not None:
            up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
        return (situ_a * up).to(gate_out.dtype)
    raise ValueError(f"unknown activation {activation!r}; expected {ACTIVATION_SILU!r} or {ACTIVATION_SITU!r}")


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
        activation: str = ACTIVATION_SILU,
        situ_beta: float = 1.0,
        situ_linear_beta: float | None = None,
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
            activation: "silu" (default, unchanged behaviour) or "situ" for Kimi-K3's SiTU-GLU.
            situ_beta / situ_linear_beta: SiTU scalars, ignored unless activation == "situ".
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta

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

        # GLU activation and element-wise multiplication
        activated = apply_glu_activation(
            gate_out,
            up_out,
            activation=self.activation,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )

        # Down projection
        output = F.linear(activated, self.down_proj)

        return output

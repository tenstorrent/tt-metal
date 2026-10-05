# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Immutable, device-independent configuration of one Qwen Gated DeltaNet (GDN) layer."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Activation of the output gate z in the gated RMSNorm, ``w * rmsnorm(o) * act(z)``. Model configs spell silu as
# ``swish`` or leave it unset; ``from_model_config`` resolves those aliases, so only canonical names reach here.
GDN_OUTPUT_GATE_ACTIVATIONS = ("silu", "sigmoid")


@dataclass(frozen=True)
class GDNConfig:
    """Dimensions and numerical policy of one GDN layer.

    q and k carry ``num_key_heads`` heads; v, z, the decay ``g``, ``beta``, the recurrent state and the output norm
    carry ``num_value_heads``. V head ``j`` reads K head ``j // group`` (transformers ``repeat_interleave`` order).
    ``output_gate_activation`` is ``silu`` (transformers ``qwen3_5`` / ``qwen3_5_moe``) or ``sigmoid`` (``qwen4_exp``
    with ``output_gate_type: sigmoid``, the KDA gate).
    """

    hidden_size: int
    num_key_heads: int
    num_value_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float
    output_gate_activation: str

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "num_key_heads": self.num_key_heads,
            "num_value_heads": self.num_value_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "conv_kernel_size": self.conv_kernel_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_value_heads % self.num_key_heads:
            raise ValueError(
                f"num_value_heads {self.num_value_heads} must be a multiple of num_key_heads {self.num_key_heads}"
            )
        if self.conv_kernel_size != 4:
            raise ValueError(f"GDN currently requires conv_kernel_size=4, got {self.conv_kernel_size}")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be finite and positive, got {self.norm_eps}")
        if self.output_gate_activation not in GDN_OUTPUT_GATE_ACTIVATIONS:
            raise ValueError(
                f"output_gate_activation must be one of {GDN_OUTPUT_GATE_ACTIVATIONS}, got {self.output_gate_activation!r}"
            )

    @property
    def group(self) -> int:
        """V heads per K head."""
        return self.num_value_heads // self.num_key_heads

    @property
    def q_dim(self) -> int:
        return self.num_key_heads * self.head_k_dim

    @property
    def k_dim(self) -> int:
        return self.num_key_heads * self.head_k_dim

    @property
    def v_dim(self) -> int:
        return self.num_value_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        """Channels of the fused ``[q | k | v]`` projection and convolution."""
        return self.q_dim + self.k_dim + self.v_dim

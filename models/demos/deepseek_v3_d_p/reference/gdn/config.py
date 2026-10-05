# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Immutable, device-independent configuration of one Qwen Gated DeltaNet (GDN) layer."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GDNConfig:
    """Dimensions and numerical policy of one GDN layer.

    q and k carry ``num_key_heads`` heads; v, z, the decay ``g``, ``beta``, the recurrent state and the output norm
    carry ``num_value_heads``. V head ``j`` reads K head ``j // group`` (transformers ``repeat_interleave`` order).
    """

    hidden_size: int
    num_key_heads: int
    num_value_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float

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

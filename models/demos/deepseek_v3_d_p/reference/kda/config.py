# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Immutable, device-independent configuration for Kimi Delta Attention."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KDAConfig:
    """Dimensions and numerical policy for one KDA layer."""

    hidden_size: int
    num_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float
    use_full_rank_gate: bool = False
    gate_lower_bound: float | None = None

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "conv_kernel_size": self.conv_kernel_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.conv_kernel_size != 4:
            raise ValueError(f"KDA currently requires conv_kernel_size=4, got {self.conv_kernel_size}")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be finite and positive, got {self.norm_eps}")
        if self.gate_lower_bound is not None and not -5.0 <= self.gate_lower_bound < 0.0:
            raise ValueError(f"gate_lower_bound must be in [-5, 0), got {self.gate_lower_bound}")

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def k_dim(self) -> int:
        return self.num_heads * self.head_k_dim

    @property
    def v_dim(self) -> int:
        return self.num_heads * self.head_v_dim

    @classmethod
    def from_model_config(cls, model_config: Mapping[str, Any]) -> "KDAConfig":
        """Build from a Kimi Linear Hugging Face configuration mapping."""
        try:
            if "text_config" in model_config:
                model_config = model_config["text_config"]
                if not isinstance(model_config, Mapping):
                    raise TypeError("text_config must be a mapping")
            linear = model_config["linear_attn_config"]
            if not isinstance(linear, Mapping):
                raise TypeError("linear_attn_config must be a mapping")
            head_dim = int(linear["head_dim"])
            return cls(
                hidden_size=int(model_config["hidden_size"]),
                num_heads=int(linear["num_heads"]),
                head_k_dim=head_dim,
                head_v_dim=head_dim,
                conv_kernel_size=int(linear["short_conv_kernel_size"]),
                norm_eps=float(model_config["rms_norm_eps"]),
                use_full_rank_gate=bool(linear.get("use_full_rank_gate", False)),
                gate_lower_bound=(
                    float(linear["gate_lower_bound"]) if linear.get("gate_lower_bound") is not None else None
                ),
            )
        except KeyError as error:
            raise ValueError(f"missing Kimi config field: {error.args[0]}") from error

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Immutable configuration for Kimi Delta Attention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import ttnn


@dataclass(frozen=True)
class KDAConfig:
    """Dimensions and numerical policy for one KDA layer."""

    hidden_size: int
    num_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    norm_eps: float
    recurrent_state_dtype: ttnn.DataType = ttnn.float32
    chunk_size: int = 64

    def __post_init__(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "conv_kernel_size": self.conv_kernel_size,
            "chunk_size": self.chunk_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if self.recurrent_state_dtype not in (ttnn.float32, ttnn.bfloat16):
            raise ValueError(
                "recurrent_state_dtype must be ttnn.float32 or ttnn.bfloat16, " f"got {self.recurrent_state_dtype}"
            )

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
        """Build from the canonical Hugging Face Kimi Linear config mapping."""
        try:
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
            )
        except KeyError as error:
            raise ValueError(f"missing Kimi config field: {error.args[0]}") from error

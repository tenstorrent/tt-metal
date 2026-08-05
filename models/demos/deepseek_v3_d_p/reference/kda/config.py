# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Immutable configuration for Kimi Delta Attention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
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
        """Build from the canonical Hugging Face Kimi Linear config mapping."""
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


@dataclass(frozen=True)
class KDARecurrenceProgramConfig:
    """Complete tensor and program contract for KDA recurrence."""

    chunk_size: int = ttnn.TILE_SIZE
    grouped_scan_min_chunks: int = 160
    summary_group_chunks: int = 20

    qkv_dtype: ttnn.DataType = ttnn.bfloat16
    gate_dtype: ttnn.DataType = ttnn.bfloat16
    beta_dtype: ttnn.DataType = ttnn.float32
    recurrent_state_dtype: ttnn.DataType = ttnn.float32
    affine_summary_dtype: ttnn.DataType = ttnn.bfloat16
    scan_output_dtype: ttnn.DataType = ttnn.bfloat16
    prep_output_bf16_mask: int = (1 << 1) | (1 << 2) | (1 << 5)

    preparation_memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    prefix_memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG
    distributed_working_memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG
    output_memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG

    affine_prefix_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    grouped_scan_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2

    def __post_init__(self) -> None:
        if self.chunk_size != ttnn.TILE_SIZE:
            raise ValueError(f"KDA recurrence requires chunk_size={ttnn.TILE_SIZE}, got {self.chunk_size}")
        if self.grouped_scan_min_chunks <= 0:
            raise ValueError("grouped_scan_min_chunks must be positive")
        if self.summary_group_chunks <= 0:
            raise ValueError("summary_group_chunks must be positive")
        fixed_dtypes = {
            "qkv_dtype": (self.qkv_dtype, ttnn.bfloat16),
            "gate_dtype": (self.gate_dtype, ttnn.bfloat16),
            "beta_dtype": (self.beta_dtype, ttnn.float32),
            "recurrent_state_dtype": (self.recurrent_state_dtype, ttnn.float32),
            "affine_summary_dtype": (self.affine_summary_dtype, ttnn.bfloat16),
            "scan_output_dtype": (self.scan_output_dtype, ttnn.bfloat16),
        }
        for name, (actual, expected) in fixed_dtypes.items():
            if actual != expected:
                raise ValueError(f"{name} is fixed to {expected}, got {actual}")
        fixed_memory_configs = {
            "preparation_memory_config": (self.preparation_memory_config, ttnn.DRAM_MEMORY_CONFIG),
            "prefix_memory_config": (self.prefix_memory_config, ttnn.DRAM_MEMORY_CONFIG),
            "distributed_working_memory_config": (
                self.distributed_working_memory_config,
                ttnn.L1_MEMORY_CONFIG,
            ),
            "output_memory_config": (self.output_memory_config, ttnn.DRAM_MEMORY_CONFIG),
        }
        for name, (actual, expected) in fixed_memory_configs.items():
            if actual != expected:
                raise ValueError(f"{name} is fixed to {expected}, got {actual}")
        expected_mask = (1 << 1) | (1 << 2) | (1 << 5)
        if self.prep_output_bf16_mask != expected_mask:
            raise ValueError(f"prep_output_bf16_mask is fixed to {expected_mask}, got {self.prep_output_bf16_mask}")


@dataclass(frozen=True)
class KDAProgramConfig:
    """Device-program tuning kept separate from checkpoint model dimensions."""

    recurrence: KDARecurrenceProgramConfig = field(default_factory=KDARecurrenceProgramConfig)
    output_projection_out_block_w: int | None = None
    tp_ccl_topology: ttnn.Topology = ttnn.Topology.Linear
    gated_rms_output_dtype: ttnn.DataType = ttnn.float32
    output_projection_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4

    def __post_init__(self) -> None:
        if self.output_projection_out_block_w is not None and self.output_projection_out_block_w <= 0:
            raise ValueError(
                "output_projection_out_block_w must be positive, " f"got {self.output_projection_out_block_w}"
            )
        if self.gated_rms_output_dtype not in (ttnn.float32, ttnn.bfloat16):
            raise ValueError("gated_rms_output_dtype must be ttnn.float32 or ttnn.bfloat16")

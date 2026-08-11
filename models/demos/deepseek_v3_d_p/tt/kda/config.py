# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-program configuration for Kimi Delta Attention."""

from __future__ import annotations

from dataclasses import dataclass, field

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config


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


def kimi_k3_program_config(*, tp_ccl_topology: ttnn.Topology) -> KDAProgramConfig:
    """Return measured K3 tuning with caller-owned per-axis CCL topology."""
    return KDAProgramConfig(
        recurrence=KDARecurrenceProgramConfig(summary_group_chunks=KimiK3Config.KDA_SUMMARY_GROUP_CHUNKS),
        output_projection_out_block_w=KimiK3Config.KDA_OUTPUT_PROJECTION_OUT_BLOCK_W,
        tp_ccl_topology=tp_ccl_topology,
        gated_rms_output_dtype=ttnn.bfloat16,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
    )

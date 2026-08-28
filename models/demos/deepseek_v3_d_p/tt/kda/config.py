# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-program configuration for Kimi Delta Attention."""

from __future__ import annotations

from dataclasses import dataclass, field

import ttnn

KDA_CHUNK_SIZE = ttnn.TILE_SIZE
KDA_QKV_DTYPE = ttnn.bfloat16
KDA_GATE_DTYPE = ttnn.bfloat16
KDA_BETA_DTYPE = ttnn.float32
KDA_RECURRENT_STATE_DTYPE = ttnn.float32
KDA_AFFINE_SUMMARY_DTYPE = ttnn.bfloat16
KDA_SCAN_OUTPUT_DTYPE = ttnn.bfloat16
KDA_PREP_OUTPUT_BF16_MASK = (1 << 1) | (1 << 2) | (1 << 5)
KDA_PREPARATION_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
KDA_LOCAL_PREFIX_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG
# DRAM is the calibrated T=5120 distributed-scan placement; keep it distinct from the local L1 path.
KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG
KDA_OUTPUT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG


@dataclass(frozen=True)
class KDARecurrenceProgramConfig:
    """Tunable recurrence strategy and compute fidelity."""

    grouped_scan_min_chunks: int = 160
    summary_group_chunks: int = 20
    affine_prefix_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    grouped_scan_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2

    def __post_init__(self) -> None:
        if self.grouped_scan_min_chunks <= 0:
            raise ValueError("grouped_scan_min_chunks must be positive")
        if self.summary_group_chunks <= 0:
            raise ValueError("summary_group_chunks must be positive")


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
        recurrence=KDARecurrenceProgramConfig(summary_group_chunks=20),
        output_projection_out_block_w=4,
        tp_ccl_topology=tp_ccl_topology,
        gated_rms_output_dtype=ttnn.bfloat16,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
    )

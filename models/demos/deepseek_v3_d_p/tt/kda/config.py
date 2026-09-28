# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-program configuration for Kimi Delta Attention."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG
KDA_OUTPUT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG


@dataclass(frozen=True)
class KDARecurrenceProgramConfig:
    """Tunable recurrence strategy and compute fidelity."""

    local_scan_strategy: Literal["direct", "grouped"] = "direct"
    # Exact number of chunks per group. Construction validates divisibility and
    # worker capacity; execution never silently changes an explicit configuration.
    summary_group_chunks: int = 20
    affine_prefix_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    scan_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2

    def __post_init__(self) -> None:
        if self.local_scan_strategy not in ("direct", "grouped"):
            raise ValueError("local_scan_strategy must be 'direct' or 'grouped'")
        if self.summary_group_chunks <= 0:
            raise ValueError("summary_group_chunks must be positive")


@dataclass(frozen=True)
class KDAProgramConfig:
    """Device-program tuning kept separate from checkpoint model dimensions."""

    recurrence: KDARecurrenceProgramConfig = field(default_factory=KDARecurrenceProgramConfig)
    # Ceiling: the effective chunk is the largest TP-local channel divisor no greater than this value.
    qkv_channel_chunk_size: int = 768
    tp_ccl_topology: ttnn.Topology = ttnn.Topology.Linear
    gated_rms_output_dtype: ttnn.DataType = ttnn.float32
    output_projection_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    # Explicit projection matmul schedules; None keeps the auto-selected ttnn.linear configs.
    input_projection_minimal_matmul_config: ttnn.MinimalMatmulConfig | None = None
    output_projection_program_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None = None
    # Stage short-lived activations (the QKV slice before its untilize, the bounded-decay sigmoid,
    # and the gated-norm output before the output projection) in L1; only for local lengths where
    # they fit comfortably.
    stage_activations_in_l1: bool = False

    def __post_init__(self) -> None:
        if self.qkv_channel_chunk_size <= 0 or self.qkv_channel_chunk_size % ttnn.TILE_SIZE:
            raise ValueError(
                "qkv_channel_chunk_size must be a positive multiple of "
                f"{ttnn.TILE_SIZE}, got {self.qkv_channel_chunk_size}"
            )
        if self.gated_rms_output_dtype not in (ttnn.float32, ttnn.bfloat16):
            raise ValueError("gated_rms_output_dtype must be ttnn.float32 or ttnn.bfloat16")


def _galaxy_projection_configs() -> tuple[ttnn.MinimalMatmulConfig, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig]:
    """Return the Galaxy SP8xTP4 projection schedules for 640 tokens per device.

    Measured on the 12x10 Blackhole worker grid with the production numerics (bf16, FP32
    destination accumulation) in tests/kda/perf/test_matmul_perf.py. Each core row owns 2 of the
    20 row tiles; subblocks stay within the 4-tile FP32 destination limit.
    """
    grid = ttnn.CoreCoord(12, 10)
    input_projection = ttnn.MinimalMatmulConfig(
        M_block_size=2,
        K_block_size=8,
        N_block_size=3,
        subblock_h=1,
        subblock_w=3,
        compute_with_storage_grid_size=grid,
    )
    # 7168 output columns over 12 grid columns: 19 tiles per core.
    output_projection = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=1,
        out_block_h=2,
        out_block_w=19,
        per_core_M=2,
        per_core_N=19,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    return input_projection, output_projection


def kimi_k3_program_config(*, active_seq_len_local: int, tp_ccl_topology: ttnn.Topology) -> KDAProgramConfig:
    """Return the production K3 program configuration with caller-owned per-axis CCL topology."""
    # Fixed production/proxy geometries; native worker capacity is validated by
    # the recurrence constructor for the actual TP-local head count and device.
    group_chunks = {32: 1, 64: 2, 128: 4, 256: 8, 320: 10, 640: 20, 1280: 20, 2560: 20, 5120: 20}
    if active_seq_len_local not in group_chunks:
        raise ValueError(f"no tuned Kimi-K3 recurrence configuration for local T={active_seq_len_local}")
    # Galaxy SP8xTP4 at T=5120; other geometries keep the auto-selected projection configs and
    # DRAM staging (the largest staged activation, 640x9216 BF16, is ~98 KB per core in L1).
    galaxy = active_seq_len_local == 640
    input_projection, output_projection = _galaxy_projection_configs() if galaxy else (None, None)
    return KDAProgramConfig(
        # Scan policy is fixed at construction. Direct scan avoids summary overhead for shorter fixed
        # sequences; grouped scan trades P local scans of N/P chunks plus a log2(P) prefix for summary
        # overhead and requires batch_heads * P worker owners. K3 at T=5120 uses grouped scan.
        recurrence=KDARecurrenceProgramConfig(
            local_scan_strategy="grouped", summary_group_chunks=group_chunks[active_seq_len_local]
        ),
        qkv_channel_chunk_size=512,
        tp_ccl_topology=tp_ccl_topology,
        gated_rms_output_dtype=ttnn.bfloat16,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        input_projection_minimal_matmul_config=input_projection,
        output_projection_program_config=output_projection,
        stage_activations_in_l1=galaxy,
    )

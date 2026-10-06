# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-program configuration for Kimi Delta Attention."""

from __future__ import annotations

import math
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
    # HiFi3: the prefix matmul reads the complement-form transition E = A - I through SrcB, which keeps only 7
    # significant bits at HiFi2; that truncation measured 1.26% state error on long-memory channels (|G_last| = 1e-3
    # per chunk) over 10240 chunks. HiFi3 adds the SrcB low-bit phase and matches HiFi4 (worst class 0.42% both;
    # tt_metal_tracker-g1b.7, tt_metal_tracker-g1b.4.15, test_weak_decay_chain T5).
    affine_prefix_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi3
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
    # Use the projection matmul schedules tuned at _TUNED_PROJECTION_ROWS; False keeps the
    # auto-selected ttnn.linear configs.
    tuned_projection_matmuls: bool = False

    def __post_init__(self) -> None:
        if self.qkv_channel_chunk_size <= 0 or self.qkv_channel_chunk_size % ttnn.TILE_SIZE:
            raise ValueError(
                "qkv_channel_chunk_size must be a positive multiple of "
                f"{ttnn.TILE_SIZE}, got {self.qkv_channel_chunk_size}"
            )
        if self.gated_rms_output_dtype not in (ttnn.float32, ttnn.bfloat16):
            raise ValueError("gated_rms_output_dtype must be ttnn.float32 or ttnn.bfloat16")


# Rows per device the projection schedules were tuned at: Galaxy SP8xTP4 at T=5120.
_TUNED_PROJECTION_ROWS = 640


def tuned_projection_matmul_configs(
    grid: ttnn.CoreCoord, rows: int, output_k: int, output_n: int
) -> tuple[ttnn.MinimalMatmulConfig, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig]:
    """Return the tuned input and output projection schedules laid out on ``grid``.

    Tuned on the 12x10 Blackhole worker grid at 640 rows per device with the production numerics
    (bf16, FP32 destination accumulation); subblocks stay within the 4-tile FP32 destination limit.
    Raises when the blocking does not fit: a configuration that requests the tuned schedules never
    silently runs the auto-selected ttnn.linear configs instead.
    """
    row_tiles = rows // ttnn.TILE_SIZE
    per_core_m = math.ceil(row_tiles / grid.y)
    per_core_n = math.ceil(output_n // ttnn.TILE_SIZE / grid.x)
    if per_core_m % 2 or (output_k // ttnn.TILE_SIZE) % 8:
        raise ValueError(
            f"tuned KDA projection schedules do not fit {rows} rows, output projection K={output_k} "
            f"on a {grid.x}x{grid.y} grid; per-core M must be even and K a multiple of 8 tiles"
        )
    input_projection = ttnn.MinimalMatmulConfig(
        M_block_size=2,
        K_block_size=8,
        N_block_size=3,
        subblock_h=1,
        subblock_w=3,
        compute_with_storage_grid_size=grid,
    )
    output_projection = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=1,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )
    return input_projection, output_projection


def kimi_k3_program_config(*, active_seq_len_local: int, tp_ccl_topology: ttnn.Topology) -> KDAProgramConfig:
    """Return the production K3 program configuration with caller-owned per-axis CCL topology."""
    # Fixed production/proxy geometries; native worker capacity is validated by
    # the recurrence constructor for the actual TP-local head count and device.
    # 640 rows (Galaxy SP8xTP4 per chip, LoudBox LB-A/LB-B): two groups of 10. One group of 20 makes the
    # recurrent scan split V across four cores per head and re-read the V-independent inputs per block;
    # measured 0.17 ms faster per layer on LoudBox (tt_metal_tracker-g1b.5.15).
    group_chunks = {32: 1, 64: 2, 128: 4, 256: 8, 320: 10, 640: 10, 1280: 20, 2560: 20, 5120: 20}
    if active_seq_len_local not in group_chunks:
        raise ValueError(f"no tuned Kimi-K3 recurrence configuration for local T={active_seq_len_local}")
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
        # Galaxy SP8xTP4 at T=5120; other geometries keep the auto-selected projection configs.
        tuned_projection_matmuls=active_seq_len_local == _TUNED_PROJECTION_ROWS,
    )


def glm_5_3_flash_program_config(*, active_seq_len_local: int, tp_ccl_topology: ttnn.Topology) -> KDAProgramConfig:
    """Return the GLM-5.3-Flash program configuration with caller-owned per-axis CCL topology.

    Configured for the Galaxy per-chip geometry its LoudBox proxies run: 640 local rows with 16 heads
    per chip (LB-A 2x4 SP2xTP4 at T=1280, LB-B 8x1 SP8xTP1 at T=5120 on a quarter-head slice).
    Recurrence and numerics follow Kimi K3 at the same local length. The projection matmuls keep the
    auto-selected ttnn.linear configs: the tuned schedules were measured at K3's projection shapes only.
    """
    if active_seq_len_local != 640:
        raise ValueError(f"no tuned GLM-5.3-Flash recurrence configuration for local T={active_seq_len_local}")
    return KDAProgramConfig(
        # Two groups of 10 chunks, as Kimi K3 at 640 rows: 0.15 ms faster per layer than one group of 20 on
        # LoudBox (tt_metal_tracker-g1b.5.15).
        recurrence=KDARecurrenceProgramConfig(local_scan_strategy="grouped", summary_group_chunks=10),
        qkv_channel_chunk_size=512,
        tp_ccl_topology=tp_ccl_topology,
        gated_rms_output_dtype=ttnn.bfloat16,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        tuned_projection_matmuls=False,
    )

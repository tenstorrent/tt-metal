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
    input_projection_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
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
    grid: ttnn.CoreCoord,
    rows: int,
    output_k: int,
    output_n: int,
    input_projection_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
) -> tuple[ttnn.MinimalMatmulConfig | None, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None]:
    """Return the tuned input and output projection schedules laid out on ``grid``.

    Tuned on the 12x10 Blackhole worker grid at 640 rows per device with the production numerics
    (bf16, FP32 destination accumulation); subblocks stay within the 4-tile FP32 destination limit.
    Returns (None, None) when the blocking does not fit, keeping the auto-selected ttnn.linear configs.
    """
    row_tiles = rows // ttnn.TILE_SIZE
    per_core_m = math.ceil(row_tiles / grid.y)
    per_core_n = math.ceil(output_n // ttnn.TILE_SIZE / grid.x)
    if per_core_m % 2 or (output_k // ttnn.TILE_SIZE) % 8:
        return None, None
    # HiFi4 is compute-bound with short blocks. At HiFi2 the math halves, and longer K and N blocks
    # cut the per-block overhead that then dominates (K3 at 640 rows: 829 -> 654 us per device).
    if input_projection_math_fidelity == ttnn.MathFidelity.HiFi4:
        k_block, n_block = 8, 3
    else:
        k_block, n_block = 16, 12
    input_projection = ttnn.MinimalMatmulConfig(
        M_block_size=2,
        K_block_size=k_block,
        N_block_size=n_block,
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
    group_chunks = {32: 1, 64: 2, 128: 4, 256: 8, 320: 10, 640: 20, 1280: 20, 2560: 20, 5120: 20}
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
        input_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        # Galaxy SP8xTP4 at T=5120; other geometries keep the auto-selected projection configs.
        tuned_projection_matmuls=active_seq_len_local == _TUNED_PROJECTION_ROWS,
    )


def decay_projection_program_config(
    grid: ttnn.CoreCoord, rows: int, decay_rank: int, output_n: int, fused_activation: ttnn.UnaryWithParam | None
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None:
    """Return a 2D multicast schedule for the low-rank decay projection that applies ``fused_activation``.

    ttnn.linear's ``activation`` argument runs as a separate eltwise op under the auto-selected
    config; an explicit schedule fuses it into the matmul's pack. Returns None when the projection
    does not tile the grid, keeping the auto-selected config.
    """
    row_tiles = rows // ttnn.TILE_SIZE
    k_tiles = decay_rank // ttnn.TILE_SIZE
    n_tiles = output_n // ttnn.TILE_SIZE
    if row_tiles % grid.y or n_tiles % grid.x:
        return None
    per_core_m = row_tiles // grid.y
    per_core_n = n_tiles // grid.x
    # FP32 destination accumulation holds four tiles.
    subblock_h = 2 if per_core_m % 2 == 0 and per_core_n % 2 == 0 else 1
    subblock_w = next(width for width in (4 // subblock_h, 2, 1) if per_core_n % width == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=k_tiles,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=True,
    )

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
KDA_RECURRENT_STATE_DTYPE = ttnn.float32
KDA_AFFINE_SUMMARY_DTYPE = ttnn.bfloat16
KDA_SCAN_OUTPUT_DTYPE = ttnn.bfloat16
KDA_PREP_OUTPUT_BF16_MASK = (1 << 1) | (1 << 2) | (1 << 5)
# The L1 budgets below are static: they are sized on Galaxy SP8xTP4 against what is live next to these tensors
# there (the local summary and prefix buffers, the scans' circular buffers and the CCL semaphores), not measured at
# run time. They assume the device is opened with an L1_SMALL region (as the model's prefill tests do), so the CCL
# semaphores the first call allocates do not land between the chunk terms and fragment L1.
#
# The chunk terms stay in interleaved L1 for both scans when their share of each worker core fits this budget;
# larger geometries keep them in DRAM. Kimi-K3 on Galaxy SP8xTP4 (24 heads, 640 rows per device) needs 256 KiB per
# core; the rest of L1 holds the summary, prefix and scan buffers that run while the chunk terms are alive.
# Single-device and longer-sequence geometries exceed the budget and would clash with those buffers.
KDA_PREPARATION_L1_BYTES_PER_CORE = 320 * 1024
# That budget assumes the summary scan splits each head's value tiles across cores down to one per core, as on
# Galaxy SP8xTP4 with one summary group per head. More groups (or heads) leave fewer cores per head, and each extra
# value tile on a core adds this much to the summary's static circular buffers (recurrent_chunk_scan_program_factory:
# 54 FP32 and 10 BF16 tiles), which the chunk terms must then make room for.
KDA_SUMMARY_L1_BYTES_PER_VALUE_TILE = 54 * 4096 + 10 * 2048
# The decay gate stays in interleaved L1 from its projection until chunk preparation has read it when its share of
# each worker core fits this budget (32 KiB for Kimi-K3 on Galaxy SP8xTP4); larger geometries keep it in DRAM.
KDA_DECAY_L1_BYTES_PER_CORE = 48 * 1024
# The convolution's Q, K and V stay in interleaved L1 until chunk preparation reads them when each core's share fits
# this budget (96 KiB for Kimi-K3 on Galaxy SP8xTP4); larger geometries keep them in DRAM. LoudBox SP2xTP4 holds four
# times the rows per device (384 KiB per core), so it keeps them in DRAM, as before the fusion: its perf gate measures
# the DRAM placement, and only Galaxy's measures the L1 one.
KDA_QKV_L1_BYTES_PER_CORE = 112 * 1024
# The QKV convolution's fabric history exchange (qkv_causal_conv1d_silu_layout.hpp): a persistent buffer of
# predecessor, carry, outgoing and end rows, three rows each, and its entry, predecessor and carry semaphores.
KDA_HISTORY_EXCHANGE_ROWS = 12
KDA_HISTORY_EXCHANGE_SEMAPHORES = 3
# The gated norm's output stays in interleaved L1 until the output projection reads it when each core's share fits
# this budget (32 KiB for Kimi-K3 on Galaxy SP8xTP4); larger geometries keep it in DRAM.
KDA_NORM_OUTPUT_L1_BYTES_PER_CORE = 48 * 1024
# The derived input projection schedule writes its output to interleaved L1, where the layer reads it in place until
# the gated norm, when each core's share fits this budget (130 KiB for Kimi-K3 on Galaxy SP8xTP4 at T=5120, 260 KiB at
# T=10240); larger geometries write it to DRAM, leaving L1 to the recurrence's circular buffers.
KDA_INPUT_PROJECTION_L1_BYTES_PER_CORE = 272 * 1024
# The output projection's partial sums stay in interleaved L1 until the TP reduce-scatter reads them when each core's
# share fits this budget (75 KiB for Kimi-K3 on Galaxy SP8xTP4); larger geometries keep them in DRAM.
KDA_PARTIAL_OUTPUT_L1_BYTES_PER_CORE = 96 * 1024
# The gathered transforms and the chain's FP32 entry state and final carry stay in interleaved L1 when each core's share
# fits this budget (131 KiB for Kimi-K3 on Galaxy SP8xTP4); larger geometries keep them in DRAM.
KDA_DISTRIBUTED_PREFIX_L1_BYTES_PER_CORE = 160 * 1024
# Links high_bw_all_gather spreads the SP prefix summaries over; meshes with fewer on the SP axis keep ttnn.all_gather.
KDA_HIGH_BW_PREFIX_GATHER_LINKS = 2
KDA_LOCAL_PREFIX_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG
KDA_DISTRIBUTED_PREFIX_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG
KDA_DISTRIBUTED_WORKING_MEMORY_CONFIG = ttnn.L1_MEMORY_CONFIG
KDA_OUTPUT_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG


def l1_when_it_fits(device, nbytes: int, bytes_per_core: int) -> ttnn.MemoryConfig:
    """Interleaved L1 when ``nbytes`` spread over the worker grid fits ``bytes_per_core`` per core, else DRAM."""
    grid = device.compute_with_storage_grid_size()
    return ttnn.L1_MEMORY_CONFIG if nbytes <= bytes_per_core * grid.x * grid.y else ttnn.DRAM_MEMORY_CONFIG


def preparation_bytes(batch_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> int:
    """Bytes of chunk preparation's seven outputs, in the order and formats KDA_PREP_OUTPUT_BF16_MASK selects."""
    key_tiles, value_tiles = key_dim // ttnn.TILE_SIZE, value_dim // ttnn.TILE_SIZE
    # v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv tiles per chunk.
    tiles = (value_tiles, key_tiles, key_tiles, 1, key_tiles, key_tiles, 1)
    tile_bytes = sum(
        count * ttnn.TILE_SIZE * ttnn.TILE_SIZE * (2 if KDA_PREP_OUTPUT_BF16_MASK & (1 << index) else 4)
        for index, count in enumerate(tiles)
    )
    return batch_heads * num_chunks * tile_bytes


@dataclass(frozen=True)
class KDARecurrenceProgramConfig:
    """Tunable recurrence strategy and compute fidelity."""

    local_scan_strategy: Literal["direct", "grouped"] = "direct"
    # Exact number of chunks per group, or None to derive it from the geometry (auto_summary_group_chunks).
    # Construction validates divisibility and worker capacity; execution never silently changes an explicit
    # configuration.
    summary_group_chunks: int | None = None
    affine_prefix_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    scan_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    # Gather the SP prefix summaries with high_bw_all_gather into a persistent DRAM buffer. The recurrence takes it only
    # when the SP axis has KDA_HIGH_BW_PREFIX_GATHER_LINKS links; other meshes keep ttnn.all_gather.
    high_bw_prefix_gather: bool = False

    def __post_init__(self) -> None:
        if self.local_scan_strategy not in ("direct", "grouped"):
            raise ValueError("local_scan_strategy must be 'direct' or 'grouped'")
        if self.summary_group_chunks is not None and self.summary_group_chunks <= 0:
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
    # Weight dtype of the input (q, k, v, f_a, f_b, g, b) and output projections.
    projection_dtype: ttnn.DataType = ttnn.bfloat16
    # Derive the projection matmul schedules from the geometry (projection_matmul_config); False keeps the
    # auto-selected ttnn.linear configs.
    derived_projection_matmuls: bool = False

    def __post_init__(self) -> None:
        if self.qkv_channel_chunk_size <= 0 or self.qkv_channel_chunk_size % ttnn.TILE_SIZE:
            raise ValueError(
                "qkv_channel_chunk_size must be a positive multiple of "
                f"{ttnn.TILE_SIZE}, got {self.qkv_channel_chunk_size}"
            )
        if self.gated_rms_output_dtype not in (ttnn.float32, ttnn.bfloat16):
            raise ValueError("gated_rms_output_dtype must be ttnn.float32 or ttnn.bfloat16")


def scan_value_tiles_per_core(grid: ttnn.CoreCoord, batch_heads: int, value_tiles: int) -> int:
    """Value tiles each scan core keeps, as kda_factory_utils.cpp distribute_value_blocks splits each head: over the
    most cores (a divisor of its value tiles, within one grid row) that still leave a core group for every head."""
    split = next(
        (
            cores
            for cores in range(min(value_tiles, grid.x), 1, -1)
            if value_tiles % cores == 0 and batch_heads <= grid.x // cores * grid.y
        ),
        1,
    )
    return value_tiles // split


# The cross-group prefix that more than one group per head adds (a reduce, a gather and an exclusive scan), in scan
# chunk steps of one value tile: ~115 us against ~7.5 us per step on Galaxy SP8xTP4.
_GROUP_PREFIX_CHUNK_STEPS = 16


def auto_summary_group_chunks(grid: ttnn.CoreCoord, batch_heads: int, num_chunks: int, value_tiles: int) -> int:
    """Chunks per summary group that minimise the scans' sequential work.

    A core of the summary and recurrent scans walks every chunk of its group over the value tiles it keeps, so its
    work is (chunks per group) x (value tiles per core). More groups shorten the walk only while their extra owners
    find free workers to split heads' value tiles over, and cost the cross-group prefix; ties keep the fewest groups.
    Each group needs one worker per head.
    """
    capacity = grid.x * grid.y
    candidates = [
        groups for groups in range(1, num_chunks + 1) if num_chunks % groups == 0 and batch_heads * groups <= capacity
    ]
    if not candidates:
        raise ValueError(f"grouped KDA needs at least {batch_heads} summary owners, but only {capacity} are supported")
    groups = min(
        candidates,
        key=lambda g: (
            num_chunks // g * scan_value_tiles_per_core(grid, batch_heads * g, value_tiles)
            + (_GROUP_PREFIX_CHUNK_STEPS if g > 1 else 0),
            g,
        ),
    )
    return num_chunks // groups


# Per-core L1 a derived projection schedule may take for its circular buffers, sized on Galaxy SP8xTP4 against the
# layer's other L1 tensors: with the device's CCL semaphores fragmenting L1, a 1192 KiB output projection clashed.
KDA_PROJECTION_CB_L1_BYTES_PER_CORE = 1170 * 1024
# A projection's circular buffers and the input projection's L1 output share each core's L1. Buffers allocated while
# that output is alive sit below it, so it bounds the output projection too, after it is freed. On Galaxy SP8xTP4
# without an L1_SMALL region, 1369 KiB (input) and 1366 KiB (output) ran at 1152 rows; 1395 KiB and 1392 KiB clashed at
# 1280 rows.
KDA_PROJECTION_TOTAL_L1_BYTES_PER_CORE = 1360 * 1024
# FP32 destination accumulation holds four tiles.
_FP32_DEST_TILES = 4
# Narrower K blocks double the per-block overhead without saving bandwidth.
_MIN_PROJECTION_K_BLOCK = 2
# Projection cost model, measured on one Blackhole Galaxy chip (bfloat8_b weights, HiFi2, FP32 accumulation) over 766
# schedules of the Kimi-K3 input and output projections at 128 to 1408 rows. A schedule is bound by its most loaded
# core's tile matmuls, by its weight stream (read once per pass over a core's rows), or, in 1D, by multicasting every
# row to every core. Who reads the weights sets the stream's rate: the 2D schedule's sender row of 12 cores, the
# transposed 2D schedule's sender column of 10 cores (2.4x slower), or every core in 1D.
_PROJECTION_US_PER_TILE_MATMUL = 27.8e-3
_PROJECTION_WEIGHT_BYTES_PER_US = {"2d": 255e3, "2d_transposed": 108e3, "1d": 330e3}
# 1D activation multicast rate, by whether the activation is in L1.
_PROJECTION_ACTIVATION_BYTES_PER_US = {True: 24e3, False: 21e3}
# A K block of w tiles adds about this / w^2 to each tile matmul (10% at two tiles).
_PROJECTION_K_BLOCK_OVERHEAD = 0.4
# Each pass after the first exposes about this fraction of its weight stream.
_PROJECTION_EXPOSED_PASS_FRACTION = 0.1


# Measured-best schedules of the Kimi-K3 Galaxy SP8xTP4 projections (bfloat8_b weights, 12 x 10 grid) at every
# prefill chunk length, from the 766-schedule sweep plus every K block of the cost model's picks. Keyed by
# (row tiles, K tiles, N tiles); values are (layout, per_core_M, out_block_h, in0_block_w). The cost model picks the
# same layout at every length but misjudges some K blocks, by up to 42 us.
_MEASURED_PROJECTION_SCHEDULES = {
    # Input projection, K = 7168, N = 12440.
    (4, 224, 389): ("1d", 4, 4, 2),  # T=1k, 258 us
    (8, 224, 389): ("1d", 8, 8, 2),  # T=2k, 273 us
    (12, 224, 389): ("1d", 12, 12, 7),  # T=3k, 306 us
    (16, 224, 389): ("1d", 16, 16, 8),  # T=4k, 400 us
    (20, 224, 389): ("2d", 2, 2, 8),  # T=5k, 413 us
    (24, 224, 389): ("1d", 24, 24, 4),  # T=6k, 605 us
    (28, 224, 389): ("2d", 3, 3, 7),  # T=7k, 615 us
    (32, 224, 389): ("1d", 32, 16, 8),  # T=8k, 797 us
    (36, 224, 389): ("2d", 4, 2, 8),  # T=9k, 824 us
    (40, 224, 389): ("2d", 4, 2, 8),  # T=10k, 824 us
    (44, 224, 389): ("2d_transposed", 4, 4, 2),  # T=11k, 1157 us
    # Output projection, K = 3072, N = 7168.
    (4, 96, 224): ("1d", 4, 4, 3),  # T=1k, 70 us
    (8, 96, 224): ("1d", 8, 8, 8),  # T=2k, 76 us
    (12, 96, 224): ("1d", 12, 12, 12),  # T=3k, 96 us
    (16, 96, 224): ("2d", 2, 2, 6),  # T=4k, 112 us
    (20, 96, 224): ("2d", 2, 2, 6),  # T=5k, 112 us
    (24, 96, 224): ("2d", 3, 3, 6),  # T=6k, 161 us
    (28, 96, 224): ("2d", 3, 3, 12),  # T=7k, 190 us
    (32, 96, 224): ("2d", 4, 4, 12),  # T=8k, 245 us
    (36, 96, 224): ("2d", 4, 4, 8),  # T=9k, 256 us
    (40, 96, 224): ("2d", 4, 4, 8),  # T=10k, 260 us
    (44, 96, 224): ("2d_transposed", 4, 4, 8),  # T=11k, 313 us
}


def _divisors(value: int, limit: int | None = None) -> list[int]:
    """Divisors of ``value`` up to ``limit``, largest first."""
    return [d for d in range(min(value, limit or value), 0, -1) if value % d == 0]


def _tile_bytes(dtype: ttnn.DataType) -> int:
    return ttnn.Tile((ttnn.TILE_SIZE, ttnn.TILE_SIZE)).get_tile_size(dtype)


def _projection_cb_bytes(block_h: int, k_block: int, per_core_n: int, weight_dtype: ttnn.DataType) -> int:
    """Circular-buffer bytes of a multicast projection: double-buffered bf16 activation and ``weight_dtype`` weight
    blocks, the bf16 output block and its FP32 accumulation block."""
    activation_tile, weight_tile = _tile_bytes(ttnn.bfloat16), _tile_bytes(weight_dtype)
    return (
        2 * block_h * k_block * activation_tile
        + 2 * k_block * per_core_n * weight_tile
        + block_h * per_core_n * (activation_tile + _tile_bytes(ttnn.float32))
    )


def _projection_cost_us(
    layout: str,
    per_core_m: int,
    per_core_n: int,
    block_h: int,
    k_block: int,
    k_tiles: int,
    weight_bytes: int,
    activation_bytes: int,
    activation_in_l1: bool,
) -> float:
    """Estimated device time of one projection schedule (see _PROJECTION_US_PER_TILE_MATMUL)."""
    passes = per_core_m // block_h
    compute = (
        _PROJECTION_US_PER_TILE_MATMUL
        * per_core_m
        * per_core_n
        * k_tiles
        * (1 + _PROJECTION_K_BLOCK_OVERHEAD / k_block**2)
    )
    weight_pass = weight_bytes / _PROJECTION_WEIGHT_BYTES_PER_US[layout]
    activation = activation_bytes / _PROJECTION_ACTIVATION_BYTES_PER_US[activation_in_l1] if layout == "1d" else 0.0
    bound = max(compute, passes * weight_pass, activation)
    # Each pass's first K block fills the pipeline before compute overlaps the stream.
    return bound * (1 + k_block / (k_tiles * passes)) + _PROJECTION_EXPOSED_PASS_FRACTION * (passes - 1) * weight_pass


def _projection_column_cores(grid: ttnn.CoreCoord, layout: str) -> int:
    """Cores that split a projection's output columns under ``layout``."""
    return {"1d": grid.x * grid.y, "2d": grid.x, "2d_transposed": grid.y}[layout]


def estimated_projection_schedule(
    grid: ttnn.CoreCoord,
    row_tiles: int,
    k_tiles: int,
    n_tiles: int,
    weight_dtype: ttnn.DataType,
    cb_bytes_per_core: int,
    activation_in_l1: bool,
) -> tuple[str, int, int, int, int] | None:
    """The (layout, per_core_M, per_core_N, out_block_h, in0_block_w) with the lowest _projection_cost_us, or None.

    Candidates are the 2D schedule (rows over the grid's rows, output columns over its columns), the transposed 2D
    schedule (rows over the columns) and the 1D schedule (every core holds all rows and its own output columns). Each
    takes the fewest row tiles per core or one more, a pass height that divides them, and any K block whose circular
    buffers fit ``cb_bytes_per_core``.
    """
    k_blocks = [w for w in _divisors(k_tiles) if w >= min(_MIN_PROJECTION_K_BLOCK, k_tiles)]
    weight_bytes = k_tiles * n_tiles * _tile_bytes(weight_dtype)
    activation_bytes = row_tiles * k_tiles * _tile_bytes(ttnn.bfloat16)
    shapes = [("1d", row_tiles, math.ceil(n_tiles / _projection_column_cores(grid, "1d")))]
    for layout, row_cores in (("2d", grid.y), ("2d_transposed", grid.x)):
        fewest = math.ceil(row_tiles / row_cores)
        per_core_n = math.ceil(n_tiles / _projection_column_cores(grid, layout))
        shapes += [(layout, per_core_m, per_core_n) for per_core_m in range(fewest, min(row_tiles, fewest + 1) + 1)]
    candidates = [
        (
            _projection_cost_us(
                layout,
                per_core_m,
                per_core_n,
                block_h,
                k_block,
                k_tiles,
                weight_bytes,
                activation_bytes,
                activation_in_l1,
            ),
            (layout, per_core_m, per_core_n, block_h, k_block),
        )
        for layout, per_core_m, per_core_n in shapes
        for block_h in _divisors(per_core_m)
        for k_block in k_blocks
        if _projection_cb_bytes(block_h, k_block, per_core_n, weight_dtype) <= cb_bytes_per_core
    ]
    return min(candidates)[1] if candidates else None


def projection_matmul_config(
    grid: ttnn.CoreCoord,
    rows: int,
    k: int,
    n: int,
    weight_dtype: ttnn.DataType,
    cb_bytes_per_core: int = KDA_PROJECTION_CB_L1_BYTES_PER_CORE,
    activation_in_l1: bool = False,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None:
    """Return the fastest multicast schedule for a [rows, k] x [k, n] projection on ``grid``, or None for auto.

    Measured geometries (_MEASURED_PROJECTION_SCHEDULES) take their measured-best schedule when its circular buffers
    fit ``cb_bytes_per_core``; any other takes estimated_projection_schedule. None only when nothing fits.
    """
    tile = ttnn.TILE_SIZE
    row_tiles, k_tiles, n_tiles = rows // tile, math.ceil(k / tile), math.ceil(n / tile)
    measured = _MEASURED_PROJECTION_SCHEDULES.get((row_tiles, k_tiles, n_tiles))
    if measured is not None and (grid.x, grid.y) == (12, 10) and weight_dtype == ttnn.bfloat8_b:
        layout, per_core_m, block_h, k_block = measured
        per_core_n = math.ceil(n_tiles / _projection_column_cores(grid, layout))
        if _projection_cb_bytes(block_h, k_block, per_core_n, weight_dtype) <= cb_bytes_per_core:
            return _projection_program_config(grid, layout, per_core_m, per_core_n, block_h, k_block)
    schedule = estimated_projection_schedule(
        grid, row_tiles, k_tiles, n_tiles, weight_dtype, cb_bytes_per_core, activation_in_l1
    )
    return None if schedule is None else _projection_program_config(grid, *schedule)


def _projection_program_config(
    grid: ttnn.CoreCoord, layout: str, per_core_m: int, per_core_n: int, block_h: int, k_block: int
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
    subblock_w = _divisors(per_core_n, _FP32_DEST_TILES)[0]
    subblock_h = _divisors(block_h, _FP32_DEST_TILES // subblock_w)[0]
    if layout == "1d":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=k_block,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            out_block_h=block_h,
            out_block_w=per_core_n,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=k_block,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        out_block_h=block_h,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=layout == "2d_transposed",
        fused_activation=None,
        fuse_batch=True,
    )


# tt-blaze decode keeps the K3 input projections in bfloat4_b after layer 0, which costs ~0.025
# layer-output PCC; bfloat8_b matches decode's layer 0 and output projection.
KIMI_K3_PROJECTION_DTYPE = ttnn.bfloat8_b


def kimi_k3_program_config(*, tp_ccl_topology: ttnn.Topology) -> KDAProgramConfig:
    """Return the production K3 program configuration with caller-owned per-axis CCL topology.

    Nothing in it depends on the sequence length: the summary groups and the projection schedules are derived from
    each layer's geometry at construction.
    """
    return KDAProgramConfig(
        # Scan policy is fixed at construction. Direct scan avoids summary overhead for shorter fixed
        # sequences; grouped scan trades P local scans of N/P chunks plus a log2(P) prefix for summary
        # overhead and requires batch_heads * P worker owners. K3 uses grouped scan.
        recurrence=KDARecurrenceProgramConfig(
            local_scan_strategy="grouped",
            # Galaxy SP8xTP4: 136 us instead of 184 us for the 1.6 MB-per-chip summary gather, whose size does not
            # depend on the sequence length. Taken only with one group per head and enough SP links.
            high_bw_prefix_gather=True,
        ),
        qkv_channel_chunk_size=64,
        tp_ccl_topology=tp_ccl_topology,
        gated_rms_output_dtype=ttnn.bfloat16,
        input_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        projection_dtype=KIMI_K3_PROJECTION_DTYPE,
        derived_projection_matmuls=True,
    )


_DECAY_OUT_BLOCK_TILES = 32
# The decay projection is bound by its fused sigmoid, so its time follows the output tiles per core; each extra output
# block costs about as much as this many more tiles (Blackhole Galaxy, 128 to 1408 rows).
_DECAY_TILES_PER_EXTRA_BLOCK = 3.5


def decay_projection_program_config(
    grid: ttnn.CoreCoord, rows: int, key_dim: int, width: int, fused_activation: ttnn.UnaryWithParam | None
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Spread the decay projection's [rows, key_dim] x [key_dim, width] output over the whole worker grid.

    Rows go over the grid's rows, or over its columns (transposed multicast) when that leaves fewer output tiles per
    core. Each core's output is computed in blocks of at most _DECAY_OUT_BLOCK_TILES tiles, which bounds the output and
    FP32 intermediate circular buffers on geometries with many rows or heads per core (one Kimi-K3 Galaxy SP8xTP4
    core holds a single 2 x 8 block).
    """
    tile = ttnn.TILE_SIZE
    row_tiles, width_tiles = rows // tile, width // tile
    best = None
    for transpose, row_cores, column_cores in ((False, grid.y, grid.x), (True, grid.x, grid.y)):
        per_core_M = math.ceil(row_tiles / row_cores)
        per_core_N = math.ceil(width_tiles / column_cores)
        out_block_w = max(w for w in range(1, min(per_core_N, _DECAY_OUT_BLOCK_TILES) + 1) if per_core_N % w == 0)
        out_block_h = max(
            h for h in range(1, per_core_M + 1) if per_core_M % h == 0 and h * out_block_w <= _DECAY_OUT_BLOCK_TILES
        )
        blocks = (per_core_M // out_block_h) * (per_core_N // out_block_w)
        cost = per_core_M * per_core_N + _DECAY_TILES_PER_EXTRA_BLOCK * (blocks - 1)
        if best is None or cost < best[0]:
            best = (cost, transpose, per_core_M, per_core_N, out_block_h, out_block_w)
    _, transpose, per_core_M, per_core_N, out_block_h, out_block_w = best
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=key_dim // tile,
        out_subblock_h=1,
        # FP32 accumulation holds four tiles in DST.
        out_subblock_w=max(w for w in range(1, 5) if out_block_w % w == 0),
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=transpose,
        fused_activation=fused_activation,
        fuse_batch=True,
    )

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
# With the entry correction, the summary's emitted output terms also live in L1, and its circular buffers double on a
# grid that splits each head's values over 2 cores instead of 4 (11x10): the chunk terms then stay in L1 only up to
# Galaxy SP8xTP4's 256 KiB per core on its 12x10 grid.
KDA_PREPARATION_WITH_ENTRY_CORRECTION_L1_BYTES_PER_CORE = 256 * 1024
# The fused input projection's output stays in interleaved L1 until the layer has read its last columns (the gated norm)
# when each core's share fits this budget (120 KiB for Kimi-K3 on Galaxy SP8xTP4); larger geometries keep it in DRAM.
KDA_INPUT_PROJECTION_L1_BYTES_PER_CORE = 128 * 1024
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
# The output projection's partial sums stay in interleaved L1 until the TP reduce-scatter reads them when each core's
# share fits this budget (75 KiB for Kimi-K3 on Galaxy SP8xTP4); larger geometries keep them in DRAM.
KDA_PARTIAL_OUTPUT_L1_BYTES_PER_CORE = 96 * 1024
# The gathered transforms and the chain's FP32 entry state and final carry stay in interleaved L1 when each core's share
# fits this budget (131 KiB for Kimi-K3 on Galaxy SP8xTP4); larger geometries keep them in DRAM.
KDA_DISTRIBUTED_PREFIX_L1_BYTES_PER_CORE = 160 * 1024
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
    # Exact number of chunks per group. Construction validates divisibility and
    # worker capacity; execution never silently changes an explicit configuration.
    summary_group_chunks: int = 20
    affine_prefix_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    scan_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    # With sequence parallelism and one group per head, the SP prefix is a relay along the SP axis (RelayAffinePrefix):
    # each rank applies its own summary to the entry state it receives and forwards the result. The summary also emits
    # each chunk's output terms, and an EntryCorrection program forms the chunk outputs and the final state from the
    # entry states, all chunks in parallel. Cores the entry correction may use: its products count toward the Blackhole
    # di/dt bound (as KDAProgramConfig.max_matmul_cores).
    entry_correction_max_cores: int = 110

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
    # Weight dtype of the input (q, k, v, f_a, f_b, g, b) and output projections.
    projection_dtype: ttnn.DataType = ttnn.bfloat16
    # At most this many cores run projection matmuls at once. The projections take the widest full-height block of
    # worker columns within it (the whole 12 x 10 Blackhole grid at 120, as main's tuned schedules).
    max_matmul_cores: int = 120
    # K-block tiles of the projections (block_projection_program_config; at most, where L1 needs smaller blocks).
    input_projection_k_block: int = 4
    output_projection_k_block: int = 4

    def __post_init__(self) -> None:
        if self.qkv_channel_chunk_size <= 0 or self.qkv_channel_chunk_size % ttnn.TILE_SIZE:
            raise ValueError(
                "qkv_channel_chunk_size must be a positive multiple of "
                f"{ttnn.TILE_SIZE}, got {self.qkv_channel_chunk_size}"
            )
        if self.gated_rms_output_dtype not in (ttnn.float32, ttnn.bfloat16):
            raise ValueError("gated_rms_output_dtype must be ttnn.float32 or ttnn.bfloat16")
        if self.max_matmul_cores <= 0:
            raise ValueError("max_matmul_cores must be positive")


def matmul_grid(grid: ttnn.CoreCoord, max_cores: int) -> ttnn.CoreCoord:
    """The widest block of full-height worker columns of ``grid`` with at most ``max_cores`` cores."""
    if max_cores < grid.y:
        raise ValueError(
            f"max_matmul_cores {max_cores} cannot hold one full-height worker column of the {grid.x}x{grid.y} grid"
        )
    return ttnn.CoreCoord(min(grid.x, max_cores // grid.y), grid.y)


# L1 a projection's circular buffers may take per core: the double-buffered in0 and in1 blocks, the output block and its
# FP32 intermediate. Kimi-K3's input projection on Galaxy SP8xTP4 takes 0.98 MiB (2 x 32 output tiles, 8-tile K blocks of
# bfloat8_b weights) next to the layer's other L1 tensors.
KDA_PROJECTION_CB_L1_BYTES_PER_CORE = 1024 * 1024
_FP32_TILE_BYTES = ttnn.TILE_SIZE * ttnn.TILE_SIZE * 4


def _projection_cb_bytes(out_h: int, out_w: int, k_block: int, in0_tile: int, in1_tile: int, out_tile: int) -> int:
    return 2 * k_block * (out_h * in0_tile + out_w * in1_tile) + out_h * out_w * (out_tile + _FP32_TILE_BYTES)


def block_projection_program_config(
    columns: int,
    rows: int,
    m_tiles: int,
    n_tiles: int,
    k_block: int,
    *,
    k_tiles: int | None = None,
    weight_dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """ttnn.linear's 2D multicast schedule for an [m_tiles, K] @ [K, n_tiles] projection on the worker block of the
    first ``columns`` columns and ``rows`` rows: one output block per core, K in ``k_block``-tile blocks, and the
    largest (then widest) output subblock within the four FP32 destination tiles.

    The KDA projections all take this schedule. Where the blocks would exceed KDA_PROJECTION_CB_L1_BYTES_PER_CORE (many
    rows or heads per device), K blocks shrink to 2 tiles first, then each core's output is computed in narrower, then
    shorter blocks.
    """
    per_core_m = -(-m_tiles // rows)
    per_core_n = -(-n_tiles // columns)
    if k_tiles is not None:
        k_block = max(d for d in range(1, min(k_block, k_tiles) + 1) if k_tiles % d == 0)
    tile = ttnn.TILE_SIZE * ttnn.TILE_SIZE
    in1_tile = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}.get(weight_dtype, 2 * tile)
    fits = lambda h, w, k: _projection_cb_bytes(h, w, k, 2 * tile, in1_tile, 2 * tile) <= (
        KDA_PROJECTION_CB_L1_BYTES_PER_CORE
    )
    while not fits(per_core_m, per_core_n, k_block) and k_block > 2 and k_block % 2 == 0:
        k_block //= 2
    divisors = lambda n: [d for d in range(n, 0, -1) if n % d == 0]
    out_block_w = next((w for w in divisors(per_core_n) if fits(per_core_m, w, k_block)), 1)
    out_block_h = next((h for h in divisors(per_core_m) if fits(h, out_block_w, k_block)), 1)
    subblock_h, subblock_w = max(
        (
            (h, w)
            for h in (1, 2, 4)
            for w in (4, 3, 2, 1)
            if h * w <= 4 and out_block_h % h == 0 and out_block_w % w == 0
        ),
        key=lambda hw: (hw[0] * hw[1], hw[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(columns, rows),
        in0_block_w=k_block,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


# tt-blaze decode keeps the K3 input projections in bfloat4_b after layer 0, which costs ~0.025
# layer-output PCC; bfloat8_b matches decode's layer 0 and output projection.
KIMI_K3_PROJECTION_DTYPE = ttnn.bfloat8_b


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
            local_scan_strategy="grouped",
            summary_group_chunks=group_chunks[active_seq_len_local],
        ),
        qkv_channel_chunk_size=64,
        tp_ccl_topology=tp_ccl_topology,
        gated_rms_output_dtype=ttnn.bfloat16,
        input_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        projection_dtype=KIMI_K3_PROJECTION_DTYPE,
        # 8-tile K blocks for the bfloat8_b weights.
        input_projection_k_block=8,
        output_projection_k_block=8,
    )


_DECAY_OUT_BLOCK_TILES = 32


def decay_projection_program_config(
    grid: ttnn.CoreCoord, rows: int, key_dim: int, width: int, fused_activation: ttnn.UnaryWithParam | None
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Spread the decay projection's [rows, key_dim] x [key_dim, width] output over the whole worker grid.

    Each core's output is computed in blocks of at most _DECAY_OUT_BLOCK_TILES tiles, which bounds the output and
    FP32 intermediate circular buffers on geometries with many rows or heads per core (one Kimi-K3 Galaxy SP8xTP4
    core holds a single 2 x 8 block).
    """
    tile = ttnn.TILE_SIZE
    per_core_M = math.ceil(rows // tile / grid.y)
    per_core_N = math.ceil(width // tile / grid.x)
    out_block_w = max(w for w in range(1, min(per_core_N, _DECAY_OUT_BLOCK_TILES) + 1) if per_core_N % w == 0)
    out_block_h = max(
        h for h in range(1, per_core_M + 1) if per_core_M % h == 0 and h * out_block_w <= _DECAY_OUT_BLOCK_TILES
    )
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
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=True,
    )

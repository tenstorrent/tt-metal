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
    # Exact number of chunks per group. Construction validates divisibility and
    # worker capacity; execution never silently changes an explicit configuration.
    summary_group_chunks: int = 20
    affine_prefix_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    scan_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    # Gather the SP prefix summaries with high_bw_all_gather into a persistent DRAM buffer. The recurrence takes it only
    # when the SP axis has KDA_HIGH_BW_PREFIX_GATHER_LINKS links; other meshes keep ttnn.all_gather.
    high_bw_prefix_gather: bool = False
    # Compose the SP prefix by relaying each head's state around the sequence-parallel ring (RelayAffinePrefix) instead
    # of gathering every summary and chaining them. Taken only when the fabric wraps the SP axis.
    relay_affine_prefix: bool = False

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
    # Use the projection matmul schedules tuned at _TUNED_PROJECTION_ROWS; False keeps the
    # auto-selected ttnn.linear configs.
    tuned_projection_matmuls: bool = False
    # Compute the output gate's projection columns in their own matmul (ttnn.linear's 2D multicast matmul) on the worker
    # cores the SP relay leaves idle (the gate is only read by the gated norm). Taken only with the ring relay.
    overlap_output_gate: bool = False
    # Blackhole di/dt: at most this many cores run projection matmuls at once. The projections take the widest
    # full-height block of worker columns within it (11 x 10 of the 12 x 10 grid at 110).
    max_matmul_cores: int = 110
    # K-block tiles of the worker-block projections (block_projection_program_config).
    input_projection_k_block: int = 4
    output_projection_k_block: int = 4
    output_gate_k_block: int = 8
    # Output-gate tile columns per core the tuned input projection takes past its own width: they shorten the gate
    # matmul beside the SP relay, whose window is on the critical path.
    input_projection_extra_gate_tiles: int = 2

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
    return ttnn.CoreCoord(max(1, min(grid.x, max_cores // grid.y)), grid.y)


# Rows per device the projection schedules were tuned at: Galaxy SP8xTP4 at T=5120.
_TUNED_PROJECTION_ROWS = 640


def block_projection_program_config(
    columns: int, rows: int, m_tiles: int, n_tiles: int, k_block: int
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """ttnn.linear's 2D multicast schedule for an [m_tiles, K] @ [K, n_tiles] projection on the worker block of the
    first ``columns`` columns and ``rows`` rows: one output block per core, K in ``k_block``-tile blocks, and the
    largest (then widest) output subblock within the four FP32 destination tiles."""
    per_core_m = -(-m_tiles // rows)
    per_core_n = -(-n_tiles // columns)
    subblock_h, subblock_w = max(
        ((h, w) for h in (1, 2, 4) for w in (4, 3, 2, 1) if h * w <= 4 and per_core_m % h == 0 and per_core_n % w == 0),
        key=lambda hw: (hw[0] * hw[1], hw[1]),
    )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(columns, rows),
        in0_block_w=k_block,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def tuned_projection_matmul_configs(
    grid: ttnn.CoreCoord,
    rows: int,
    input_n: int,
    output_k: int,
    output_n: int,
) -> tuple[
    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None
]:
    """Return the tuned input and output projection schedules laid out on ``grid``.

    Tuned on the 12x10 Blackhole worker grid at 640 rows per device with the production numerics
    (bf16, FP32 destination accumulation); subblocks stay within the 4-tile FP32 destination limit.
    Returns (None, None) when the blocking does not fit, keeping the auto-selected ttnn.linear configs.
    """
    row_tiles = rows // ttnn.TILE_SIZE
    per_core_m = math.ceil(row_tiles / grid.y)
    per_core_n = math.ceil(output_n // ttnn.TILE_SIZE / grid.x)
    input_per_core_n = math.ceil(math.ceil(input_n / ttnn.TILE_SIZE) / grid.x)
    if per_core_m % 2 or (output_k // ttnn.TILE_SIZE) % 8 or (output_n // ttnn.TILE_SIZE) % 4:
        return None, None
    # 1x3 output subblocks when the per-core width allows, else 2x2 over a width padded to even (the input projection
    # without the output gate's columns: 293 tiles, 26 per core).
    input_subblock = (1, 3) if input_per_core_n % 3 == 0 else (2, 2)
    if input_subblock == (2, 2):
        input_per_core_n += input_per_core_n % 2
    # One output block per core: each core keeps its whole [per_core_m, input_per_core_n] block resident and
    # streams K in 4-tile blocks (K3 at 640 rows: 2 x 33 tiles per core, 56 K blocks), the widest that fits in
    # L1 next to the L1 output with bf16 weights. At HiFi2 the bf16 weight stream from DRAM bounds it.
    input_projection = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=4,
        out_subblock_h=input_subblock[0],
        out_subblock_w=input_subblock[1],
        out_block_h=per_core_m,
        out_block_w=input_per_core_n,
        per_core_M=per_core_m,
        per_core_N=input_per_core_n,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
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
            # Galaxy SP8xTP4 at T=5120: 136 us instead of 184 us for the 1.6 MB-per-chip summary gather.
            high_bw_prefix_gather=active_seq_len_local == _TUNED_PROJECTION_ROWS,
            relay_affine_prefix=True,
        ),
        qkv_channel_chunk_size=64,
        tp_ccl_topology=tp_ccl_topology,
        gated_rms_output_dtype=ttnn.bfloat16,
        input_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
        projection_dtype=KIMI_K3_PROJECTION_DTYPE,
        # Galaxy SP8xTP4 at T=5120; other geometries keep the auto-selected projection configs.
        tuned_projection_matmuls=active_seq_len_local == _TUNED_PROJECTION_ROWS,
        # Tuned at T=5120 on the 110-core matmul grid with bfloat8_b weights: 8-tile K blocks, and the input
        # projection takes 5 gate tile columns per core past its width, 32 tiles per core (1 x 4 subblocks; 29 or
        # 31 leave only 2 x 1), so the gate beside the relay gets 37 tiles on 8 columns.
        input_projection_k_block=8,
        output_projection_k_block=8,
        input_projection_extra_gate_tiles=5,
        # The output gate beside the relay (and the projections on the matmul program) also at T=1024, 2048 and
        # 2560: 0.752 -> 0.740 ms, 0.913 -> 0.839 ms and 1.021 -> 0.871 ms.
        overlap_output_gate=active_seq_len_local in (128, 256, 320, _TUNED_PROJECTION_ROWS),
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

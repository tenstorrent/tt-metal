# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``final_mixer``: ``Qwen38TTNNFinalMixer.__call__`` (21 programs: rms_norm_pre, reshape, all_gather, rms_norm_post,
gamma multiply, down linear, S2I, the fp32 all_reduce composite's 7 programs, typecast, silu, up linear, S2I, sigmoid,
gate multiply, fast_reduce_nc, reshape) on F2's gr_read programs: ``stats`` and ``normalize`` as they are (the mixer
is the GR read pattern without the injection), ``down`` and ``gate`` with the mixer's shapes (K = 2560 -> 320 tiles
10, spill 8; K = 320 -> 2560, spill 5), and a new ``low_rank`` that sums the four gathered fp32 partials the way the
chain's composite does: ``local_sum_float32`` reshapes the gathered rows to a leading device dim and ``ttnn::sum``
transposes that dim into H (zero pad) and reduces with ReduceOpDim::H; for fp32 with the reduce's default
fp32_dest_acc_en that is the ACCURATE SFPU path (input unpacked to the fp32 dest, rows folded by the fp32 SFPU;
``reduce_op.cpp`` fp32_sfpu_eligible), which the kernel calls through the same ``compute_kernel_lib::reduce`` helper
on a tile whose rows 0..3 are the four partial rows; then the chain's typecast (RNE in the dest) and bf16 silu.  The two collectives stay (``all_gather`` of the stats, ``all_gather`` of
the partials along a new dim 0 = the composite's own all_broadcast + concat bytes).  Split form: 5 programs + 2
collectives.  Merged form (the default, ``QWEN38_FUSED_FINAL_MIXER_MERGED=0`` selects the split form): F2's two
multicast programs with the mixer's shapes -- ``normalize_down`` (the four norm cores multicast the normalized row
into the ten down cores' CB while those prefetch their weight column) and ``low_rank_gate`` (five row cores fold two
stacked tiles each and multicast the low-rank row into the twenty gate cores' CB while those prefetch their weight
columns and normalized tiles); 3 programs + 2 collectives.  The tolerance class is BITWISE: every op is the chain's
op or its LLK sequence.  Rows = 1 (the decode step)."""

from __future__ import annotations

import warnings

import torch

import ttnn

from .. import gr_read as gr
from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "final_mixer"
TP_SIZE, TP_AXIS, BRANCHES = gr.TP_SIZE, gr.TP_AXIS, gr.BRANCHES
LOCAL_HIDDEN, FLAT_WIDTH, FLAT_TILES, HIDDEN_TILES = gr.LOCAL_HIDDEN, gr.FLAT_WIDTH, gr.FLAT_TILES, gr.HIDDEN_TILES
RANK = 320
RANK_TILES = RANK // fp.TILE  # 10
DOWN_SPILL = 8  # dram_sharded_matmul_configs(K=2560, N=320, num_cores=5): 16 K tiles per core -> in0_block_w 8 (gr_read's down too)
UP_SPILL = (
    5  # dram_sharded_matmul_configs(K=320, N=2560, num_cores=2): 5 K tiles per core -> in0_block_w 5 (gr_read's up: 6)
)
BF16, FP32, TILE_BF16, TILE_FP32 = gr.BF16, gr.FP32, gr.TILE_BF16, gr.TILE_FP32
LOW_RANK_CORES = 5  # the merged low_rank_gate's row cores: RANK_TILES / 2 stacked tiles each (one 5-wide producer row)
LOW_RANK_TILES_PER_CORE = RANK_TILES // LOW_RANK_CORES  # 2
DOWN_GRID = (5, 2)  # the merged normalize_down's consumers: one down column tile each (RANK_TILES = 10)
GATE_GRID = (5, 4)  # the merged low_rank_gate's consumers: one output column tile each (HIDDEN_TILES = 20), F2's grid
MERGED_ENV = "QWEN38_FUSED_FINAL_MIXER_MERGED"  # "0": the split form (normalize, down, low_rank, gate as four programs)
LOWRANK_COMPUTE = fp.kernel_source(NAME, "lowrank_reduce_compute.cpp")
LOWRANK_READER = fp.kernel_source(NAME, "lowrank_reduce_reader.cpp")
LOWRANK_WRITER = fp.kernel_source(NAME, "lowrank_reduce_writer.cpp")
_ZERO_FP32: dict[int, object] = {}
_NORM_ROWS: dict[int, object] = {}


def _mixer_module():
    from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer

    return final_mixer


def zero_tile(mesh):
    """One fp32 zero tile per mesh (the stacked tiles' padding rows), uploaded once."""

    key = id(mesh)
    if key not in _ZERO_FP32:
        _ZERO_FP32[key] = ttnn.from_torch(
            torch.zeros(1, 1, fp.TILE, fp.TILE),
            dtype=FP32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return _ZERO_FP32[key]


def norm_scale_rows(mesh, norm_scale):
    """The mixer's fp32 gamma/4 ``[1, 4, 1, 640]`` (per device) repeated over the 32 tile rows -> ``[1, 4, 32, 640]``
    fp32, the row-repeated operand F2's ``normalize`` streams (the chain row-broadcasts the one-row tensor in its
    multiply; the values are the same).  Built once from the resident weight; a 1x1 mesh takes the plain upload."""

    hosts = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(norm_scale)]
    rows = [h.expand(-1, -1, fp.TILE, -1).contiguous() for h in hosts]
    if len(rows) == 1:
        return ttnn.from_torch(
            rows[0], dtype=FP32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=_mixer_module().MESH_SHAPE, dims=(None, 3))
    return ttnn.from_torch(
        torch.cat(rows, dim=3),
        dtype=FP32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def module_norm_scale_rows(module):
    key = id(module)
    if key not in _NORM_ROWS:
        _NORM_ROWS[key] = norm_scale_rows(module.mesh_device, module.weights.norm_scale)
    return _NORM_ROWS[key]


def down(normalized, weight):
    """Stage 2b: the flat row times the mixer's down weight ``[1, 1, 2560, 320]`` -> the fp32 partial
    ``[1, 1, rows, 320]`` (F2's DOWN kernel: fp32-dest accumulation in K order with the DRAM-sharded matmul's spill)."""

    rows = fp.rows_of(normalized)
    gr._expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "final mixer normalized row")
    gr._expect(weight, (1, 1, FLAT_WIDTH, RANK), BF16, "final mixer down")
    mesh = normalized.device()
    out = fp.allocate((1, 1, rows, RANK), FP32, ttnn.TILE_LAYOUT, mesh)
    n_tiles = gr._tiles_wide(weight)
    blk = 8
    work = fp.split_work(RANK_TILES, mesh)
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, FLAT_TILES, cores),
        fp.cb_descriptor(1, BF16, TILE_BF16, 2 * blk, cores),
        fp.cb_descriptor(2, FP32, TILE_FP32, 1, cores),
        fp.cb_descriptor(16, FP32, TILE_FP32, 1, cores),
    ]
    reader = gr._reader(
        cores,
        [(normalized, 0), (weight, 1)],
        [],
        [
            (
                w.core,
                (
                    [
                        gr._stream(normalized, 1, FLAT_TILES, 0, 1, 0, blk),
                        gr._stream(weight, 1, FLAT_TILES, w.start, n_tiles, 0, blk),
                    ],
                    [],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(gr.DOWN, cores, [FLAT_TILES, blk, 0, 1, 16, DOWN_SPILL, 2], fp32_dest=True)
    writer = gr._writer(cores, [(out, 16)], [(w.core, [(1, w.start, 1, 1)]) for w in work])
    # the weight once (each core its column), the normalized row once per core, the fp32 partial out; the matmul
    meta = fp.program_meta(
        NAME,
        "down",
        rows,
        reads=(weight,),
        writes=(out,),
        dram_bytes=len(work) * fp.tensor_bytes(normalized),
        flops=2 * rows * FLAT_WIDTH * RANK,
        cores=len(work),
    )
    return fp.run_program(
        [normalized, weight, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta
    )


def low_rank(gathered_partials):
    """Stage 3a: the four gathered fp32 partial rows ``[4, 1, 1, 320]`` summed as the chain's composite sums them,
    typecast bf16, silu -> ``[1, 1, 1, 320]`` bf16 (one core)."""

    shape = tuple(int(v) for v in gathered_partials.shape)
    if (
        shape != (TP_SIZE, 1, 1, RANK)
        or gathered_partials.dtype != FP32
        or gathered_partials.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(f"final mixer gathered partials must be fp32 TILE [4, 1, 1, {RANK}], got {shape}")
    mesh = gathered_partials.device()
    zero = zero_tile(mesh)
    out = fp.allocate((1, 1, 1, RANK), BF16, ttnn.TILE_LAYOUT, mesh)
    core = ttnn.CoreCoord(0, 0)
    one = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    cbs = [
        fp.cb_descriptor(0, FP32, TILE_FP32, RANK_TILES, one),
        fp.cb_descriptor(1, FP32, TILE_FP32, 1, one),
        fp.cb_descriptor(2, FP32, TILE_FP32, RANK_TILES, one),
        fp.cb_descriptor(3, BF16, TILE_BF16, RANK_TILES, one),
        fp.cb_descriptor(16, BF16, TILE_BF16, RANK_TILES, one),
    ]
    reader = fp.reader_kernel(
        LOWRANK_READER,
        one,
        [RANK_TILES, TP_SIZE, RANK_TILES] + fp.accessor_args(gathered_partials) + fp.accessor_args(zero),
        [(core, [gathered_partials.buffer_address(), zero.buffer_address(), 0])],
    )
    compute = fp.compute_kernel(LOWRANK_COMPUTE, one, [RANK_TILES], fp32_dest=True, unpack_to_dest_fp32=(0, 2))
    writer = fp.writer_kernel(
        LOWRANK_WRITER, one, [RANK_TILES] + fp.accessor_args(out), [(core, [out.buffer_address()])]
    )
    # the four partial rows and the zero pad in, the bf16 row out; the fold over the devices, the typecast, the silu
    meta = fp.program_meta(
        NAME, "low_rank", 1, reads=(gathered_partials, zero), writes=(out,), flops=(TP_SIZE + 2) * RANK, cores=1
    )
    return fp.run_program(
        [gathered_partials, zero, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta
    )


def gate(low_rank_row, normalized, up):
    """Stage 3b: sigmoid(low rank x up) times the normalized row, the four branches summed -> ``[1, 1, rows, 640]``
    (F2's GATE kernel with the mixer's K = 320 and its up matmul's 5-tile K blocks)."""

    rows = fp.rows_of(low_rank_row)
    gr._expect(low_rank_row, (1, 1, rows, RANK), BF16, "final mixer low-rank row")
    gr._expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "final mixer normalized row")
    gr._expect(up, (1, 1, RANK, FLAT_WIDTH), BF16, "final mixer up")
    mesh = low_rank_row.device()
    out = fp.allocate((1, 1, rows, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    n_tiles = gr._tiles_wide(up)
    work = fp.split_work(HIDDEN_TILES, mesh)
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, RANK_TILES, cores),
        fp.cb_descriptor(1, BF16, TILE_BF16, BRANCHES * RANK_TILES, cores),
        fp.cb_descriptor(2, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(3, BF16, TILE_BF16, 1, cores),
        fp.cb_descriptor(4, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(5, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(6, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(7, FP32, TILE_FP32, BRANCHES, cores),
        fp.cb_descriptor(16, BF16, TILE_BF16, 1, cores),
    ]
    reader = gr._reader(
        cores,
        [(low_rank_row, 0), (normalized, 2), (up, 1)],
        [(gr.CONST_ZERO, 3)],
        [
            (
                w.core,
                (
                    [
                        gr._stream(low_rank_row, 1, RANK_TILES, 0, 1, 0, RANK_TILES),
                        gr._stream(normalized, 1, BRANCHES, w.start, HIDDEN_TILES, 0, BRANCHES),
                        gr._stream(up, BRANCHES, RANK_TILES, w.start, n_tiles, HIDDEN_TILES, RANK_TILES),
                    ],
                    [0],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(
        gr.GATE,
        cores,
        [RANK_TILES, BRANCHES, 0, 1, 2, 3, 4, 5, 6, 16, UP_SPILL, 7],
        fp32_dest=True,
        unpack_to_dest_fp32=(2, 4, 5),
    )
    writer = gr._writer(cores, [(out, 16)], [(w.core, [(1, w.start, 1, 1)]) for w in work])
    # the up weight and the normalized row once (each core its columns), the low-rank row once per core, the block
    # out; the up matmul, then sigmoid, gate multiply and branch sum per element
    meta = fp.program_meta(
        NAME,
        "gate",
        rows,
        reads=(up, normalized),
        writes=(out,),
        dram_bytes=len(work) * fp.tensor_bytes(low_rank_row),
        flops=2 * rows * RANK * FLAT_WIDTH + 3 * rows * FLAT_WIDTH,
        cores=len(work),
    )
    return fp.run_program(
        [low_rank_row, normalized, up, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta
    )


def merged_enabled(environ=None) -> bool:
    import os

    return (environ if environ is not None else os.environ).get(MERGED_ENV, "1") != "0"


def normalize_down(residual, gathered_stats, norm_scale, weight):
    """Stages 2a+2b as one program (F2's ``normalize_down`` with the mixer's down ``[1, 1, 2560, 320]``): the four
    norm cores multicast the normalized row into the ten down cores' CB 8 (and write it for stage 3) while those
    cores prefetch their weight column; returns (normalized ``[1, 1, rows, 2560]`` bf16, partial ``[1, 1, rows, 320]``
    fp32)."""

    rows = gr.residual_rows(residual)
    gr._expect(gathered_stats, (1, BRANCHES, rows, fp.TILE * gr.STATS_TILES), BF16, "final mixer gathered stats")
    gr._expect(norm_scale, (1, BRANCHES, fp.TILE, LOCAL_HIDDEN), FP32, "final mixer norm_scale rows")
    gr._expect(weight, (1, 1, FLAT_WIDTH, RANK), BF16, "final mixer down")
    mesh = residual.device()
    normalized = fp.allocate((1, 1, rows, FLAT_WIDTH), BF16, ttnn.TILE_LAYOUT, mesh)
    partial = fp.allocate((1, 1, rows, RANK), FP32, ttnn.TILE_LAYOUT, mesh)
    scaler_bits, scaler_dtype = gr.avg_scaler("chain")
    n_tiles = gr._tiles_wide(weight)
    workers, producers, rect = gr._rectangle(mesh, *DOWN_GRID, rows_above=BRANCHES)
    assert len(workers) == RANK_TILES
    w_set, p_set, all_set = gr._core_set(workers), gr._core_set(producers), gr._core_set(workers + producers)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, HIDDEN_TILES, p_set),
        fp.cb_descriptor(1, BF16, TILE_BF16, gr.STATS_TILES, p_set),
        fp.cb_descriptor(2, scaler_dtype, fp.TILE_BYTES[scaler_dtype], 1, p_set),
        fp.cb_descriptor(3, BF16, TILE_BF16, 1, p_set),
        fp.cb_descriptor(4, FP32, TILE_FP32, HIDDEN_TILES, p_set),
        fp.cb_descriptor(5, FP32, TILE_FP32, 1, p_set),
        fp.cb_descriptor(6, FP32, TILE_FP32, 1, p_set),
        fp.cb_descriptor(7, BF16, TILE_BF16, HIDDEN_TILES, p_set),
        fp.cb_descriptor(16, BF16, TILE_BF16, HIDDEN_TILES, p_set),
        fp.cb_descriptor(8, BF16, TILE_BF16, FLAT_TILES, all_set),
        fp.cb_descriptor(9, BF16, TILE_BF16, FLAT_TILES, w_set),
        fp.cb_descriptor(10, FP32, TILE_FP32, 1, w_set),
        fp.cb_descriptor(17, FP32, TILE_FP32, 1, w_set),
    ]
    reader = gr._reader(
        p_set,
        [(gathered_stats, 1), (residual, 0), (norm_scale, 4)],
        [(gr.CONST_SCALER, 2), (gr.CONST_COL_SCALAR, 3)],
        [
            (
                core,
                (
                    [
                        gr._stream(gathered_stats, 1, gr.STATS_TILES, b * gr.STATS_TILES, 1, 0, gr.STATS_TILES),
                        gr._stream(residual, 1, HIDDEN_TILES, b * HIDDEN_TILES, 1, 0, 4),
                        gr._stream(norm_scale, 1, HIDDEN_TILES, b * HIDDEN_TILES, 1, 0, 4),
                    ],
                    [scaler_bits, gr._bits(gr.EPS)],
                ),
            )
            for b, core in enumerate(producers)
        ],
    )
    norm = fp.compute_kernel(
        gr.NORM, p_set, [HIDDEN_TILES, gr.STATS_TILES, 4, 16], fp32_dest=True, unpack_to_dest_fp32=(4, 7)
    )
    sender = gr._mcast_writer(
        p_set,
        [(core, (b * HIDDEN_TILES, rect, (b * HIDDEN_TILES, 1), (0, 0, 1, 1))) for b, core in enumerate(producers)],
        src_cb=16,
        dst_cb=8,
        tiles=HIDDEN_TILES,
        tiles_tensor=normalized,
    )
    receiver = gr._mcast_reader(
        w_set,
        [(weight, 9)],
        [(core, [gr._stream(weight, 1, FLAT_TILES, w, n_tiles, 0, 8)]) for w, core in enumerate(workers)],
        recv_cb=8,
        recv_tiles=FLAT_TILES,
        senders=BRANCHES,
    )
    down_k = fp.compute_kernel(gr.DOWN, w_set, [FLAT_TILES, 8, 8, 9, 17, DOWN_SPILL, 10], fp32_dest=True)
    writer = gr._writer(w_set, [(partial, 17)], [(core, [(1, w, 1, 1)]) for w, core in enumerate(workers)])
    # the norm operands and the weight once, the normalized row and the partial out; the normalized row multicast
    # into the down cores' CB (L1); the norm's four passes per element, then the down matmul
    meta = fp.program_meta(
        NAME,
        "normalize_down",
        rows,
        reads=(residual, gathered_stats, norm_scale, weight),
        writes=(normalized, partial),
        l1_bytes=len(workers) * fp.tensor_bytes(normalized),
        flops=4 * rows * FLAT_WIDTH + 2 * rows * FLAT_WIDTH * RANK,
        cores=len(workers) + len(producers),
    )
    fp.run_program(
        [residual, gathered_stats, norm_scale, weight, normalized, partial],
        fp.program_descriptor(
            [reader, norm, sender, receiver, down_k, writer], cbs=cbs, semaphores=[fp.semaphore_descriptor(0, all_set)]
        ),
        meta=meta,
    )
    return normalized, partial


def low_rank_gate(gathered_partials, normalized, up):
    """Stages 3a+3b as one program (F2's ``low_rank_gate`` with the mixer's shapes): five row cores each build two
    stacked tiles, fold them (``low_rank``'s kernels), and multicast the bf16 low-rank tiles into the twenty gate
    cores' CB 8 while those prefetch their weight columns and normalized tiles -> ``[1, 1, rows, 640]`` bf16."""

    shape = tuple(int(v) for v in gathered_partials.shape)
    if (
        shape != (TP_SIZE, 1, 1, RANK)
        or gathered_partials.dtype != FP32
        or gathered_partials.layout != ttnn.TILE_LAYOUT
    ):
        raise ValueError(f"final mixer gathered partials must be fp32 TILE [4, 1, 1, {RANK}], got {shape}")
    rows = 1
    gr._expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "final mixer normalized row")
    gr._expect(up, (1, 1, RANK, FLAT_WIDTH), BF16, "final mixer up")
    mesh = gathered_partials.device()
    zero = zero_tile(mesh)
    block = fp.allocate((1, 1, rows, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    t = LOW_RANK_TILES_PER_CORE
    n_tiles = gr._tiles_wide(up)
    workers, producers, rect = gr._rectangle(mesh, *GATE_GRID, rows_above=LOW_RANK_CORES)
    assert len(workers) == HIDDEN_TILES and len(producers) * t == RANK_TILES
    w_set, p_set, all_set = gr._core_set(workers), gr._core_set(producers), gr._core_set(workers + producers)
    cbs = [
        fp.cb_descriptor(0, FP32, TILE_FP32, t, p_set),
        fp.cb_descriptor(1, FP32, TILE_FP32, 1, p_set),
        fp.cb_descriptor(2, FP32, TILE_FP32, t, p_set),
        fp.cb_descriptor(3, BF16, TILE_BF16, t, p_set),
        fp.cb_descriptor(16, BF16, TILE_BF16, t, p_set),
        fp.cb_descriptor(8, BF16, TILE_BF16, RANK_TILES, all_set),
        fp.cb_descriptor(9, BF16, TILE_BF16, BRANCHES * RANK_TILES, w_set),
        fp.cb_descriptor(10, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(11, BF16, TILE_BF16, 1, w_set),
        fp.cb_descriptor(12, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(13, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(14, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(15, FP32, TILE_FP32, BRANCHES, w_set),
        fp.cb_descriptor(18, BF16, TILE_BF16, 1, w_set),
    ]
    reader = fp.reader_kernel(
        LOWRANK_READER,
        p_set,
        [t, TP_SIZE, RANK_TILES] + fp.accessor_args(gathered_partials) + fp.accessor_args(zero),
        [
            (core, [gathered_partials.buffer_address(), zero.buffer_address(), c * t])
            for c, core in enumerate(producers)
        ],
    )
    fold = fp.compute_kernel(LOWRANK_COMPUTE, p_set, [t], fp32_dest=True, unpack_to_dest_fp32=(0, 2))
    sender = gr._mcast_writer(
        p_set,
        [(core, (c * t, rect, (0, 1), (0, 0, 1, 1))) for c, core in enumerate(producers)],
        src_cb=16,
        dst_cb=8,
        tiles=t,
        extra=(gathered_partials, gr.NONE_CB),  # no DRAM write of the low-rank row; the accessor slot needs a tensor
    )
    receiver = gr._mcast_reader(
        w_set,
        [(normalized, 10), (up, 9)],
        [
            (
                core,
                [
                    gr._stream(normalized, 1, BRANCHES, j, HIDDEN_TILES, 0, BRANCHES),
                    gr._stream(up, BRANCHES, RANK_TILES, j, n_tiles, HIDDEN_TILES, RANK_TILES),
                ],
            )
            for j, core in enumerate(workers)
        ],
        recv_cb=8,
        recv_tiles=RANK_TILES,
        senders=LOW_RANK_CORES,
        zero_cb=11,
    )
    gate_k = fp.compute_kernel(
        gr.GATE,
        w_set,
        [RANK_TILES, BRANCHES, 8, 9, 10, 11, 12, 13, 14, 18, UP_SPILL, 15],
        fp32_dest=True,
        unpack_to_dest_fp32=(10, 12, 13),
    )
    writer = gr._writer(w_set, [(block, 18)], [(core, [(1, j, 1, 1)]) for j, core in enumerate(workers)])
    # the partials, the pad, the normalized row and the up weight once, the block out; the low-rank row multicast
    # into the gate cores' CB (L1); the fold, typecast and silu, the up matmul, sigmoid, gate multiply and branch sum
    meta = fp.program_meta(
        NAME,
        "low_rank_gate",
        rows,
        reads=(gathered_partials, zero, normalized, up),
        writes=(block,),
        l1_bytes=len(workers) * RANK_TILES * TILE_BF16,
        flops=(TP_SIZE + 2) * RANK + 2 * rows * RANK * FLAT_WIDTH + 3 * rows * FLAT_WIDTH,
        cores=len(workers) + len(producers),
    )
    fp.run_program(
        [gathered_partials, zero, normalized, up, block],
        fp.program_descriptor(
            [reader, fold, sender, receiver, gate_k, writer], cbs=cbs, semaphores=[fp.semaphore_descriptor(0, all_set)]
        ),
        meta=meta,
    )
    return block


def final_mixer_fused(module, residual, *, merged: bool | None = None):
    """``Qwen38TTNNFinalMixer.__call__`` on the fused programs around the chain's two collectives; ``merged``
    (default: the QWEN38_FUSED_FINAL_MIXER_MERGED switch, on) runs normalize+down and low_rank+gate as one program
    each."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement

    fm = _mixer_module()
    if tuple(int(v) for v in residual.shape) != fm.RESIDUAL_LOCAL_SHAPE or residual.dtype != BF16:
        raise ValueError(f"final mixer input must be TILE BF16 {fm.RESIDUAL_LOCAL_SHAPE}")
    module.mesh_contract.validate_tensor(residual, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    stats_local = gr.stats(residual)
    gr._topology(module, stats_local, 3)
    gathered_stats = ttnn.all_gather(stats_local, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(stats_local)
    module.mesh_contract.validate_tensor(gathered_stats, placement=TensorPlacement.REPLICATED)
    merged = merged_enabled() if merged is None else merged
    if merged:
        try:
            gr.noc_map(
                residual.device()
            )  # the multicast rectangles, once per mesh; devices that disagree take the split form
        except gr.NocMapMismatch as error:
            warnings.warn(f"final_mixer: split form ({error})", RuntimeWarning, stacklevel=2)
            merged = False
    if merged:
        normalized, partial = normalize_down(
            residual, gathered_stats, module_norm_scale_rows(module), module.weights.down
        )
    else:
        normalized = gr.normalize(residual, gathered_stats, module_norm_scale_rows(module))
        partial = down(normalized, module.weights.down)
    ttnn.deallocate(gathered_stats)
    gr._topology(module, normalized, 3)
    module.mesh_contract.mark_local_partial(
        partial, replicated_reference=module.weights.replicated_anchor, expected_shape=(1, 1, 1, RANK)
    )
    gathered_partials = ttnn.all_gather(partial, dim=0, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(partial)
    module.mesh_contract.validate_tensor(gathered_partials, placement=TensorPlacement.REPLICATED)
    if merged:
        block = low_rank_gate(gathered_partials, normalized, module.weights.up)
    else:
        low = low_rank(gathered_partials)
        low.update_tensor_topology(gathered_partials.tensor_topology())
        block = gate(low, normalized, module.weights.up)
        ttnn.deallocate(low)
    ttnn.deallocate(gathered_partials)
    ttnn.deallocate(normalized)
    gr._topology(module, block, 3)
    module.mesh_contract.validate_tensor(block, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    if tuple(int(v) for v in block.shape) != fm.OUTPUT_LOCAL_SHAPE or block.dtype != BF16:
        raise RuntimeError(f"fused final mixer produced {tuple(block.shape)} {block.dtype}")
    return block


def final_mixer_composed(module, residual):
    """The chain as written in ttnn/final_mixer.py (the class method, whatever the instance resolved to)."""

    return type(module).__call__(module, residual)


register(
    FusedKernel(
        name=NAME,
        replaces="Qwen38TTNNFinalMixer.__call__: rms_norm_pre/post_all_gather, gamma multiply, down linear, S2I, the fp32 "
        "all_reduce composite (7 programs), typecast, silu, up linear, S2I, sigmoid, gate multiply, fast_reduce_nc "
        "(21 programs once per step -> 3 merged or 5 split; the two collectives stay)",
        tolerance=BITWISE,
        fused=final_mixer_fused,
        composed=final_mixer_composed,
        gate=None,  # needs the TP4 collectives: component gate = the single-chip replica test (four slices, host gathers)
    )
)

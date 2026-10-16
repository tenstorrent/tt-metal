# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gr_read``: the TP4 gated-residual read as five compute programs around the chain's two collectives.

The composed chain (``Qwen38TTNNGatedResidual._read_composed``) is 18 programs per read: FillPad,
rms_norm_pre_all_gather, all_gather, rms_norm_post_all_gather, multiply (gamma), linear (down+inject, fp32),
to_memory_config, all_gather_async, fast_reduce_nc, typecast, slice, silu, linear (up), to_memory_config, sigmoid,
multiply (gate), fast_reduce_nc, multiply (2 sigmoid).  Here the same arithmetic runs as ``stats`` -> all_gather ->
``normalize`` -> ``down_project`` -> all_gather_async -> ``low_rank`` -> ``gate``; every phase is the chain kernel's
LLK sequence on the same tiles (see work/FUSE-GR-READ-20260913.md section 1), so the output is bitwise against the
chain.  Rows 1..32: the residual ``[1, 4, rows, 640]`` is four branch row-blocks of one row tile each; every program
works on whole tiles.

The five program builders take device tensors only (no module), so a single chip can run the four device slices in
turn with host gathers between them (the single-chip replica tool in the dev tree); ``gr_read_fused`` is the model-level read
(the module supplies the weights, the mesh contract and the TT-CCL collectives).

``scaler_mode``: the chain's post-norm AVG scaler tile is bf16 1/2560 written by truncation (0x39CC = 1/2569.6,
layernorm_post_all_gather_program_factory.cpp ``calculate_and_prepare_reduce_scaler`` ->
reduce_helpers_dataflow.inl ``float_to_scaler_bits<Float16_b>``).  ``"chain"`` keeps it (bitwise); ``"rne"`` writes
the round-to-nearest bf16 0x39CD; ``"fp32"`` writes the exact 1/2560 as an fp32 scaler tile.  The latter two are the
component-error measurement for the numerics re-pin, not the default.
"""

from __future__ import annotations

import struct

import ttnn

from .. import program as fp
from ..registry import BITWISE, FusedKernel, register

NAME = "gr_read"
TP_SIZE = 4
TP_AXIS = 1
BRANCHES = 4
LOCAL_HIDDEN = 640
FLAT_WIDTH = BRANCHES * LOCAL_HIDDEN  # 2560
RANK = 320
PARTIAL_WIDTH = 384
TILE = fp.TILE
HIDDEN_TILES = LOCAL_HIDDEN // TILE  # 20
FLAT_TILES = FLAT_WIDTH // TILE  # 80
PARTIAL_TILES = PARTIAL_WIDTH // TILE  # 12
INJECT_TILE = RANK // TILE  # 10: the inject columns 320-323 sit in this partial tile
STATS_TILES = TP_SIZE  # gathered stats tiles per branch row-block
LOW_RANK_CORES = 4
LOW_RANK_TILES_PER_CORE = PARTIAL_TILES // LOW_RANK_CORES  # 3
EPS = 1e-6
SCALER_MODES = ("chain", "rne", "fp32")
BF16, FP32 = ttnn.bfloat16, ttnn.float32
TILE_BF16, TILE_FP32 = fp.TILE_BYTES[BF16], fp.TILE_BYTES[FP32]
READER = fp.kernel_source(NAME, "reader.cpp")
WRITER = fp.kernel_source(NAME, "writer.cpp")
STATS = fp.kernel_source(NAME, "stats_compute.cpp")
NORM = fp.kernel_source(NAME, "norm_compute.cpp")
DOWN = fp.kernel_source(NAME, "down_compute.cpp")
LOWRANK = fp.kernel_source(NAME, "lowrank_compute.cpp")
GATE = fp.kernel_source(NAME, "gate_compute.cpp")
MCAST_WRITER = fp.kernel_source(NAME, "mcast_writer.cpp")
MCAST_READER = fp.kernel_source(NAME, "mcast_reader.cpp")
NOC_PROBE = fp.kernel_source(NAME, "noc_probe.cpp")
_NOC_MAPS: dict[int, dict[tuple[int, int], tuple[int, int]]] = {}
MERGED_ENV = "QWEN38_FUSED_GR_READ_MERGED"  # default "1": normalize+down and low_rank+gate as one program each (5 programs per read); "0": the split forms (7)
NONE_CB = 0xFF
MATMUL_ENV = "QWEN38_FUSED_GR_READ_MATMUL"
MATMUL_MODES = ("chain", "exact")
# The chain's DRAM-sharded linears spill the fp32 running partial to an intermediate CB between K blocks (8 tiles for
# down+inject, 6 for up) and reload it through SrcA, which rounds it (the factory marks no UnpackToDestFp32 there):
# "chain" reproduces that (bitwise), "exact" accumulates in the dest throughout (the default matmul program's
# arithmetic; measured 2026-09-14: rms rel 3.2e-4 vs the exact sum against the chain's 2.0e-3).
DOWN_SPILL, UP_SPILL = 8, 6


def matmul_mode(environ=None) -> str:
    import os

    mode = (environ if environ is not None else os.environ).get(MATMUL_ENV, "chain")
    if mode not in MATMUL_MODES:
        raise ValueError(f"{MATMUL_ENV} must be one of {MATMUL_MODES}, got {mode!r}")
    return mode


CONST_SCALER, CONST_COL_SCALAR, CONST_ZERO = 1, 2, 3


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def avg_scaler(mode: str) -> tuple[int, object]:
    """(bits handed to the reader's reduce-scaler generator, scaler CB dtype) for 1/2560 in the given mode."""

    if mode == "chain":  # the generator truncates the fp32 bits to bf16: 0x39CCCCCD -> 0x39CC
        return _bits(1.0 / FLAT_WIDTH), BF16
    if mode == "rne":  # a float whose truncation is the round-to-nearest bf16 of 1/2560
        return 0x39CD0000, BF16
    if mode == "fp32":
        return _bits(1.0 / FLAT_WIDTH), FP32
    raise ValueError(f"scaler_mode must be one of {SCALER_MODES}, got {mode!r}")


def residual_rows(residual) -> int:
    shape, padded = tuple(residual.shape), tuple(residual.padded_shape)
    if len(shape) != 4 or shape[:2] != (1, BRANCHES) or shape[3] != LOCAL_HIDDEN or padded[2] != TILE:
        raise ValueError(
            f"GR residual must be [1, {BRANCHES}, 1..{TILE}, {LOCAL_HIDDEN}] padded to one row tile, got {shape}"
        )
    if residual.layout != ttnn.TILE_LAYOUT or residual.dtype != BF16:
        raise ValueError("GR residual must be TILE bfloat16")
    return shape[2]


def _expect(tensor, shape: tuple[int, ...], dtype, label: str) -> None:
    if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            f"{label} must be TILE {dtype} {shape}, got {tensor.layout} {tensor.dtype} {tuple(tensor.shape)}"
        )


def _tiles_wide(tensor) -> int:
    return tensor.padded_shape[-1] // TILE


def _stream(tensor, outer: int, inner: int, first: int, inner_stride: int, outer_stride: int, batch: int) -> list[int]:
    if (outer * inner) % batch:
        raise ValueError(f"stream of {outer * inner} tiles is not a multiple of batch {batch}")
    return [tensor.buffer_address(), outer, inner, first, inner_stride, outer_stride, batch]


def _reader(cores, streams, consts, runtime, gate=None):
    """streams: [(tensor, cb)]; consts: [(kind, cb)]; runtime: per core -> (stream args lists, const bits list);
    ``gate`` = (stream index, program semaphore id, count): that stream is read once the semaphore reached the count
    (a producer of the same program signals that its pages are complete), then the semaphore is reset."""

    compile_args = [len(streams), *[cb for _, cb in streams], *[0] * (4 - len(streams)), len(consts)]
    for kind, cb in consts:
        compile_args += [kind, cb]
    compile_args += [0, 0] * (3 - len(consts))
    compile_args += list(gate) if gate is not None else [NONE_CB, 0, 0]
    for slot in range(4):  # the kernel defines four accessor sets; unused slots repeat the first tensor's
        compile_args += fp.accessor_args(streams[min(slot, len(streams) - 1)][0])
    return fp.reader_kernel(
        READER,
        cores,
        compile_args,
        [(core, [a for args in stream_args for a in args] + list(bits)) for core, (stream_args, bits) in runtime],
    )


def _writer(cores, streams, runtime):
    """streams: [(tensor, cb)]; runtime: per core -> list of (count, first, stride, batch) per stream."""

    compile_args = [len(streams), *[cb for _, cb in streams], *[0] * (3 - len(streams))]
    for slot in range(3):
        compile_args += fp.accessor_args(streams[min(slot, len(streams) - 1)][0])
    return fp.writer_kernel(
        WRITER,
        cores,
        compile_args,
        [
            (
                core,
                [
                    a
                    for (tensor, _), (count, first, stride, batch) in zip(streams, per_stream)
                    for a in (tensor.buffer_address(), count, first, stride, batch)
                ],
            )
            for core, per_stream in runtime
        ],
    )


def stats(residual):
    """Stage 1: the per-branch sum of squares over the local 640 -> ``[1, 4, rows, 32]`` bf16 (4 tiles)."""

    rows = residual_rows(residual)
    mesh = residual.device()
    out = fp.allocate((1, BRANCHES, rows, TILE), BF16, ttnn.TILE_LAYOUT, mesh)
    work = fp.split_work(BRANCHES, mesh)
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, HIDDEN_TILES, cores),
        fp.cb_descriptor(1, FP32, TILE_FP32, 1, cores),
        fp.cb_descriptor(2, FP32, TILE_FP32, HIDDEN_TILES, cores),
        fp.cb_descriptor(16, BF16, TILE_BF16, 1, cores),
    ]
    reader = _reader(
        cores,
        [(residual, 0)],
        [(CONST_SCALER, 1)],
        [(w.core, ([_stream(residual, 1, HIDDEN_TILES, w.start * HIDDEN_TILES, 1, 0, 4)], [_bits(1.0)])) for w in work],
    )
    compute = fp.compute_kernel(STATS, cores, [HIDDEN_TILES], fp32_dest=True)
    writer = _writer(cores, [(out, 16)], [(w.core, [(1, w.start, 1, 1)]) for w in work])
    meta = fp.program_meta(  # the residual in (each core its branch), the stats tiles out; square and sum per element
        NAME,
        "stats",
        rows,
        reads=(residual,),
        writes=(out,),
        flops=2 * rows * FLAT_WIDTH,
        cores=len(work),
        outputs=((out, 3),),
    )
    return fp.run_program([residual, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta)


def normalize(residual, gathered_stats, norm_scale, *, scaler_mode: str = "chain"):
    """Stage 2a: x * rsqrt(mean of the gathered sums + eps) * gamma/4 -> the flat ``[1, 1, rows, 2560]`` bf16 row.
    ``norm_scale`` is the module's ``norm_scale_rows`` (gamma/4 repeated over the 32 tile rows) so the reader streams
    it as it is; the chain row-broadcasts a one-row tensor in its multiply, the values are the same."""

    rows = residual_rows(residual)
    _expect(gathered_stats, (1, BRANCHES, rows, TILE * STATS_TILES), BF16, "GR gathered stats")
    _expect(norm_scale, (1, BRANCHES, TILE, LOCAL_HIDDEN), FP32, "GR norm_scale rows")
    mesh = residual.device()
    out = fp.allocate((1, 1, rows, FLAT_WIDTH), BF16, ttnn.TILE_LAYOUT, mesh)
    scaler_bits, scaler_dtype = avg_scaler(scaler_mode)
    work = fp.split_work(BRANCHES, mesh)
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, HIDDEN_TILES, cores),
        fp.cb_descriptor(1, BF16, TILE_BF16, STATS_TILES, cores),
        fp.cb_descriptor(2, scaler_dtype, fp.TILE_BYTES[scaler_dtype], 1, cores),
        fp.cb_descriptor(3, BF16, TILE_BF16, 1, cores),
        fp.cb_descriptor(4, FP32, TILE_FP32, HIDDEN_TILES, cores),
        fp.cb_descriptor(5, FP32, TILE_FP32, 1, cores),
        fp.cb_descriptor(6, FP32, TILE_FP32, 1, cores),
        fp.cb_descriptor(7, BF16, TILE_BF16, HIDDEN_TILES, cores),
        fp.cb_descriptor(16, BF16, TILE_BF16, HIDDEN_TILES, cores),
    ]
    reader = _reader(
        cores,
        [(gathered_stats, 1), (residual, 0), (norm_scale, 4)],
        [(CONST_SCALER, 2), (CONST_COL_SCALAR, 3)],
        [
            (
                w.core,
                (
                    [
                        _stream(gathered_stats, 1, STATS_TILES, w.start * STATS_TILES, 1, 0, STATS_TILES),
                        _stream(residual, 1, HIDDEN_TILES, w.start * HIDDEN_TILES, 1, 0, 4),
                        _stream(norm_scale, 1, HIDDEN_TILES, w.start * HIDDEN_TILES, 1, 0, 4),
                    ],
                    [scaler_bits, _bits(EPS)],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(
        NORM, cores, [HIDDEN_TILES, STATS_TILES, 4, 16], fp32_dest=True, unpack_to_dest_fp32=(4, 7)
    )
    writer = _writer(cores, [(out, 16)], [(w.core, [(HIDDEN_TILES, w.start * HIDDEN_TILES, 1, 4)]) for w in work])
    meta = fp.program_meta(  # the residual, the gathered stats and gamma in, the flat row out; four passes per element
        NAME,
        "normalize",
        rows,
        reads=(residual, gathered_stats, norm_scale),
        writes=(out,),
        flops=4 * rows * FLAT_WIDTH,
        cores=len(work),
        outputs=((out, 3),),
    )
    return fp.run_program(
        [residual, gathered_stats, norm_scale, out],
        fp.program_descriptor([reader, compute, writer], cbs=cbs),
        meta=meta,
    )


def down_project(normalized, down_inject, *, matmul: str = "chain"):
    """Stage 2b: the flat row times the fused down+inject weight -> the fp32 partial ``[1, 1, rows, 384]``."""

    rows = fp.rows_of(normalized)
    _expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "GR normalized row")
    _expect(down_inject, (1, 1, FLAT_WIDTH, PARTIAL_WIDTH), BF16, "GR down_inject")
    mesh = normalized.device()
    out = fp.allocate((1, 1, rows, PARTIAL_WIDTH), FP32, ttnn.TILE_LAYOUT, mesh)
    n_tiles = _tiles_wide(down_inject)
    blk = 8
    work = fp.split_work(PARTIAL_TILES, mesh)
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, FLAT_TILES, cores),
        fp.cb_descriptor(1, BF16, TILE_BF16, 2 * blk, cores),
        fp.cb_descriptor(2, FP32, TILE_FP32, 1, cores),
        fp.cb_descriptor(16, FP32, TILE_FP32, 1, cores),
    ]
    reader = _reader(
        cores,
        [(normalized, 0), (down_inject, 1)],
        [],
        [
            (
                w.core,
                (
                    [
                        _stream(normalized, 1, FLAT_TILES, 0, 1, 0, blk),
                        _stream(down_inject, 1, FLAT_TILES, w.start, n_tiles, 0, blk),
                    ],
                    [],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(
        DOWN, cores, [FLAT_TILES, blk, 0, 1, 16, DOWN_SPILL if matmul == "chain" else 0, 2], fp32_dest=True
    )
    writer = _writer(cores, [(out, 16)], [(w.core, [(1, w.start, 1, 1)]) for w in work])
    meta = fp.program_meta(  # the weight once (each core its column), the flat row once per core, the partial out
        NAME,
        "down_project",
        rows,
        reads=(down_inject,),
        writes=(out,),
        dram_bytes=len(work) * fp.tensor_bytes(normalized),
        flops=2 * rows * FLAT_WIDTH * PARTIAL_WIDTH,
        cores=len(work),
    )
    return fp.run_program(
        [normalized, down_inject, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta
    )


def low_rank(gathered_partials):
    """Stage 3a: the four device partials summed, typecast bf16, silu -> ``[1, 1, rows, 384]`` bf16; the injection
    ``[1, 1, rows, 4]`` bf16 = 2 sigmoid of the typecast inject columns."""

    shape = tuple(gathered_partials.shape)
    if len(shape) != 4 or shape[:2] != (TP_SIZE, 1) or shape[3] != PARTIAL_WIDTH or gathered_partials.dtype != FP32:
        raise ValueError(f"GR gathered partials must be fp32 [4, 1, rows, {PARTIAL_WIDTH}], got {shape}")
    rows = shape[2]
    mesh = gathered_partials.device()
    out = fp.allocate((1, 1, rows, PARTIAL_WIDTH), BF16, ttnn.TILE_LAYOUT, mesh)
    injection = fp.allocate((1, 1, rows, BRANCHES), BF16, ttnn.TILE_LAYOUT, mesh)
    t = LOW_RANK_TILES_PER_CORE
    inject_core, inject_local = divmod(INJECT_TILE, t)
    work = fp.split_work(LOW_RANK_CORES, mesh)
    cores = fp.core_set(work)
    plain = fp.core_set([w for w in work if w.start != inject_core])
    inject = fp.core_set([w for w in work if w.start == inject_core])
    cbs = [
        fp.cb_descriptor(0, FP32, TILE_FP32, t * TP_SIZE, cores),
        fp.cb_descriptor(1, FP32, TILE_FP32, 1, cores),
        fp.cb_descriptor(2, BF16, TILE_BF16, t, cores),
        fp.cb_descriptor(16, BF16, TILE_BF16, t, cores),
        fp.cb_descriptor(17, BF16, TILE_BF16, 1, inject),
    ]
    reader = _reader(
        cores,
        [(gathered_partials, 0)],
        [(CONST_ZERO, 1)],
        [
            (w.core, ([_stream(gathered_partials, t, TP_SIZE, w.start * t, PARTIAL_TILES, 1, TP_SIZE)], [0]))
            for w in work
        ],
    )
    computes = [
        fp.compute_kernel(LOWRANK, plain, [t, TP_SIZE, t, 16, 17], fp32_dest=True),
        fp.compute_kernel(LOWRANK, inject, [t, TP_SIZE, inject_local, 16, 17], fp32_dest=True),
    ]
    writers = [
        _writer(plain, [(out, 16)], [(w.core, [(t, w.start * t, 1, 1)]) for w in work if w.start != inject_core]),
        _writer(
            inject,
            [(out, 16), (injection, 17)],
            [(w.core, [(t, w.start * t, 1, 1), (1, 0, 1, 1)]) for w in work if w.start == inject_core],
        ),
    ]
    meta = fp.program_meta(  # the four partial rows in, the bf16 row and the injection out; fold, typecast, silu, 2 sigmoid
        NAME,
        "low_rank",
        rows,
        reads=(gathered_partials,),
        writes=(out, injection),
        flops=(TP_SIZE + 3) * rows * PARTIAL_WIDTH,
        cores=len(work),
        outputs=((out, None), (injection, None)),
    )
    fp.run_program(
        [gathered_partials, out, injection], fp.program_descriptor([reader, *computes, *writers], cbs=cbs), meta=meta
    )
    return out, injection


def gate(low_rank_row, normalized, up, *, matmul: str = "chain", debug: bool = False):
    """Stage 3b: sigmoid(low rank x up) times the normalized row, the four branches summed -> ``[1, 1, rows, 640]``.
    ``debug`` also returns the packed up tiles and the gated products as ``[1, 4, rows, 640]`` (dev only)."""

    rows = fp.rows_of(low_rank_row)
    _expect(low_rank_row, (1, 1, rows, PARTIAL_WIDTH), BF16, "GR low-rank row")
    _expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "GR normalized row")
    _expect(up, (1, 1, PARTIAL_WIDTH, FLAT_WIDTH), BF16, "GR up")
    mesh = low_rank_row.device()
    out = fp.allocate((1, 1, rows, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    n_tiles = _tiles_wide(up)
    work = fp.split_work(HIDDEN_TILES, mesh)
    cores = fp.core_set(work)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, PARTIAL_TILES, cores),
        fp.cb_descriptor(1, BF16, TILE_BF16, BRANCHES * PARTIAL_TILES, cores),
        fp.cb_descriptor(2, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(3, BF16, TILE_BF16, 1, cores),
        fp.cb_descriptor(4, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(5, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(6, BF16, TILE_BF16, BRANCHES, cores),
        fp.cb_descriptor(7, FP32, TILE_FP32, BRANCHES, cores),
        fp.cb_descriptor(16, BF16, TILE_BF16, 1, cores),
    ]
    reader = _reader(
        cores,
        [(low_rank_row, 0), (normalized, 2), (up, 1)],
        [(CONST_ZERO, 3)],
        [
            (
                w.core,
                (
                    [
                        _stream(low_rank_row, 1, PARTIAL_TILES, 0, 1, 0, PARTIAL_TILES),
                        _stream(normalized, 1, BRANCHES, w.start, HIDDEN_TILES, 0, BRANCHES),
                        _stream(up, BRANCHES, PARTIAL_TILES, w.start, n_tiles, HIDDEN_TILES, PARTIAL_TILES),
                    ],
                    [0],
                ),
            )
            for w in work
        ],
    )
    compute = fp.compute_kernel(
        GATE,
        cores,
        [PARTIAL_TILES, BRANCHES, 0, 1, 2, 3, 4, 5, 6, 16, UP_SPILL if matmul == "chain" else 0, 7],
        defines=[("DEBUG_KEEP", "1")] if debug else (),
        fp32_dest=True,
        unpack_to_dest_fp32=(2, 4, 5),
    )

    # the up weight and the flat row once (each core its columns), the low-rank row once per core, the block out; the
    # up matmul, then sigmoid, gate multiply and branch sum per element
    def meta(*extra_writes):
        return fp.program_meta(
            NAME,
            "gate",
            rows,
            reads=(up, normalized),
            writes=(out, *extra_writes),
            dram_bytes=len(work) * fp.tensor_bytes(low_rank_row),
            flops=2 * rows * PARTIAL_WIDTH * FLAT_WIDTH + 3 * rows * FLAT_WIDTH,
            cores=len(work),
            outputs=((out, 3),),
        )

    if not debug:
        writer = _writer(cores, [(out, 16)], [(w.core, [(1, w.start, 1, 1)]) for w in work])
        return fp.run_program(
            [low_rank_row, normalized, up, out], fp.program_descriptor([reader, compute, writer], cbs=cbs), meta=meta()
        )
    up_tiles = fp.allocate((1, BRANCHES, rows, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    gated = fp.allocate((1, BRANCHES, rows, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    writer = _writer(
        cores,
        [(out, 16), (up_tiles, 4), (gated, 6)],
        [
            (
                w.core,
                [
                    (1, w.start, 1, 1),
                    (BRANCHES, w.start, HIDDEN_TILES, BRANCHES),
                    (BRANCHES, w.start, HIDDEN_TILES, BRANCHES),
                ],
            )
            for w in work
        ],
    )
    fp.run_program(
        [low_rank_row, normalized, up, out, up_tiles, gated],
        fp.program_descriptor([reader, compute, writer], cbs=cbs),
        meta=meta(up_tiles, gated),
    )
    return out, up_tiles, gated


class NocMapMismatch(RuntimeError):
    """The mesh's devices map logical cores to different NoC coordinates: one program descriptor (one runtime-argument
    set, replicated to every device) cannot carry the multicast rectangles, so the merged forms cannot run on it."""


def noc_map(mesh) -> dict[tuple[int, int], tuple[int, int]]:
    """Logical compute core -> its NoC-0 coordinates, measured once per mesh by a probe program (Python has no
    logical-to-NoC binding; the dispatcher's coordinate virtualization decides the mapping).  Every device of the mesh
    runs the probe; the maps must agree (the 4-chip acceptance ON run at 10cdff7d8a failed here on ttnn.to_torch of the
    mesh tensor, which needs a per-device readback)."""

    key = id(mesh)
    if key not in _NOC_MAPS:
        grid = mesh.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(x, y) for x in range(grid.x) for y in range(grid.y)]
        out = fp.allocate((1, 1, TILE, TILE * len(cores)), ttnn.uint32, ttnn.TILE_LAYOUT, mesh)
        core_set = _core_set(cores)
        probe = fp.writer_kernel(
            NOC_PROBE, core_set, fp.accessor_args(out), [(c, [out.buffer_address(), i]) for i, c in enumerate(cores)]
        )
        fp.run_program(
            [out, out],
            fp.program_descriptor(
                [probe], cbs=[fp.cb_descriptor(0, ttnn.uint32, fp.TILE_BYTES[ttnn.uint32], 1, core_set)]
            ),
            meta=fp.program_meta(NAME, "noc_probe", 1, writes=(out,), cores=len(cores)),  # once per mesh
        )
        maps = []
        for shard in ttnn.get_device_tensors(out):  # one probe tile per device of the mesh
            words = ttnn.to_torch(shard).reshape(TILE, len(cores), TILE)[0]  # row 0 of each tile: words 0..3
            maps.append({(c.x, c.y): (int(words[i, 0]), int(words[i, 1])) for i, c in enumerate(cores)})
        ttnn.deallocate(out)
        if any(m != maps[0] for m in maps[1:]):
            raise NocMapMismatch(f"logical-to-NoC maps differ across the {len(maps)} devices of the mesh")
        _NOC_MAPS[key] = maps[0]
    return _NOC_MAPS[key]


def _rectangle(mesh, width: int, height: int, *, rows_above: int):
    """Consumers on the logical rectangle [0, width) x [0, height) and producers on row ``height`` (x < rows_above)."""

    grid = mesh.compute_with_storage_grid_size()
    if grid.x < max(width, rows_above) or grid.y < height + 1:
        raise RuntimeError(
            f"compute grid {grid.x}x{grid.y} cannot hold {width}x{height} consumers and {rows_above} producers"
        )
    consumers = [ttnn.CoreCoord(x, y) for x in range(width) for y in range(height)]
    producers = [ttnn.CoreCoord(x, height) for x in range(rows_above)]
    noc = noc_map(mesh)
    xs = sorted({noc[(c.x, c.y)][0] for c in consumers})
    ys = sorted({noc[(c.x, c.y)][1] for c in consumers})
    if xs != list(range(xs[0], xs[0] + width)) or ys != list(range(ys[0], ys[0] + height)):
        raise RuntimeError(f"consumer cores are not a contiguous NoC rectangle: x {xs} y {ys}")
    return consumers, producers, (xs[0], ys[0], xs[-1], ys[-1])


def _core_set(cores) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _mcast_writer(cores, runtime, *, src_cb, dst_cb, tiles, tiles_tensor=None, extra=None, sem=0):
    """runtime: per core -> (dst tile offset, rect, (first, stride) for the tiles tensor, (count, first, stride, batch) extra)."""

    extra_tensor, extra_cb = extra if extra is not None else (tiles_tensor, NONE_CB)
    accessor_tensor = tiles_tensor if tiles_tensor is not None else extra_tensor
    compile_args = [src_cb, dst_cb, tiles, 1 if tiles_tensor is not None else 0, extra_cb, sem]
    compile_args += fp.accessor_args(accessor_tensor) + fp.accessor_args(extra_tensor)
    args = []
    for core, (offset, rect, tiles_args, extra_args) in runtime:
        first, stride = tiles_args
        count, efirst, estride, batch = extra_args
        args.append(
            (
                core,
                [
                    offset,
                    *rect,
                    tiles_tensor.buffer_address() if tiles_tensor is not None else 0,
                    first,
                    stride,
                    extra_tensor.buffer_address() if extra is not None else 0,
                    count,
                    efirst,
                    estride,
                    batch,
                ],
            )
        )
    return fp.writer_kernel(MCAST_WRITER, cores, compile_args, args)


def _mcast_reader(cores, streams, runtime, *, recv_cb, recv_tiles, senders, zero_cb=NONE_CB, sem=0):
    """streams: [(tensor, cb)] (<= 2, each CB holds its whole stream); runtime: per core -> stream arg lists."""

    compile_args = [
        recv_cb,
        recv_tiles,
        senders,
        sem,
        len(streams),
        *[cb for _, cb in streams],
        *[0] * (2 - len(streams)),
        zero_cb,
    ]
    for slot in range(2):
        compile_args += fp.accessor_args(streams[min(slot, len(streams) - 1)][0])
    return fp.reader_kernel(
        MCAST_READER,
        cores,
        compile_args,
        [(core, [a for args in stream_args for a in args]) for core, stream_args in runtime],
    )


def normalize_down(
    residual, gathered_stats, norm_scale, down_inject, *, scaler_mode: str = "chain", matmul: str = "chain"
):
    """Stages 2a+2b as one program: the four norm cores multicast the normalized row into the twelve matmul cores'
    CB (and write it for stage 3) while those cores prefetch their weight column."""

    rows = residual_rows(residual)
    _expect(gathered_stats, (1, BRANCHES, rows, TILE * STATS_TILES), BF16, "GR gathered stats")
    _expect(norm_scale, (1, BRANCHES, TILE, LOCAL_HIDDEN), FP32, "GR norm_scale rows")
    _expect(down_inject, (1, 1, FLAT_WIDTH, PARTIAL_WIDTH), BF16, "GR down_inject")
    mesh = residual.device()
    normalized = fp.allocate((1, 1, rows, FLAT_WIDTH), BF16, ttnn.TILE_LAYOUT, mesh)
    partial = fp.allocate((1, 1, rows, PARTIAL_WIDTH), FP32, ttnn.TILE_LAYOUT, mesh)
    scaler_bits, scaler_dtype = avg_scaler(scaler_mode)
    n_tiles = _tiles_wide(down_inject)
    workers, producers, rect = _rectangle(mesh, 6, 2, rows_above=BRANCHES)
    w_set, p_set, all_set = _core_set(workers), _core_set(producers), _core_set(workers + producers)
    cbs = [
        fp.cb_descriptor(0, BF16, TILE_BF16, HIDDEN_TILES, p_set),
        fp.cb_descriptor(1, BF16, TILE_BF16, STATS_TILES, p_set),
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
    reader = _reader(
        p_set,
        [(gathered_stats, 1), (residual, 0), (norm_scale, 4)],
        [(CONST_SCALER, 2), (CONST_COL_SCALAR, 3)],
        [
            (
                core,
                (
                    [
                        _stream(gathered_stats, 1, STATS_TILES, b * STATS_TILES, 1, 0, STATS_TILES),
                        _stream(residual, 1, HIDDEN_TILES, b * HIDDEN_TILES, 1, 0, 4),
                        _stream(norm_scale, 1, HIDDEN_TILES, b * HIDDEN_TILES, 1, 0, 4),
                    ],
                    [scaler_bits, _bits(EPS)],
                ),
            )
            for b, core in enumerate(producers)
        ],
    )
    norm = fp.compute_kernel(
        NORM, p_set, [HIDDEN_TILES, STATS_TILES, 4, 16], fp32_dest=True, unpack_to_dest_fp32=(4, 7)
    )
    sender = _mcast_writer(
        p_set,
        [(core, (b * HIDDEN_TILES, rect, (b * HIDDEN_TILES, 1), (0, 0, 1, 1))) for b, core in enumerate(producers)],
        src_cb=16,
        dst_cb=8,
        tiles=HIDDEN_TILES,
        tiles_tensor=normalized,
    )
    receiver = _mcast_reader(
        w_set,
        [(down_inject, 9)],
        [(core, [_stream(down_inject, 1, FLAT_TILES, w, n_tiles, 0, 8)]) for w, core in enumerate(workers)],
        recv_cb=8,
        recv_tiles=FLAT_TILES,
        senders=BRANCHES,
    )
    down = fp.compute_kernel(
        DOWN, w_set, [FLAT_TILES, 8, 8, 9, 17, DOWN_SPILL if matmul == "chain" else 0, 10], fp32_dest=True
    )
    writer = _writer(w_set, [(partial, 17)], [(core, [(1, w, 1, 1)]) for w, core in enumerate(workers)])
    # the norm operands and the weight once, the flat row and the partial out; the flat row multicast into the twelve
    # down cores' CB (L1); the norm's four passes per element, then the down+inject matmul
    meta = fp.program_meta(
        NAME,
        "normalize_down",
        rows,
        reads=(residual, gathered_stats, norm_scale, down_inject),
        writes=(normalized, partial),
        l1_bytes=len(workers) * fp.tensor_bytes(normalized),
        flops=4 * rows * FLAT_WIDTH + 2 * rows * FLAT_WIDTH * PARTIAL_WIDTH,
        cores=len(workers) + len(producers),
        outputs=((normalized, 3),),  # the partial is a partial sum the caller hands to its gather
    )
    fp.run_program(
        [residual, gathered_stats, norm_scale, down_inject, normalized, partial],
        fp.program_descriptor(
            [reader, norm, sender, receiver, down, writer], cbs=cbs, semaphores=[fp.semaphore_descriptor(0, all_set)]
        ),
        meta=meta,
    )
    return normalized, partial


def low_rank_gate(gathered_partials, normalized, up, *, matmul: str = "chain"):
    """Stages 3a+3b as one program: the four row cores multicast the low-rank row into the twenty gate cores' CB
    while those cores prefetch their weight columns and normalized tiles; the injection tile comes from row core 3."""

    shape = tuple(gathered_partials.shape)
    if len(shape) != 4 or shape[:2] != (TP_SIZE, 1) or shape[3] != PARTIAL_WIDTH or gathered_partials.dtype != FP32:
        raise ValueError(f"GR gathered partials must be fp32 [4, 1, rows, {PARTIAL_WIDTH}], got {shape}")
    rows = shape[2]
    _expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "GR normalized row")
    _expect(up, (1, 1, PARTIAL_WIDTH, FLAT_WIDTH), BF16, "GR up")
    mesh = gathered_partials.device()
    block = fp.allocate((1, 1, rows, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    injection = fp.allocate((1, 1, rows, BRANCHES), BF16, ttnn.TILE_LAYOUT, mesh)
    t = LOW_RANK_TILES_PER_CORE
    inject_core, inject_local = divmod(INJECT_TILE, t)
    n_tiles = _tiles_wide(up)
    # 5 x 4, not 10 x 2: the harvested p150's contiguous virtual column span is 7 wide (x 0..6 -> 1..7, then a gap),
    # and a multicast needs one NoC rectangle; the consumers' order (x-major) is all the runtime args depend on.
    workers, producers, rect = _rectangle(mesh, 5, 4, rows_above=LOW_RANK_CORES)
    plain = [c for i, c in enumerate(producers) if i != inject_core]
    inject = [producers[inject_core]]
    w_set, p_set, all_set = _core_set(workers), _core_set(producers), _core_set(workers + producers)
    plain_set, inject_set = _core_set(plain), _core_set(inject)
    cbs = [
        fp.cb_descriptor(0, FP32, TILE_FP32, t * TP_SIZE, p_set),
        fp.cb_descriptor(1, FP32, TILE_FP32, 1, p_set),
        fp.cb_descriptor(2, BF16, TILE_BF16, t, p_set),
        fp.cb_descriptor(16, BF16, TILE_BF16, t, p_set),
        fp.cb_descriptor(17, BF16, TILE_BF16, 1, inject_set),
        fp.cb_descriptor(8, BF16, TILE_BF16, PARTIAL_TILES, all_set),
        fp.cb_descriptor(9, BF16, TILE_BF16, BRANCHES * PARTIAL_TILES, w_set),
        fp.cb_descriptor(10, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(11, BF16, TILE_BF16, 1, w_set),
        fp.cb_descriptor(12, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(13, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(14, BF16, TILE_BF16, BRANCHES, w_set),
        fp.cb_descriptor(15, FP32, TILE_FP32, BRANCHES, w_set),
        fp.cb_descriptor(18, BF16, TILE_BF16, 1, w_set),
    ]
    reader = _reader(
        p_set,
        [(gathered_partials, 0)],
        [(CONST_ZERO, 1)],
        [
            (core, ([_stream(gathered_partials, t, TP_SIZE, c * t, PARTIAL_TILES, 1, TP_SIZE)], [0]))
            for c, core in enumerate(producers)
        ],
    )
    computes = [
        fp.compute_kernel(LOWRANK, plain_set, [t, TP_SIZE, t, 16, 17], fp32_dest=True),
        fp.compute_kernel(LOWRANK, inject_set, [t, TP_SIZE, inject_local, 16, 17], fp32_dest=True),
    ]
    senders = [
        _mcast_writer(
            plain_set,
            [(core, (c * t, rect, (0, 1), (0, 0, 1, 1))) for c, core in enumerate(producers) if c != inject_core],
            src_cb=16,
            dst_cb=8,
            tiles=t,
            extra=(injection, NONE_CB),
        ),
        _mcast_writer(
            inject_set,
            [(producers[inject_core], (inject_core * t, rect, (0, 1), (1, 0, 1, 1)))],
            src_cb=16,
            dst_cb=8,
            tiles=t,
            extra=(injection, 17),
        ),
    ]
    receiver = _mcast_reader(
        w_set,
        [(normalized, 10), (up, 9)],
        [
            (
                core,
                [
                    _stream(normalized, 1, BRANCHES, j, HIDDEN_TILES, 0, BRANCHES),
                    _stream(up, BRANCHES, PARTIAL_TILES, j, n_tiles, HIDDEN_TILES, PARTIAL_TILES),
                ],
            )
            for j, core in enumerate(workers)
        ],
        recv_cb=8,
        recv_tiles=PARTIAL_TILES,
        senders=LOW_RANK_CORES,
        zero_cb=11,
    )
    gate_k = fp.compute_kernel(
        GATE,
        w_set,
        [PARTIAL_TILES, BRANCHES, 8, 9, 10, 11, 12, 13, 14, 18, UP_SPILL if matmul == "chain" else 0, 15],
        fp32_dest=True,
        unpack_to_dest_fp32=(10, 12, 13),
    )
    writer = _writer(w_set, [(block, 18)], [(core, [(1, j, 1, 1)]) for j, core in enumerate(workers)])
    # the partials, the flat row and the up weight once, the block and the injection out; the low-rank row multicast
    # into the twenty gate cores' CB (L1); the fold, typecast, silu and 2 sigmoid, the up matmul, sigmoid, gate
    # multiply and branch sum
    meta = fp.program_meta(
        NAME,
        "low_rank_gate",
        rows,
        reads=(gathered_partials, normalized, up),
        writes=(block, injection),
        l1_bytes=len(workers) * PARTIAL_TILES * TILE_BF16,
        flops=(TP_SIZE + 3) * rows * PARTIAL_WIDTH + 2 * rows * PARTIAL_WIDTH * FLAT_WIDTH + 3 * rows * FLAT_WIDTH,
        cores=len(workers) + len(producers),
        outputs=((block, 3), (injection, None)),
    )
    fp.run_program(
        [gathered_partials, normalized, up, block, injection],
        fp.program_descriptor(
            [reader, *computes, *senders, receiver, gate_k, writer],
            cbs=cbs,
            semaphores=[fp.semaphore_descriptor(0, all_set)],
        ),
        meta=meta,
    )
    return block, injection


def merged_enabled(environ=None) -> bool:
    """The merged forms (5 programs per read) unless QWEN38_FUSED_GR_READ_MERGED=0 selects the split forms (7)."""

    import os

    value = (environ if environ is not None else os.environ).get(MERGED_ENV, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{MERGED_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


def _topology(module, tensor, shard_dim: int | None) -> None:
    """Record the placement the chain's ops would have given ``tensor`` (a stock op's output the module re-reads; the
    fused launches declare their outputs' placements in their program meta and ``fp.run_program`` stamps them)."""

    reference = module.weights.replicated_anchor.tensor_topology()
    placements = [
        ttnn.PlacementReplicate(),
        ttnn.PlacementReplicate() if shard_dim is None else ttnn.PlacementShard(shard_dim),
    ]
    tensor.update_tensor_topology(
        ttnn.TensorTopology(reference.distribution_shape(), placements, reference.mesh_coords())
    )


def gr_read_fused(
    module,
    residual,
    *,
    scaler_mode: str = "chain",
    merged: bool | None = None,
    matmul: str | None = None,
    stats_gather=None,
    partial_gather=None,
    read_front=None,
):
    """The model-level read: the fused programs around the chain's two collectives; returns (block input, state).
    ``merged`` (default: on, unless the QWEN38_FUSED_GR_READ_MERGED=0 switch) runs normalize+down and low_rank+gate
    as one program each.  ``stats_gather(residual)`` replaces ``stats`` + ``ttnn.all_gather`` with a program that
    gathers inside itself (ttnn/fused/gr_fold); it returns the same ``[1, 4, rows, 128]`` bf16 tiles.
    ``partial_gather(residual, gathered_stats, gamma_rows, down_inject)`` likewise replaces ``normalize_down`` +
    ``all_gather_async`` (merged forms only); it returns ``(normalized, gathered_partials)`` with the same
    ``[4, 1, rows, 384]`` fp32 tiles.  ``read_front(residual, gamma_rows, down_inject)`` replaces all four (stats,
    all_gather, normalize_down, all_gather_async) with one program and returns ``(normalized, gathered_partials)``."""

    from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import TensorPlacement
    from models.demos.blackhole.qwen38_flash_next.ttnn.gr import Qwen38TTNNGatedResidualState

    rows = residual_rows(residual)
    module.mesh_contract.validate_tensor(residual, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    merged = merged_enabled() if merged is None else merged
    matmul = matmul_mode() if matmul is None else matmul
    gamma_rows = module.weights.norm_scale_rows
    if gamma_rows is None:
        raise RuntimeError("the fused GR read needs weights.norm_scale_rows (gamma/4 repeated over the tile rows)")
    if (partial_gather is not None or read_front is not None) and not merged:
        raise ValueError(
            "partial_gather / read_front fold the merged normalize_down form; the split forms keep the collectives"
        )
    if read_front is not None:
        normalized, gathered_partials = read_front(residual, gamma_rows, module.weights.down_inject)
    else:
        if stats_gather is None:
            stats_local = stats(residual)
            gathered_stats = ttnn.all_gather(
                stats_local, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(stats_local)
        else:
            gathered_stats = stats_gather(residual)
        module.mesh_contract.validate_tensor(gathered_stats, placement=TensorPlacement.REPLICATED)
    if read_front is not None:
        pass
    elif partial_gather is not None:
        normalized, gathered_partials = partial_gather(residual, gathered_stats, gamma_rows, module.weights.down_inject)
        ttnn.deallocate(gathered_stats)
    else:
        if merged:
            normalized, partial = normalize_down(
                residual, gathered_stats, gamma_rows, module.weights.down_inject, scaler_mode=scaler_mode, matmul=matmul
            )
        else:
            normalized = normalize(residual, gathered_stats, gamma_rows, scaler_mode=scaler_mode)
            partial = down_project(normalized, module.weights.down_inject, matmul=matmul)
        ttnn.deallocate(gathered_stats)
        module._mark_partial(partial, (1, 1, rows, PARTIAL_WIDTH))
        if module.collective_topology != ttnn.Topology.Linear or module.tt_ccl is None:
            raise RuntimeError("GR partial gather requires the TP4 Linear topology and the TT-CCL manager")
        gathered_partials = ttnn.experimental.all_gather_async(
            partial,
            persistent_output_buffer=None,
            dim=0,
            multi_device_global_semaphore=module.tt_ccl.get_and_cycle_ag_semaphore_handles(TP_AXIS),
            num_links=module.tt_ccl.get_num_links(TP_AXIS),
            cluster_axis=TP_AXIS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            barrier_semaphore=module.tt_ccl.get_and_cycle_barrier_semaphore_handle(TP_AXIS),
            chunks_per_sync=1,
            num_workers_per_link=1,
            num_buffers_per_channel=2,
        )
        ttnn.deallocate(partial)
    if merged:
        block, injection = low_rank_gate(gathered_partials, normalized, module.weights.up, matmul=matmul)
    else:
        low_rank_row, injection = low_rank(gathered_partials)
        block = gate(low_rank_row, normalized, module.weights.up, matmul=matmul)
        ttnn.deallocate(low_rank_row)
    ttnn.deallocate(gathered_partials)
    ttnn.deallocate(normalized)
    module.mesh_contract.validate_tensor(block, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
    module.mesh_contract.validate_tensor(injection, placement=TensorPlacement.REPLICATED)
    return block, Qwen38TTNNGatedResidualState(residual=residual, injection=injection)


def gr_read_composed(module, residual):
    """The chain as written in ttnn/gr.py (the class method, whatever the instance resolved to)."""

    return type(module).read(module, residual)


register(
    FusedKernel(
        name=NAME,
        replaces="the GR read chain: FillPad, rms_norm_pre/post_all_gather, gamma multiply, down+inject linear, "
        "to_memory_config x2, fast_reduce_nc x2, typecast, slice, silu, up linear, sigmoid, gate multiply, "
        "2*sigmoid (18 programs per read, 96 reads per step; the two collectives stay)",
        tolerance=BITWISE,
        fused=gr_read_fused,
        composed=gr_read_composed,
        gate=None,  # needs the TP4 collectives: component gate = the single-chip replica tool (one chip, four slices in turn), model gate = the acceptance table
    )
)

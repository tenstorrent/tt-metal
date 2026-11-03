# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The GR gate program v3 (``QWEN38_GR_GATE_V3=1``, default off): ``gr_read.low_rank_gate`` with
  * the low-rank cores' 24 gathered partial tiles read in ONE batch (one DRAM read barrier instead of three);
  * the SFPU stages (silu, the injection sigmoid, the gate sigmoid, the gate multiply) on rows 0..15 only (VectorMode::R);
decode rows <= 16 only (anything else runs ``gr_read.low_rank_gate``).  Bit-identical valid rows.
"""

from __future__ import annotations

import ttnn

from ...contracts import TP_SIZE
from .. import program as fp
from . import (
    BF16,
    BRANCHES,
    CONST_ZERO,
    FLAT_WIDTH,
    FP32,
    GATE_HEIGHT,
    GATE_WIDTH,
    HIDDEN_TILES,
    INJECT_TILE,
    LOCAL_HIDDEN,
    LOW_RANK_TILES_PER_CORE,
    NAME,
    NONE_CB,
    PARTIAL_TILES,
    PARTIAL_WIDTH,
    TILE_BF16,
    TILE_FP32,
    UP_SPILL,
    _core_set,
    _expect,
    _mcast_reader,
    _mcast_writer,
    _reader,
    _rectangle,
    _stream,
    _tiles_wide,
    _writer,
)

ENV = "QWEN38_GR_GATE_V3"
MAX_ROWS = 16
MAX_LIVE_ROWS = 5  # the verify rows (k + 1) the live-row form serves; the front v3 admits the same
LOWRANK3 = fp.kernel_source(NAME, "lowrank_compute3.cpp")
GATE3 = fp.kernel_source(NAME, "gate_compute3.cpp")

# QWEN38_GR_GATE_TINY (default "1"), rows == 1: every CB of the program except the up weight's holds 1 x 32 tiles (row 0
# only: face 0 row 0 then face 1 row 0, 64 B BF16 / 128 B FP32) instead of 32 x 32 tiles.  The compute kernels are the
# same files; the readers and writers move the two valid-row chunks of a DRAM page (reader_t / mcast_reader_t / writer_t /
# mcast_writer_t).  Row 0 of the block and the injection is bit for bit today's; the other rows of their pages are not written.
TINY_ENV = "QWEN38_GR_GATE_TINY"
TINY_BF16, TINY_FP32 = 2 * fp.TILE, 4 * fp.TILE
FACE_FP32_BYTES, FACE_BF16_BYTES = 16 * 16 * 4, 16 * 16 * 2  # face 1 sits one face after face 0 in a DRAM tile page
SCRATCH_CB = 19  # the worker reader's staging of face 1 row 0 (BRANCHES x 64 B)
READER_T = fp.kernel_source(NAME, "reader_t.cpp")
MCAST_READER_T = fp.kernel_source(NAME, "mcast_reader_t.cpp")
MCAST_WRITER_T = fp.kernel_source(NAME, "mcast_writer_t.cpp")
WRITER_T = fp.kernel_source(NAME, "writer_t.cpp")

# QWEN38_GR_ROWS_FAST (default on), rows 2..8: the tiny-tile scheme for more than one row.  Every CB of the program except the up weight's holds
# 8 x 32 tiles (two 8-row faces: BF16 512 B, FP32 1024 B); the readers and writers move only the live rows (rows 0..rows-1 of face 0 and
# of face 1, rows * 32 B BF16 / rows * 64 B FP32) of a DRAM page (reader_r / mcast_reader_r / writer_r / mcast_writer_r).  In the live-row
# form (out_rows = 32 > rows) the writers also zero the dead rows of the L1 tile and write the rest of the 32-row page from a zero tile in
# one barrier (gr_rows_out.h), so the pages equal the one-row program's.  Valid rows of the block and the injection are bit for bit the one-row program's.
ROWS_FAST_MIN, ROWS_FAST_MAX = 2, 8
READER_R = fp.kernel_source(NAME, "reader_r.cpp")
MCAST_READER_R = fp.kernel_source(NAME, "mcast_reader_r.cpp")
MCAST_WRITER_R = fp.kernel_source(NAME, "mcast_writer_r.cpp")
WRITER_R = fp.kernel_source(NAME, "writer_r.cpp")
ROWS_BF16, ROWS_FP32 = 8 * 32 * 2, 8 * 32 * 4


def rows_fast_enabled(rows: int) -> bool:
    from . import rows_fast

    return rows_fast() and ROWS_FAST_MIN <= rows <= ROWS_FAST_MAX


ROWTRIM_ENV = "QWEN38_GR_GATE_ROWTRIM"
ROWTRIM_MAX_ROWS = 8
ROWTRIM_ITERATIONS = 2  # rows 0..3 of faces 0 and 1 (both column parities); the default kernels run 8


def trim_iterations(rows: int) -> int:
    """One iteration is half of 4 rows (one column parity): 2 iterations per 4 valid rows (rows 1..4 -> 2, 5..8 -> 4)."""
    return ROWTRIM_ITERATIONS * -(-rows // 4)


def rowtrim_defines(rows: int):
    """Kernel defines of the two GR compute kernels: with ``QWEN38_GR_GATE_ROWTRIM=1`` (default; "0" off) and ``rows <= 4`` every
    SFPU op runs ``ROWTRIM_ITERATIONS`` iterations per face (VectorMode::R) instead of 8; empty otherwise (kernels as before).
    """
    import os

    if os.environ.get(ROWTRIM_ENV, "1") == "1" and 1 <= rows <= ROWTRIM_MAX_ROWS:
        return [("QWEN38_GR_ROWTRIM_ITERS", str(trim_iterations(rows)))]
    return []


def tiny_enabled(rows: int) -> bool:
    import os

    return os.environ.get(TINY_ENV, "1") == "1" and rows == 1


def enabled(rows: int) -> bool:
    import os

    return os.environ.get(ENV, "1") == "1" and 1 <= rows <= MAX_ROWS  # default on (x49 A/B -0.41 ms)


def low_rank_gate3(gathered_partials, normalized, up, *, matmul: str = "chain", out_rows: int | None = None):
    """Stages 3a+3b as one program: the four row cores multicast the low-rank row into the gate cores' CB
    while those cores prefetch their weight columns and normalized tiles; the injection tile comes from row core 3.

    ``out_rows`` (the verify's 32-row tile, ``rows < out_rows``, rows >= 2): the block and the injection are allocated with
    ``out_rows`` rows and their tiles are zero-filled past the ``rows`` live rows before they are written, so the consumers see
    the 32-row form (dead rows exact zeros; the injection's dead rows are 0 where the 32-row program computes 1.0)."""

    shape = tuple(gathered_partials.shape)
    if len(shape) != 4 or shape[:2] != (TP_SIZE, 1) or shape[3] != PARTIAL_WIDTH or gathered_partials.dtype != FP32:
        raise ValueError(f"GR gathered partials must be fp32 [{TP_SIZE}, 1, rows, {PARTIAL_WIDTH}], got {shape}")
    rows = shape[2]
    from .. import row_gather3  # QWEN38_GATE_GATHER=1: this program with the block-input gather folded in (row_gather3)

    gather_config = row_gather3.take_context()
    if (
        gather_config is not None
        and out_rows in (None, rows)
        and row_gather3.admits(gathered_partials, rows, gather_config)
    ):
        return row_gather3.gate_gather(gathered_partials, normalized, up, gather_config, matmul=matmul)
    _expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "GR normalized row")
    _expect(up, (1, 1, PARTIAL_WIDTH, FLAT_WIDTH), BF16, "GR up")
    mesh = gathered_partials.device()
    live = out_rows is not None and out_rows != rows
    if live and not (2 <= rows <= MAX_LIVE_ROWS and rows < out_rows):
        raise ValueError(f"GR gate live rows must be 2..{MAX_LIVE_ROWS} below the {out_rows}-row tile, got {rows}")
    out = out_rows if live else rows
    block = fp.allocate((1, 1, out, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    injection = fp.allocate((1, 1, out, BRANCHES), BF16, ttnn.TILE_LAYOUT, mesh)
    v2 = False
    t = LOW_RANK_TILES_PER_CORE
    n_producers = PARTIAL_TILES // t
    inject_core, inject_local = divmod(INJECT_TILE, t)
    n_tiles = _tiles_wide(up)
    # 5 wide (5 x 4 at TP4, 5 x 2 at TP8), not 10 wide: the harvested p150's contiguous virtual column span is 7 wide (x 0..6 -> 1..7, then a gap),
    # and a multicast needs one NoC rectangle; the consumers' order (x-major) is all the runtime args depend on.
    workers, producers, rect = _rectangle(mesh, GATE_WIDTH, GATE_HEIGHT, rows_above=n_producers, wrap=v2)
    plain = [c for i, c in enumerate(producers) if i != inject_core]
    inject = [producers[inject_core]]
    w_set, p_set, all_set = _core_set(workers), _core_set(producers), _core_set(workers + producers)
    plain_set, inject_set = _core_set(plain), _core_set(inject)
    tiny = tiny_enabled(rows)
    fast = rows_fast_enabled(rows)
    bf16_page, fp32_page = (
        (TINY_BF16, TINY_FP32) if tiny else (ROWS_BF16, ROWS_FP32) if fast else (TILE_BF16, TILE_FP32)
    )
    shape_rows = 1 if tiny else 8 if fast else None  # the 1 x 32 or 8 x 32 tile
    cbs = [
        fp.cb_descriptor(0, FP32, fp32_page, t * TP_SIZE, p_set, shape_rows),
        fp.cb_descriptor(1, FP32, fp32_page, 1, p_set, shape_rows),
        fp.cb_descriptor(2, BF16, bf16_page, t, p_set, shape_rows),
        fp.cb_descriptor(16, BF16, bf16_page, t, p_set, shape_rows),
        fp.cb_descriptor(17, BF16, bf16_page, 1, inject_set, shape_rows),
        fp.cb_descriptor(8, BF16, bf16_page, PARTIAL_TILES, all_set, shape_rows),
        fp.cb_descriptor(9, BF16, TILE_BF16, BRANCHES * PARTIAL_TILES, w_set),  # the up weight: whole tiles
        fp.cb_descriptor(10, BF16, bf16_page, BRANCHES, w_set, shape_rows),
        fp.cb_descriptor(11, BF16, bf16_page, 1, w_set, shape_rows),
        fp.cb_descriptor(12, BF16, bf16_page, BRANCHES, w_set, shape_rows),
        fp.cb_descriptor(13, BF16, bf16_page, BRANCHES, w_set, shape_rows),
        fp.cb_descriptor(14, BF16, bf16_page, BRANCHES, w_set, shape_rows),
        fp.cb_descriptor(15, FP32, fp32_page, BRANCHES, w_set, shape_rows),
        fp.cb_descriptor(18, BF16, bf16_page, 1, w_set, shape_rows),
    ]
    if tiny:
        cbs.append(fp.cb_descriptor(SCRATCH_CB, BF16, TINY_BF16, BRANCHES, w_set))
    reader_kw = (
        {"source": READER_T, "defines": [("GT_CHUNK", str(TINY_FP32 // 2)), ("GT_SRC_FACE", str(FACE_FP32_BYTES))]}
        if tiny
        else {}
    )
    mcast_writer_kw = {"source": MCAST_WRITER_T, "defines": [("GT_DST_FACE", str(FACE_BF16_BYTES))]} if tiny else {}
    mcast_reader_kw = (
        {
            "source": MCAST_READER_T,
            "defines": [("GT_SRC_FACE", str(FACE_BF16_BYTES)), ("GT_SCRATCH_CB", str(SCRATCH_CB))],
        }
        if tiny
        else {}
    )
    writer_kw = {"source": WRITER_T, "defines": [("GT_DST_FACE", str(FACE_BF16_BYTES))]} if tiny else {}
    if fast:
        common = [("GT_ROWS", str(rows)), ("GT_DST_FACE", str(FACE_BF16_BYTES))]
        reader_kw = {
            "source": READER_R,
            "defines": [("GT_CHUNK", str(rows * 64)), ("GT_SRC_FACE", str(FACE_FP32_BYTES))],
        }
        mcast_reader_kw = {
            "source": MCAST_READER_R,
            "defines": [("GT_CHUNK", str(rows * 32)), ("GT_SRC_FACE", str(FACE_BF16_BYTES))],
        }
        writer_kw = {"source": WRITER_R, "defines": common + ([("GT_ZERO_CB", "11")] if live else [])}
        mcast_writer_kw = {"source": MCAST_WRITER_R, "defines": common + ([("GT_ZERO_CB", "1")] if live else [])}
    elif (
        live
    ):  # not tiny (rows >= 2): the stock writer and mcast writer with the dead-row zero fill (kernels/writer.cpp, mcast_phase.h)
        writer_kw = {"defines": [("GR_LIVE_ROWS", str(rows)), ("GR_LIVE_ZERO_CB", "11")]}
        mcast_writer_kw = {"defines": [("GR_LIVE_ROWS", str(rows)), ("GR_LIVE_ZERO_CB", "1")]}
    reader = _reader(
        p_set,
        [(gathered_partials, 0)],
        [(CONST_ZERO, 1)],
        [
            (core, ([_stream(gathered_partials, t, TP_SIZE, c * t, PARTIAL_TILES, 1, t * TP_SIZE)], [0]))
            for c, core in enumerate(producers)
        ],
        **reader_kw,
    )
    trim = rowtrim_defines(rows)
    computes = [
        fp.compute_kernel(LOWRANK3, plain_set, [t, TP_SIZE, t, 16, 17], defines=trim, fp32_dest=True),
        fp.compute_kernel(LOWRANK3, inject_set, [t, TP_SIZE, inject_local, 16, 17], defines=trim, fp32_dest=True),
    ]
    senders = [
        _mcast_writer(
            plain_set,
            [(core, (c * t, rect, (0, 1), (0, 0, 1, 1))) for c, core in enumerate(producers) if c != inject_core],
            src_cb=16,
            dst_cb=8,
            tiles=t,
            extra=(injection, NONE_CB),
            **mcast_writer_kw,
        ),
        _mcast_writer(
            inject_set,
            [(producers[inject_core], (inject_core * t, rect, (0, 1), (1, 0, 1, 1)))],
            src_cb=16,
            dst_cb=8,
            tiles=t,
            extra=(injection, 17),
            **mcast_writer_kw,
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
                    _stream(up, BRANCHES, PARTIAL_TILES, j, n_tiles, HIDDEN_TILES, BRANCHES * PARTIAL_TILES),
                ],
            )
            for j, core in enumerate(workers)
        ],
        recv_cb=8,
        recv_tiles=PARTIAL_TILES,
        senders=n_producers,
        zero_cb=11,
        **mcast_reader_kw,
    )
    gate_k = fp.compute_kernel(
        GATE3,
        w_set,
        [PARTIAL_TILES, BRANCHES, 8, 9, 10, 11, 12, 13, 14, 18, UP_SPILL if matmul == "chain" else 0, 15],
        defines=trim,
        fp32_dest=True,
        unpack_to_dest_fp32=(10, 12, 13),
    )
    writer = _writer(w_set, [(block, 18)], [(core, [(1, j, 1, 1)]) for j, core in enumerate(workers)], **writer_kw)
    # the partials, the flat row and the up weight once, the block and the injection out; the low-rank row multicast
    # into the gate cores' CB (L1); the fold, typecast, silu and 2 sigmoid, the up matmul, sigmoid, gate
    # multiply and branch sum
    meta = fp.program_meta(
        NAME,
        "low_rank_gate_v3",
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

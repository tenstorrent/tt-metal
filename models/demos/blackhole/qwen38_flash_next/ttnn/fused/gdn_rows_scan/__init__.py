# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``gdn_rows_scan``: the MTP verify rows' GDN body as ONE program per layer -- the serial fp32 recurrence of the
fused decode step ``gdn_step``, row after row over the k + 1 verify rows, with every prefix state kept.

Stage B of the verify-rows scan design (the fold).  One GDN layer's verify body becomes

    projection linear + S2I -> qkv landing slice -> gdn_rows_scan -> out-projection linear + reduce-scatter

where the wrap (``gdn_rows_wrap``, today's default) runs six programs between the S2I and the out-projection
(``gdn_pre_rows`` -> ``chunk_gdn_prep`` -> ``chunk_gdn_scan`` -> ``post_cast`` -> ``post_norm`` on the chunk / WY form
at TF32 operand precision) and the chain 53.  The program runs gdn_step's compute body in a row loop: conv + SiLU and
both l2 norms once over the 32-row tile, the gate scalars of every row in one SFPU pass over the a / b tiles (element
(row, head) of beta_all / decay_all, which the reader replicates into one exact fp32 tile per row), then per row
``S' = S * exp(g)``, ``v_read = k S'``, ``delta = (v - v_read) * beta`` on that row alone (the one-hot row mask), ``S =
S' + k^T delta`` and ``o = (q S) * 128^-0.5``, the state carried row to row in L1; then the gated RMSNorm once on the
assembled o rows.  Every LLK call, rounding point and DST accumulation order is gdn_step's (the gate ops are
lane-wise, so a whole-tile pass gives each element the bits the per-row scalar chain gave it), so the program is
BITWISE with ``rows`` sequential fused ``gdn_step`` calls from the same state (the die proof), and its numerics class against today's MTP stream (the chunk
form) is COMPONENT.

Outputs per layer: the gated tile ``[1, 1, 32, 1536]`` bf16 straight into the out-projection's activation shard (rows
>= ``rows`` exact zeros) and the PREFIX STATES ``prefix[a]`` = the state after ``a + 1`` rows, ``a = 0 .. rows-1``,
``[rows, 12, 128, 128]`` fp32, persistent with the rows state.  Row 0 (the base token) is always committed, so slot
``a`` is exactly the committed state for ``accepted = a``: the commit becomes one data-movement program (``pick``:
read the pass's device accept count, copy that slot into ``state.recurrent``) plus the chain's own history advance,
instead of the wrap's masked re-run of the two prims (its TF32-class passthrough disappears).  ``forward_rows``
under the fold returns ``final_state=None``: the state after all rows is prefix slot ``rows - 1``, owned by the rows
state, and the verify body only frees the result's final state.

On by default since 2026-09-26 (class COMPONENT with its component-gate record: the probe's rows column against the
CPU oracle; ``QWEN38_FUSED_OFF=gdn_rows_scan`` restores the wrap, and the served stream's pins are this kernel's:
plain decode's divergence on 12 of 12 records).  Admission is a property of the rows state, decided when it is
allocated (``attach``): the 32-row verify form with ``1 <= rows <= MAX_ROWS``, its own body, device tensors.  With
the fold attached the wrap is not; ``QWEN38_FUSED_OFF`` semantics are unchanged.

Work items: ``SPLIT`` state column blocks per value head -- 1 (12 items, one head per core; the bring-up form) or 4
(48 items, one 32-column block of the head's state per core; every per-tile op of the recurrence touches only its
own column block, so the split changes which tiles a core computes and never the order inside a tile; the four cores
of a head exchange their o-row tiles before the gated norm).  ``reference_rows`` is the claim in torch: ``rows``
sequential ``gdn_step.reference_step`` calls with the FIR ring emulated from the history rows.
"""

from __future__ import annotations

import os

import torch

import ttnn

from .. import program as fp
from .. import gdn_rows_wrap as wrap
from .. import gdn_step
from .. import gr_read
from ..gdn_step import (
    A_COLUMN,
    B_COLUMN,
    HEAD_DIM,
    HEADS,
    PROJECTION_WIDTH,
    QK_WIDTH,
    QKV_WIDTH,
    STATE_TILES,
    VALUE_WIDTH,
)
from ..registry import COMPONENT, FusedKernel, enabled, register

NAME = "gdn_rows_scan"
READER = fp.kernel_source(NAME, "reader.cpp")
WRITER = fp.kernel_source(NAME, "writer.cpp")
COMPUTE = fp.kernel_source(NAME, "compute.cpp")
PICK = fp.kernel_source(NAME, "pick.cpp")
# the lanes form (one (value head, lane) item per core, the whole head per core, compute.cpp shared)
READER_LANES = fp.kernel_source(NAME, "reader_lanes.cpp")
WRITER_LANES = fp.kernel_source(NAME, "writer_lanes.cpp")
PICK_LANES = fp.kernel_source(NAME, "pick_lanes.cpp")

TILE = fp.TILE  # the verify tile: 32 rows
HT = HEAD_DIM // TILE  # 4 column tiles per head
HISTORY_ROWS = 3  # the FIR history rows before row 0
# State column blocks per value head: 1 = one head per core (12 items), 4 = one 32-column block per core (48 items).
# The served form is 4 (MEASURED on one p150 die at R = 5, traced: 113 us per call against 160 us for the 12-item form,
# both bitwise the sequential gdn_step calls); ``run(..., split=1)`` keeps the 12-item form for the device test.
SPLIT = 4
# The most verify rows the fold admits (k + 1; the prefix states are rows x 786 KB per layer per device).
MAX_ROWS = 8
# The lanes form: B lanes x R rows (R = k + 1 <= MAX_ROWS) lane-major in one tile, B * R <= 32, one (value head,
# lane) item per core (12 B cores, the whole head per core), the state [B, 12, 128, 128] and the prefix states
# [B * R, 12, 128, 128] (slot u * R + j = lane u after j + 1 rows; the single form's tensor with rows = B * R).
MAX_LANES = 8
# The layer's verify segment with the fold on: the all-gather, the projection linear, its S2I, the qkv landing slice,
# this program, the out-projection linear, the reduce-scatter (the wrap runs 11, the chain 58); the commit: the pick,
# the history concat and its select matmul (the wrap runs 5).
PROGRAMS_PER_LAYER = 7
COMMIT_PROGRAMS_PER_LAYER = 3

BF16, FP32 = ttnn.bfloat16, ttnn.float32

# Circular buffers of the scan program (index, dtype, pages), the indices shared with the three kernels; the state
# hand-off CB_SNEWC (30) / CB_OUTS (25) is one allocation under two indices (the compute reads the next row's state
# back, the writer drains it), at twice the item's state tiles so a row's pack overlaps the previous state's drain.
CB_STATE, CB_SNEW, CB_OUTS, CB_DEBUG, CB_SNEWC, CB_OROWS, CB_OBF = 7, 24, 25, 28, 30, 29, 31
# debug taps (DEBUG_TAPS builds) in program order, one fp32 tile each: per item, then per row
ITEM_TAPS = ("conv_q0", "conv_v0", *(f"q{t}" for t in range(4)), *(f"k{t}" for t in range(4)))  # the conv and the norms
GATE_TAPS = ("beta", "decay")  # per row, in the gates block after the norms (the reader's replicated tiles)
LATE_TAPS = ("kcol0",)  # after the gates: the k columns
ROW_TAPS = ("sdec0", "vread0", "deltab0", "snew0", "o0")  # per row, in the recurrence
# fp32 CBs consumed only by copy_tile (exact unpack to DST); matmul, reduce and broadcast operands stay Default
FP32_COPY_CBS = (5, 7, 18, 19, 21, 22, 30)


def cb_table(rows: int, vbt: int) -> tuple[tuple[int, object, int], ...]:
    """The plain CBs of the scan program for ``rows`` real rows and ``vbt`` state column tiles per item."""

    nt = 2 * HT + vbt  # conv tiles per item: q, k, the item's v tiles
    st = HT * vbt  # state tiles per item
    return (
        (0, BF16, nt),  # P: the projection's q|k|v tiles (conv slot 3)
        (1, BF16, 3 * nt),  # S: the three shifted slots per tile
        (2, BF16, 4 * nt),  # T: the four taps per tile
        (3, BF16, HT),  # Z
        (4, BF16, 2),  # AB: the a and b tiles (every row's gate scalars; the gates take all rows in one pass)
        (5, FP32, 2),  # DTNA
        (6, BF16, HT),  # W
        (7, FP32, st),  # STATE (S at pass start)
        (8, BF16, rows),  # MASK: one-hot row masks
        (9, FP32, 2),  # SCALER
        (10, BF16, 4),  # CONVSUM
        (11, BF16, nt),  # QKV: silu(conv)
        (12, BF16, HT),  # SQ / NRMW
        (13, FP32, 2),  # GALL: beta_all / decay_all (compute -> reader, which replicates them per row)
        (14, BF16, HT),  # UNIT / NRM
        (15, FP32, HT),  # QROW
        (16, FP32, HT),  # KROW
        (17, FP32, HT),  # KCOL
        (18, FP32, rows),  # BETA: one tile per row
        (19, FP32, rows),  # DECAY: one tile per row
        (20, FP32, st),  # SDEC
        (21, FP32, st),  # SDECC
        (22, FP32, HT),  # VREAD / SIG
        (23, FP32, HT),  # DELTAB / SQO
        (24, FP32, st),  # SNEW (the o matmul's operand)
        (26, BF16, HT),  # OUTG
        (27, FP32, 1),  # RS
        (29, BF16, HT),  # OROWS: the assembled o rows (writer -> compute)
        (31, BF16, 2 * vbt),  # OBF: each row's o tiles (compute -> writer)
    )


def _release(*tensors) -> None:
    for tensor in tensors:
        if tensor is not None and tensor.is_allocated():
            ttnn.deallocate(tensor)


def _items(split: int) -> list[tuple[int, int]]:
    return [(head, vb) for head in range(HEADS) for vb in range(split)]


def debug_tiles(rows: int) -> int:
    """Tap tiles per item of a DEBUG_TAPS build: the item taps, then the row taps for every row."""

    return len(ITEM_TAPS) + len(LATE_TAPS) + (len(GATE_TAPS) + len(ROW_TAPS)) * rows


def run(projected, history, taps, constants, norm, recurrent, prefix, out, *, split: int | None = None, debug=None):
    """The scan program on explicit tensors (local shapes): ``projected`` [1,1,32,4160] bf16 TILE (the rows past the
    real ones may hold anything finite); ``history`` [1,1,32,2560] bf16 (rows 0..2 valid); ``taps`` 4 x [1,1,1,2560]
    bf16; ``constants`` from ``gdn_step.constant_tiles``; ``norm`` [1,1,1,128] bf16; ``recurrent`` [1,12,128,128]
    fp32 (read only); ``prefix`` [rows,12,128,128] fp32 (written: slot a = the state after a + 1 rows); ``out``
    [1,1,32,1536] bf16 (any memory layout the TensorAccessor addresses; rows >= rows exact zeros).  ``split`` overrides
    the module's ``SPLIT`` (the device test measures both forms); ``debug`` an optional ``[items, debug_tiles(rows), 32,
    32]`` fp32 tensor that receives the DEBUG_TAPS build's intermediates."""

    rows = int(prefix.shape[0])
    split = SPLIT if split is None else int(split)
    if split not in (1, 4):
        raise ValueError(f"{NAME} split is 1 (one head per core) or 4 (one column block per core), got {split}")
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"{NAME} admits 1..{MAX_ROWS} rows, got {rows}")
    if not fp.is_row_tile(projected) or fp.tile_width_of(projected) != PROJECTION_WIDTH:
        raise ValueError(f"{NAME} projection must be one row tile of width {PROJECTION_WIDTH}, got {projected.shape}")
    if tuple(recurrent.shape) != (1, HEADS, HEAD_DIM, HEAD_DIM) or recurrent.dtype != FP32:
        raise ValueError(f"{NAME} state must be [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}] fp32, got {recurrent.shape}")
    if tuple(prefix.shape) != (rows, HEADS, HEAD_DIM, HEAD_DIM) or prefix.dtype != FP32:
        raise ValueError(
            f"{NAME} prefix states must be [rows, {HEADS}, {HEAD_DIM}, {HEAD_DIM}] fp32, got {prefix.shape}"
        )
    if tuple(history.shape) != (1, 1, TILE, QKV_WIDTH) or history.dtype != BF16:
        raise ValueError(f"{NAME} history must be [1, 1, {TILE}, {QKV_WIDTH}] bf16, got {history.shape}")
    if len(taps) != 4:
        raise ValueError(f"{NAME} takes four conv taps")
    mesh = projected.device()
    vbt = HT // split
    items = _items(split)
    work = fp.split_work(len(items), mesh)
    cores = fp.core_rectangle(work, mesh)
    if len(work) != len(items):
        raise RuntimeError(f"{NAME} places one item per core; {len(items)} items got {len(work)} cores")

    def pairs(w):
        return [value for item in items[w.start : w.start + w.count] for value in item]

    # The column-block form: the four cores of a head exchange their o-row tiles through L1, so each item's writer
    # takes its three peers' NoC coordinates (measured once per mesh: the logical -> NoC map is not linear).
    peers_of = {}
    if split > 1:
        noc = gr_read.noc_map(mesh)
        core_of = {item: w.core for item, w in zip(items, work)}
        for head, vb in items:
            peers_of[(head, vb)] = [
                value
                for other in range(split)
                if other != vb
                for value in noc[(core_of[(head, other)].x, core_of[(head, other)].y)]
            ]

    def writer_args(w):
        return [value for item in items[w.start : w.start + w.count] for value in (*item, *peers_of.get(item, ()))]

    reader_cta = [rows, vbt]
    for tensor in (projected, history, *taps, constants, norm, recurrent):
        reader_cta.extend(fp.accessor_args(tensor))
    reader_addrs = [
        projected.buffer_address(),
        history.buffer_address(),
        *(t.buffer_address() for t in taps),
        constants.buffer_address(),
        norm.buffer_address(),
        recurrent.buffer_address(),
    ]
    writer_cta = [rows, vbt, *fp.accessor_args(prefix), *fp.accessor_args(out)]
    writer_addrs = [prefix.buffer_address(), out.buffer_address()]
    defines = [("INP_FLOAT32", "1")]
    if debug is not None:
        writer_cta.extend(fp.accessor_args(debug))
        writer_addrs.append(debug.buffer_address())
        defines.append(("DEBUG_TAPS", "1"))
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores)
        for index, dtype, pages in cb_table(rows, vbt)
    ]
    if debug is not None:
        cbs.append(fp.cb_descriptor(CB_DEBUG, FP32, fp.TILE_BYTES[FP32], debug_tiles(rows), cores))
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * HT * vbt * fp.TILE_BYTES[FP32],
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SNEWC, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
                ttnn.CBFormatDescriptor(buffer_index=CB_OUTS, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
            ],
        )
    )
    reader = fp.reader_kernel(
        READER, cores, reader_cta, [(w.core, [*reader_addrs, w.count, *pairs(w)]) for w in work], defines=defines
    )
    writer = fp.writer_kernel(
        WRITER, cores, writer_cta, [(w.core, [*writer_addrs, w.count, *writer_args(w)]) for w in work], defines=defines
    )
    compute = fp.compute_kernel(
        COMPUTE,
        cores,
        [rows, vbt],
        [(w.core, [w.count]) for w in work],
        defines=defines,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=FP32_COPY_CBS,
    )
    io = [projected, history, *taps, constants, norm, recurrent, *([debug] if debug is not None else []), prefix, out]
    # the projection's q/k/v/z/a/b tiles and the history once per item (the q/k tiles once per item of a key head),
    # the taps, the fp32 state once, the prefix states and the gated tile written; the 4-tap conv and the two l2
    # norms over the 32-row tile, per row the fp32 delta-rule update and read-out on 12 heads of 128 x 128, the gated
    # norm over the tile
    meta = fp.program_meta(
        NAME,
        "verify_rows",
        rows,
        reads=(projected, history, *taps, constants, norm, recurrent),
        writes=(prefix, out, *([debug] if debug is not None else [])),
        flops=TILE * (2 * 4 * QKV_WIDTH + 6 * 2 * QK_WIDTH + 6 * VALUE_WIDTH) + rows * 8 * HEADS * HEAD_DIM * HEAD_DIM,
        cores=len(work),
        # the gated tile head-sharded on dim 3 as the wrap stamps it, the prefix states on the heads (dim 1) as the
        # recurrent state is; the debug taps are a host readback (replicated)
        outputs=((out, 3), (prefix, 1), *([(debug, None)] if debug is not None else [])),
    )
    descriptor = fp.program_descriptor([reader, writer, compute], cbs, semaphores=[fp.semaphore_descriptor(0, cores)])
    fp.run_program(io, descriptor, meta=meta)
    return out


def run_pick(accepted, prefix, recurrent):
    """The commit under the fold: ``recurrent[...] = prefix[accepted]`` for the pass's fp32 ``[1,1,1,1]`` device accept
    count (``0 <= accepted < rows``; the kernel clamps), one data-movement program on 48 cores."""

    rows = int(prefix.shape[0])
    if tuple(accepted.shape) != (1, 1, 1, 1) or accepted.dtype != FP32:
        raise ValueError(f"{NAME} pick needs the fp32 [1, 1, 1, 1] accept count, got {accepted.shape} {accepted.dtype}")
    if tuple(prefix.shape) != (rows, HEADS, HEAD_DIM, HEAD_DIM) or prefix.dtype != FP32:
        raise ValueError(
            f"{NAME} prefix states must be [rows, {HEADS}, {HEAD_DIM}, {HEAD_DIM}] fp32, got {prefix.shape}"
        )
    if tuple(recurrent.shape) != (1, HEADS, HEAD_DIM, HEAD_DIM) or recurrent.dtype != FP32:
        raise ValueError(f"{NAME} state must be [1, {HEADS}, {HEAD_DIM}, {HEAD_DIM}] fp32, got {recurrent.shape}")
    mesh = prefix.device()
    items = [(head, col) for head in range(HEADS) for col in range(HT)]
    work = fp.split_work(len(items), mesh)
    cores = fp.core_rectangle(work, mesh)

    def pairs(w):
        return [value for item in items[w.start : w.start + w.count] for value in item]

    cta = [rows, *fp.accessor_args(accepted), *fp.accessor_args(prefix), *fp.accessor_args(recurrent)]
    addrs = [accepted.buffer_address(), prefix.buffer_address(), recurrent.buffer_address()]
    cbs = [
        fp.cb_descriptor(0, FP32, fp.TILE_BYTES[FP32], 1, cores),
        fp.cb_descriptor(1, FP32, fp.TILE_BYTES[FP32], HT, cores),
    ]
    kernel = fp.reader_kernel(PICK, cores, cta, [(w.core, [*addrs, w.count, *pairs(w)]) for w in work])
    # the accept scalar once per core, one prefix slot read, the state written
    meta = fp.program_meta(
        NAME,
        "commit_pick",
        rows,
        writes=(recurrent,),
        partial=((prefix, HEADS * STATE_TILES * fp.TILE_BYTES[FP32]),),
        dram_bytes=len(work) * fp.TILE_BYTES[FP32],
        cores=len(work),
        outputs=((recurrent, 1),),  # the state slot written in place: head-sharded on dim 1, as allocated
    )
    fp.run_program([accepted, prefix, recurrent], fp.program_descriptor([kernel], cbs), meta=meta)
    return recurrent


# ------------------------------------------------------------------------------------------ the lanes form


def lanes_admitted(lanes: int, rows: int) -> bool:
    """Whether ``lanes`` x ``rows`` is a shape the lanes form runs: 1..MAX_LANES lanes of 1..MAX_ROWS rows, B * R rows
    in one tile.  (The lane body itself runs 2..8 lanes; one lane is the die test's degenerate case.)"""

    return 1 <= lanes <= MAX_LANES and 1 <= rows <= MAX_ROWS and lanes * rows <= TILE


def lanes_prefix_shape(lanes: int, rows: int) -> tuple[int, int, int, int]:
    return (lanes * rows, HEADS, HEAD_DIM, HEAD_DIM)


def prefix_states_bytes(rows: int, layers: int = 1) -> int:
    """The prefix states' bytes per device for ``rows`` slots (the lanes form: rows = B * R) over ``layers`` GDN
    layers: rows x 12 x 16 tile pages of 4 KiB per layer (786,432 B per slot)."""

    return layers * rows * HEADS * STATE_TILES * fp.TILE_BYTES[FP32]


def _lane_items(lanes: int) -> list[tuple[int, int]]:
    return [(head, lane) for lane in range(lanes) for head in range(HEADS)]


def run_lanes(projected, history, taps, constants, norm, recurrent, prefix, out, *, lanes: int, rows: int):
    """The scan program's lanes form on explicit tensors (local shapes): ``projected`` [1,1,32,4160] bf16 TILE, the
    lane-major verify tile (tile row u * rows + j = lane u's row j; every row finite); ``history`` [1,lanes,32,2560] bf16
    (rows 0..2 of every lane valid, the rest finite); ``taps`` 4 x [1,1,1,2560] bf16; ``constants`` from
    ``gdn_step.constant_tiles``; ``norm`` [1,1,1,128] bf16; ``recurrent`` [lanes,12,128,128] fp32 (read only);
    ``prefix`` [lanes*rows,12,128,128] fp32 (written: slot u*rows + j = lane u's state after j + 1 rows); ``out``
    [1,1,32,1536] bf16 (rows >= lanes*rows exact zeros).  One (value head, lane) item per core: 12 * lanes cores, the
    whole head per core; the compute kernel is the single form's with ROWS = ``rows`` (one lane's rows)."""

    lanes, rows = int(lanes), int(rows)
    if not lanes_admitted(lanes, rows):
        raise ValueError(
            f"{NAME} lanes form admits 1..{MAX_LANES} lanes x 1..{MAX_ROWS} rows within one tile, got {lanes} x {rows}"
        )
    if not fp.is_row_tile(projected) or fp.tile_width_of(projected) != PROJECTION_WIDTH:
        raise ValueError(f"{NAME} projection must be one row tile of width {PROJECTION_WIDTH}, got {projected.shape}")
    if tuple(recurrent.shape) != (lanes, HEADS, HEAD_DIM, HEAD_DIM) or recurrent.dtype != FP32:
        raise ValueError(
            f"{NAME} lanes state must be [{lanes}, {HEADS}, {HEAD_DIM}, {HEAD_DIM}] fp32, got {recurrent.shape}"
        )
    if tuple(prefix.shape) != lanes_prefix_shape(lanes, rows) or prefix.dtype != FP32:
        raise ValueError(
            f"{NAME} lanes prefix states must be {lanes_prefix_shape(lanes, rows)} fp32, got {prefix.shape} {prefix.dtype}"
        )
    if tuple(history.shape) != (1, lanes, TILE, QKV_WIDTH) or history.dtype != BF16:
        raise ValueError(f"{NAME} lanes history must be [1, {lanes}, {TILE}, {QKV_WIDTH}] bf16, got {history.shape}")
    if len(taps) != 4:
        raise ValueError(f"{NAME} takes four conv taps")
    mesh = projected.device()
    vbt = HT  # the whole head per core
    items = _lane_items(lanes)
    work = fp.split_work(len(items), mesh)
    cores = fp.core_rectangle(work, mesh)
    if len(work) != len(items):
        raise RuntimeError(f"{NAME} lanes form places one item per core; {len(items)} items got {len(work)} cores")

    def pairs(w):
        return [value for item in items[w.start : w.start + w.count] for value in item]

    reader_cta = [rows, vbt, lanes]
    for tensor in (projected, history, *taps, constants, norm, recurrent):
        reader_cta.extend(fp.accessor_args(tensor))
    reader_addrs = [
        projected.buffer_address(),
        history.buffer_address(),
        *(t.buffer_address() for t in taps),
        constants.buffer_address(),
        norm.buffer_address(),
        recurrent.buffer_address(),
    ]
    writer_cta = [rows, vbt, lanes, *fp.accessor_args(prefix), *fp.accessor_args(out)]
    writer_addrs = [prefix.buffer_address(), out.buffer_address()]
    defines = [("INP_FLOAT32", "1")]
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, cores)
        for index, dtype, pages in cb_table(rows, vbt)
    ]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * HT * vbt * fp.TILE_BYTES[FP32],
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SNEWC, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
                ttnn.CBFormatDescriptor(buffer_index=CB_OUTS, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
            ],
        )
    )
    reader = fp.reader_kernel(
        READER_LANES, cores, reader_cta, [(w.core, [*reader_addrs, w.count, *pairs(w)]) for w in work], defines=defines
    )
    writer = fp.writer_kernel(
        WRITER_LANES, cores, writer_cta, [(w.core, [*writer_addrs, w.count, *pairs(w)]) for w in work], defines=defines
    )
    compute = fp.compute_kernel(
        COMPUTE,
        cores,
        [rows, vbt],
        [(w.core, [w.count]) for w in work],
        defines=defines,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=FP32_COPY_CBS,
    )
    io = [projected, history, *taps, constants, norm, recurrent, prefix, out]
    # per item the projection's q/k/v/z/a/b tiles and the lane's history tiles, the taps, the lane's fp32 state once;
    # the prefix states and the gated rows written; per item the 4-tap conv and the two l2 norms over the 32-row
    # tile, per lane row the fp32 delta-rule update and read-out on one head of 128 x 128, the gated norm over the tile
    meta = fp.program_meta(
        NAME,
        "verify_rows_lanes",
        lanes * rows,
        reads=(projected, *taps, constants, norm, recurrent),
        writes=(prefix, out),
        partial=((history, lanes * (2 * HT + vbt) * HEADS * fp.TILE_BYTES[BF16]),),
        dram_bytes=(lanes - 1) * (fp.tensor_bytes(projected) + sum(fp.tensor_bytes(t) for t in taps)),
        flops=lanes
        * (TILE * (2 * 4 * QKV_WIDTH + 6 * 2 * QK_WIDTH + 6 * VALUE_WIDTH) + rows * 8 * HEADS * HEAD_DIM * HEAD_DIM),
        cores=len(work),
        outputs=((out, 3), (prefix, 1)),
    )
    descriptor = fp.program_descriptor([reader, writer, compute], cbs)
    fp.run_program(io, descriptor, meta=meta)
    return out


def run_pick_lanes(counts, prefix, recurrent, *, lanes: int, rows: int):
    """The lanes commit under the fold: for every lane u with ``c_u >= 1``, ``recurrent[u] = prefix[u * rows + c_u - 1]``
    (lane u's state after ``c_u`` committed rows); ``c_u == 0`` writes nothing (the lane keeps its state bitwise).
    ``counts`` is the pass's fp32 TILE ``[1, lanes, 1, 1]`` committed-rows tensor (the lane selectors' ``(a_u + 1) *
    active_u``; the kernel clamps to ``rows``), read on the device.  One data-movement program, (lane, head, column)
    items over the grid."""

    lanes, rows = int(lanes), int(rows)
    if not lanes_admitted(lanes, rows):
        raise ValueError(
            f"{NAME} lanes pick admits 1..{MAX_LANES} lanes x 1..{MAX_ROWS} rows within one tile, got {lanes} x {rows}"
        )
    if tuple(counts.shape) != (1, lanes, 1, 1) or counts.dtype != FP32 or counts.layout != ttnn.TILE_LAYOUT:
        raise ValueError(
            f"{NAME} lanes pick needs the fp32 TILE [1, {lanes}, 1, 1] committed counts, got {counts.shape} {counts.dtype}"
        )
    if tuple(prefix.shape) != lanes_prefix_shape(lanes, rows) or prefix.dtype != FP32:
        raise ValueError(
            f"{NAME} lanes prefix states must be {lanes_prefix_shape(lanes, rows)} fp32, got {prefix.shape}"
        )
    if tuple(recurrent.shape) != (lanes, HEADS, HEAD_DIM, HEAD_DIM) or recurrent.dtype != FP32:
        raise ValueError(
            f"{NAME} lanes state must be [{lanes}, {HEADS}, {HEAD_DIM}, {HEAD_DIM}] fp32, got {recurrent.shape}"
        )
    mesh = prefix.device()
    items = [(lane, head, col) for lane in range(lanes) for head in range(HEADS) for col in range(HT)]
    work = fp.split_work(len(items), mesh)
    cores = fp.core_rectangle(work, mesh)

    def triples(w):
        return [value for item in items[w.start : w.start + w.count] for value in item]

    cta = [rows, lanes, *fp.accessor_args(counts), *fp.accessor_args(prefix), *fp.accessor_args(recurrent)]
    addrs = [counts.buffer_address(), prefix.buffer_address(), recurrent.buffer_address()]
    cbs = [
        fp.cb_descriptor(0, FP32, fp.TILE_BYTES[FP32], lanes, cores),
        fp.cb_descriptor(1, FP32, fp.TILE_BYTES[FP32], HT, cores),
    ]
    kernel = fp.reader_kernel(PICK_LANES, cores, cta, [(w.core, [*addrs, w.count, *triples(w)]) for w in work])
    # the counts once per core, one prefix slot read per lane (at most), the lane states written
    meta = fp.program_meta(
        NAME,
        "commit_pick_lanes",
        lanes * rows,
        writes=(recurrent,),
        partial=((prefix, lanes * HEADS * STATE_TILES * fp.TILE_BYTES[FP32]),),
        dram_bytes=len(work) * lanes * fp.TILE_BYTES[FP32],
        cores=len(work),
        outputs=((recurrent, 1),),
    )
    fp.run_program([counts, prefix, recurrent], fp.program_descriptor([kernel], cbs), meta=meta)
    return recurrent


# ------------------------------------------------------------------------------- the per-layer buffers (the rows state)


class Buffers:
    """The fold's per-layer tensors, allocated with the rows state and freed with it: the prefix states, and the
    program's constants (gdn_step's dt_bias / neg_exp_A full tiles, shared with the decode step through the layer's
    own cache and not owned here).  The gated tile does not outlive one body and is allocated per call."""

    __slots__ = ("prefix", "constants", "gated_memory_config")

    def __init__(self, *, prefix, constants, gated_memory_config):
        self.prefix, self.constants, self.gated_memory_config = prefix, constants, gated_memory_config

    @property
    def rows(self) -> int:
        return int(self.prefix.shape[0])

    def deallocate(self) -> None:
        _release(self.prefix)


def qualifies(rows_state) -> bool:
    """Whether a rows state is the form the fold serves: the 32-row verify tile with its own body, up to MAX_ROWS real
    rows, not the slab's flat q/k, device tensors.  Shapes only, read once when the state is allocated."""

    constants = getattr(rows_state, "constants", None)
    if constants is None or getattr(rows_state, "flat_qk", False) or not getattr(rows_state, "owns_body", True):
        return False
    if not callable(getattr(getattr(rows_state, "v", None), "buffer_address", None)):
        return False
    return getattr(constants, "tile_rows", 0) == TILE and 1 <= getattr(constants, "rows", 0) <= MAX_ROWS


def attach(gdn, rows_state, environ=None) -> Buffers | None:
    """Allocate the fold's buffers onto a newly allocated rows state and return them, or None when the fold is off
    (not named by ``QWEN38_FUSED``, or named by ``QWEN38_FUSED_OFF``) or the state is not the form it serves.  Called
    from ``allocate_rows_state`` before the wrap's ``attach``, so the form is fixed before the warm pass and the
    commit that reads the prefix states cannot find itself on the other side of the switch from the forward pass."""

    environ = os.environ if environ is None else environ
    if not enabled(NAME, environ) or not qualifies(rows_state):
        return None
    mesh = gdn.mesh_device
    prefix = fp.allocate((rows_state.constants.rows, HEADS, HEAD_DIM, HEAD_DIM), FP32, ttnn.TILE_LAYOUT, mesh)
    fp.stamp_topology(prefix, rows_state.v, shard_dim=1)
    buffers = Buffers(
        prefix=prefix, constants=gdn_step._constants(gdn), gated_memory_config=gdn.out_proj_act_memory_config
    )
    rows_state.scan_buffers = buffers
    return buffers


def buffers_of(rows_state) -> Buffers | None:
    """The fold's buffers of a rows state, or None when it runs the wrap or the chain."""

    return getattr(rows_state, "scan_buffers", None)


class LaneBuffers:
    """The lanes form's per-layer tensors, allocated with the lane rows state and freed with it: the prefix states
    ``[B * R, 12, 128, 128]`` fp32 (slot u * R + j = lane u after j + 1 rows) and the program's constants (shared with
    the decode step, not owned here)."""

    __slots__ = ("prefix", "constants", "gated_memory_config", "lanes", "rows")

    def __init__(self, *, prefix, constants, gated_memory_config, lanes: int, rows: int):
        self.prefix, self.constants, self.gated_memory_config = prefix, constants, gated_memory_config
        self.lanes, self.rows = int(lanes), int(rows)

    def deallocate(self) -> None:
        _release(self.prefix)


def qualifies_lanes(rows_state) -> bool:
    """Whether a lane rows state is the form the lanes fold serves: the 32-row lane-major verify tile, ``lanes`` x
    ``rows`` within the admission, device tensors.  Shapes only, read once when the state is allocated."""

    constants = getattr(rows_state, "constants", None)
    lane_constants = getattr(rows_state, "lane_constants", None)
    if constants is None or lane_constants is None:
        return False
    if not callable(getattr(getattr(rows_state, "v", None), "buffer_address", None)):
        return False
    lanes, rows = getattr(rows_state, "lanes", 0), getattr(constants, "rows", 0)
    if getattr(lane_constants, "lanes", None) != lanes or getattr(lane_constants, "rows", None) != rows:
        return False
    return getattr(constants, "tile_rows", 0) == TILE and lanes_admitted(lanes, rows)


def attach_lanes(gdn, rows_state, environ=None) -> LaneBuffers | None:
    """Allocate the lanes form's buffers onto a newly allocated lane rows state and return them, or None when the fold
    is off (the same switch as the single form: ``QWEN38_FUSED_OFF=gdn_rows_scan``) or the state is not the form it
    serves.  Called from ``allocate_lane_rows_state``: the form is fixed before the warm pass and the capture, and the
    commit that picks the prefix states cannot find itself on the other side of the switch from the forward pass."""

    environ = os.environ if environ is None else environ
    if not enabled(NAME, environ) or not qualifies_lanes(rows_state):
        return None
    mesh = gdn.mesh_device
    lanes, rows = rows_state.lanes, rows_state.constants.rows
    prefix = fp.allocate(lanes_prefix_shape(lanes, rows), FP32, ttnn.TILE_LAYOUT, mesh)
    fp.stamp_topology(prefix, rows_state.v, shard_dim=1)
    buffers = LaneBuffers(
        prefix=prefix,
        constants=gdn_step._constants(gdn),
        gated_memory_config=gdn.out_proj_act_memory_config,
        lanes=lanes,
        rows=rows,
    )
    rows_state.scan_buffers = buffers
    return buffers


def lane_buffers_of(rows_state) -> LaneBuffers | None:
    """The lanes form's buffers of a lane rows state, or None when it runs the chain."""

    buffers = getattr(rows_state, "scan_buffers", None)
    return buffers if isinstance(buffers, LaneBuffers) else None


# ---------------------------------------------------------------------------------------------- the bodies


def rows_body_scan(gdn, full_hidden, rows_state, state, *, full_tile: bool = False):
    """The folded body: the projection's linear and S2I, the qkv landing the history advance reads, the scan program
    (the gated tile into the out-projection's activation shard, the prefix states into the rows state's buffer),
    then the chain's own out-projection.  Returns ``(output, None)``: the state after all rows is prefix slot
    ``rows - 1``.  On a lane rows state that carries the lanes buffers, the lanes form (``rows_body_scan_lanes``)."""

    if lane_buffers_of(rows_state) is not None:
        return rows_body_scan_lanes(gdn, full_hidden, rows_state, state)
    buffers = buffers_of(rows_state)
    projected = gdn._project_rows_linear(full_hidden, rows_state)
    gdn._land_rows_qkv(projected, rows_state)
    gated = fp.allocate((1, 1, TILE, VALUE_WIDTH), BF16, ttnn.TILE_LAYOUT, gdn.mesh_device, buffers.gated_memory_config)
    run(
        projected,
        rows_state.history,
        gdn.weights.conv_taps,
        buffers.constants,
        gdn.weights.norm,
        state.recurrent,
        buffers.prefix,
        gated,
    )
    _release(projected)
    # the gated tile and the prefix states carry their placements from the launch (the program meta's outputs)
    output = gdn._out_proj_tile(gated, full_hidden)  # consumes the gated tile
    _release(full_hidden)
    return gdn._rows_output_tile(output, rows_state, full_tile=full_tile), None


def rows_body_fallback(gdn, full_hidden, rows_state, state, *, full_tile: bool = False):
    """Today's stream where the fold does not serve: the wrap on a rows state that carries its buffers, the chain
    otherwise (the layer's resolved wrap-or-chain body); on a lane rows state, the lanes chain (one batched chunk
    call from the lanes' states)."""

    if is_lane_rows_state(rows_state):
        return gdn._rows_body_lanes_chain(full_hidden, rows_state, state)
    return gdn._rows_body_wrap()(gdn, full_hidden, rows_state, state, full_tile=full_tile)


def is_lane_rows_state(rows_state) -> bool:
    """Whether ``rows_state`` is the lane rows form (B lanes lane-major; ``Qwen38TTNNGDNLaneRowsState``): it carries
    a lane count and the lane constants; the single stream's rows state carries neither."""

    return getattr(rows_state, "lane_constants", None) is not None and getattr(rows_state, "lanes", None) is not None


def admits(gdn, full_hidden, rows_state, state, *, full_tile: bool = False) -> bool:
    """The fold's input contract for one ``forward_rows`` body: the rows state carries the fold's buffers (which
    ``attach`` gives only the admitted verify form, and only when the kernel is on) and the call's state is the
    single-lane fp32 recurrent tensor.  Shapes only: host fakes and every other form take today's stream.  A lane
    rows state is admitted by the lanes form's contract (``admits_lanes``), the same registry step serving both."""

    if is_lane_rows_state(rows_state):
        return admits_lanes(rows_state, state)
    if buffers_of(rows_state) is None or not qualifies(rows_state):
        return False
    recurrent = getattr(state, "recurrent", None)
    if recurrent is None:
        return False
    try:
        return tuple(recurrent.shape) == (1, HEADS, HEAD_DIM, HEAD_DIM) and recurrent.dtype == FP32
    except (AttributeError, TypeError):
        return False


def commit(gdn, rows_state, buffers, state, selectors) -> None:
    """The commit under the fold: prefix slot ``accepted`` into ``state.recurrent`` by the pick program.  The accept
    count is the pass's device scalar the selectors were built from."""

    accepted = getattr(selectors, "accepted", None)
    if accepted is None:
        raise RuntimeError(
            f"the {NAME} commit needs the pass's accept count on its selectors (build_rows_selectors sets it)"
        )
    run_pick(accepted, buffers.prefix, state.recurrent)


def commit_all_rows(buffers, state) -> None:
    """Every row committed (``commit_rows_full`` under the fold): prefix slot ``rows - 1`` into ``state.recurrent``."""

    rows = buffers.rows
    landed = ttnn.slice(
        buffers.prefix, (rows - 1, 0, 0, 0), (rows, HEADS, HEAD_DIM, HEAD_DIM), output_tensor=state.recurrent
    )
    if landed is not None and landed.buffer_address() != state.recurrent.buffer_address():
        raise RuntimeError(f"{NAME} committed-state slice did not land in the recurrent state")


def admits_lanes(rows_state, state) -> bool:
    """The lanes form's input contract for one ``forward_rows_lanes`` body: the lane rows state carries the lanes
    buffers (which ``attach_lanes`` gives only the admitted form, and only when the kernel is on) and the call's state
    is the ``[B, 12, 128, 128]`` fp32 lane recurrent tensor."""

    buffers = lane_buffers_of(rows_state)
    if buffers is None or not qualifies_lanes(rows_state):
        return False
    recurrent = getattr(state, "recurrent", None)
    if recurrent is None:
        return False
    try:
        return tuple(recurrent.shape) == (buffers.lanes, HEADS, HEAD_DIM, HEAD_DIM) and recurrent.dtype == FP32
    except (AttributeError, TypeError):
        return False


def rows_body_scan_lanes(gdn, full_hidden, rows_state, state):
    """The folded lane body: the projection's linear and S2I, the q|k|v landing and its per-lane expand (the commit's
    history advance reads both), the lanes scan program (the gated tile into the out-projection's activation shard,
    the prefix states into the lane rows state's buffer), then the chain's own out-projection.  Returns ``(output,
    None)``: lane u's state after all rows is prefix slot ``u * R + R - 1``.  The state must be the lane state the
    buffers were attached for (``admits_lanes``); anything else is a wiring error, not a fallback."""

    buffers = lane_buffers_of(rows_state)
    if buffers is None or not admits_lanes(rows_state, state):
        raise RuntimeError(
            f"{NAME} lanes body called on a state outside its contract (lanes {getattr(rows_state, 'lanes', None)})"
        )
    projected = gdn._project_rows_linear(full_hidden, rows_state)
    gdn._land_rows_qkv(projected, rows_state)
    gdn._select_rows(
        rows_state.lane_constants.expand_select,
        rows_state.qkv,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        label="GDN lane rows expand",
        output_tensor=rows_state.qkv_lanes,
    )
    gated = fp.allocate((1, 1, TILE, VALUE_WIDTH), BF16, ttnn.TILE_LAYOUT, gdn.mesh_device, buffers.gated_memory_config)
    run_lanes(
        projected,
        rows_state.history,
        gdn.weights.conv_taps,
        buffers.constants,
        gdn.weights.norm,
        state.recurrent,
        buffers.prefix,
        gated,
        lanes=buffers.lanes,
        rows=buffers.rows,
    )
    _release(projected)
    output = gdn._out_proj_tile(gated, full_hidden)  # consumes the gated tile
    _release(full_hidden)
    return output, None


def commit_lanes(gdn, rows_state, buffers, state, selectors) -> None:
    """The lanes commit under the fold: lane u's prefix slot ``u * R + c_u - 1`` into ``state.recurrent[u]`` by the
    lanes pick program (``c_u == 0`` keeps the lane's state).  The counts are the selectors' committed-rows tensor
    (``build_rows_selectors_lanes`` keeps it)."""

    counts = getattr(selectors, "committed_counts", None)
    if counts is None:
        raise RuntimeError(
            f"the {NAME} lanes commit needs the per-lane committed counts on its selectors "
            "(build_rows_selectors_lanes sets committed_counts)"
        )
    run_pick_lanes(counts, buffers.prefix, state.recurrent, lanes=buffers.lanes, rows=buffers.rows)


# ---------------------------------------------------------------------------------------------- the reference


def reference_rows(projected, history, taps, dt_bias, neg_exp_A, norm, state, rows: int):
    """The program's claim in torch: ``rows`` sequential ``gdn_step.reference_step`` calls (tt/gdn.py's arithmetic per
    row) with the FIR ring emulated from the history rows.  ``projected`` [32, 4160] bf16 (rows >= ``rows`` unused),
    ``history`` [32, 2560] bf16 (rows 0..2), ``taps`` 4 x [2560] bf16, ``dt_bias`` / ``neg_exp_A`` [12] fp32, ``norm``
    [128] bf16, ``state`` [12, 128, 128] fp32.  Returns (prefix [rows, 12, 128, 128] fp32: slot a = the state after
    a + 1 rows; gated [rows, 1536] bf16)."""

    window = torch.cat([history[:HISTORY_ROWS], projected[:, :QKV_WIDTH]], dim=0)  # rows before row 0, then the rows
    current = state.unsqueeze(0)
    states, gated_rows = [], []
    for row in range(rows):
        older = [window[row + i : row + i + 1] for i in range(HISTORY_ROWS)]
        current, gated, *_ = gdn_step.reference_step(
            projected[row : row + 1], older, taps, dt_bias, neg_exp_A, norm, current
        )
        states.append(current[0])
        gated_rows.append(gated[0])
    return torch.stack(states), torch.stack(gated_rows)


def reference_rows_lanes(projected, history, taps, dt_bias, neg_exp_A, norm, state, lanes: int, rows: int):
    """The lanes form's claim in torch: per lane, ``reference_rows`` on that lane's rows (tile rows u*rows .. u*rows +
    rows - 1 of the lane-major ``projected`` [32, 4160]), its history ``history[u]`` [32, 2560] (rows 0..2) and its
    state ``state[u]`` [12, 128, 128].  Returns (prefix [lanes*rows, 12, 128, 128] fp32: slot u*rows + j = lane u
    after j + 1 rows; gated [lanes*rows, 1536] bf16)."""

    states, gated_rows = [], []
    for lane in range(lanes):
        lane_tile = torch.zeros_like(projected)
        lane_tile[:rows] = projected[lane * rows : (lane + 1) * rows]
        prefix, gated = reference_rows(lane_tile, history[lane], taps, dt_bias, neg_exp_A, norm, state[lane], rows)
        states.append(prefix)
        gated_rows.append(gated)
    return torch.cat(states), torch.cat(gated_rows)


register(
    FusedKernel(
        name=NAME,
        replaces="the GDN verify-rows body between the projection's sharded-to-interleaved and the out-projection "
        "matmul (the wrap's six programs: gdn_pre_rows, chunk_gdn_prep, chunk_gdn_scan, post_cast, post_norm on the "
        "chunk form; the chain's 53) as one serial-recurrence program with every prefix state, and the commit's masked "
        "prep + scan re-run as one prefix-state pick",
        tolerance=COMPONENT,
        fused=rows_body_scan,
        composed=rows_body_fallback,
        admits=admits,
        # COMPONENT against the wrap's stream (the chunk form); the component gate is the fused GDN step probe's rows
        # column (fused_gdn_step_probe --rows 5): the fold's slots against the CPU oracle, and bitwise the fused step
        # run sequentially in every window (the die proof's identity on real inputs)
        component_proof=(
            "the fused GDN step probe's rows column against the CPU oracle, one p150 die, 2026-09-26 (run "
            "probe-rows5-8edcaf9279c-162454, --rows 5, 64 steps): state error 0.0048 and gated output 0.0098 for "
            "the fold's five rows, open and closed loop, 0.0048 and 0.0098 for the fused step and 0.0081 and "
            "0.0234 for the composed chain; every window's slots and rows bitwise the fused step run sequentially"
        ),
        gate=None,
    )
)

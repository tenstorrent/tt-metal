# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 of the GDN step split (``QWEN38_GDN_CONV_SPLIT=2``): the V split on top of the stage-1 conv fan-out.

A head's 128 x 128 state is split into VB column blocks of VT = 4 / VB value-column tiles, one block per core
(``compute_vb.cpp``; a state column never reads another column, so the per-column ops are the default ones).  The conv
cores of stage 1 (same three kernels) send q/k tiles to every block core of the key head and each v tile to the one block
core that owns its column.  The gated norm needs the whole o row: the VB block cores of a head exchange their bf16 o tiles
through L1 (``writer_vb.cpp``) and each runs the default squares and reduce over all 4 tiles.  Bit-identical to the default
program (design x30_gdn_split/DESIGN.md sections 3 and 4).

Layout (``vb_for``): VB = 4 for rows 1, 2 and 2 for rows 4 (rows*VB <= grid rows and rows*6*VB + 40 <= the grid).  Head core
(lane, head, s) at (head, lane*VB + s): head rectangle (0,0)-(5, rows*VB-1) (a key head's 3 heads are 3 adjacent columns);
conv rectangle (6,0)-(9,9), tile T at (6 + T // 10, T % 10).  CB 11 (conv tiles), CB 10 (the o row) and the semaphores
(0 = q/k tiles, 8 per block core; 1 = v tiles, VT; 2 = o exchange, VB) have one address over all cores they span.
``QWEN38_GDN_CS_DECAY_FIRST=1`` selects the decay-first order (the state decay before the wait for the conv tiles): bit-identical,
but no faster in the unit test (x34), so off by default.
"""

from __future__ import annotations

import os

import ttnn

from .. import gr_read
from .. import program as fp
from .conv_split import CONV_ROWS, CONV_TILES, SEM_QK, SEM_V

SEM_X = 2
HEAD_COLUMNS = 6
CONV_X0, CONV_COLUMNS = 6, 4
CB_QKV, CB_OBF, CB_OMINE, CB_SQO = 11, 10, 29, 30
CONV_ONLY_CBS = (0, 1, 2, 11)  # the default table's conv CBs (11 is declared over the union)
ROWS_SUPPORTED = (1, 2, 4)


def decay_first() -> bool:
    return os.environ.get("QWEN38_GDN_CS_DECAY_FIRST", "0") == "1"


def vb_for(rows: int, grid_x: int, grid_y: int) -> int:
    """The column blocks per head for ``rows`` on a grid_x x grid_y grid, 0 when stage 2 does not fit."""

    if rows not in ROWS_SUPPORTED or grid_y < CONV_ROWS or grid_x < CONV_X0 + CONV_COLUMNS:
        return 0
    for vb in (4, 2):
        if rows * vb <= grid_y and rows * 6 * vb + CONV_TILES <= grid_x * grid_y:
            return vb
    return 0


def admits(rows: int, mesh) -> bool:
    grid = mesh.compute_with_storage_grid_size()
    return vb_for(rows, grid.x, grid.y) > 0


def head_core(lane: int, head: int, s: int, vb: int) -> tuple[int, int]:
    return head, lane * vb + s


def conv_core(tile: int) -> tuple[int, int]:
    return CONV_X0 + tile // CONV_ROWS, tile % CONV_ROWS


def destinations(tile: int, rows: int, vb: int, *, k_tile0: int, v_tile0: int, qk_repeat: int):
    """[((lane, head, s), slot, sem)] the block cores that receive conv tile ``tile``: q/k tiles go to every block of
    the key head's heads (slot c / 4 + c), a v tile (h, cc) to block cc // VT of head h only (slot 8 + cc % VT)."""

    vt = 4 // vb
    if tile < v_tile0:
        g, slot = (tile // 4, tile % 4) if tile < k_tile0 else ((tile - k_tile0) // 4, 4 + (tile - k_tile0) % 4)
        return [
            ((lane, g * qk_repeat + r, s), slot, SEM_QK)
            for lane in range(rows)
            for r in range(qk_repeat)
            for s in range(vb)
        ]
    head, cc = (tile - v_tile0) // 4, (tile - v_tile0) % 4
    return [((lane, head, cc // vt), 8 + cc % vt, SEM_V) for lane in range(rows)]


def _range(x0, y0, x1, y1) -> ttnn.CoreRange:
    return ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))


def run_vb(m, projected, older, newest, taps, constants, norm, recurrent, out):
    """The stage-2 program; ``m`` is the ``gdn_step`` module."""

    rows = fp.rows_of(projected)
    mesh = projected.device()
    grid = mesh.compute_with_storage_grid_size()
    vb = vb_for(rows, grid.x, grid.y)
    vt = 4 // vb
    st = 4 * vt
    HEADS, QK_REPEAT = m.HEADS, m.QK_REPEAT
    BF16, FP32 = m.BF16, m.FP32
    items = [(lane, head, s) for lane in range(rows) for head in range(HEADS) for s in range(vb)]
    cores_of = {item: head_core(*item, vb) for item in items}
    assert len(set(cores_of.values())) == len(items)
    head_range = _range(0, 0, HEAD_COLUMNS - 1, rows * vb - 1)
    conv_range = _range(CONV_X0, 0, CONV_X0 + CONV_COLUMNS - 1, CONV_ROWS - 1)
    head_set, conv_set = ttnn.CoreRangeSet([head_range]), ttnn.CoreRangeSet([conv_range])
    union = ttnn.CoreRangeSet([head_range, conv_range])
    noc = gr_read.noc_map(mesh)

    reader_cta = [rows, HEADS, m.K_TILE0, m.V_TILE0, m.Z_TILE0, m.A_TILE, m.B_TILE, vt]
    for tensor in (projected, constants, norm, recurrent):
        reader_cta.extend(fp.accessor_args(tensor))
    reader_addrs = [
        projected.buffer_address(),
        constants.buffer_address(),
        norm.buffer_address(),
        recurrent.buffer_address(),
    ]
    writer_cta = [rows, HEADS, vt, vb, *fp.accessor_args(recurrent), *fp.accessor_args(out)]
    writer_addrs = [recurrent.buffer_address(), out.buffer_address()]
    defines = [("INP_FLOAT32", "1")]

    resized = {7: st, 20: st, 21: st}  # state, decayed state (two copies): this block's tiles only
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], resized.get(index, pages), head_set)
        for index, dtype, pages in m.CBS
        if index not in CONV_ONLY_CBS
    ]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=st * fp.TILE_BYTES[FP32],
            core_ranges=head_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=m.CB_SNEW, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
                ttnn.CBFormatDescriptor(buffer_index=m.CB_OUTS, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
            ],
        )
    )
    cbs.append(fp.cb_descriptor(CB_OMINE, BF16, fp.TILE_BYTES[BF16], vt, head_set))
    cbs.append(
        fp.cb_descriptor(CB_SQO, FP32, fp.TILE_BYTES[FP32], 4, head_set)
    )  # compute.cpp's CB_SQO = CB_DELTAB (fp32)
    cbs.append(fp.cb_descriptor(CB_QKV, BF16, fp.TILE_BYTES[BF16], 12, union))
    for index, pages in ((0, 1), (1, 3), (2, 4), (10, 1)):
        cbs.append(fp.cb_descriptor(index, BF16, fp.TILE_BYTES[BF16], pages, conv_set))

    def core(xy):
        return ttnn.CoreCoord(*xy)

    def peers(item):
        lane, head, _ = item
        return [v for s in range(vb) for v in noc[cores_of[(lane, head, s)]]]

    reader = fp.reader_kernel(
        fp.kernel_source(m.NAME, "reader_vb.cpp"),
        head_set,
        reader_cta,
        [(core(cores_of[i]), [*reader_addrs, 1, *i]) for i in items],
        defines=defines,
    )
    writer = fp.writer_kernel(
        fp.kernel_source(m.NAME, "writer_vb.cpp"),
        head_set,
        writer_cta,
        [(core(cores_of[i]), [*writer_addrs, 1, *i, *peers(i)]) for i in items],
        defines=defines,
    )
    compute = fp.compute_kernel(
        fp.kernel_source(m.NAME, "compute_vb.cpp"),
        head_set,
        [vt],
        [(core(cores_of[i]), [1, i[2]]) for i in items],
        defines=[*defines, *(("DECAY_FIRST", "1"),) * decay_first()],
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=m.FP32_COPY_CBS,
    )

    conv_defines = [*defines, *(("CS_NOWORK", "1"),) * (os.environ.get("QWEN38_GDN_CS_NOWORK") == "1")]
    conv_cta = []
    for tensor in (projected, *older, *taps, newest):
        conv_cta.extend(fp.accessor_args(tensor))
    conv_addrs = [t.buffer_address() for t in (projected, *older, *taps, newest)]
    tiles = range(CONV_TILES)
    conv_xy = [conv_core(t) for t in tiles]
    conv_reader = fp.reader_kernel(
        fp.kernel_source(m.NAME, "conv_reader.cpp"),
        conv_set,
        conv_cta,
        [(core(c), [*conv_addrs, t]) for t, c in zip(tiles, conv_xy)],
        defines=conv_defines,
    )
    dest_args = {}
    for t in tiles:
        flat = []
        dests = destinations(t, rows, vb, k_tile0=m.K_TILE0, v_tile0=m.V_TILE0, qk_repeat=QK_REPEAT)
        for item, slot, sem in dests:
            x, y = noc[cores_of[item]]
            flat.extend([x, y, slot, sem])
        dest_args[t] = [len(dests), *flat]
    conv_writer = fp.writer_kernel(
        fp.kernel_source(m.NAME, "conv_writer.cpp"),
        conv_set,
        [],
        [(core(c), dest_args[t]) for t, c in zip(tiles, conv_xy)],
        defines=conv_defines,
    )
    conv_compute = fp.compute_kernel(
        fp.kernel_source(m.NAME, "conv_compute.cpp"),
        conv_set,
        [],
        [],
        defines=conv_defines,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=m.FP32_COPY_CBS,
    )

    io = [projected, *older, *taps, constants, norm, recurrent, newest]
    meta = fp.program_meta(
        m.NAME,
        "step_v_split",
        rows,
        reads=(projected, *older, *taps, norm, recurrent),
        writes=(recurrent, newest, out),
        dram_bytes=len(items) * 2 * st * fp.TILE_BYTES[FP32],
        flops=rows
        * (2 * 4 * m.QKV_WIDTH + 6 * 2 * m.QK_WIDTH + 8 * HEADS * m.HEAD_DIM * m.HEAD_DIM + 6 * m.VALUE_WIDTH),
        cores=len(items) + CONV_TILES,
    )
    descriptor = fp.program_descriptor(
        [reader, writer, compute, conv_reader, conv_writer, conv_compute],
        cbs,
        semaphores=[fp.semaphore_descriptor(i, union) for i in (SEM_QK, SEM_V, SEM_X)],
    )
    fp.run_program([*io, out], descriptor, meta=meta)
    return out

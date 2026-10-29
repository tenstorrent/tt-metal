# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 of the GDN step split (``QWEN38_GDN_CONV_SPLIT=1``): the conv fan-out.

Today every head core convolves its own 12 q/k/v tiles (72 tile-convs for 40 unique tiles, ~31 us of the ~71 us step).
Here 40 conv cores convolve one tile each (bf16 conv + SiLU, the same ops on the same operand bytes as ``compute.cpp``'s
``conv_silu``) and unicast the tile into the CB_QKV slots of the head cores that consume it; a head core is one
(lane, value head) item as before and runs the default recurrence / norm code (``compute_cs.cpp`` = ``compute.cpp``
minus the conv, gates first).  The result is bit-identical to the default program (note x30_gdn_split/DESIGN.md).

Layout (``cs_layout``): head cores in the rectangle (0,0)-(2,2*rows-1), item (lane, head) at (head % 3, 2*lane + head // 3)
(a key head's three heads share a row of the rectangle); conv cores in the rectangle (4,0)-(7,9), tile T at
(4 + T // 10, T % 10).  Two rectangles, two kernel groups of three kernels.  CB 11 (one 12-page descriptor over both
rectangles) and the two semaphores (0 = q/k tiles, 8 increments per head core; 1 = v tiles, 4) have the same L1 address on
every core of the program, so a conv core addresses a head core's slot as ``cb_base + slot * 2048``.
"""

from __future__ import annotations

import os

import ttnn

from .. import gr_read
from .. import program as fp

ENV = "QWEN38_GDN_CONV_SPLIT"
ROWS_SUPPORTED = (1, 2, 4)
HEAD_COLUMNS, CONV_X0, CONV_COLUMNS, CONV_ROWS = 3, 4, 4, 10
CONV_TILES = 40  # 8 q + 8 k + 24 v
SEM_QK, SEM_V = 0, 1
CB_QKV = 11
QK_INCREMENTS, V_INCREMENTS = 8, 4
HEAD_ONLY_CBS_DROPPED = (0, 1, 2)  # the conv inputs live on the conv cores


def enabled() -> bool:
    return os.environ.get(ENV, "2") in ("1", "2")


def grid_fits(grid_x: int, grid_y: int, rows: int) -> bool:
    return rows in ROWS_SUPPORTED and grid_x >= CONV_X0 + CONV_COLUMNS and grid_y >= max(CONV_ROWS, 2 * rows)


def admits(rows: int, mesh) -> bool:
    grid = mesh.compute_with_storage_grid_size()
    return grid_fits(grid.x, grid.y, rows)


def head_core(lane: int, head: int) -> tuple[int, int]:
    return head % HEAD_COLUMNS, 2 * lane + head // HEAD_COLUMNS


def conv_core(tile: int) -> tuple[int, int]:
    return CONV_X0 + tile // CONV_ROWS, tile % CONV_ROWS


def tile_target(tile: int, *, k_tile0: int, v_tile0: int, qk_repeat: int):
    """(kind, key-or-value head, slot) of conv tile ``tile``: kind 'qk' goes to every head of key head g (slot c for q,
    4 + c for k), kind 'v' to value head h only (slot 8 + c)."""

    if tile < k_tile0:
        return "qk", tile // 4, tile % 4
    if tile < v_tile0:
        return "qk", (tile - k_tile0) // 4, 4 + (tile - k_tile0) % 4
    return "v", (tile - v_tile0) // 4, 8 + (tile - v_tile0) % 4


def destinations(tile: int, rows: int, *, k_tile0: int, v_tile0: int, qk_repeat: int):
    """[((lane, head), slot, sem)] head items that receive conv tile ``tile``."""

    kind, index, slot = tile_target(tile, k_tile0=k_tile0, v_tile0=v_tile0, qk_repeat=qk_repeat)
    if kind == "qk":
        heads = [index * qk_repeat + r for r in range(qk_repeat)]
        return [((lane, head), slot, SEM_QK) for lane in range(rows) for head in heads]
    return [((lane, index), slot, SEM_V) for lane in range(rows)]


def _range(x0, y0, x1, y1) -> ttnn.CoreRange:
    return ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))


def run_cs(m, projected, older, newest, taps, constants, norm, recurrent, out):
    """The split program; ``m`` is the ``gdn_step`` module (constants and the default CB table)."""

    rows = fp.rows_of(projected)
    mesh = projected.device()
    if os.environ.get(ENV, "2") == "2":  # stage 2 (V split, the default) where its layout fits, else this stage
        from . import conv_split_v

        if conv_split_v.admits(rows, mesh):
            return conv_split_v.run_vb(m, projected, older, newest, taps, constants, norm, recurrent, out)
    HEADS, QK_REPEAT = m.HEADS, m.QK_REPEAT
    BF16, FP32 = m.BF16, m.FP32
    items = [(lane, head) for lane in range(rows) for head in range(HEADS)]
    head_cores = [head_core(*item) for item in items]
    assert len(set(head_cores)) == len(items)
    head_range = _range(0, 0, HEAD_COLUMNS - 1, 2 * rows - 1)
    conv_range = _range(CONV_X0, 0, CONV_X0 + CONV_COLUMNS - 1, CONV_ROWS - 1)
    head_set, conv_set = ttnn.CoreRangeSet([head_range]), ttnn.CoreRangeSet([conv_range])
    union = ttnn.CoreRangeSet([head_range, conv_range])
    noc = gr_read.noc_map(mesh)

    # ---- head group
    reader_cta = [rows, HEADS, m.K_TILE0, m.V_TILE0, m.Z_TILE0, m.A_TILE, m.B_TILE]
    for tensor in (projected, constants, norm, recurrent):
        reader_cta.extend(fp.accessor_args(tensor))
    reader_addrs = [
        projected.buffer_address(),
        constants.buffer_address(),
        norm.buffer_address(),
        recurrent.buffer_address(),
    ]
    writer_cta = [rows, HEADS, *fp.accessor_args(recurrent), *fp.accessor_args(out)]
    writer_addrs = [recurrent.buffer_address(), out.buffer_address()]
    defines = [("INP_FLOAT32", "1")]
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], pages, head_set)
        for index, dtype, pages in m.CBS
        if index not in (*HEAD_ONLY_CBS_DROPPED, CB_QKV)
    ]
    cbs.append(
        ttnn.CBDescriptor(
            total_size=m.STATE_TILES * fp.TILE_BYTES[FP32],
            core_ranges=head_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=m.CB_SNEW, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
                ttnn.CBFormatDescriptor(buffer_index=m.CB_OUTS, data_format=FP32, page_size=fp.TILE_BYTES[FP32]),
            ],
        )
    )
    # CB 11: one descriptor over head and conv cores (12 bf16 pages) -> the same L1 base everywhere
    cbs.append(fp.cb_descriptor(CB_QKV, BF16, fp.TILE_BYTES[BF16], 12, union))
    # conv-core CBs
    for index, pages in ((0, 1), (1, 3), (2, 4), (10, 1)):
        cbs.append(fp.cb_descriptor(index, BF16, fp.TILE_BYTES[BF16], pages, conv_set))

    def core(xy):
        return ttnn.CoreCoord(*xy)

    pairs = {c: list(item) for c, item in zip(head_cores, items)}
    reader = fp.reader_kernel(
        fp.kernel_source(m.NAME, "reader_cs.cpp"),
        head_set,
        reader_cta,
        [(core(c), [*reader_addrs, 1, *pairs[c]]) for c in head_cores],
        defines=defines,
    )
    writer = fp.writer_kernel(
        m.WRITER,
        head_set,
        writer_cta,
        [(core(c), [*writer_addrs, 1, *pairs[c]]) for c in head_cores],
        defines=defines,
    )
    compute = fp.compute_kernel(
        fp.kernel_source(m.NAME, "compute_cs.cpp"),
        head_set,
        [],
        [(core(c), [1]) for c in head_cores],
        defines=defines,
        fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest=True,
        unpack_to_dest_fp32=m.FP32_COPY_CBS,
    )

    # ---- conv group (QWEN38_GDN_CS_NOWORK=1: the dispatch-cost probe, conv cores launch and do nothing)
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
        dests = destinations(t, rows, k_tile0=m.K_TILE0, v_tile0=m.V_TILE0, qk_repeat=QK_REPEAT)
        for item, slot, sem in dests:
            x, y = noc[head_core(*item)]
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
        "step_conv_split",
        rows,
        reads=(projected, *older, *taps, norm, recurrent),
        writes=(recurrent, newest, out),
        dram_bytes=len(items) * 2 * fp.TILE_BYTES[FP32],
        flops=rows
        * (2 * 4 * m.QKV_WIDTH + 6 * 2 * m.QK_WIDTH + 8 * HEADS * m.HEAD_DIM * m.HEAD_DIM + 6 * m.VALUE_WIDTH),
        cores=len(items) + CONV_TILES,
    )
    descriptor = fp.program_descriptor(
        [reader, writer, compute, conv_reader, conv_writer, conv_compute],
        cbs,
        semaphores=[fp.semaphore_descriptor(SEM_QK, union), fp.semaphore_descriptor(SEM_V, union)],
    )
    fp.run_program([*io, out], descriptor, meta=meta)
    return out

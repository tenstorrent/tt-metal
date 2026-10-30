# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Split 4 of the GDN step (``QWEN38_GDN_CONV_SPLIT=4``): split 3 (``reader_vb3.cpp``'s single read batch, decay first) with
the gates and the l2 norms moved off the head cores onto helper cores.  Every value and every op order is the one of
``compute_vb.cpp`` (the helper kernels copy ``l2_norm()`` and ``gate_scalars()`` verbatim), so the result is bit-identical
to the default program.

Cores (rows = 1 only; for rows 2 and 4 the helper rectangles would overlap the head rectangle, so ``run_vb`` falls back to
split 2):
  head cores  (0,0)-(5,3)   as split 2: (head, s); compute ``compute_vb4.cpp`` (no gates, no l2), reader ``reader_vb4.cpp``,
                            writer ``writer_vb.cpp`` (unchanged)
  conv cores  (6,0)-(9,9)   the three stage-1 kernels, unchanged; the destinations differ: a q (k) tile of key head g goes to
                            the norm core (g, q|k), CB_QKV slot tile % 4; a v tile to its block core, slot cc % VT
  norm cores  (0,4)-(3,4)   core (2 * g + is_k, 4): ``reader_norm4.cpp``, ``compute_norm4.cpp``, ``writer_norm4.cpp``; the 4
                            normed fp32 tiles (CB_QROW) go to CB_QROW / CB_KROW of the 12 head cores of key head g
  gate cores  (0,5)-(5,5)   core (h, 5) for value head h: ``reader_gate4.cpp``, ``compute_gate4.cpp``, ``writer_gate4.cpp``;
                            CB_BETA and CB_DECAY go to the same CBs of the 4 block cores of head h
Semaphores (all over every core): 0 = conv tiles on a norm core (4), 1 = v tiles on a head core (VT), 2 = o exchange (VB),
3 = norm tiles on a head core (2), 4 = gate tiles on a head core (1).
CBs 11 (conv tiles), 15, 16 (q / k rows), 18, 19 (beta, decay) are one descriptor over every core that writes or reads them, so
a remote write uses the sender's own L1 address.
"""

from __future__ import annotations

import os

import ttnn

from .. import gr_read
from .. import program as fp
from . import conv_split_v as v
from .conv_split import CONV_ROWS, CONV_TILES, SEM_QK, SEM_V

SEM_X, SEM_NORM, SEM_GATE = 2, 3, 4
NORM_Y, GATE_Y = 4, 5
CB_QROW, CB_KROW, CB_BETA, CB_DECAY = 15, 16, 18, 19


def active() -> bool:
    return os.environ.get(v.SPLIT_ENV, "4") == "4"


def admits(rows: int, grid_x: int, grid_y: int) -> bool:
    """Split 4 runs at rows 1 on a grid with the head, conv and helper rectangles; elsewhere split 2 runs."""

    return rows == 1 and v.vb_for(rows, grid_x, grid_y) == 4 and grid_y > GATE_Y


def norm_core(g: int, is_k: int) -> tuple[int, int]:
    return 2 * g + is_k, NORM_Y


def gate_core(head: int) -> tuple[int, int]:
    return head, GATE_Y


def destinations(tile: int, *, k_tile0: int, v_tile0: int, vb: int):
    """[(core, slot, sem)] for conv tile ``tile`` (rows 1): q / k tile -> the norm core of its key head, slot tile % 4;
    v tile (h, cc) -> the block core cc // VT of head h, slot cc % VT."""

    vt = 4 // vb
    if tile < v_tile0:
        is_k = int(tile >= k_tile0)
        rel = tile - k_tile0 * is_k
        return [(norm_core(rel // 4, is_k), rel % 4, SEM_QK)]
    head, cc = (tile - v_tile0) // 4, (tile - v_tile0) % 4
    return [(v.head_core(0, head, cc // vt, vb), cc % vt, SEM_V)]


def run_vb4(m, projected, older, newest, taps, constants, norm, recurrent, out):
    rows = fp.rows_of(projected)
    mesh = projected.device()
    grid = mesh.compute_with_storage_grid_size()
    vb = 4
    vt = 1
    st = 4
    HEADS, QK_REPEAT = m.HEADS, m.QK_REPEAT
    BF16, FP32 = m.BF16, m.FP32
    items = [(0, head, s) for head in range(HEADS) for s in range(vb)]
    cores_of = {item: v.head_core(*item, vb) for item in items}
    head_range = v._range(0, 0, v.HEAD_COLUMNS - 1, vb - 1)
    conv_range = v._range(v.CONV_X0, 0, v.CONV_X0 + v.CONV_COLUMNS - 1, CONV_ROWS - 1)
    norm_range = v._range(0, NORM_Y, 2 * (m.QK_HEADS) - 1, NORM_Y)
    gate_range = v._range(0, GATE_Y, HEADS - 1, GATE_Y)
    CS = ttnn.CoreRangeSet
    head_set, conv_set, norm_set, gate_set = (CS([r]) for r in (head_range, conv_range, norm_range, gate_range))
    everything = CS([head_range, conv_range, norm_range, gate_range])
    head_norm = CS([head_range, norm_range])
    head_gate = CS([head_range, gate_range])
    head_conv_norm = CS([head_range, conv_range, norm_range])
    helpers = CS([norm_range, gate_range])
    noc = gr_read.noc_map(mesh)

    scalers, masks = extra_io = list(v._reader_constants(mesh))
    reader_cta = [rows, HEADS, m.K_TILE0, m.V_TILE0, m.Z_TILE0, m.A_TILE, m.B_TILE, vt]
    for tensor in (projected, norm, recurrent, scalers, masks):
        reader_cta.extend(fp.accessor_args(tensor))
    writer_cta = [rows, HEADS, vt, vb, *fp.accessor_args(recurrent), *fp.accessor_args(out)]
    defines = [("INP_FLOAT32", "1")]

    # head CB table: split 2's, with the CBs the helper cores share declared over the cores that share them
    shared = {
        4: head_gate,  # a, b (gate cores)
        5: head_gate,  # dt_bias, neg_exp_A
        9: head_norm,  # scalers
        12: head_norm,  # squares
        13: CS([head_range, norm_range, gate_range]),  # stat
        14: head_norm,  # unit
        CB_QROW: head_norm,
        CB_KROW: head_norm,
        CB_BETA: head_gate,
        CB_DECAY: head_gate,
    }
    resized = {7: st, 20: st, 21: st}
    cbs = [
        fp.cb_descriptor(index, dtype, fp.TILE_BYTES[dtype], resized.get(index, pages), shared.get(index, head_set))
        for index, dtype, pages in m.CBS
        if index not in v.CONV_ONLY_CBS
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
    cbs.append(fp.cb_descriptor(v.CB_OMINE, BF16, fp.TILE_BYTES[BF16], vt, head_set))
    cbs.append(fp.cb_descriptor(v.CB_SQO, FP32, fp.TILE_BYTES[FP32], 4, head_set))
    cbs.append(fp.cb_descriptor(v.CB_QKV, BF16, fp.TILE_BYTES[BF16], 12, head_conv_norm))
    for index, pages in ((0, 1), (1, 3), (2, 4), (10, 1)):
        cbs.append(fp.cb_descriptor(index, BF16, fp.TILE_BYTES[BF16], pages, conv_set))

    def core(xy):
        return ttnn.CoreCoord(*xy)

    def peers(item):
        _, head, _ = item
        return [c for s in range(vb) for c in noc[cores_of[(0, head, s)]]]

    reader = fp.reader_kernel(
        fp.kernel_source(m.NAME, "reader_vb4.cpp"),
        head_set,
        reader_cta,
        [
            (
                core(cores_of[i]),
                [
                    projected.buffer_address(),
                    norm.buffer_address(),
                    recurrent.buffer_address(),
                    scalers.buffer_address(),
                    masks.buffer_address(),
                    *i,
                ],
            )
            for i in items
        ],
        defines=defines,
    )
    writer = fp.writer_kernel(
        fp.kernel_source(m.NAME, "writer_vb.cpp"),
        head_set,
        writer_cta,
        [(core(cores_of[i]), [recurrent.buffer_address(), out.buffer_address(), 1, *i, *peers(i)]) for i in items],
        defines=defines,
    )
    compute_kwargs = dict(fidelity=ttnn.MathFidelity.HiFi4, fp32_dest=True, unpack_to_dest_fp32=m.FP32_COPY_CBS)
    compute = fp.compute_kernel(
        fp.kernel_source(m.NAME, "compute_vb4.cpp"),
        head_set,
        [vt],
        [(core(cores_of[i]), [1, i[2]]) for i in items],
        defines=defines,
        **compute_kwargs,
    )

    # norm cores: core (g, is_k) l2 norms the 4 q (k) tiles of key head g for the 12 head cores of g
    norm_cores = [(g, is_k) for g in range(m.QK_HEADS) for is_k in range(2)]
    heads_of = lambda g: [cores_of[(0, h, s)] for h in range(g * QK_REPEAT, (g + 1) * QK_REPEAT) for s in range(vb)]
    norm_reader = fp.reader_kernel(
        fp.kernel_source(m.NAME, "reader_norm4.cpp"),
        norm_set,
        fp.accessor_args(scalers),
        [(core(norm_core(*c)), [scalers.buffer_address()]) for c in norm_cores],
        defines=defines,
    )
    norm_writer = fp.writer_kernel(
        fp.kernel_source(m.NAME, "writer_norm4.cpp"),
        norm_set,
        [],
        [
            (
                core(norm_core(g, is_k)),
                [CB_KROW if is_k else CB_QROW, len(heads_of(g)), *(c for xy in heads_of(g) for c in noc[xy])],
            )
            for g, is_k in norm_cores
        ],
        defines=defines,
    )
    norm_compute = fp.compute_kernel(
        fp.kernel_source(m.NAME, "compute_norm4.cpp"), norm_set, [vt], [], defines=defines, **compute_kwargs
    )

    # gate cores: core (h, 5) for value head h
    gate_reader = fp.reader_kernel(
        fp.kernel_source(m.NAME, "reader_gate4.cpp"),
        gate_set,
        [HEADS, m.A_TILE, m.B_TILE, *fp.accessor_args(projected), *fp.accessor_args(constants)],
        [(core(gate_core(h)), [projected.buffer_address(), constants.buffer_address(), 0, h]) for h in range(HEADS)],
        defines=defines,
    )
    gate_writer = fp.writer_kernel(
        fp.kernel_source(m.NAME, "writer_gate4.cpp"),
        gate_set,
        [],
        [(core(gate_core(h)), [vb, *(c for s in range(vb) for c in noc[cores_of[(0, h, s)]])]) for h in range(HEADS)],
        defines=defines,
    )
    gate_compute = fp.compute_kernel(
        fp.kernel_source(m.NAME, "compute_gate4.cpp"), gate_set, [vt], [], defines=defines, **compute_kwargs
    )

    conv_defines = [*defines, *(("CS_NOWORK", "1"),) * (os.environ.get("QWEN38_GDN_CS_NOWORK") == "1")]
    conv_cta = []
    for tensor in (projected, *older, *taps, newest):
        conv_cta.extend(fp.accessor_args(tensor))
    conv_addrs = [t.buffer_address() for t in (projected, *older, *taps, newest)]
    tiles = range(CONV_TILES)
    conv_xy = [v.conv_core(t) for t in tiles]
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
        dests = destinations(t, k_tile0=m.K_TILE0, v_tile0=m.V_TILE0, vb=vb)
        for xy, slot, sem in dests:
            x, y = noc[xy]
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
        fp.kernel_source(m.NAME, "conv_compute.cpp"), conv_set, [], [], defines=conv_defines, **compute_kwargs
    )

    io = [projected, *older, *taps, constants, *extra_io, norm, recurrent, newest]
    meta = fp.program_meta(
        m.NAME,
        "step_v_split4",
        rows,
        reads=(projected, *older, *taps, norm, recurrent),
        writes=(recurrent, newest, out),
        dram_bytes=len(items) * 2 * st * fp.TILE_BYTES[FP32],
        flops=rows
        * (2 * 4 * m.QKV_WIDTH + 6 * 2 * m.QK_WIDTH + 8 * HEADS * m.HEAD_DIM * m.HEAD_DIM + 6 * m.VALUE_WIDTH),
        cores=len(items) + CONV_TILES + len(norm_cores) + HEADS,
    )
    descriptor = fp.program_descriptor(
        [
            reader,
            writer,
            compute,
            conv_reader,
            conv_writer,
            conv_compute,
            norm_reader,
            norm_writer,
            norm_compute,
            gate_reader,
            gate_writer,
            gate_compute,
        ],
        cbs,
        semaphores=[fp.semaphore_descriptor(i, everything) for i in (SEM_QK, SEM_V, SEM_X, SEM_NORM, SEM_GATE)],
    )
    fp.run_program([*io, out], descriptor, meta=meta)
    return out

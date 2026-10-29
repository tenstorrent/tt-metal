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
LOWRANK3 = fp.kernel_source(NAME, "lowrank_compute3.cpp")
GATE3 = fp.kernel_source(NAME, "gate_compute3.cpp")


def enabled(rows: int) -> bool:
    import os

    return os.environ.get(ENV, "1") == "1" and 1 <= rows <= MAX_ROWS  # default on (x49 A/B -0.41 ms)


def low_rank_gate3(gathered_partials, normalized, up, *, matmul: str = "chain"):
    """Stages 3a+3b as one program: the four row cores multicast the low-rank row into the gate cores' CB
    while those cores prefetch their weight columns and normalized tiles; the injection tile comes from row core 3."""

    shape = tuple(gathered_partials.shape)
    if len(shape) != 4 or shape[:2] != (TP_SIZE, 1) or shape[3] != PARTIAL_WIDTH or gathered_partials.dtype != FP32:
        raise ValueError(f"GR gathered partials must be fp32 [{TP_SIZE}, 1, rows, {PARTIAL_WIDTH}], got {shape}")
    rows = shape[2]
    from .. import row_gather3  # QWEN38_GATE_GATHER=1: this program with the block-input gather folded in (row_gather3)

    gather_config = row_gather3.take_context()
    if gather_config is not None and row_gather3.admits(gathered_partials, rows, gather_config):
        return row_gather3.gate_gather(gathered_partials, normalized, up, gather_config, matmul=matmul)
    _expect(normalized, (1, 1, rows, FLAT_WIDTH), BF16, "GR normalized row")
    _expect(up, (1, 1, PARTIAL_WIDTH, FLAT_WIDTH), BF16, "GR up")
    mesh = gathered_partials.device()
    block = fp.allocate((1, 1, rows, LOCAL_HIDDEN), BF16, ttnn.TILE_LAYOUT, mesh)
    injection = fp.allocate((1, 1, rows, BRANCHES), BF16, ttnn.TILE_LAYOUT, mesh)
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
            (core, ([_stream(gathered_partials, t, TP_SIZE, c * t, PARTIAL_TILES, 1, t * TP_SIZE)], [0]))
            for c, core in enumerate(producers)
        ],
    )
    computes = [
        fp.compute_kernel(LOWRANK3, plain_set, [t, TP_SIZE, t, 16, 17], fp32_dest=True),
        fp.compute_kernel(LOWRANK3, inject_set, [t, TP_SIZE, inject_local, 16, 17], fp32_dest=True),
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
                    _stream(up, BRANCHES, PARTIAL_TILES, j, n_tiles, HIDDEN_TILES, BRANCHES * PARTIAL_TILES),
                ],
            )
            for j, core in enumerate(workers)
        ],
        recv_cb=8,
        recv_tiles=PARTIAL_TILES,
        senders=n_producers,
        zero_cb=11,
    )
    gate_k = fp.compute_kernel(
        GATE3,
        w_set,
        [PARTIAL_TILES, BRANCHES, 8, 9, 10, 11, 12, 13, 14, 18, UP_SPILL if matmul == "chain" else 0, 15],
        fp32_dest=True,
        unpack_to_dest_fp32=(10, 12, 13),
    )
    writer = _writer(w_set, [(block, 18)], [(core, [(1, j, 1, 1)]) for j, core in enumerate(workers)])
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

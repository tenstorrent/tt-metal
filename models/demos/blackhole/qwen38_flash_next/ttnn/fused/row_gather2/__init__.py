# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``row_gather2``: the barrier-free packed row all-gather of the per-block decode hidden.

Same contract as ``row_gather`` (``ttnn.all_gather(x, dim=3, cluster_axis=1, memory_config=<consumer's>)`` of a local
``[1, 1, rows, W]`` bf16 TILE tensor, rows 1..2 here, valid rows only), but one fabric traversal instead of two: the
packed rows of every device land in a PERSISTENT L1 landing buffer (parity double buffered, one call counter) with the
data semaphores of two parities, so no per-call barrier is needed (``kernels/transport.cpp`` has the safety argument),
and the unpack into the gathered tensor's pages is shared by both RISCs.  The first call per mesh creates the persistent
buffer and the semaphores: make it eagerly, before any trace capture (``prepare``).  Tolerance class BITWISE.

Live rows 3..8 (QWEN38_ROW_GATHER_ROWS_MAX, default 8; 2 = off) take ``row_gather_rows`` (``kernels/transport8.cpp``): the
packed rows of 3..8 rows do not fit the persistent landing buffer, which must not grow, so that mode lands in the program's
own CB behind v1's per-call barrier, one core per link, and writes zeros into rows >= the live rows of every gathered page
(``live_rows=``: the verify pass's R real rows on the 32-row tile).  Rows 1..2 keep the program above unchanged.
"""

from __future__ import annotations

import atexit
import os
import sys

import ttnn

from ...contracts import TP_SIZE
from .. import gr_fold
from .. import program as fp
from ..registry import BITWISE

NAME = "row_gather2"
PHASE = "rowgather2"
KERNEL = fp.kernel_source(NAME, "transport.cpp")
PACK_CB = 0
CHUNK_PITCH = 64
LAND_HEADER_BYTES = 16
ROWS_MAX = 2
# rows 3..8: the barrier + CB-landing program (kernels/transport8.cpp); the persistent landing buffer stays sized for 2 rows
ROWS_ENV = "QWEN38_ROW_GATHER_ROWS_MAX"
ROWS_BARRIER_MAX = 8
KERNEL_ROWS = fp.kernel_source(NAME, "transport8.cpp")
PHASE_ROWS = "rowgather"  # the (barrier, data) semaphore pair of the line transport's row gather (shared with v1: one call at a time)
ROWS_LINKS = 2
ZERO_BLOCK_BYTES = 1536  # tile bytes - one face: the longest zero write of the dead rows
_STATE: dict[int, dict] = {}
CALLS: dict[
    str, int
] = {}  # row_gather() launches (program builds, eager or captured) by calling function, for the A/B logs


def _report_calls() -> None:
    if CALLS:
        print(f"ROW_GATHER_CALLS total={sum(CALLS.values())} {dict(sorted(CALLS.items()))}", flush=True)


atexit.register(_report_calls)


def enabled_by_env(environ=None) -> bool:
    return (os.environ if environ is None else environ).get("QWEN38_ROW_GATHER", "1") != "0"


def _tiles(x) -> int:
    return int(x.padded_shape[3]) // fp.TILE


def rows_max(environ=None) -> int:
    """The largest live row count served (QWEN38_ROW_GATHER_ROWS_MAX, default 8): 2 = rows 3..8 take ``ttnn.all_gather``."""

    raw = (os.environ if environ is None else environ).get(ROWS_ENV, str(ROWS_BARRIER_MAX))
    try:
        value = int(raw)
    except ValueError:
        return ROWS_MAX
    return min(max(value, ROWS_MAX), ROWS_BARRIER_MAX)


def _geometry(hidden, live: int):
    """(links, pitch, slot bytes) of the rows program for ``live`` rows: one core per link, the shard's tiles dealt round
    robin, each chunk (rows [0, live) of a face) on a 64 B pitch, 2 chunks per tile."""

    line = gr_fold.line(hidden.device())
    links = min(ROWS_LINKS, int(getattr(line, "links", ROWS_LINKS)), _tiles(hidden))
    pitch = -(-(live * (fp.TILE_BYTES[hidden.dtype] // 64)) // 64) * 64
    my_tiles = -(-_tiles(hidden) // links)
    return links, pitch, my_tiles * 2 * pitch


def admits_rows(hidden, *, cluster_axis=1, live_rows=None) -> bool:
    """Live rows 3..``rows_max()`` of a ``[1, 1, S, W]`` bf16 TILE shard on the TP line (S = the live rows, or the whole
    tile with ``live_rows`` given), whose packed rows fit one fabric packet per core."""

    if not enabled_by_env():
        return False
    try:
        shape = tuple(int(v) for v in hidden.shape)
        padded = tuple(int(v) for v in hidden.padded_shape)
        live = shape[2] if live_rows is None else int(live_rows)
        if not (
            cluster_axis == gr_fold.TP_AXIS
            and len(shape) == 4
            and shape[:2] == (1, 1)
            and ROWS_MAX < live <= min(rows_max(), shape[2])
            and padded[2] == fp.TILE
            and padded[3] % fp.TILE == 0
            and hidden.dtype == ttnn.bfloat16
            and hidden.layout == ttnn.TILE_LAYOUT
            and tuple(int(v) for v in hidden.device().shape) == (1, TP_SIZE)
            and (gr_fold.line(hidden.device()) is not None)
        ):
            return False
        return _geometry(hidden, live)[2] <= int(ttnn.get_tt_fabric_max_payload_size_bytes())
    except Exception:
        return False


def admits(hidden, *, cluster_axis=1, memory_config=None, live_rows=None) -> bool:
    if not enabled_by_env():
        return False
    try:
        shape = tuple(int(v) for v in hidden.shape)
        padded = tuple(int(v) for v in hidden.padded_shape)
        live = shape[2] if live_rows is None else int(live_rows)
        if not (shape[2] <= ROWS_MAX and live == shape[2]):
            return admits_rows(hidden, cluster_axis=cluster_axis, live_rows=live)
        return (
            cluster_axis == gr_fold.TP_AXIS
            and len(shape) == 4
            and shape[:2] == (1, 1)
            and 1 <= shape[2] <= ROWS_MAX
            and padded[2] == fp.TILE
            and padded[3] % fp.TILE == 0
            and hidden.dtype == ttnn.bfloat16
            and hidden.layout == ttnn.TILE_LAYOUT
            and tuple(int(v) for v in hidden.device().shape) == (1, TP_SIZE)
            and (gr_fold.line(hidden.device()) is not None)
            and _tiles(hidden) * 2 * CHUNK_PITCH <= int(ttnn.get_tt_fabric_max_payload_size_bytes())
        )
    except Exception:
        return False


def composed(hidden, *, cluster_axis=1, memory_config=None, live_rows=None):
    return ttnn.all_gather(hidden, dim=3, cluster_axis=cluster_axis, memory_config=memory_config)


def prepare(mesh, tiles: int = 10) -> dict:
    """The persistent landing buffer ([call counter 16 B][2 parity sets x TP_SIZE slots]) on the transport core of every
    device and the two parity data semaphores (one pair per mesh), created once, before any trace capture.  The slot
    size is that of ``tiles`` shard tiles; a mesh serves one shard width."""

    state = _STATE.get(id(mesh))
    if state is not None:
        if state["tiles"] != tiles:
            raise ValueError(f"row_gather2 landing buffer was made for {state['tiles']} shard tiles, got {tiles}")
        return state
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sem0, sem1 = ttnn.create_global_semaphore(mesh, cores, 0), ttnn.create_global_semaphore(mesh, cores, 0)
    slot = tiles * 2 * CHUNK_PITCH
    words = (LAND_HEADER_BYTES + 2 * TP_SIZE * slot) // 4
    core = gr_fold.TRANSPORT["partials"][0]
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), [1, words], ttnn.ShardOrientation.ROW_MAJOR),
    )
    land = ttnn.zeros([1, words], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=memory)
    state = {
        "tiles": tiles,
        "sems": (sem0, sem1),
        "data0": int(ttnn.get_global_semaphore_address(sem0)),
        "data1": int(ttnn.get_global_semaphore_address(sem1)),
        "land": land,
        "land_addr": int(land.buffer_address()),
    }
    _STATE[id(mesh)] = state
    table = gr_fold.line_semaphores(mesh)
    table.addresses[PHASE] = (state["data0"], state["data0"])  # (barrier: unused, data parity 0)
    return state


def _count_call() -> None:
    frame = sys._getframe(2)
    while frame.f_back is not None and frame.f_code.co_name in ("row_gather", "row_gather_or_all_gather", "__call__"):
        frame = frame.f_back
    CALLS[frame.f_code.co_name] = CALLS.get(frame.f_code.co_name, 0) + 1


def row_gather_rows(hidden, *, cluster_axis=1, memory_config=None, live_rows=None):
    """The gather of live rows 3..8: ``transport8.cpp`` on one core per link, a per-call barrier, packets landing in the
    program's CB (no persistent buffer), rows >= the live rows of every gathered tile page written as zeros.  The output has
    the input's logical rows (``[1, 1, S, W * TP_SIZE]``), live rows bit-equal to ``ttnn.all_gather``."""

    shape = tuple(int(v) for v in hidden.shape)
    if not admits_rows(hidden, cluster_axis=cluster_axis, live_rows=live_rows):
        raise ValueError(
            f"row_gather2 rows mode does not admit {shape} live {live_rows} {hidden.dtype} {hidden.layout}"
        )
    _count_call()
    live = shape[2] if live_rows is None else int(live_rows)
    mesh = hidden.device()
    links, pitch, slot = _geometry(hidden, live)
    tiles = _tiles(hidden)
    tile_bytes = fp.TILE_BYTES[hidden.dtype]
    out = fp.allocate(
        shape[:3] + (shape[3] * TP_SIZE,),
        hidden.dtype,
        ttnn.TILE_LAYOUT,
        mesh,
        memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )
    transports = gr_fold.TRANSPORT["partials"][:links]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in transports])
    spec = gr_fold.Transport(
        out,
        hidden,
        PACK_CB,
        tiles,
        gr_fold.SOURCE_TENSOR,
        PHASE_ROWS,
        transports,
        page_strides=(1, tiles),
        rows=live,
        pack_cb=PACK_CB,
        kernel=KERNEL_ROWS,
        extra_args=(live, tile_bytes),
    )
    cbs = [fp.cb_descriptor(PACK_CB, hidden.dtype, (TP_SIZE + 2) * slot + 2 * ZERO_BLOCK_BYTES, 1, cores)]
    semaphores = [
        fp.semaphore_descriptor(sem, cores) for sem in (gr_fold.SEM_GO, gr_fold.SEM_SCRATCH, gr_fold.SEM_DONE)
    ]
    meta = fp.program_meta(
        NAME,
        "row_gather2_rows",
        live,
        reads=(hidden,),
        dram_bytes=TP_SIZE * tiles * tile_bytes,
        l1_bytes=(TP_SIZE + 2) * slot + 2 * ZERO_BLOCK_BYTES,
        cores=links,
        outputs=((out, None),),
    )
    fp.run_program(
        [hidden, out],
        gr_fold.transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, links=links),
        meta=meta,
    )
    return fp.stamp_topology(out, hidden)


def row_gather(hidden, *, cluster_axis=1, memory_config=None, live_rows=None):
    shape = tuple(int(v) for v in hidden.shape)
    if admits_rows(hidden, cluster_axis=cluster_axis, live_rows=live_rows):
        return row_gather_rows(hidden, cluster_axis=cluster_axis, memory_config=memory_config, live_rows=live_rows)
    if not admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config, live_rows=live_rows):
        raise ValueError(f"row_gather2 does not admit {shape} {hidden.dtype} {hidden.layout}")
    frame = sys._getframe(1)
    while frame.f_back is not None and frame.f_code.co_name in ("row_gather", "row_gather_or_all_gather", "__call__"):
        frame = frame.f_back
    CALLS[frame.f_code.co_name] = CALLS.get(frame.f_code.co_name, 0) + 1
    from .. import (
        row_gather3,
    )  # the GR gate program may already have gathered this block's input (QWEN38_GATE_GATHER=1)

    handed = row_gather3.take_gathered(hidden, memory_config)
    if handed is not None:
        return handed
    mesh = hidden.device()
    rows, tiles = shape[2], _tiles(hidden)
    state = prepare(mesh, tiles)
    tile_bytes = fp.TILE_BYTES[hidden.dtype]
    out = fp.allocate(
        (1, 1, rows, shape[3] * TP_SIZE),
        hidden.dtype,
        ttnn.TILE_LAYOUT,
        mesh,
        memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )
    transports = gr_fold.TRANSPORT["partials"][:1]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in transports])
    spec = gr_fold.Transport(
        out,
        hidden,
        PACK_CB,
        tiles,
        gr_fold.SOURCE_TENSOR,
        PHASE,
        transports,
        page_strides=(1, tiles),
        rows=rows,
        pack_cb=PACK_CB,
        kernel=KERNEL,
        extra_args=(rows, state["land_addr"], state["data1"], tile_bytes),
    )
    slot = tiles * 2 * CHUNK_PITCH
    cbs = [fp.cb_descriptor(PACK_CB, hidden.dtype, 2 * slot, 1, cores)]
    semaphores = [
        fp.semaphore_descriptor(sem, cores) for sem in (gr_fold.SEM_GO, gr_fold.SEM_SCRATCH, gr_fold.SEM_DONE)
    ]
    meta = fp.program_meta(
        NAME,
        "row_gather2",
        rows,
        reads=(hidden,),
        dram_bytes=TP_SIZE * tiles * 2 * CHUNK_PITCH,
        l1_bytes=2 * slot + TP_SIZE * slot,
        cores=1,
        outputs=((out, None),),
    )
    fp.run_program(
        [hidden, out],
        gr_fold.transport_mesh_program(mesh, [spec], semaphores=semaphores, cbs=cbs, links=1),
        meta=meta,
    )
    return fp.stamp_topology(out, hidden)


def row_gather_or_all_gather(hidden, *, cluster_axis=1, memory_config=None, live_rows=None):
    if admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config, live_rows=live_rows):
        return row_gather(hidden, cluster_axis=cluster_axis, memory_config=memory_config, live_rows=live_rows)
    return composed(hidden, cluster_axis=cluster_axis, memory_config=memory_config)

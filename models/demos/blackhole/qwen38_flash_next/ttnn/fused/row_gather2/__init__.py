# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``row_gather2``: the barrier-free packed row all-gather of the per-block decode hidden.

Same contract as ``row_gather`` (``ttnn.all_gather(x, dim=3, cluster_axis=1, memory_config=<consumer's>)`` of a local
``[1, 1, rows, W]`` bf16 TILE tensor, rows 1..2 here, valid rows only), but one fabric traversal instead of two: the
packed rows of every device land in a PERSISTENT L1 landing buffer (parity double buffered, one call counter) with the
data semaphores of two parities, so no per-call barrier is needed (``kernels/transport.cpp`` has the safety argument),
and the unpack into the gathered tensor's pages is shared by both RISCs.  The first call per mesh creates the persistent
buffer and the semaphores: make it eagerly, before any trace capture (``prepare``).  Tolerance class BITWISE.
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


def admits(hidden, *, cluster_axis=1, memory_config=None) -> bool:
    if not enabled_by_env():
        return False
    try:
        shape = tuple(int(v) for v in hidden.shape)
        padded = tuple(int(v) for v in hidden.padded_shape)
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


def composed(hidden, *, cluster_axis=1, memory_config=None):
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


def row_gather(hidden, *, cluster_axis=1, memory_config=None):
    shape = tuple(int(v) for v in hidden.shape)
    if not admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config):
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


def row_gather_or_all_gather(hidden, *, cluster_axis=1, memory_config=None):
    if admits(hidden, cluster_axis=cluster_axis, memory_config=memory_config):
        return row_gather(hidden, cluster_axis=cluster_axis, memory_config=memory_config)
    return composed(hidden, cluster_axis=cluster_axis, memory_config=memory_config)

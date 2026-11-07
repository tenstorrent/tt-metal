# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Core placement for the MoE block's fused programs.

Programs never take reserved cores: the decode linears' activation storage cores ``(0..4, 0)``, every device's primary
DRAM bank reader cores and moe_compute's tilize drain core. The scan is deterministic (x from ``grid_x - width`` down
to 0 outer, y down to 0 inner), so a program lands on the same cores standalone and hosted.
"""

from __future__ import annotations

from typing import Callable, Iterable

import ttnn

Core = tuple[int, int]
Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 inclusive

STORAGE_CORES: tuple[Core, ...] = tuple((x, 0) for x in range(5))  # the decode linears' activation shard
HIDDEN = 2560
_RESERVED: dict[int, frozenset[Core]] = {}


def rectangle_cores(rect: Rect) -> list[Core]:
    x0, y0, x1, y1 = rect
    return [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def core_range(rect: Rect) -> ttnn.CoreRange:
    x0, y0, x1, y1 = rect
    return ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))


def range_rect(core_range: ttnn.CoreRange) -> Rect:
    return (core_range.start.x, core_range.start.y, core_range.end.x, core_range.end.y)


def free_rectangle_in(
    grid_x: int,
    grid_y: int,
    reserved: Iterable[Core],
    width: int,
    height: int,
    *,
    accept: Callable[[Rect], bool] | None = None,
) -> Rect:
    """The first ``width x height`` rectangle in the grid disjoint from ``reserved``, in the module's scan order.

    ``accept`` may refuse a candidate (e.g. one that is not a contiguous NoC rectangle). Raises when nothing fits."""

    taken = frozenset(reserved)
    if width < 1 or height < 1 or width > grid_x or height > grid_y:
        raise ValueError(f"a {width}x{height} rectangle does not fit the {grid_x}x{grid_y} compute grid")
    for x0 in range(grid_x - width, -1, -1):
        for y0 in range(grid_y - height, -1, -1):
            rect = (x0, y0, x0 + width - 1, y0 + height - 1)
            if any(core in taken for core in rectangle_cores(rect)):
                continue
            if accept is not None and not accept(rect):
                continue
            return rect
    raise RuntimeError(
        f"no free {width}x{height} rectangle in the {grid_x}x{grid_y} compute grid outside {len(taken)} reserved cores"
    )


def drain_core(mesh) -> ttnn.CoreCoord:
    """moe_compute's tilize drain core for the model's arguments (one token-parallel core, auto width shard)."""

    from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim

    width_shard = auto_output_width_shard_dim(HIDDEN)
    return ttnn.experimental.get_moe_tilize_drain_core(mesh, 1, width_shard, HIDDEN)


def reserved_cores(mesh) -> frozenset[Core]:
    """The reserved cores of every device in the mesh; memoized per mesh, so call before a trace capture."""

    key = id(mesh)
    if key not in _RESERVED:
        from ..decode_matmul import mesh_dram_bank_worker_signatures

        cores: set[Core] = set(STORAGE_CORES)
        for signature in mesh_dram_bank_worker_signatures(mesh).values():
            cores.update(signature)
        drain = drain_core(mesh)
        cores.add((int(drain.x), int(drain.y)))
        _RESERVED[key] = frozenset(cores)
    return _RESERVED[key]


def free_rectangle(
    mesh, width: int, height: int, *, avoid: Iterable[Core] = (), accept: Callable[[Rect], bool] | None = None
) -> ttnn.CoreRange:
    """``free_rectangle_in`` on the mesh's compute grid with ``reserved_cores(mesh) | avoid``."""

    grid = mesh.compute_with_storage_grid_size()
    return core_range(
        free_rectangle_in(grid.x, grid.y, reserved_cores(mesh) | frozenset(avoid), width, height, accept=accept)
    )


def cores_of(core_range: ttnn.CoreRange) -> list[ttnn.CoreCoord]:
    """The cores of a range in row-major order (y outer, x inner)."""

    return [ttnn.CoreCoord(x, y) for x, y in rectangle_cores(range_rect(core_range))]

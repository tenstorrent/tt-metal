# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The compute grid a custom op plans its work on, and the planners' shared rules.

Every op derives its work split (cores per row tile, column tiles per core, units per core, exchange groups)
from the device's ``compute_with_storage_grid_size`` -- never from a constant -- so the same kernels run on
Wormhole (8x8 worker grid) and Blackhole (13x10). ``GEMMA4_TEST_GRID=WxH`` (a test knob) caps the grid an op may
use, e.g. ``8x8`` on Blackhole exercises the Wormhole split on a Blackhole chip. The planners here are pure
python so a host test can check them for both grids.
"""

import os

import ttnn


def compute_grid(device):
    """``(x, y)`` of the grid an op may use: the device grid, capped by ``GEMMA4_TEST_GRID`` when set."""
    g = device.compute_with_storage_grid_size()
    gx, gy = int(g.x), int(g.y)
    cap = os.environ.get("GEMMA4_TEST_GRID", "").strip().lower()
    if cap:
        cx, cy = (int(v) for v in cap.split("x"))
        gx, gy = min(gx, cx), min(gy, cy)
    return gx, gy


def grid_cores(grid) -> int:
    """Core count of a ``(x, y)`` grid."""
    return int(grid[0]) * int(grid[1])


def row_major_cores(grid, n):
    """The first ``n`` cores of the grid in row-major order (x fastest), as ``ttnn.CoreCoord``."""
    gx, gy = grid
    assert n <= gx * gy, f"{n} cores on a {gx}x{gy} grid"
    return [ttnn.CoreCoord(i % gx, i // gx) for i in range(n)]


def core_range_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def plan_qkv_heads_units(grid, batch):
    """Work split of ``qkv_heads_norm_rope``: ``(n_cores, tensors_per_core)``. One (user, tensor) per core when
    the grid holds 3 B cores, else one user per core with the three tensors in sequence (needs B cores)."""
    gx, gy = grid
    cores = gx * gy
    if 3 * batch <= cores:
        return 3 * batch, 1
    if batch <= cores:
        return batch, 3
    raise ValueError(f"batch {batch} does not fit {cores} cores")


def plan_concat_heads_units(grid, num_heads, cols, heads_per_unit=None):
    """Work split of ``concat_heads_decode``: ``(n_cores, heads_per_unit, groups, per-core unit counts)``."""
    gx, gy = grid
    n_grid = gx * gy
    if heads_per_unit is None:
        heads_per_unit = next(hg for hg in range(1, num_heads + 1) if cols * ((num_heads + hg - 1) // hg) <= n_grid)
    heads_per_unit = max(1, min(num_heads, heads_per_unit))
    groups = (num_heads + heads_per_unit - 1) // heads_per_unit
    n_units = cols * groups
    n_cores = min(n_units, n_grid)
    per = [(n_units // n_cores) + (1 if i < n_units % n_cores else 0) for i in range(n_cores)]
    return n_cores, heads_per_unit, groups, per


def plan_norm_cores(grid, cols_t, nin=1, cap=None):
    """Work split of the fused decode norms (``fused_norm_decode`` / ``fused_norm_chain``): the number of cores
    (a divisor of the column tiles, at most ``cap`` -- GEMMA4_FUSED_NORM_CORES, default 22 -- and the grid, with
    at most 8 / nin tiles per core so one fp32 dest window holds them)."""
    gx, gy = grid
    limit = min(gx * gy, 64)  # two fp32 receive tiles of 32 columns
    cap = min(limit, max(1, int(os.environ.get("GEMMA4_FUSED_NORM_CORES", "22")) if cap is None else cap))
    max_per = 8 // nin
    for n in range(min(cap, cols_t), 0, -1):
        if cols_t % n == 0 and cols_t // n <= max_per:
            return n
    for n in range(cap + 1, min(limit, cols_t) + 1):
        if cols_t % n == 0 and cols_t // n <= max_per:
            return n
    raise ValueError(f"no core count up to {limit} divides {cols_t} column tiles into <= {max_per}")

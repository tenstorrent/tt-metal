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

# The Blackhole worker L1 (the Python MeshDevice exposes no l1_size_per_core); the Blackhole-only ops size their
# resident CBs against it from the allocator base.
BLACKHOLE_L1_PER_CORE = 1536 * 1024


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


NORM_MAX_CORES = 64  # an exchange group: two fp32 receive tiles of 32 columns
NORM_DEFAULT_CORES = 22  # the preferred group size (GEMMA4_FUSED_NORM_CORES caps it)


def plan_norm_split(grid_cores_, cols_t, nin=1, rt_in=1, cap=None):
    """The one work split of the fused norms (``fused_norm_decode``) for ``rt_in`` row tiles on a grid of
    ``grid_cores_`` cores: ``(N, per, groups, passes)`` -- ``N`` cores per row tile (a divisor of ``cols_t`` with
    ``per = cols_t / N <= 8 / nin`` tiles each so one fp32 dest window holds them), ``groups`` row tiles per pass
    (``min(rt_in, grid_cores_ // N)``) and ``passes = ceil(rt_in / groups)`` launches of the program. The split
    minimizes the passes, prefers a core count up to the cap (``cap``, else ``GEMMA4_FUSED_NORM_CORES``, default
    22) and then the largest count; it is derived from the grid, never from a constant: 2816 wide at 128 rows =
    11 cores x 4 row tiles in one pass on Wormhole's 8x8, 22 x 4 on Blackhole's 13x10. Raises ValueError when no
    split fits (the callers fall back to the composite path)."""
    if cap is None:
        cap = int(os.environ.get("GEMMA4_FUSED_NORM_CORES", str(NORM_DEFAULT_CORES)))
    cap = min(NORM_MAX_CORES, max(1, int(cap)))
    max_per = 8 // nin
    best = None
    for n in range(1, min(NORM_MAX_CORES, cols_t) + 1):
        if cols_t % n or cols_t // n > max_per:
            continue
        groups = min(rt_in, grid_cores_ // n)
        if groups < 1:
            continue
        passes = -(-rt_in // groups)
        key = (passes, 0 if n <= cap else 1, -n)
        if best is None or key < best[0]:
            best = (key, (n, cols_t // n, groups, passes))
    if best is None:
        raise ValueError(
            f"no core count up to {NORM_MAX_CORES} divides {cols_t} column tiles into <= {max_per} on "
            f"{grid_cores_} cores"
        )
    return best[1]


def plan_norm_cores(grid, cols_t, nin=1, cap=None):
    """The fused norms' core count for one row tile on ``grid`` (``plan_norm_split``'s ``N``)."""
    return plan_norm_split(grid_cores(grid), cols_t, nin, 1, cap)[0]

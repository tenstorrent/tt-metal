# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Core sets for Bonsai's generic ops as rectangles.

A CoreRangeSet of one single-core range per core makes the dispatcher send each core its own launch commands: a
130-core ternary decode linear took ~52 us per traced call whatever its size (16 cores: 9 us). The first n cores in
row-major order are at most two rectangles (the full rows, then the partial row), which the dispatcher multicasts."""

import ttnn


def row_major_crs(n: int, grid_x: int) -> ttnn.CoreRangeSet:
    """Cores 0 .. n - 1, core c at (c % grid_x, c // grid_x), as at most two rectangles."""
    assert n > 0
    rows, rest = divmod(n, grid_x)
    ranges = []
    if rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, rows - 1)))
    if rest:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, rows), ttnn.CoreCoord(rest - 1, rows)))
    return ttnn.CoreRangeSet(ranges)


def virtual_rects(device, crs):
    """[x0, y0, x1, y1, cores] multicast rectangles (virtual coordinates) covering crs, split where the virtual columns
    are not adjacent (a multicast rectangle must not span non-Tensix columns)."""
    rects = []
    for r in crs.ranges():
        vx = [
            device.worker_core_from_logical_core(ttnn.CoreCoord(x, r.start.y)).x for x in range(r.start.x, r.end.x + 1)
        ]
        vy0 = device.worker_core_from_logical_core(ttnn.CoreCoord(r.start.x, r.start.y)).y
        vy1 = device.worker_core_from_logical_core(ttnn.CoreCoord(r.start.x, r.end.y)).y
        rows = r.end.y - r.start.y + 1
        assert vy1 - vy0 == rows - 1, (vy0, vy1, rows)
        start = 0
        for i in range(1, len(vx) + 1):
            if i == len(vx) or vx[i] != vx[i - 1] + 1:
                rects.append([vx[start], vy0, vx[i - 1], vy1, (i - start) * rows])
                start = i
    return rects


def row_major_range_crs(first: int, last: int, grid_x: int) -> ttnn.CoreRangeSet:
    """Cores first .. last - 1 (row-major, grid_x wide) as at most three rectangles."""
    assert last > first
    ranges, c = [], first
    while c < last:
        y, x = divmod(c, grid_x)
        if x == 0 and last - c >= grid_x:
            rows = (last - c) // grid_x
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(grid_x - 1, y + rows - 1)))
            c += rows * grid_x
        else:
            end = min(last, (y + 1) * grid_x)
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(end - 1 - y * grid_x, y)))
            c = end
    return ttnn.CoreRangeSet(ranges)

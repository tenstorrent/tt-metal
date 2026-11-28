# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Core sets for Bonsai's generic ops as rectangles.

A CoreRangeSet of one single-core range per core makes the dispatcher send each core its own launch commands: a
130-core ternary decode linear took ~52 us per traced call whatever its size (16 cores: 9 us). The first n cores in
row-major order are at most two rectangles (the full rows, then the partial row), which the dispatcher multicasts."""

import ttnn

_LAYOUT = None  # receiver order (stream.py): core i is _LAYOUT[i] instead of (i % grid_x, i // grid_x)


_DEFINES = []
_GENERIC_OP = None


def set_layout(order, defines=()):
    """Place cores by `order` (a list of CoreCoord) instead of row-major, and add `defines` (the kernels' way to find
    their place, e.g. stream.py RX_INFO) to every generic_op kernel; None: row-major again."""
    global _LAYOUT, _DEFINES, _GENERIC_OP
    _LAYOUT = list(order) if order is not None else None
    _DEFINES = list(defines) if order is not None else []
    if _GENERIC_OP is None:
        _GENERIC_OP = ttnn.generic_op

        def generic_op(tensors, program, *args, **kwargs):
            if _DEFINES:
                for k in program.kernels:
                    have = {d[0] for d in k.defines}
                    k.defines = list(k.defines) + [d for d in _DEFINES if d[0] not in have]
            return _GENERIC_OP(tensors, program, *args, **kwargs)

        ttnn.generic_op = generic_op


def available(grid_cores: int) -> int:
    """Cores the ops may use: the layout's, else the whole grid."""
    return len(_LAYOUT) if _LAYOUT is not None else grid_cores


def core_xy(i: int, grid_x: int) -> ttnn.CoreCoord:
    if _LAYOUT is not None:
        return _LAYOUT[i]
    return ttnn.CoreCoord(i % grid_x, i // grid_x)


def rects(cores) -> ttnn.CoreRangeSet:
    """Cells as rectangles: runs along each row, then equal runs of adjacent rows merged."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    runs = []  # (y, x0, x1)
    for y in sorted(rows):
        xs = sorted(rows[y])
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            runs.append((y, start, prev))
            if x is not None:
                start = prev = x
    merged = []  # [x0, x1, y0, y1]
    for y, x0, x1 in runs:
        for m in merged:
            if m[0] == x0 and m[1] == x1 and m[3] == y - 1:
                m[3] = y
                break
        else:
            merged.append([x0, x1, y, y])
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(a, c), ttnn.CoreCoord(b, d)) for a, b, c, d in merged])


def row_major_crs(n: int, grid_x: int) -> ttnn.CoreRangeSet:
    """Cores 0 .. n - 1, core c at (c % grid_x, c // grid_x), as at most two rectangles."""
    assert n > 0
    if _LAYOUT is not None:
        return rects(_LAYOUT[:n])
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
    if _LAYOUT is not None:
        return rects(_LAYOUT[first:last])
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

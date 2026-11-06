# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``rect_core_ranges``: the smallest set of rectangular ``CoreRange``s covering a list of cores. A generic op whose
program descriptor lists one ``CoreRange`` per core (80-120 ranges) costs 15-35 us of dispatch per launch; the same
cores as 1-3 rectangles dispatch in < 1 us (doc/mesh/README.md, the decode ledger of 2026-10-21). Per-core runtime
args are keyed by ``CoreCoord`` and do not change."""

from __future__ import annotations

import ttnn


def rect_core_ranges(cores) -> ttnn.CoreRangeSet:
    """Cores (any order, duplicates allowed) -> ``CoreRangeSet`` of column-run rectangles: every column's
    contiguous y-runs, with adjacent columns that share a run merged into one rectangle."""
    by_x: dict[int, set[int]] = {}
    for c in cores:
        by_x.setdefault(int(c.x), set()).add(int(c.y))
    runs: dict[tuple[int, int], list[int]] = {}  # (y0, y1) -> columns holding that run
    for x, ys in by_x.items():
        ys_sorted = sorted(ys)
        start = prev = ys_sorted[0]
        for y in ys_sorted[1:] + [None]:
            if y is not None and y == prev + 1:
                prev = y
                continue
            runs.setdefault((start, prev), []).append(x)
            if y is not None:
                start = prev = y
    ranges = []
    for (y0, y1), xs in runs.items():
        xs_sorted = sorted(xs)
        x0 = xp = xs_sorted[0]
        for x in xs_sorted[1:] + [None]:
            if x is not None and x == xp + 1:
                xp = x
                continue
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(xp, y1)))
            if x is not None:
                x0 = xp = x
    return ttnn.CoreRangeSet(ranges)


def num_ranges(core_set: ttnn.CoreRangeSet) -> int:
    return len(core_set.ranges())

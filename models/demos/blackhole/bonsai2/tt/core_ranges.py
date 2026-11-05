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

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/core_ranges.rect_core_ranges``: covers exactly the given cores with few rectangles (no device)."""

import pytest

import ttnn
from models.demos.glm53_flash.custom_ops.core_ranges import num_ranges, rect_core_ranges


def _cover(core_set):
    out = set()
    for r in core_set.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                out.add((x, y))
    return out


@pytest.mark.parametrize(
    "cores, max_ranges",
    [
        ([(c // 10, c % 10) for c in range(120)], 1),  # the full 12 x 10 grid
        ([(c // 10, c % 10) for c in range(85)], 2),  # 8 whole columns + a partial one (paged_pool_gather)
        ([(x, y0 + p) for x in range(8) for y0 in (0, 5) for p in range(5)], 1),  # 16 vertical runs of 5 = 8 x 10
        ([(x, y) for x in range(3) for y in (0, 1, 2, 7, 8)], 2),  # two runs per column
        ([(0, 0), (2, 0), (4, 0)], 3),  # isolated cores
        ([(1, 1), (1, 1), (1, 2)], 1),  # duplicates
    ],
)
def test_rect_core_ranges(cores, max_ranges):
    coords = [ttnn.CoreCoord(x, y) for x, y in cores]
    cs = rect_core_ranges(coords)
    assert _cover(cs) == set(cores)
    assert num_ranges(cs) <= max_ranges, num_ranges(cs)

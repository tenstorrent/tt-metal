# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


@pytest.mark.parametrize(
    "num_cores, grid_x, grid_y, row_wise, expected_count",
    [
        (4, 8, 8, False, 4),
        (4, 8, 8, True, 4),
        (16, 4, 4, False, 16),
        (1, 1, 1, False, 1),
    ],
)
def test_grid_to_cores_grid_overload(num_cores, grid_x, grid_y, row_wise, expected_count):
    cores = ttnn.grid_to_cores(num_cores, grid_x, grid_y, row_wise=row_wise)
    assert len(cores) == expected_count
    assert cores[0] == ttnn.CoreCoord(0, 0)


@pytest.mark.parametrize(
    "start, end, row_wise, expected_count",
    [
        (ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1), False, 4),
        (ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0), False, 1),
        (ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3), True, 16),
    ],
)
def test_grid_to_cores_corecoord_overload(start, end, row_wise, expected_count):
    cores = ttnn.grid_to_cores(start, end, row_wise=row_wise)
    assert len(cores) == expected_count


def _crs(cols, rows):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})


def test_split_work_to_cores_coreset_even():
    # Sanity: even division should put all work in group_1, none in group_2.
    num_cores, all_cores, g1, g2, g1_units, g2_units = ttnn.split_work_to_cores(_crs(4, 4), 16)
    assert num_cores == 16
    assert g1.num_cores() == 16
    assert g2.num_cores() == 0
    assert g1_units == 1
    assert g2_units == 0


def test_split_work_to_cores_coreset_uneven_column_wise():
    # Regression test for group 2 start coord bug in column-wise mode:
    # when group 1 fills exactly one or more complete columns, group 2 start
    # used end_coord.y (column bottom) instead of start_coord.y (column top).
    #
    # 8x8 (64-core) grid, 72 units: remainder=8 → group_1 fills col 0 (8 cores),
    # last=(0,7)==end_coord.y → "next column" branch fires.
    # Bug: group_2 starts at (1,7) → can only assign 1+8*6=49 of 56 needed → TT_FATAL.
    # Fix: group_2 starts at (1,0) → assigns 8*7=56 ✓.
    num_cores, all_cores, g1, g2, g1_units, g2_units = ttnn.split_work_to_cores(_crs(8, 8), 72)
    assert num_cores == 64
    assert g1.num_cores() + g2.num_cores() == 64
    assert g1.num_cores() == 8  # 72 % 64 = 8 cores get one extra unit
    assert g2.num_cores() == 56
    assert g1_units == 2  # 72 // 64 + 1
    assert g2_units == 1  # 72 // 64

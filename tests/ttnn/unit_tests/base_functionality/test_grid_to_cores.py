# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

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

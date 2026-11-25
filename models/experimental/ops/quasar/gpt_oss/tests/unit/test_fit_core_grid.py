# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of fit_core_grid, which fits an expert matmul core grid to the device grid."""

import math

import pytest

from models.experimental.ops.quasar.gpt_oss.tt.experts.config import fit_core_grid


# Each expected grid is the one with the most cores, inside the device grid and at most the requested count, on
# which every core gets output columns. For example, for (5, 6) and 90 columns on 8x4: every grid with 19 to 30
# cores leaves cores idle (for example 28 cores take 4 columns each, so only 23 get work), and (6, 3) gives 18
# cores 5 columns each.
@pytest.mark.parametrize(
    "cores, num_tiles_n, max_grid, expected",
    [
        ((3, 4), 90, (8, 4), (3, 4)),  # fits: unchanged
        ((5, 6), 90, (8, 4), (6, 3)),  # the gpt-oss down projection on the 8x4 Quasar grid
        ((5, 6), 90, (2, 1), (2, 1)),
        ((3, 4), 90, (2, 1), (2, 1)),
        ((5, 6), 90, (1, 1), (1, 1)),
        ((5, 6), 7, (8, 4), (7, 1)),  # 7 columns: only 7 cores or fewer can all get work
    ],
)
def test_fit_core_grid(cores, num_tiles_n, max_grid, expected):
    assert fit_core_grid(cores, num_tiles_n, max_grid) == expected


@pytest.mark.parametrize("max_grid", [(8, 4), (2, 1), (3, 3)])
@pytest.mark.parametrize("num_tiles_n", [1, 7, 90, 96, 180])
@pytest.mark.parametrize("cores", [(5, 6), (12, 10)])
def test_fit_core_grid_result_is_usable(cores, num_tiles_n, max_grid):
    """For a requested grid that does not fit, the result fits in max_grid, has at most as many cores as requested,
    and gives every core output columns: each core takes ceil(num_tiles_n / num_cores) columns in order, so the
    last core must still get at least one."""
    grid = fit_core_grid(cores, num_tiles_n, max_grid)
    num_cores = grid[0] * grid[1]
    assert grid[0] <= max_grid[0] and grid[1] <= max_grid[1]
    assert num_cores <= cores[0] * cores[1]
    per_core_n = math.ceil(num_tiles_n / num_cores)
    assert num_tiles_n - (num_cores - 1) * per_core_n >= 1

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.tt_transformers.tt.distributed_norm import galaxy_distributed_norm_core_grid


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        (3072, (3, 8)),
        (4096, (4, 8)),
        (5120, (5, 2)),
        (8192, (4, 8)),
    ],
)
def test_galaxy_distributed_norm_core_grid(dim, expected):
    core_grid = galaxy_distributed_norm_core_grid(dim)
    num_cores = core_grid[0] * core_grid[1]

    assert core_grid == expected
    assert (dim // 4) % (num_cores * 32) == 0


def test_galaxy_distributed_norm_core_grid_rejects_unaligned_width(expect_error):
    with expect_error(ValueError, "is not tile-shardable"):
        galaxy_distributed_norm_core_grid(4608)

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


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        # dim // 4 not divisible by num_cores * 32: keep the legacy grid rather than
        # raising, so prefill-only Galaxy runs of these models still construct.
        (4608, (4, 8)),  # 1152 % 1024 != 0
        (2560, (2, 8)),  # gemma-3-4b:  640 %  512 != 0
        (2304, (2, 8)),  # gemma-2-2b:  576 %  512 != 0
        (3584, (3, 8)),  # gemma-2-9b:  896 %  768 != 0
        (5376, (4, 8)),  # gemma-3-27b: 1344 % 1024 != 0
    ],
)
def test_galaxy_distributed_norm_core_grid_falls_back_on_unaligned_width(dim, expected):
    assert galaxy_distributed_norm_core_grid(dim) == expected


@pytest.mark.parametrize("dim", [2048, 3072, 4096, 8192])
def test_galaxy_distributed_norm_core_grid_preserves_legacy_grid(dim):
    """Dims the pre-existing expression handled must be unchanged."""
    legacy = (min(4, dim // 4 // 32 // 8), 8)
    assert galaxy_distributed_norm_core_grid(dim) == legacy

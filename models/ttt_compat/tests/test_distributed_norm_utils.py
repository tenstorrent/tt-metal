# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.ttt_compat.tt.distributed_norm import galaxy_distributed_norm_core_grid


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
        # dim // 4 not divisible by num_cores * 32 on the legacy grid: use the largest grid (up to 4x8)
        # that tile-aligns the per-device width instead of a misaligned legacy grid, so the sharded decode
        # norm config is usable (Qwen2.5-VL-7B on Galaxy, dim 3584, needs this). Still no raising, so
        # prefill-only Galaxy runs of these models construct.
        (4608, (4, 3)),  # 1152 % 1024 != 0 -> 12 cores x 96
        (2560, (4, 5)),  # gemma-3-4b:  640 %  512 != 0 -> 20 cores x 32
        (2304, (3, 6)),  # gemma-2-2b:  576 %  512 != 0 -> 18 cores x 32
        (3584, (4, 7)),  # gemma-2-9b / Qwen2.5-VL-7B: 896 % 768 != 0 -> 28 cores x 32
        (5376, (3, 7)),  # gemma-3-27b: 1344 % 1024 != 0 -> 21 cores x 64
    ],
)
def test_galaxy_distributed_norm_core_grid_falls_back_on_unaligned_width(dim, expected):
    core_grid = galaxy_distributed_norm_core_grid(dim)
    assert core_grid == expected
    assert (dim // 4) % (core_grid[0] * core_grid[1] * 32) == 0


@pytest.mark.parametrize("dim", [2048, 3072, 4096, 8192])
def test_galaxy_distributed_norm_core_grid_preserves_legacy_grid(dim):
    """Dims the pre-existing expression handled must be unchanged."""
    legacy = (min(4, dim // 4 // 32 // 8), 8)
    assert galaxy_distributed_norm_core_grid(dim) == legacy

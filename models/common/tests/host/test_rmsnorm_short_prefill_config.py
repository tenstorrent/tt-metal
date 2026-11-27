# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of RMSNorm's short-prefill block-shard config against the device grid.

The shard grid is cols x rows cores with rows = seq / 32; it must fit the device's compute grid, or allocating
the sharded input fails (a 2x1 grid has no core (0, 1)). No device is opened: the input only needs a shape and
a device that reports its grid.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.common.rmsnorm import RMSNorm


def _fake_input(seq, dim, grid_x, grid_y):
    device = SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(grid_x, grid_y))
    return SimpleNamespace(shape=(1, 1, seq, dim), device=lambda: device)


def _shard_grid_end(mem_config):
    end = mem_config.shard_spec.grid.bounding_box().end
    return end.x, end.y


@pytest.mark.parametrize(
    "seq, dim, grid, expected_end",
    [
        (128, 2048, (8, 4), (7, 3)),  # 4 rows fit exactly
        (128, 2048, (13, 10), (7, 3)),
        (32, 2048, (2, 1), (1, 0)),  # 1 row fits a 1-row grid
        (128, 2048, (2, 1), None),  # 4 rows on a 1-row grid
        (96, 2048, (8, 2), None),  # 3 rows on a 2-row grid
    ],
)
def test_short_prefill_block_config_fits_grid(seq, dim, grid, expected_end):
    norm = object.__new__(RMSNorm)  # the config builder needs no weights or device
    config = norm._build_short_prefill_block_config(_fake_input(seq, dim, *grid))
    if expected_end is None:
        assert config is None
        return
    mem_config, _ = config
    assert _shard_grid_end(mem_config) == expected_end

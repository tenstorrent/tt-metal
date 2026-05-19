# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim


@pytest.mark.parametrize(
    "hidden_size, token_parallel",
    [
        (1024, 4),
        (4096, 4),
        (2880, 4),
    ],
)
def test_get_moe_combine_cores_count(device, hidden_size, token_parallel):
    data_parallel = auto_output_width_shard_dim(hidden_size)
    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, data_parallel, hidden_size)
    assert len(cores) == token_parallel * data_parallel


def test_get_moe_combine_cores_within_worker_grid(device):
    hidden_size = 4096
    token_parallel = 4
    data_parallel = auto_output_width_shard_dim(hidden_size)
    grid = device.compute_with_storage_grid_size()

    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, data_parallel, hidden_size)
    assert len(cores) == token_parallel * data_parallel

    for core in cores:
        assert 0 <= core.x < grid.x
        assert 0 <= core.y < grid.y

    assert len({(c.x, c.y) for c in cores}) == len(cores)


def test_get_moe_combine_cores_width_shard_auto_helper(device):
    hidden_size = 4096
    token_parallel = 4
    expected_width = auto_output_width_shard_dim(hidden_size)

    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, expected_width, hidden_size)
    assert len(cores) == token_parallel * expected_width


def test_get_moe_combine_cores_not_fixed_legacy_pool(device):
    """Combine cores should not all lie in the legacy hardcoded pool (5,0)-(6,7)."""
    hidden_size = 4096
    token_parallel = 4
    data_parallel = auto_output_width_shard_dim(hidden_size)

    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, data_parallel, hidden_size)

    legacy_pool = {(x, y) for x in range(5, 7) for y in range(0, 8)}
    legacy_hits = sum(1 for c in cores if (c.x, c.y) in legacy_pool)

    # Dynamic placement may coincidentally use some legacy coords, but not the entire grid.
    assert legacy_hits < len(cores)


def test_get_moe_tilize_drain_core_on_wh_grid(device):
    hidden_size = 4096
    token_parallel = 4
    data_parallel = auto_output_width_shard_dim(hidden_size)

    drain_core = ttnn.experimental.get_moe_tilize_drain_core(device, token_parallel, data_parallel, hidden_size)
    grid = device.compute_with_storage_grid_size()

    assert 0 <= drain_core.x < grid.x
    assert 0 <= drain_core.y < grid.y
    # Legacy WH drain was (6, 9) on the 8×10 worker grid.
    if grid.x == 8 and grid.y == 10:
        assert drain_core.x == 6
        assert drain_core.y == 9

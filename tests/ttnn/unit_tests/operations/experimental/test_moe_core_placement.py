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


def test_get_moe_combine_cores_disjoint_from_tilize(device):
    """Combine cores must be spatially disjoint from the tilize drain core."""
    hidden_size = 4096
    token_parallel = 4
    data_parallel = auto_output_width_shard_dim(hidden_size)

    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, data_parallel, hidden_size)
    drain = ttnn.experimental.get_moe_tilize_drain_core(device, token_parallel, data_parallel, hidden_size)

    combine_set = {(c.x, c.y) for c in cores}
    assert (drain.x, drain.y) not in combine_set, "tilize drain core must not overlap combine cores"


def test_get_moe_tilize_drain_core_structural(device):
    hidden_size = 4096
    token_parallel = 4
    data_parallel = auto_output_width_shard_dim(hidden_size)

    drain_core = ttnn.experimental.get_moe_tilize_drain_core(device, token_parallel, data_parallel, hidden_size)
    grid = device.compute_with_storage_grid_size()

    assert 0 <= drain_core.x < grid.x
    assert 0 <= drain_core.y < grid.y

    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, data_parallel, hidden_size)
    combine_set = {(c.x, c.y) for c in cores}
    assert (drain_core.x, drain_core.y) not in combine_set, "drain core must not overlap combine cores"

    assert drain_core.y >= grid.y - 2, "drain core should be in the top two worker rows (tilize region)"


def test_get_moe_combine_cores_avoids_mux_cores(device):
    """Combine and tilize cores must not overlap a caller-specified mux region."""
    hidden_size = 4096
    token_parallel = 4
    data_parallel = auto_output_width_shard_dim(hidden_size)
    mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])

    cores = ttnn.experimental.get_moe_combine_cores(
        device, token_parallel, data_parallel, hidden_size, mux_core_range_set=mux
    )
    combine_set = {(c.x, c.y) for c in cores}
    for y in range(1, 4):
        for x in range(1, 4):
            assert (x, y) not in combine_set, f"combine core ({x},{y}) overlaps mux region"

    drain = ttnn.experimental.get_moe_tilize_drain_core(
        device, token_parallel, data_parallel, hidden_size, mux_core_range_set=mux
    )
    assert (drain.x, drain.y) not in {
        (x, y) for y in range(1, 4) for x in range(1, 4)
    }, "tilize drain core overlaps mux region"

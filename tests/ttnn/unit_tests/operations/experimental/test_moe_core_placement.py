# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from models.common.utility_functions import is_wormhole_b0
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size


def _auto_output_width_shard_dim_for_device(device, hidden_size):
    return auto_output_width_shard_dim(hidden_size, matmul_ring_size=effective_matmul_ring_size(device))


@pytest.mark.parametrize(
    "hidden_size, token_parallel",
    [
        (1024, 4),
        (4096, 4),
        (2880, 4),
    ],
)
def test_get_moe_combine_cores_count(device, hidden_size, token_parallel):
    data_parallel = _auto_output_width_shard_dim_for_device(device, hidden_size)
    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, data_parallel, hidden_size)
    assert len(cores) == token_parallel * data_parallel


def test_get_moe_combine_cores_within_worker_grid(device):
    hidden_size = 4096
    token_parallel = 4
    data_parallel = _auto_output_width_shard_dim_for_device(device, hidden_size)
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
    expected_width = _auto_output_width_shard_dim_for_device(device, hidden_size)

    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, expected_width, hidden_size)
    assert len(cores) == token_parallel * expected_width


def test_get_moe_combine_cores_disjoint_from_tilize(device):
    """Combine cores must be spatially disjoint from the tilize drain core."""
    hidden_size = 4096
    token_parallel = 4
    data_parallel = _auto_output_width_shard_dim_for_device(device, hidden_size)

    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, data_parallel, hidden_size)
    drain = ttnn.experimental.get_moe_tilize_drain_core(device, token_parallel, data_parallel, hidden_size)

    combine_set = {(c.x, c.y) for c in cores}
    assert (drain.x, drain.y) not in combine_set, "tilize drain core must not overlap combine cores"


def test_get_moe_tilize_drain_core_structural(device):
    hidden_size = 4096
    token_parallel = 4
    data_parallel = _auto_output_width_shard_dim_for_device(device, hidden_size)

    drain_core = ttnn.experimental.get_moe_tilize_drain_core(device, token_parallel, data_parallel, hidden_size)
    grid = device.compute_with_storage_grid_size()

    assert 0 <= drain_core.x < grid.x
    assert 0 <= drain_core.y < grid.y

    cores = ttnn.experimental.get_moe_combine_cores(device, token_parallel, data_parallel, hidden_size)
    combine_set = {(c.x, c.y) for c in cores}
    assert (drain_core.x, drain_core.y) not in combine_set, "drain core must not overlap combine cores"

    assert drain_core.y >= grid.y - 2, "drain core should be in the top two worker rows (tilize region)"


@pytest.mark.skipif(is_wormhole_b0(), reason="mux placement geometry is Blackhole-only")
def test_get_moe_combine_cores_avoids_mux_cores(device):
    """Combine and tilize cores must not overlap a caller-specified mux region."""
    hidden_size = 4096
    token_parallel = 4
    data_parallel = _auto_output_width_shard_dim_for_device(device, hidden_size)
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


@pytest.mark.skipif(is_wormhole_b0(), reason="mux placement geometry is Blackhole-only")
def test_moe_worker_mcast_bbox_consistent_with_mux_placement(device):
    """get_moe_worker_mcast_bounding_box must agree with the worker placement
    the op actually uses when mux cores are present.

    The helper accepts mux_core_range_set and forwards it to
    select_moe_compute_cores, so the bbox reflects the real worker layout
    regardless of mux avoidance.
    """
    hidden_size = 4096
    token_parallel = 4
    data_parallel = _auto_output_width_shard_dim_for_device(device, hidden_size)
    grid = device.compute_with_storage_grid_size()

    # Block the eastern 2 columns — the combine strip's preferred location
    # and the tilize 2x2 block's preferred location.
    mux_x_start = grid.x - 2
    mux = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(mux_x_start, 0),
                ttnn.CoreCoord(grid.x - 1, grid.y - 1),
            )
        ]
    )

    combine_with_mux = ttnn.experimental.get_moe_combine_cores(
        device, token_parallel, data_parallel, hidden_size, mux_core_range_set=mux
    )
    drain_with_mux = ttnn.experimental.get_moe_tilize_drain_core(
        device, token_parallel, data_parallel, hidden_size, mux_core_range_set=mux
    )

    bbox = ttnn.experimental.get_moe_worker_mcast_bounding_box(
        device, token_parallel, data_parallel, hidden_size, mux_core_range_set=mux
    )

    def in_bbox(cx, cy):
        return bbox.start.x <= cx <= bbox.end.x and bbox.start.y <= cy <= bbox.end.y

    def combine_bbox_area(cores):
        min_x = min(c.x for c in cores)
        max_x = max(c.x for c in cores)
        min_y = min(c.y for c in cores)
        max_y = max(c.y for c in cores)
        return (max_x - min_x + 1) * (max_y - min_y + 1)

    for c in combine_with_mux:
        assert in_bbox(c.x, c.y), (
            f"combine core ({c.x},{c.y}) falls outside mcast bbox "
            f"[({bbox.start.x},{bbox.start.y})-({bbox.end.x},{bbox.end.y})]"
        )
    assert combine_bbox_area(combine_with_mux) == len(combine_with_mux), "combine placement bbox must be dense"

    assert in_bbox(drain_with_mux.x, drain_with_mux.y), (
        f"tilize drain ({drain_with_mux.x},{drain_with_mux.y}) falls outside mcast bbox "
        f"[({bbox.start.x},{bbox.start.y})-({bbox.end.x},{bbox.end.y})]"
    )

    # The bbox must NOT extend into the mux region (no mux core should be a
    # worker core, so the bbox should have shrunk away from the mux columns).
    combine_no_mux = ttnn.experimental.get_moe_combine_cores(device, token_parallel, data_parallel, hidden_size)
    bbox_no_mux = ttnn.experimental.get_moe_worker_mcast_bounding_box(
        device, token_parallel, data_parallel, hidden_size
    )
    with_mux_set = {(c.x, c.y) for c in combine_with_mux}
    no_mux_set = {(c.x, c.y) for c in combine_no_mux}
    assert combine_bbox_area(combine_no_mux) == len(combine_no_mux), "combine placement bbox must be dense"

    # Precondition: mux actually forced a different placement.
    assert no_mux_set != with_mux_set, (
        "Test setup failure: mux region did not change the combine placement. "
        "Choose a mux region that overlaps the preferred combine strip."
    )

    # The mux-aware bbox must differ from the no-mux bbox.
    mux_bbox_tuple = (bbox.start.x, bbox.start.y, bbox.end.x, bbox.end.y)
    no_mux_bbox_tuple = (bbox_no_mux.start.x, bbox_no_mux.start.y, bbox_no_mux.end.x, bbox_no_mux.end.y)
    assert mux_bbox_tuple != no_mux_bbox_tuple, (
        f"mux shifted combine cores but the bbox did not change — " f"mux_core_range_set is not being forwarded"
    )

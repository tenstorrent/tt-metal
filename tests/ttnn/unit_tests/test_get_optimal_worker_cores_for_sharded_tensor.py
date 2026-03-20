# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
import ttnn


def corerange_to_cores_python(core_range_set, max_cores=None, row_wise=False):
    """
    Pure-Python reimplementation of corerange_to_cores for computing expected core order.
    Iterates each CoreRange in the set, enumerating coordinates in row-major or column-major order.
    """
    cores = []
    for cr in core_range_set.ranges():
        x_start, y_start = cr.start.x, cr.start.y
        x_end, y_end = cr.end.x, cr.end.y
        if row_wise:
            for y in range(y_start, y_end + 1):
                for x in range(x_start, x_end + 1):
                    cores.append(ttnn.CoreCoord(x, y))
        else:
            for x in range(x_start, x_end + 1):
                for y in range(y_start, y_end + 1):
                    cores.append(ttnn.CoreCoord(x, y))
    if max_cores is not None:
        cores = cores[:max_cores]
    return cores


def compute_num_shards_nd(tensor_shape, shard_shape):
    """Compute number of shards for an ND-sharded tensor."""
    num_shards = 1
    for t, s in zip(tensor_shape, shard_shape):
        num_shards *= math.ceil(t / s)
    return num_shards


def compute_num_shards_legacy(tensor_shape, shard_shape, memory_layout):
    """Compute number of shards for a legacy 2D sharded tensor."""
    total_height = 1
    for dim in tensor_shape[:-1]:
        total_height *= dim
    width = tensor_shape[-1]
    shard_h, shard_w = shard_shape

    if memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        return math.ceil(total_height / shard_h)
    elif memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        return math.ceil(width / shard_w)
    elif memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        return math.ceil(total_height / shard_h) * math.ceil(width / shard_w)
    else:
        raise ValueError(f"Unsupported memory layout: {memory_layout}")


def compute_expected_cores_round_robin(core_range_set, num_shards, row_major):
    """
    For HEIGHT_SHARDED, WIDTH_SHARDED, and ND_SHARDED with ROUND_ROBIN_1D:
    cores are traversed in row-major or column-major order and shards are distributed round-robin.
    Only cores that actually hold at least one shard appear in the result.
    """
    all_cores = corerange_to_cores_python(core_range_set, row_wise=row_major)
    num_cores = len(all_cores)
    num_cores_with_data = min(num_cores, num_shards)
    return all_cores[:num_cores_with_data]


def compute_expected_cores_grid_2d(core_range_set, num_shards_height, num_shards_width, row_major):
    """
    For BLOCK_SHARDED with GRID_2D distribution:
    The grid is trimmed to num_shards_width x num_shards_height, then traversed in shard orientation order.
    If orientation is COL_MAJOR, width/height are swapped for grid trimming.
    """
    assert len(core_range_set.ranges()) == 1, "GRID_2D requires a single contiguous CoreRange"
    cr = core_range_set.ranges()[0]
    x_start, y_start = cr.start.x, cr.start.y

    if row_major:
        trim_x = num_shards_width
        trim_y = num_shards_height
    else:
        trim_x = num_shards_height
        trim_y = num_shards_width

    trimmed = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(x_start, y_start),
                ttnn.CoreCoord(x_start + trim_x - 1, y_start + trim_y - 1),
            )
        }
    )
    return corerange_to_cores_python(trimmed, row_wise=row_major)


def assert_cores_match(actual_cores, expected_cores):
    assert len(actual_cores) == len(
        expected_cores
    ), f"Length mismatch: got {len(actual_cores)}, expected {len(expected_cores)}"
    for i, (a, e) in enumerate(zip(actual_cores, expected_cores)):
        assert a == e, f"Core mismatch at index {i}: got ({a.x}, {a.y}), expected ({e.x}, {e.y})"


# ============================================================================
#  ND Sharded Tests (ROUND_ROBIN_1D)
# ============================================================================


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid, orientation, description",
    [
        # grid == num_shards (4 shards on 4 cores)
        (
            [2, 1, 128, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "nd_grid_eq_shards_row_major",
        ),
        (
            [2, 1, 128, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "nd_grid_eq_shards_col_major",
        ),
        # grid > num_shards (4 shards on 8 cores)
        (
            [2, 1, 128, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "nd_grid_gt_shards_row_major",
        ),
        (
            [2, 1, 128, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "nd_grid_gt_shards_col_major",
        ),
        # grid < num_shards (8 shards on 4 cores => 2 shards per core)
        (
            [2, 1, 256, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "nd_grid_lt_shards_row_major",
        ),
        (
            [2, 1, 256, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "nd_grid_lt_shards_col_major",
        ),
        # Disjoint CoreRangeSet, grid > num_shards
        (
            [2, 1, 64, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "nd_disjoint_grid_gt_shards_row_major",
        ),
        # Disjoint CoreRangeSet, grid == num_shards
        (
            [2, 1, 128, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "nd_disjoint_grid_eq_shards_row_major",
        ),
        # Disjoint CoreRangeSet, grid < num_shards
        (
            [2, 1, 256, 64],
            [2, 1, 32, 64],
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "nd_disjoint_grid_lt_shards_row_major",
        ),
        # 2D grid, grid > num_shards, row major
        (
            [2, 1, 64, 64],
            [2, 1, 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "nd_2d_grid_gt_shards_row_major",
        ),
        # 2D grid, grid > num_shards, col major
        (
            [2, 1, 64, 64],
            [2, 1, 32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "nd_2d_grid_gt_shards_col_major",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_nd_sharded_round_robin(device, tensor_shape, shard_shape, grid, orientation, description):
    num_shards = compute_num_shards_nd(tensor_shape, shard_shape)
    row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
    expected_cores = compute_expected_cores_round_robin(grid, num_shards, row_major)

    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=shard_shape,
        grid=grid,
        orientation=orientation,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd_shard_spec)

    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, device, memory_config=mem_config)

    actual_cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(tt_tensor)
    assert_cores_match(actual_cores, expected_cores)


# ============================================================================
#  Legacy HEIGHT_SHARDED Tests
# ============================================================================


@pytest.mark.parametrize(
    "tensor_shape, shard_h, grid, orientation, description",
    [
        # grid == num_shards
        (
            [1, 1, 128, 64],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "height_grid_eq_shards_row_major",
        ),
        (
            [1, 1, 128, 64],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "height_grid_eq_shards_col_major",
        ),
        # grid > num_shards
        (
            [1, 1, 128, 64],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "height_grid_gt_shards_row_major",
        ),
        # Disjoint grid == num_shards
        (
            [1, 1, 128, 64],
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "height_disjoint_grid_eq_shards",
        ),
        # Disjoint grid > num_shards
        (
            [1, 1, 64, 64],
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "height_disjoint_grid_gt_shards",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_legacy_height_sharded(device, tensor_shape, shard_h, grid, orientation, description):
    shard_w = tensor_shape[-1]
    shard_shape = [shard_h, shard_w]
    num_shards = compute_num_shards_legacy(tensor_shape, shard_shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
    expected_cores = compute_expected_cores_round_robin(grid, num_shards, row_major)

    shard_spec = ttnn.ShardSpec(grid, shard_shape, orientation)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, device, memory_config=mem_config)

    actual_cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(tt_tensor)
    assert_cores_match(actual_cores, expected_cores)


# ============================================================================
#  Legacy WIDTH_SHARDED Tests
# ============================================================================


@pytest.mark.parametrize(
    "tensor_shape, shard_w, grid, orientation, description",
    [
        # grid == num_shards
        (
            [1, 1, 32, 128],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "width_grid_eq_shards_row_major",
        ),
        (
            [1, 1, 32, 128],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "width_grid_eq_shards_col_major",
        ),
        # grid > num_shards
        (
            [1, 1, 32, 128],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "width_grid_gt_shards_row_major",
        ),
        # Disjoint grid == num_shards
        (
            [1, 1, 32, 128],
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "width_disjoint_grid_eq_shards",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_legacy_width_sharded(device, tensor_shape, shard_w, grid, orientation, description):
    total_height = 1
    for dim in tensor_shape[:-1]:
        total_height *= dim
    shard_shape = [total_height, shard_w]
    num_shards = compute_num_shards_legacy(tensor_shape, shard_shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED)
    row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
    expected_cores = compute_expected_cores_round_robin(grid, num_shards, row_major)

    shard_spec = ttnn.ShardSpec(grid, shard_shape, orientation)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, device, memory_config=mem_config)

    actual_cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(tt_tensor)
    assert_cores_match(actual_cores, expected_cores)


# ============================================================================
#  Legacy BLOCK_SHARDED Tests (GRID_2D distribution)
# ============================================================================


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid, orientation, description",
    [
        # grid == num_shards (2x2 grid, 2x2 shards)
        (
            [1, 1, 64, 64],
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "block_grid_eq_shards_row_major",
        ),
        (
            [1, 1, 64, 64],
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "block_grid_eq_shards_col_major",
        ),
        # grid > num_shards (4x4 grid, 2x2 shards)
        (
            [1, 1, 64, 64],
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "block_grid_gt_shards_row_major",
        ),
        (
            [1, 1, 64, 64],
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "block_grid_gt_shards_col_major",
        ),
        # Rectangular: 4 height shards x 2 width shards on 2x4 grid (ROW_MAJOR)
        (
            [1, 1, 128, 64],
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "block_rect_grid_eq_shards_row_major",
        ),
        # Rectangular: 4 height shards x 2 width shards on 4x2 grid (COL_MAJOR, axes swapped)
        (
            [1, 1, 128, 64],
            [32, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "block_rect_grid_eq_shards_col_major",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_legacy_block_sharded(device, tensor_shape, shard_shape, grid, orientation, description):
    total_height = 1
    for dim in tensor_shape[:-1]:
        total_height *= dim
    width = tensor_shape[-1]
    shard_h, shard_w = shard_shape

    num_shards_h = math.ceil(total_height / shard_h)
    num_shards_w = math.ceil(width / shard_w)
    row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
    expected_cores = compute_expected_cores_grid_2d(grid, num_shards_h, num_shards_w, row_major)

    shard_spec = ttnn.ShardSpec(grid, shard_shape, orientation)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)

    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, device, memory_config=mem_config)

    actual_cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(tt_tensor)
    assert_cores_match(actual_cores, expected_cores)


# ============================================================================
#  DRAM Sharded Helper
# ============================================================================


def compute_expected_dram_worker_cores(device, dram_grid, num_shards, row_major, noc=ttnn.NOC.NOC_0):
    """
    For DRAM-sharded tensors the API returns optimal Tensix worker cores, not DRAM cores.
    Steps:
      1. Enumerate DRAM logical cores from the grid in traversal order.
      2. Keep only the first min(num_cores, num_shards) that have data.
      3. Map each DRAM core -> channel -> optimal worker via the device mapping.
    """
    all_dram_cores = corerange_to_cores_python(dram_grid, row_wise=row_major)
    cores_with_data = all_dram_cores[: min(len(all_dram_cores), num_shards)]

    all_dram_workers = device.get_optimal_dram_bank_to_logical_worker_assignment(noc)

    expected_workers = []
    for dram_core in cores_with_data:
        channel = dram_core.x
        expected_workers.append(all_dram_workers[channel])
    return expected_workers


# ============================================================================
#  DRAM Legacy HEIGHT_SHARDED Tests
# ============================================================================


@pytest.mark.parametrize(
    "tensor_shape, shard_h, grid, orientation, description",
    [
        # All 12 DRAM banks, grid == num_shards (12 shards on 12 banks)
        (
            [1, 1, 384, 64],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_height_all_banks_eq_shards_row_major",
        ),
        (
            [1, 1, 384, 64],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_height_all_banks_eq_shards_col_major",
        ),
        # All 12 DRAM banks, num_shards < num_banks (6 shards on 12 banks)
        (
            [1, 1, 192, 64],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_height_all_banks_gt_shards_row_major",
        ),
        (
            [1, 1, 192, 64],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_height_all_banks_gt_shards_col_major",
        ),
        # Disjoint DRAM banks (banks 0-3 and 8-11), grid == num_shards
        (
            [1, 1, 256, 64],
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_height_disjoint_eq_shards_row_major",
        ),
        (
            [1, 1, 256, 64],
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_height_disjoint_eq_shards_col_major",
        ),
        # Disjoint DRAM banks, num_shards < num_banks (4 shards on 8 disjoint banks)
        (
            [1, 1, 128, 64],
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_height_disjoint_gt_shards_row_major",
        ),
        (
            [1, 1, 128, 64],
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_height_disjoint_gt_shards_col_major",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_dram_legacy_height_sharded(device, tensor_shape, shard_h, grid, orientation, description):
    num_device_dram_banks = device.dram_grid_size().x
    required_banks = grid.num_cores()
    if required_banks > num_device_dram_banks:
        pytest.skip(f"This architecture has fewer than {required_banks} DRAM banks ({num_device_dram_banks} available)")

    shard_w = tensor_shape[-1]
    shard_shape = [shard_h, shard_w]
    num_shards = compute_num_shards_legacy(tensor_shape, shard_shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
    row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
    expected_cores = compute_expected_dram_worker_cores(device, grid, num_shards, row_major)

    shard_spec = ttnn.ShardSpec(grid, shard_shape, orientation)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, device, memory_config=mem_config)

    actual_cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(tt_tensor)
    assert_cores_match(actual_cores, expected_cores)


# ============================================================================
#  DRAM Legacy WIDTH_SHARDED Tests
# ============================================================================


@pytest.mark.parametrize(
    "tensor_shape, shard_w, grid, orientation, description",
    [
        # All 12 DRAM banks, grid == num_shards (12 shards on 12 banks)
        (
            [1, 1, 32, 384],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_width_all_banks_eq_shards_row_major",
        ),
        (
            [1, 1, 32, 384],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_width_all_banks_eq_shards_col_major",
        ),
        # All 12 DRAM banks, num_shards < num_banks (6 shards on 12 banks)
        (
            [1, 1, 32, 192],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_width_all_banks_gt_shards_row_major",
        ),
        (
            [1, 1, 32, 192],
            32,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_width_all_banks_gt_shards_col_major",
        ),
        # Disjoint DRAM banks, grid == num_shards
        (
            [1, 1, 32, 256],
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_width_disjoint_eq_shards_row_major",
        ),
        (
            [1, 1, 32, 256],
            32,
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_width_disjoint_eq_shards_col_major",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_dram_legacy_width_sharded(device, tensor_shape, shard_w, grid, orientation, description):
    num_device_dram_banks = device.dram_grid_size().x
    required_banks = grid.num_cores()
    if required_banks > num_device_dram_banks:
        pytest.skip(f"This architecture has fewer than {required_banks} DRAM banks ({num_device_dram_banks} available)")

    total_height = 1
    for dim in tensor_shape[:-1]:
        total_height *= dim
    shard_shape = [total_height, shard_w]
    num_shards = compute_num_shards_legacy(tensor_shape, shard_shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED)
    row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
    expected_cores = compute_expected_dram_worker_cores(device, grid, num_shards, row_major)

    shard_spec = ttnn.ShardSpec(grid, shard_shape, orientation)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, device, memory_config=mem_config)

    actual_cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(tt_tensor)
    assert_cores_match(actual_cores, expected_cores)


# ============================================================================
#  DRAM ND Sharded Tests (ROUND_ROBIN_1D)
# ============================================================================


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid, orientation, description",
    [
        # All 12 DRAM banks, grid == num_shards (12 shards on 12 banks)
        (
            [1, 1, 384, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_nd_all_banks_eq_shards_row_major",
        ),
        (
            [1, 1, 384, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_nd_all_banks_eq_shards_col_major",
        ),
        # All 12 DRAM banks, num_shards < num_banks (6 shards on 12 banks)
        (
            [1, 1, 192, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_nd_all_banks_gt_shards_row_major",
        ),
        (
            [1, 1, 192, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_nd_all_banks_gt_shards_col_major",
        ),
        # All 12 DRAM banks, num_shards > num_banks (24 shards on 12 banks, 2 per bank)
        (
            [1, 1, 768, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_nd_all_banks_lt_shards_row_major",
        ),
        (
            [1, 1, 768, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))}),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_nd_all_banks_lt_shards_col_major",
        ),
        # Disjoint DRAM banks (banks 0-3 and 8-11), grid == num_shards
        (
            [1, 1, 256, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_nd_disjoint_eq_shards_row_major",
        ),
        (
            [1, 1, 256, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_nd_disjoint_eq_shards_col_major",
        ),
        # Disjoint DRAM banks, num_shards < num_banks (4 shards on 8 disjoint banks)
        (
            [1, 1, 128, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            "dram_nd_disjoint_gt_shards_row_major",
        ),
        (
            [1, 1, 128, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 0)),
                ]
            ),
            ttnn.ShardOrientation.COL_MAJOR,
            "dram_nd_disjoint_gt_shards_col_major",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_dram_nd_sharded_round_robin(device, tensor_shape, shard_shape, grid, orientation, description):
    num_device_dram_banks = device.dram_grid_size().x
    required_banks = grid.num_cores()
    if required_banks > num_device_dram_banks:
        pytest.skip(f"This architecture has fewer than {required_banks} DRAM banks ({num_device_dram_banks} available)")

    num_shards = compute_num_shards_nd(tensor_shape, shard_shape)
    row_major = orientation == ttnn.ShardOrientation.ROW_MAJOR
    expected_cores = compute_expected_dram_worker_cores(device, grid, num_shards, row_major)

    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=shard_shape,
        grid=grid,
        orientation=orientation,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    torch_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, device, memory_config=mem_config)

    actual_cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(tt_tensor)
    assert_cores_match(actual_cores, expected_cores)


def test_get_optimal_worker_cores_for_sharded_tensor_rejects_host_tensor():
    torch_tensor = torch.randn([1, 1, 32, 64], dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

    with pytest.raises(RuntimeError, match="Tensor must be on device and sharded to compute optimal worker cores"):
        ttnn.get_optimal_worker_cores_for_sharded_tensor(tt_tensor)


def test_get_optimal_worker_cores_for_sharded_tensor_rejects_interleaved_tensor(device):
    torch_tensor = torch.randn([1, 1, 32, 64], dtype=torch.bfloat16)
    tt_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_tensor = ttnn.to_device(tt_tensor, device)

    with pytest.raises(RuntimeError, match="Tensor must be on device and sharded to compute optimal worker cores"):
        ttnn.get_optimal_worker_cores_for_sharded_tensor(tt_tensor)

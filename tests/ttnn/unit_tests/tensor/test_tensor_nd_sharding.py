# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype, assert_with_pcc


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, layout",
    [
        # Row Major Layout
        ([3, 4, 5], [3, 4, 5], ttnn.ROW_MAJOR_LAYOUT),  # All data on a single core
        ([3, 4, 5], [3, 4, 1], ttnn.ROW_MAJOR_LAYOUT),  # Each core gets full batch and height dimension
        ([3, 4, 5], [3, 1, 5], ttnn.ROW_MAJOR_LAYOUT),  # Each core gets full batch and width dimension
        ([3, 4, 5], [1, 4, 5], ttnn.ROW_MAJOR_LAYOUT),  # Each core gets full height and width dimension
        ([3, 4, 5], [3, 1, 1], ttnn.ROW_MAJOR_LAYOUT),  # Each core gets full batch dimension
        ([3, 4, 5], [1, 4, 1], ttnn.ROW_MAJOR_LAYOUT),  # Each core gets full height dimension
        ([3, 4, 5], [1, 1, 5], ttnn.ROW_MAJOR_LAYOUT),  # Each core gets full width dimension
        ([3, 4, 5], [1, 1, 1], ttnn.ROW_MAJOR_LAYOUT),  # Data is distributed equally across all cores
        # Tile Layout
        ([3, 128, 160], [3, 128, 160], ttnn.TILE_LAYOUT),  # All data on a single core
        ([3, 128, 160], [3, 128, 32], ttnn.TILE_LAYOUT),  # Each core gets full batch and height dimension
        ([3, 128, 160], [3, 32, 160], ttnn.TILE_LAYOUT),  # Each core gets full batch and width dimension
        ([3, 128, 160], [1, 128, 160], ttnn.TILE_LAYOUT),  # Each core gets full height and width dimension
        ([3, 128, 160], [3, 32, 32], ttnn.TILE_LAYOUT),  # Each core gets full batch dimension
        ([3, 128, 160], [1, 128, 32], ttnn.TILE_LAYOUT),  # Each core gets full height dimension
        ([3, 128, 160], [1, 32, 160], ttnn.TILE_LAYOUT),  # Each core gets full width dimension
        ([3, 128, 160], [1, 32, 32], ttnn.TILE_LAYOUT),  # Data is distributed equally across all cores
        # Uneven shards
        ([30, 40, 55], [30, 40, 10], ttnn.ROW_MAJOR_LAYOUT),
        ([30, 45, 50], [30, 10, 50], ttnn.ROW_MAJOR_LAYOUT),
        ([35, 40, 50], [10, 40, 50], ttnn.ROW_MAJOR_LAYOUT),
        ([30, 45, 50], [30, 10, 50], ttnn.ROW_MAJOR_LAYOUT),
        ([35, 40, 50], [10, 40, 50], ttnn.ROW_MAJOR_LAYOUT),
        ([35, 45, 50], [10, 10, 50], ttnn.ROW_MAJOR_LAYOUT),
        ([35, 45, 55], [10, 10, 10], ttnn.ROW_MAJOR_LAYOUT),
        ([3, 128, 165], [3, 128, 32], ttnn.TILE_LAYOUT),
        ([3, 130, 160], [3, 32, 160], ttnn.TILE_LAYOUT),
        ([5, 128, 160], [2, 128, 160], ttnn.TILE_LAYOUT),
        ([3, 130, 165], [3, 32, 32], ttnn.TILE_LAYOUT),
        ([5, 128, 165], [2, 128, 32], ttnn.TILE_LAYOUT),
        ([5, 130, 160], [2, 32, 160], ttnn.TILE_LAYOUT),
        ([5, 130, 165], [2, 32, 32], ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM])
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.uint8,
        ttnn.uint16,
        ttnn.uint32,
        ttnn.int32,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
    ],
)
def test_tensor_nd_sharding_loopback(tensor_shape, shard_shape, layout, buffer_type, tt_dtype, device):
    torch.manual_seed(0)

    if tt_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("{} is only valid for ttnn.TILE_LAYOUT!".format(tt_dtype))

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.uint8, torch.int16, torch.int32}:
        py_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, tensor_shape, dtype=dtype)
    else:
        py_tensor = torch.rand(tensor_shape, dtype=dtype)

    if buffer_type == ttnn.BufferType.L1:
        grid_size = device.compute_with_storage_grid_size()
    else:
        grid_size = device.dram_grid_size()
    core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))
    grid = ttnn.CoreRangeSet([core_range])

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid)
    memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)
    assert memory_config.is_sharded()
    assert memory_config.nd_shard_spec == nd_shard_spec

    tt_tensor = ttnn.from_torch(py_tensor, dtype=tt_dtype, device=device, layout=layout, memory_config=memory_config)
    py_tensor_after_round_trip = ttnn.to_torch(tt_tensor)

    if tt_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        assert_with_pcc(py_tensor, py_tensor_after_round_trip, 0.95)
    else:
        assert torch.allclose(py_tensor, py_tensor_after_round_trip)


core_ranges = ttnn.num_cores_to_corerangeset(16, [4, 4], True)


@pytest.mark.parametrize(
    "shape, layout, dims, expected_shard_shape",
    [
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [-1], [5, 3, 4 * 32, 32]),
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [-2], [5, 3, 32, 4 * 32]),
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [-1, -2], [5, 3, 32, 32]),
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [0], [1, 3, 4 * 32, 4 * 32]),
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [0, 1], [1, 1, 4 * 32, 4 * 32]),
        ((1, 1, 1, 1), ttnn.ROW_MAJOR_LAYOUT, [0], [1, 1, 1, 4]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [-1], [5, 3, 8, 4]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [-2], [5, 3, 1, 8]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [-1, -2], [5, 3, 1, 4]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [0], [1, 3, 8, 8]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [0, 1], [1, 1, 8, 8]),
    ],
)
def test_sharded_across_dims(shape, layout, dims, expected_shard_shape):
    spec = ttnn.TensorSpec(shape, ttnn.float32, layout, buffer_type=ttnn.BufferType.L1).sharded_across_dims(
        dims, core_ranges
    )
    assert spec.memory_config.nd_shard_spec.shard_shape == expected_shard_shape


@pytest.mark.parametrize(
    "shape, layout, dims, expected_shard_shape",
    [
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [-1], [1, 1, 32, 4 * 32]),
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [-2], [1, 1, 4 * 32, 32]),
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [-1, -2], [1, 1, 4 * 32, 4 * 32]),
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [0], [5, 1, 32, 32]),
        ((5, 3, 4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [0, 1], [5, 3, 32, 32]),
        ((1, 1, 1, 1), ttnn.ROW_MAJOR_LAYOUT, [0], [1, 1, 1, 4]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [-1], [1, 1, 1, 8]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [-2], [1, 1, 8, 4]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [-1, -2], [1, 1, 8, 8]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [0], [5, 1, 1, 4]),
        ((5, 3, 8, 8), ttnn.ROW_MAJOR_LAYOUT, [0, 1], [5, 3, 1, 4]),
    ],
)
def test_sharded_across_dims_except(shape, layout, dims, expected_shard_shape):
    spec = ttnn.TensorSpec(shape, ttnn.float32, layout, buffer_type=ttnn.BufferType.L1).sharded_across_dims_except(
        dims, core_ranges
    )
    assert spec.memory_config.nd_shard_spec.shard_shape == expected_shard_shape


@pytest.mark.parametrize(
    "shape, layout, expected_shard_shape",
    [
        ((4 * 32, 4 * 32), ttnn.TILE_LAYOUT, [32, 32]),
        ((8 * 32, 4 * 32), ttnn.TILE_LAYOUT, [2 * 32, 32]),
        ((4 * 32, 8 * 32), ttnn.TILE_LAYOUT, [32, 2 * 32]),
        ((8 * 32, 8 * 32), ttnn.TILE_LAYOUT, [2 * 32, 2 * 32]),
        ((500, 500), ttnn.TILE_LAYOUT, [4 * 32, 4 * 32]),
        ((4, 5, 30, 500), ttnn.TILE_LAYOUT, [5 * 32, 4 * 32]),
        ((1, 1), ttnn.ROW_MAJOR_LAYOUT, [1, 4]),
        ((800, 1200), ttnn.ROW_MAJOR_LAYOUT, [200, 300]),
        ((333, 555), ttnn.ROW_MAJOR_LAYOUT, [84, 140]),
        ((2, 3, 333, 555), ttnn.ROW_MAJOR_LAYOUT, [500, 140]),
    ],
)
def test_auto_block_sharded(shape, layout, expected_shard_shape):
    spec = ttnn.TensorSpec(shape, ttnn.float32, layout, buffer_type=ttnn.BufferType.L1).block_sharded(core_ranges)
    assert spec.memory_config.nd_shard_spec.shard_shape == expected_shard_shape


@pytest.mark.parametrize(
    "shape, layout, expected_shard_shape",
    [
        ((32 * 32, 128), ttnn.TILE_LAYOUT, [2 * 32, 128]),
        ((37 * 32, 128), ttnn.TILE_LAYOUT, [3 * 32, 128]),
        ((5, 37 * 32, 128), ttnn.TILE_LAYOUT, [12 * 32, 128]),
        ((160, 17), ttnn.ROW_MAJOR_LAYOUT, [10, 17]),
        ((167, 17), ttnn.ROW_MAJOR_LAYOUT, [11, 17]),
        ((5, 167, 17), ttnn.ROW_MAJOR_LAYOUT, [53, 17]),
    ],
)
def test_height_sharded(shape, layout, expected_shard_shape):
    spec = ttnn.TensorSpec(shape, ttnn.float32, layout, buffer_type=ttnn.BufferType.L1).height_sharded(core_ranges)
    assert spec.memory_config.nd_shard_spec.shard_shape == expected_shard_shape


@pytest.mark.parametrize(
    "shape, layout, expected_shard_shape",
    [
        ((128, 32 * 32), ttnn.TILE_LAYOUT, [128, 2 * 32]),
        ((128, 37 * 32), ttnn.TILE_LAYOUT, [128, 3 * 32]),
        ((5, 128, 37 * 32), ttnn.TILE_LAYOUT, [5 * 128, 3 * 32]),
        ((17, 160), ttnn.ROW_MAJOR_LAYOUT, [17, 10]),
        ((17, 167), ttnn.ROW_MAJOR_LAYOUT, [17, 11]),
        ((5, 17, 167), ttnn.ROW_MAJOR_LAYOUT, [5 * 17, 11]),
    ],
)
def test_width_sharded(shape, layout, expected_shard_shape):
    spec = ttnn.TensorSpec(shape, ttnn.float32, layout, buffer_type=ttnn.BufferType.L1).width_sharded(core_ranges)
    assert spec.memory_config.nd_shard_spec.shard_shape == expected_shard_shape

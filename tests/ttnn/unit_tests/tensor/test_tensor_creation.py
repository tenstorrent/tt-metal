# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np

import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype


@pytest.mark.parametrize(
    "layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
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
@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
def test_tensor_creation(shape, tt_dtype, layout, device):
    torch.manual_seed(0)

    if tt_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("{} is only valid for ttnn.TILE_LAYOUT!".format(tt_dtype))

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.uint8, torch.int16, torch.int32}:
        py_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
    else:
        py_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttnn.Tensor(py_tensor, tt_dtype, device, layout)

    tt_tensor = tt_tensor.cpu()

    py_tensor_after_round_trip = tt_tensor.to_torch()

    assert py_tensor.dtype == py_tensor_after_round_trip.dtype
    assert py_tensor.shape == py_tensor_after_round_trip.shape

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(py_tensor, py_tensor_after_round_trip, **allclose_kwargs)
    assert passing


@pytest.mark.parametrize(
    "layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
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
@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
def test_tensor_creation_api_parity(shape, tt_dtype, layout, device):
    torch.manual_seed(0)

    if tt_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("{} is only valid for ttnn.TILE_LAYOUT!".format(tt_dtype))

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.uint8, torch.int16, torch.int32}:
        py_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
    else:
        py_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor_1 = ttnn.Tensor(py_tensor, tt_dtype, device, layout)
    tt_tensor_2 = ttnn.from_torch(py_tensor, tt_dtype, device=device, layout=layout)

    tt_tensor_1 = tt_tensor_1.cpu()
    tt_tensor_2 = tt_tensor_2.cpu()

    py_tensor_after_round_trip_1 = tt_tensor_1.to_torch()
    py_tensor_after_round_trip_2 = tt_tensor_2.to_torch()
    py_tensor_after_round_trip_3 = ttnn.to_torch(tt_tensor_1)
    py_tensor_after_round_trip_4 = ttnn.to_torch(tt_tensor_2)

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(py_tensor, py_tensor_after_round_trip_1, **allclose_kwargs)
    passing = torch.allclose(py_tensor, py_tensor_after_round_trip_2, **allclose_kwargs)
    passing = torch.allclose(py_tensor, py_tensor_after_round_trip_3, **allclose_kwargs)
    passing = torch.allclose(py_tensor, py_tensor_after_round_trip_4, **allclose_kwargs)
    assert passing


grid_size = [8, 7]
core_ranges = ttnn.num_cores_to_corerangeset(56, grid_size, True)


@pytest.mark.parametrize(
    "layout, tile",
    [
        (ttnn.ROW_MAJOR_LAYOUT, None),
        (ttnn.TILE_LAYOUT, ttnn.Tile([32, 32])),
        (ttnn.TILE_LAYOUT, ttnn.Tile([16, 16])),
        (ttnn.TILE_LAYOUT, ttnn.Tile([32, 16])),
        (ttnn.TILE_LAYOUT, ttnn.Tile([16, 16])),
    ],
)
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
@pytest.mark.parametrize(
    "shape, memory_config",
    [
        (
            (1, 2, 3, 4),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED,
                ttnn.BufferType.DRAM,
            ),
        ),
        (
            (1, 48, 56, 32),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.num_cores_to_corerangeset(56, grid_size, True),
                    [48, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardMode.LOGICAL,
                ),
            ),
        ),
        (
            (1, 2, 10, 5),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.num_cores_to_corerangeset(3, grid_size, True),
                    [20, 2],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardMode.LOGICAL,
                ),
            ),
        ),
        (
            (1, 1, 5, 96),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
                    [5, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        ),
        (
            (2, 3, 64, 96),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 5))}),
                    [64, 64],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardMode.LOGICAL,
                ),
            ),
        ),
        (
            (1, 8, 36, 32),
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 5))}),
                    [48, 10],
                    [64, 64],  # NOTE: This value is compatible with all PageConfigs in this sweep
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        ),
    ],
    ids=[
        "interleaved",
        "height_sharded",
        "width_sharded",
        "width_sharded_uneven",
        "block_sharded",
        "block_sharded_with_custom_physical_shard_shape",
    ],
)
def test_tensor_creation_with_memory_config(shape, memory_config, tt_dtype, layout, tile, device):
    torch.manual_seed(0)

    if tt_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("{} is only valid for ttnn.TILE_LAYOUT!".format(tt_dtype))

    if (
        memory_config.shard_spec is not None
        and memory_config.shard_spec.mode == ttnn.ShardMode.PHYSICAL
        and tile is not None
    ):
        shard_shape = memory_config.shard_spec.shape
        if shard_shape[0] % tile.tile_shape[0] != 0 or shard_shape[1] % tile.tile_shape[1] != 0:
            pytest.skip(
                "Shard shape {} is not divisible by tile {} {}!".format(
                    shard_shape, tile.tile_shape[0], tile.tile_shape[1]
                )
            )

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in {torch.uint8, torch.int16, torch.int32}:
        py_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
    else:
        py_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor_1 = ttnn.Tensor(py_tensor, tt_dtype, device, layout, memory_config, tile)
    tt_tensor_2 = ttnn.from_torch(
        py_tensor, tt_dtype, device=device, layout=layout, memory_config=memory_config, tile=tile
    )

    tt_tensor_1 = tt_tensor_1.cpu()
    tt_tensor_2 = tt_tensor_2.cpu()

    py_tensor_after_round_trip_1 = tt_tensor_1.to_torch()
    py_tensor_after_round_trip_2 = tt_tensor_2.to_torch()
    py_tensor_after_round_trip_3 = ttnn.to_torch(tt_tensor_1)
    py_tensor_after_round_trip_4 = ttnn.to_torch(tt_tensor_2)

    allclose_kwargs = {}
    if tt_dtype == ttnn.bfloat8_b:
        allclose_kwargs = dict(atol=1e-2)
    elif tt_dtype == ttnn.bfloat4_b:
        allclose_kwargs = dict(atol=0.2)

    passing = torch.allclose(py_tensor, py_tensor_after_round_trip_1, **allclose_kwargs)
    passing = torch.allclose(py_tensor, py_tensor_after_round_trip_2, **allclose_kwargs)
    passing = torch.allclose(py_tensor, py_tensor_after_round_trip_3, **allclose_kwargs)
    passing = torch.allclose(py_tensor, py_tensor_after_round_trip_4, **allclose_kwargs)
    assert passing


@pytest.mark.parametrize(
    "tensor_spec",
    [
        # Basic tests for using TensorSpec to create a Tensor
        ttnn.TensorSpec((1, 2, 3, 4), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        ttnn.TensorSpec((2, 3, 10, 20), ttnn.float32, ttnn.TILE_LAYOUT),
        ttnn.TensorSpec((2, 3, 10, 20), ttnn.float32, ttnn.TILE_LAYOUT, tile=ttnn.Tile([16, 16])),
        ttnn.TensorSpec((2, 3, 10, 20), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1),
        # Manually specifying sharding
        ttnn.TensorSpec(
            (2, 3, 10, 20),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardSpec(core_ranges, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
            buffer_type=ttnn.BufferType.L1,
        ),
        ttnn.TensorSpec(
            (2, 3, 40, 50),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            ttnn.NdShardSpec(
                [1, 1, 32, 32],
                core_ranges,
                ttnn.ShardOrientation.ROW_MAJOR,
                ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
            ),
            buffer_type=ttnn.BufferType.L1,
        ),
        # Sharding using TensorSpec methods
        # Batch sharding
        ttnn.TensorSpec(
            (2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
        ).sharded_across_dims_except([0], core_ranges),
        # Sharding preserving last two dimensions
        ttnn.TensorSpec(
            (2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
        ).sharded_across_dims_except([-1, -2], core_ranges),
        # Sharding across last dimension
        ttnn.TensorSpec(
            (2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1
        ).sharded_across_dims([-1], core_ranges),
        # 2D block sharding
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).block_sharded(
            core_ranges
        ),
        # 2D height sharding
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).height_sharded(
            core_ranges
        ),
        # 2D width sharding
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).width_sharded(
            core_ranges
        ),
        # Customized ND sharding
        ttnn.TensorSpec((2, 3, 40, 50), ttnn.float32, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1).sharded(
            (1, 37, 37), core_ranges, ttnn.ShardShapeAlignment.RECOMMENDED
        ),
    ],
)
def test_tensor_creation_with_tensor_spec(tensor_spec, device):
    torch.manual_seed(0)
    dtype = tt_dtype_to_torch_dtype[tensor_spec.dtype]
    py_tensor = torch.rand(list(tensor_spec.shape), dtype=dtype)
    tt_tensor = ttnn.from_torch(py_tensor, spec=tensor_spec, device=device)
    assert tt_tensor.spec == tensor_spec
    py_tensor_after_round_trip = ttnn.to_torch(tt_tensor)
    assert torch.allclose(py_tensor, py_tensor_after_round_trip)

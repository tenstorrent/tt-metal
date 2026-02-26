# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import os
import pathlib

import torch
import numpy as np

import ttnn
from tests.ttnn.utils_for_testing import (
    align_tensor_dtype,
    tt_dtype_to_torch_dtype,
    TORCH_INTEGER_DTYPES,
)

pytestmark = pytest.mark.use_module_device


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

    if dtype in TORCH_INTEGER_DTYPES:
        py_tensor = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype)
    else:
        py_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = ttnn.Tensor(py_tensor, tt_dtype, device, layout)

    tt_tensor = tt_tensor.cpu()

    py_tensor_after_round_trip = tt_tensor.to_torch()
    py_tensor_after_round_trip = align_tensor_dtype(py_tensor_after_round_trip, py_tensor.dtype)

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

    if dtype in TORCH_INTEGER_DTYPES:
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

    py_tensor_after_round_trip_1 = py_tensor_after_round_trip_1.to(py_tensor.dtype)
    py_tensor_after_round_trip_2 = py_tensor_after_round_trip_2.to(py_tensor.dtype)
    py_tensor_after_round_trip_3 = py_tensor_after_round_trip_3.to(py_tensor.dtype)
    py_tensor_after_round_trip_4 = py_tensor_after_round_trip_4.to(py_tensor.dtype)

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
    ],
    ids=[
        "interleaved",
        "width_sharded",
    ],
)
def test_tensor_creation_with_memory_config(shape, memory_config, tt_dtype, layout, tile, device):
    torch.manual_seed(0)

    if tt_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("{} is only valid for ttnn.TILE_LAYOUT!".format(tt_dtype))

    if memory_config.shard_spec is not None and tile is not None:
        shard_shape = memory_config.shard_spec.shape
        if shard_shape[0] % tile.tile_shape[0] != 0 or shard_shape[1] % tile.tile_shape[1] != 0:
            pytest.skip(
                "Shard shape {} is not divisible by tile {} {}!".format(
                    shard_shape, tile.tile_shape[0], tile.tile_shape[1]
                )
            )

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    if dtype in TORCH_INTEGER_DTYPES:
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

    py_tensor_after_round_trip_1 = py_tensor_after_round_trip_1.to(py_tensor.dtype)
    py_tensor_after_round_trip_2 = py_tensor_after_round_trip_2.to(py_tensor.dtype)
    py_tensor_after_round_trip_3 = py_tensor_after_round_trip_3.to(py_tensor.dtype)
    py_tensor_after_round_trip_4 = py_tensor_after_round_trip_4.to(py_tensor.dtype)

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


@pytest.mark.parametrize(
    "dtype,shape,buffer",
    [
        # 1D tensors
        (ttnn.uint8, [1, 3], [1, 2, 3]),
        (ttnn.uint16, [1, 3], [1000, 2000, 3000]),
        (ttnn.int32, [1, 3], [-100, -200, -300]),
        (ttnn.uint32, [1, 3], [1000000, 2000000, 3000000]),
        (ttnn.float32, [1, 3], [1.5, 2.7, 3.14]),
        (ttnn.bfloat16, [1, 3], [1.25, 2.75, 3.125]),
        # 2D tensors
        (ttnn.uint8, [2, 3], [1, 2, 3, 4, 5, 6]),
        (ttnn.uint16, [2, 3], [1000, 2000, 3000, 4000, 5000, 6000]),
        (ttnn.int32, [2, 3], [-100, -200, -300, 400, 500, 600]),
        (ttnn.uint32, [2, 3], [1000000, 2000000, 3000000, 4000000, 5000000, 6000000]),
        (ttnn.float32, [2, 3], [1.5, 2.7, 3.14, 4.2, 5.8, 6.9]),
        (ttnn.bfloat16, [2, 3], [1.25, 2.75, 3.125, 4.375, 5.625, 6.875]),
        # 3D tensors
        (ttnn.uint8, [2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]),
        (ttnn.uint16, [2, 2, 2], [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]),
        (ttnn.int32, [2, 2, 2], [-100, -200, -300, -400, -500, -600, -700, -800]),
        (ttnn.uint32, [2, 2, 2], [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000]),
        (ttnn.float32, [2, 2, 2], [1.5, 2.7, 3.14, 4.2, 5.8, 6.9, 7.1, 8.2]),
        (ttnn.bfloat16, [2, 2, 2], [1.25, 2.75, 3.125, 4.375, 5.625, 6.875, 7.125, 8.25]),
        # 4D tensors
        (ttnn.uint8, [2, 2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        (
            ttnn.uint16,
            [2, 2, 2, 2],
            [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000],
        ),
        (
            ttnn.int32,
            [2, 2, 2, 2],
            [-100, -200, -300, -400, -500, -600, -700, -800, -900, -1000, -1100, -1200, -1300, -1400, -1500, -1600],
        ),
        (
            ttnn.uint32,
            [2, 2, 2, 2],
            [
                1000000,
                2000000,
                3000000,
                4000000,
                5000000,
                6000000,
                7000000,
                8000000,
                9000000,
                10000000,
                11000000,
                12000000,
                13000000,
                14000000,
                15000000,
                16000000,
            ],
        ),
        (
            ttnn.float32,
            [2, 2, 2, 2],
            [1.5, 2.7, 3.14, 4.2, 5.8, 6.9, 7.1, 8.2, 9.3, 10.4, 11.5, 12.6, 13.7, 14.8, 15.9, 16.1],
        ),
        (
            ttnn.bfloat16,
            [2, 2, 2, 2],
            [
                1.25,
                2.75,
                3.125,
                4.375,
                5.625,
                6.875,
                7.125,
                8.25,
                9.375,
                10.625,
                11.875,
                12.125,
                13.375,
                14.625,
                15.875,
                16.125,
            ],
        ),
    ],
)
def test_tensor_creation_from_buffer(dtype, shape, buffer, device):
    tt_tensor = ttnn.from_buffer(buffer, shape, dtype=dtype, device=device)
    assert tt_tensor.shape == shape
    assert tt_tensor.dtype == dtype
    assert tt_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    def flatten_list(data):
        if isinstance(data, list):
            return [
                item for sublist in data for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])
            ]
        return [data]

    tensor_list = tt_tensor.to_list()
    flattened = flatten_list(tensor_list)

    if dtype in [ttnn.float32, ttnn.bfloat16]:
        rounded = [round(v, 5) for v in flattened]
        assert rounded == buffer
    else:
        assert flattened == buffer


@pytest.mark.parametrize(
    "dtype,buffer",
    [
        (ttnn.bfloat8_b, [[1, 2, 3], [4, 5, 6]]),
        (ttnn.bfloat4_b, [[1, 2, 3], [4, 5, 6]]),
        (ttnn.DataType.INVALID, [[1, 2, 3], [4, 5, 6]]),
    ],
)
def test_tensor_creation_from_buffer_with_unsupported_dtype(dtype, buffer, device):
    try:
        tt_tensor = ttnn.from_buffer(buffer, [2, 3], dtype, device, ttnn.TILE_LAYOUT)
    except Exception as e:
        assert "Unreachable" in str(e)
    else:
        pytest.fail("Expected an exception, but got none")


@pytest.mark.parametrize(
    "dtype,buffer,shape",
    [
        (ttnn.float32, [1, 2, 3], [1, 1, 3]),
        (ttnn.float32, [1, 2, 3, 4, 5, 6], [1, 2, 3]),  # 3 x 2
        (ttnn.float32, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3]),  # 2 x 3 x 2
        (ttnn.float32, [1, 2, 3, 4], [1, 1, 1, 4]),  # 1 x 1 x 1 x4
        (ttnn.float32, [1, 2, 3, 4], [1, 1, 4]),  # 1 x 1 x 4
        (ttnn.float32, [1, 2, 3, 4], [1, 4]),  # 1 x 4
        (ttnn.float32, [1, 2, 3, 4], [4]),  # 4
        (ttnn.float32, [1, 2, 3, 4], [1, 1, 1, 1, 4]),  # 1 x 1 x 1 x 1 x 4 # 5d!
        (ttnn.float32, [1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 1, 1, 1, 4]),  # 2 x 1 x 1 x 1 x 4 # 6d!
    ],
)
def test_tensor_creation_with_multiple_buffer_sizes(dtype, buffer, device, shape):
    tt_tensor = ttnn.Tensor(
        buffer,
        shape,
        dtype,
        ttnn.Layout.TILE,
        device,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )


def flatten_list(nested_list):
    if isinstance(nested_list, list):
        return [
            item
            for sublist in nested_list
            for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])
        ]
    return [nested_list]


@pytest.mark.parametrize(
    "layout",
    [
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "tt_dtype,data,atol",
    [
        (ttnn.float32, [1.5, 2.7, 3.14, 4.2, 5.8, 6.9, 7.1, 8.2, 9.3, 10.4, 11.5, 12.6, 13.7, 14.8, 15.9, 16.1], 1e-6),
        (
            ttnn.float32,
            [
                0.0,
                1e-10,
                1e-5,
                -1e-5,
                1e5,
                -1e5,
                3.14159,
                -2.71828,
                0.0,
                1e-10,
                1e-5,
                -1e-5,
                1e5,
                -1e5,
                3.14159,
                -2.71828,
            ],
            1e-6,
        ),
        (
            ttnn.bfloat16,
            [
                1.25,
                2.75,
                3.125,
                4.375,
                5.625,
                6.875,
                7.125,
                8.25,
                9.375,
                10.625,
                11.875,
                12.125,
                13.375,
                14.625,
                15.875,
                16.125,
            ],
            1e-3,
        ),
        (
            ttnn.bfloat16,
            [0.0, 0.001, 0.1, -0.1, 100.0, -100.0, 1.5, -1.5, 0.0, 0.001, 0.1, -0.1, 100.0, -100.0, 1.5, -1.5],
            1e-3,
        ),
        (
            ttnn.int32,
            [-100, -200, -300, -400, -500, -600, -700, -800, -900, -1000, -1100, -1200, -1300, -1400, -1500, -1600],
            0,
        ),
        (
            ttnn.int32,
            [
                -2147483648,
                -2147483647,
                -1,
                0,
                1,
                2147483646,
                2147483647,
                -1000000,
                -2147483648,
                -2147483647,
                -1,
                0,
                1,
                2147483646,
                2147483647,
                -1000000,
            ],
            0,
        ),
        (
            ttnn.uint32,
            [
                1000000,
                2000000,
                3000000,
                4000000,
                5000000,
                6000000,
                7000000,
                8000000,
                9000000,
                10000000,
                11000000,
                12000000,
                13000000,
                14000000,
                15000000,
                16000000,
            ],
            0,
        ),
        (
            ttnn.uint32,
            [
                0,
                1,
                2,
                1000,
                1000000,
                2147483646,
                2147483647,
                100000000,
                0,
                1,
                2,
                1000,
                1000000,
                2147483646,
                2147483647,
                100000000,
            ],
            0,
        ),
        (
            ttnn.uint16,
            [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000],
            0,
        ),
        (ttnn.uint16, [0, 1, 2, 255, 256, 32767, 32768, 65535, 0, 1, 2, 255, 256, 32767, 32768, 65535], 0),
        (ttnn.uint8, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 0),
        (ttnn.uint8, [0, 1, 2, 127, 128, 254, 255, 100, 0, 1, 2, 127, 128, 254, 255, 100], 0),
    ],
    ids=[
        "float32",
        "float32_edges",
        "bfloat16",
        "bfloat16_edges",
        "int32",
        "int32_edges",
        "uint32",
        "uint32_edges",
        "uint16",
        "uint16_edges",
        "uint8",
        "uint8_edges",
    ],
)
@pytest.mark.parametrize("shape", [(2, 8), (2, 2, 2, 2)])
def test_tensor_creation_from_list(shape, tt_dtype, data, layout, atol, device):
    if tt_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("{} is only valid for ttnn.TILE_LAYOUT!".format(tt_dtype))

    # Adjust data length to match shape
    total_elements = int(np.prod(shape))
    test_data = data[:total_elements]

    # Test host-only tensor (no device)
    tt_tensor_host = ttnn.Tensor(test_data, shape, tt_dtype, layout)

    assert tt_tensor_host.shape == list(shape)
    assert tt_tensor_host.dtype == tt_dtype
    assert tt_tensor_host.layout == layout

    # Verify data round-trip
    tensor_list = tt_tensor_host.to_list()

    flattened = flatten_list(tensor_list)

    # Check values
    if tt_dtype in [ttnn.float32, ttnn.bfloat16]:
        # For floating point, use approximate comparison
        for i, (a, b) in enumerate(zip(test_data, flattened)):
            assert abs(a - b) < atol, f"Mismatch at index {i}: expected {a}, got {b}"
    else:
        # For integers, use exact comparison
        assert flattened == test_data, f"Data mismatch: expected {test_data}, got {flattened}"

    # Test tensor with device
    tt_tensor_device = ttnn.Tensor(test_data, shape, tt_dtype, layout, device)
    tt_tensor_device = tt_tensor_device.cpu()

    assert tt_tensor_device.shape == list(shape)
    assert tt_tensor_device.dtype == tt_dtype
    assert tt_tensor_device.layout == layout

    tensor_list_device = tt_tensor_device.to_list()
    flattened_device = flatten_list(tensor_list_device)

    # Check values for device tensor
    if tt_dtype in [ttnn.float32, ttnn.bfloat16]:
        for i, (a, b) in enumerate(zip(test_data, flattened_device)):
            assert abs(a - b) < atol, f"Device tensor mismatch at index {i}: expected {a}, got {b}"
    else:
        assert (
            flattened_device == test_data
        ), f"Device tensor data mismatch: expected {test_data}, got {flattened_device}"


@pytest.mark.parametrize(
    "tt_dtype,data,tol",
    [
        # Block float types with float data - only TILE_LAYOUT supported
        (
            ttnn.bfloat8_b,
            [1.5, 2.7, 3.14, 4.2, 5.8, 6.9, 7.1, 8.2, 9.3, 10.4, 11.5, 12.6, 13.7, 14.8, 15.9, 16.1] * 64,
            2**-7,
        ),  # tolerance = scale * 2^-7 (7 bits for mantissa)
        (
            ttnn.bfloat4_b,
            [
                1.25,
                2.75,
                3.125,
                4.375,
                5.625,
                6.875,
                7.125,
                8.25,
                9.375,
                10.625,
                11.875,
                12.125,
                13.375,
                14.625,
                15.875,
                16.125,
            ]
            * 64,
            2**-3,
        ),  # tolerance = scale * 2^-3 (3 bits for mantissa)
    ],
    ids=["bfloat8_b", "bfloat4_b"],
)
def test_tensor_creation_from_list_block_float(tt_dtype, data, tol, device):
    """Test creating block float tensors (bfloat8_b, bfloat4_b) from Python lists"""

    # Block float types require TILE_LAYOUT and tile-aligned shapes
    shape = (1, 1, 32, 32)
    layout = ttnn.TILE_LAYOUT
    scale = max(data)
    atol = tol * scale

    # Test host-only tensor
    tt_tensor_host = ttnn.Tensor(data, shape, tt_dtype, layout)

    assert tt_tensor_host.shape == list(shape)
    assert tt_tensor_host.dtype == tt_dtype
    assert tt_tensor_host.layout == layout

    tensor_list_host = tt_tensor_host.to_list()
    assert len(tensor_list_host) == shape[0]

    flattened_host = flatten_list(tensor_list_host)
    # Verify at least some values match within tolerance
    for i in range(len(data)):
        assert (
            abs(data[i] - flattened_host[i]) < atol
        ), f"Host tensor mismatch at index {i}: expected {data[i]}, got {flattened_host[i]}"

    # Test tensor with device
    tt_tensor_device = ttnn.Tensor(data, shape, tt_dtype, layout, device)
    tt_tensor_device = tt_tensor_device.cpu()

    assert tt_tensor_device.shape == list(shape)
    assert tt_tensor_device.dtype == tt_dtype

    # Verify data round-trip for device tensor
    tensor_list_device = tt_tensor_device.to_list()
    flattened_device = flatten_list(tensor_list_device)

    # Check values within tolerance
    for i in range(len(data)):
        assert (
            abs(data[i] - flattened_device[i]) < atol
        ), f"Device tensor mismatch at index {i}: expected {data[i]}, got {flattened_device[i]}"


@pytest.mark.parametrize(
    "memory_config",
    [
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.num_cores_to_corerangeset(
                    target_num_cores=8,
                    grid_size=[8, 7],
                    row_wise=True,
                ),
                shard_shape=[64, 32],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.num_cores_to_corerangeset(
                    target_num_cores=8,
                    grid_size=[8, 7],
                    row_wise=True,
                ),
                shard_shape=[32, 64],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid=ttnn.num_cores_to_corerangeset(
                    target_num_cores=8 * 2,
                    grid_size=[8, 7],
                    row_wise=True,
                ),
                shard_shape=[32, 32],
                shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),
        (2, 32, 64),
    ],
)
@pytest.mark.parametrize(
    "tt_dtype,data_type",
    [
        (ttnn.float32, float),
        (ttnn.int32, int),
    ],
)
def test_tensor_creation_from_list_with_mem_config(shape, tt_dtype, data_type, memory_config, device):
    """Test creating tensors from Python lists with different memory configs"""

    total_elements = int(np.prod(shape))

    if data_type == float:
        data = [float(i) * 1.5 for i in range(total_elements)]
    else:
        data = [i * 10 for i in range(total_elements)]

    layout = ttnn.ROW_MAJOR_LAYOUT

    # Test tensor with device and memory config
    tt_tensor = ttnn.Tensor(data, shape, tt_dtype, layout, device, memory_config)
    tt_tensor = tt_tensor.cpu()

    assert tt_tensor.shape == list(shape)
    assert tt_tensor.dtype == tt_dtype
    assert tt_tensor.layout == layout

    # Verify data round-trip
    tensor_list = tt_tensor.to_list()

    flattened = flatten_list(tensor_list)

    if data_type == float:
        for i, (a, b) in enumerate(zip(data, flattened)):
            assert abs(a - b) < 1e-5, f"Mismatch at index {i}: expected {a}, got {b}"
    else:
        assert flattened == data

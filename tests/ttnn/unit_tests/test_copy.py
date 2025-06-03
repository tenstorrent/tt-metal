# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import math

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from models.utility_functions import skip_for_grayskull


# Test for int types
@pytest.mark.parametrize("shape", [[1, 1, 32, 256], [64, 64], [9, 32, 768], [128]])
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.int32])
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
def test_copy(shape, layout, dtype, device):
    torch.manual_seed(2005)
    torch_dtype = torch.int32

    input = torch.randint(1, 100, shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, dtype, layout=layout, device=device)

    input_b = torch.zeros(shape, dtype=torch_dtype)
    input_b = ttnn.from_torch(input_b, dtype, layout=layout, device=device)

    ttnn.copy(input, input_b)
    assert input_b.shape == input.shape
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(input_b), 1)
    assert_equal(ttnn.to_torch(input), ttnn.to_torch(input_b))


# Test for block sharding
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
@pytest.mark.parametrize("shape", [[128, 64]])
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ],
)
def test_copy_block_sharded(device, layout, shape, shard_scheme, dtype):
    torch.manual_seed(1234)
    if dtype == ttnn.uint32:
        input_torch = torch.randint(1, 100, shape, dtype=torch.int32)
    else:
        input_torch = torch.randn(shape)
    output_torch = torch.zeros(shape)
    ttnn_input = ttnn.from_torch(input_torch, dtype, layout=layout)
    ttnn_output = ttnn.from_torch(output_torch, dtype, layout=layout)

    num_cores_x = 2
    num_cores_y = 2
    num_cores = num_cores_x * num_cores_y
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])

    dims_b4_last_dim = 1
    for i in range(len(shape) - 1):
        dims_b4_last_dim *= shape[i]
    tile_widths_per_core = math.ceil(dims_b4_last_dim / num_cores_x)
    shard_shape = (tile_widths_per_core, 32 * math.ceil((shape[-1] / 32 / num_cores_y)))
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        shard_scheme,
        ttnn.BufferType.L1,
        shard_spec,
    )

    input_tensor = ttnn.to_device(
        ttnn_input,
        device,
        memory_config=output_mem_config,
    )
    output_tensor = ttnn.to_device(
        ttnn_output,
        device,
        memory_config=output_mem_config,
    )

    ttnn.copy(input_tensor, output_tensor)
    input_tensor = ttnn.to_torch(input_tensor)
    outout_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_tensor, outout_tensor, 1)
    assert_equal(input_tensor, outout_tensor)


# Test for width sharding
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
@pytest.mark.parametrize("shape", [[1, 2, 96, 128]])
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ],
)
def test_copy_width_sharded(device, layout, shape, shard_scheme, dtype):
    torch.manual_seed(1234)
    if dtype == ttnn.uint32:
        input_torch = torch.randint(1, 100, shape, dtype=torch.int32)
    else:
        input_torch = torch.randn(shape)
    output_torch = torch.zeros(shape)
    ttnn_input = ttnn.from_torch(input_torch, dtype, layout=layout)
    ttnn_output = ttnn.from_torch(output_torch, dtype, layout=layout)

    num_cores = 2
    shard_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(0, 4)),
        ]
    )

    dims_b4_last_dim = 1
    for i in range(len(shape) - 1):
        dims_b4_last_dim *= shape[i]

    shard_shape = (dims_b4_last_dim, 32 * math.ceil((math.ceil(shape[-1] / 32) / num_cores)))
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

    output_mem_config = ttnn.MemoryConfig(
        shard_scheme,
        ttnn.BufferType.L1,
        shard_spec,
    )

    input_tensor = ttnn.to_device(
        ttnn_input,
        device,
        memory_config=output_mem_config,
    )
    output_tensor = ttnn.to_device(
        ttnn_output,
        device,
        memory_config=output_mem_config,
    )

    ttnn.copy(input_tensor, output_tensor)
    input_tensor = ttnn.to_torch(input_tensor)
    outout_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_tensor, outout_tensor, 1)
    assert_equal(input_tensor, outout_tensor)


# Test for height sharding
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.bfloat16])
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
@pytest.mark.parametrize("shape", [[512, 64]])
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
def test_copy_height_sharded(device, layout, shape, shard_scheme, dtype):
    torch.manual_seed(1234)
    if dtype == ttnn.uint32:
        input_torch = torch.randint(1, 100, shape, dtype=torch.int32)
    else:
        input_torch = torch.randn(shape)
    output_torch = torch.zeros(shape)
    ttnn_input = ttnn.from_torch(input_torch, dtype=dtype, layout=layout)
    ttnn_output = ttnn.from_torch(output_torch, dtype=dtype, layout=layout)

    num_cores = 8
    shard_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(0, 5)),
        ]
    )

    dims_b4_last_dim = 1
    for i in range(len(shape) - 1):
        dims_b4_last_dim *= shape[i]
    tile_widths_per_core = math.ceil(dims_b4_last_dim / num_cores)
    shard_shape = (tile_widths_per_core, shape[-1])

    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(
        shard_scheme,
        ttnn.BufferType.L1,
        shard_spec,
    )

    input_tensor = ttnn.to_device(
        ttnn_input,
        device,
        memory_config=output_mem_config,
    )
    output_tensor = ttnn.to_device(
        ttnn_output,
        device,
        memory_config=output_mem_config,
    )

    ttnn.copy(input_tensor, output_tensor)
    input_tensor = ttnn.to_torch(input_tensor)
    outout_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_tensor, outout_tensor, 1)
    assert_equal(input_tensor, outout_tensor)

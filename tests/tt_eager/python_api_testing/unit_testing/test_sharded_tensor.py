# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib

import torch

import tt_lib as ttl


tt_dtype_to_torch_dtype = {
    ttl.tensor.DataType.UINT32: torch.int32,
    ttl.tensor.DataType.BFLOAT16: torch.bfloat16,
    ttl.tensor.DataType.BFLOAT8_B: torch.float,
}


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.DataType.BFLOAT8_B,
    ],
)
@pytest.mark.parametrize(
    "tensor_shape, shard_scheme, shard_shape, compute_grid",
    [
        ([1, 4, 64, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (64, 64), (0, 3)),
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.HEIGHT_SHARDED, (1024, 64), None),
        ([1, 1, 128, 50176], ttl.tensor.TensorMemoryLayout.WIDTH_SHARDED, (128, 512), None),
        ([1, 1, 100352, 64], ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, (2048, 32), None),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation", [ttl.tensor.ShardOrientation.ROW_MAJOR, ttl.tensor.ShardOrientation.COL_MAJOR]
)
def test_tensor_conversion_between_torch_and_tt_tile(
    tt_dtype, device, tensor_shape, shard_scheme, shard_shape, compute_grid, shard_orientation
):
    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    if compute_grid == None:
        compute_grid = ttl.tensor.CoreCoord(
            device.compute_with_storage_grid_size().x - 1, device.compute_with_storage_grid_size().y - 1
        )
    else:
        compute_grid = ttl.tensor.CoreCoord(compute_grid[0], compute_grid[1])

    shard_grid = ttl.tensor.CoreRangeSet({ttl.tensor.CoreRange(ttl.tensor.CoreCoord(0, 0), compute_grid)})
    shard_halo = False
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    if dtype == torch.int32:
        torch_tensor = torch.randint(0, 1024, tensor_shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(tensor_shape, dtype=dtype)
    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype).to(ttl.tensor.Layout.TILE)

    mem_config = ttl.tensor.MemoryConfig(shard_scheme, ttl.tensor.BufferType.L1)
    tt_tensor = tt_tensor.to(device, mem_config, shard_spec)
    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttl.tensor.DataType.UINT32,
        ttl.tensor.DataType.BFLOAT16,
    ],
)
def test_tensor_conversion_between_torch_and_tt_rm(tt_dtype, device):
    TILE_HEIGHT = 32
    TILE_WIDTH = 32

    dtype = tt_dtype_to_torch_dtype[tt_dtype]

    num_cores_height = 8
    num_cores_width = 8

    shard_grid = ttl.tensor.CoreRangeSet(
        {
            ttl.tensor.CoreRange(
                ttl.tensor.CoreCoord(0, 0), ttl.tensor.CoreCoord(num_cores_height - 1, num_cores_width - 1)
            )
        }
    )
    shard_shape = [72, 128]

    tensor_shape = (1, 1, 2304, 256)
    shard_orientation = ttl.tensor.ShardOrientation.ROW_MAJOR
    shard_halo = False
    shard_spec = ttl.tensor.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    if dtype == torch.int32:
        torch_tensor = torch.randint(0, 1024, tensor_shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(tensor_shape, dtype=dtype)
    tt_tensor = ttl.tensor.Tensor(torch_tensor, tt_dtype)

    mem_config = ttl.tensor.MemoryConfig(ttl.tensor.TensorMemoryLayout.BLOCK_SHARDED, ttl.tensor.BufferType.L1)
    tt_tensor = tt_tensor.to(device, mem_config, shard_spec)
    tt_tensor = tt_tensor.cpu().to(ttl.tensor.Layout.ROW_MAJOR)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing

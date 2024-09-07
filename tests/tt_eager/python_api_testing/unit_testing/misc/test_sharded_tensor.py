# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import pathlib

import torch

import ttnn

from models.utility_functions import get_debug_tensor
from enum import Enum

tt_dtype_to_torch_dtype = {
    ttnn.uint32: torch.int32,
    ttnn.uint16: torch.int16,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.float,
}
TILE_WIDTH = 32
TILE_HEIGHT = 32

debug = False


class DirectReadWriteType(Enum):
    READ_ONLY = 0
    WRITE_ONLY = 1
    READ_WRITE = 2


def print_tiles(tiled_tensor, num_tiles_height, num_tiles_width):
    tile_torch_rows = torch.chunk(tiled_tensor, int(num_tiles_height), dim=2)
    row_idx = 0
    for row in tile_torch_rows:
        tiles = torch.chunk(row, int(num_tiles_width), dim=3)
        col_idx = 0
        for tile in tiles:
            tile_idx = row_idx * num_tiles_width + col_idx
            print("Trip Tile " + str(int(tile_idx)) + " with shape " + str(tile.shape))
            print(tile)
            col_idx = col_idx + 1
        row_idx = row_idx + 1


def get_tensor(shape, dtype):
    if dtype in {torch.int16, torch.int32}:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)
    return torch_tensor


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "shard_orientation, tensor_shape, shard_scheme, shard_shape, grid_override, direct_read_write_type",
    [
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 4, 64, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            None,
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),  # 4 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 2048, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (512, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),  # 4 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 2048, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1024, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),  # 2 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 4096, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1024, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0)),  # 4 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 8192, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1024, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),  # 8 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 14336, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (1024, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0)),  # 8 cores
                    ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2)),  # 4 cores
                    ttnn.CoreRange(ttnn.CoreCoord(4, 4), ttnn.CoreCoord(5, 4)),  # 2 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 256, 32],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (256, 32),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),  # 1 core
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (128, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),  # 1 core
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 2048, 64],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (1024, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),  # 2 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 32, 16256],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (32, 512),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 4)),  # 25 cores
                    ttnn.CoreRange(ttnn.CoreCoord(5, 5), ttnn.CoreCoord(6, 6)),  # 4 cores
                    ttnn.CoreRange(ttnn.CoreCoord(6, 7), ttnn.CoreCoord(7, 7)),  # 2 cores
                    ttnn.CoreRange(ttnn.CoreCoord(5, 4), ttnn.CoreCoord(5, 4)),  # 1 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 64, 96],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),  # 4 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.COL_MAJOR,
            [1, 1, 256, 288],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),  # 48 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.COL_MAJOR,
            [1, 1, 8192, 320],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (1024, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4)),  # 40 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        pytest.param(
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 8192, 1536],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (1024, 320),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 7)),  # 40 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
            marks=pytest.mark.xfail(reason="7740: Test case doesn't work anymore after flipping the to correct grid."),
        ),
        (
            ttnn.ShardOrientation.ROW_MAJOR,
            [1, 1, 64, 96],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),  # 2 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
        (
            ttnn.ShardOrientation.COL_MAJOR,
            [1, 1, 8192, 512],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (1024, 320),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4)),  # 40 cores
                }
            ),
            DirectReadWriteType.READ_WRITE,
        ),
    ],
)
def test_tensor_conversion_between_torch_and_tt_tile(
    tt_dtype, device, shard_orientation, tensor_shape, shard_scheme, shard_shape, grid_override, direct_read_write_type
):
    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    compute_grid = ttnn.CoreCoord(
        device.compute_with_storage_grid_size().x - 1, device.compute_with_storage_grid_size().y - 1
    )

    if grid_override == None:
        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), compute_grid)})
    else:
        shard_grid = grid_override

    shard_halo = False
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    two_d_shape = (tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3])
    num_tiles_width = (two_d_shape[1]) / TILE_WIDTH
    num_tiles_height = (two_d_shape[0]) / TILE_HEIGHT

    if debug:
        torch_tensor = get_debug_tensor(num_tiles_width, num_tiles_height, dtype)
    else:
        torch_tensor = get_tensor(tensor_shape, dtype)
    tt_tensor = ttnn.Tensor(torch_tensor, tt_dtype).to(ttnn.TILE_LAYOUT)
    interleaved_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config = ttnn.MemoryConfig(shard_scheme, ttnn.BufferType.L1, shard_spec)

    # test not doing direct write
    if direct_read_write_type == DirectReadWriteType.READ_ONLY:
        tt_tensor = tt_tensor.to(device, interleaved_mem_config)
        tt_tensor = ttnn.interleaved_to_sharded(tt_tensor, mem_config)
    else:
        tt_tensor = tt_tensor.to(device, mem_config)
    ttnn.synchronize_device(device)
    # not doing direct read
    if direct_read_write_type == DirectReadWriteType.WRITE_ONLY:
        tt_tensor = ttnn.sharded_to_interleaved(tt_tensor, interleaved_mem_config)
    tt_tensor = tt_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT)
    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.allclose(torch_tensor, torch_tensor_after_round_trip)
    assert passing


@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.uint32,
        ttnn.uint16,
        ttnn.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "tensor_shape, shard_scheme, shard_shape",
    [
        ([1, 1, 4, 256], ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1, 256)),
    ],
)
@pytest.mark.parametrize(
    "buffer_type",
    [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
)
def test_tensor_conversion_between_torch_and_tt_rm(
    tt_dtype, device, tensor_shape, shard_scheme, shard_shape, buffer_type
):
    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    num_pages_width = tensor_shape[2] / shard_shape[0]
    num_pages_height = tensor_shape[3] / shard_shape[1]

    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    shard_halo = False
    if buffer_type == ttnn.BufferType.DRAM:
        shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    else:
        shard_grid = ttnn.CoreCoord(
            device.compute_with_storage_grid_size().x - 1, device.compute_with_storage_grid_size().y - 1
        )
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), shard_grid)})
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, shard_orientation, shard_halo)

    if debug:
        torch_tensor = get_debug_tensor(
            num_pages_width, num_pages_height, dtype, page_width=shard_shape[0], page_height=shard_shape[1]
        )
    else:
        torch_tensor = get_tensor(tensor_shape, dtype)

    torch_tensor = torch_tensor.reshape(tensor_shape)

    tt_tensor = ttnn.Tensor(torch_tensor, tt_dtype)

    assert list(torch_tensor.size()) == tensor_shape

    mem_config = ttnn.MemoryConfig(shard_scheme, buffer_type, shard_spec)
    tt_tensor = tt_tensor.to(device, mem_config)
    tt_tensor = tt_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT)

    torch_tensor_after_round_trip = tt_tensor.to_torch()

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    passing = torch.equal(torch_tensor, torch_tensor_after_round_trip)
    assert passing

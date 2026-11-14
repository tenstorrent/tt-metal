# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.skipif(not is_wormhole_b0(), reason="Wormhole-only program")


def _grid(x, y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def _skip_if_grid_smaller(device, grid):
    g = device.compute_with_storage_grid_size()
    if g.x < grid[0] or g.y < grid[1]:
        pytest.skip(f"needs a {grid[0]}x{grid[1]} core grid")


def _height_sharded(grid, shard_shape, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    shard_spec = ttnn.ShardSpec(_grid(*grid), shard_shape, orientation)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


@pytest.mark.parametrize(
    "tensor_shape, dtype, input_mem_config, grid, shard_shape",
    [
        ((1, 1, 409600, 3), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (6400, 8)),
        ((1, 1, 262144, 3), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (4096, 8)),
        ((1, 1, 409600, 8), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (6400, 8)),
        ((1, 1, 64000, 1), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (1000, 8)),
        ((1, 1, 64000, 2), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (1000, 8)),
        ((1, 1, 64000, 4), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (1000, 8)),
        ((1, 1, 64000, 5), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (1000, 8)),
        ((1, 1, 64000, 3), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (1000, 4)),
        ((1, 1, 100000, 3), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (1563, 8)),
        ((1, 1, 65536, 24), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (1024, 24)),
        ((1, 1, 1000, 3), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (4, 4), (63, 8)),
        ((1, 1, 102400, 3), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG, (8, 8), (1600, 8)),
        ((1, 1, 102400, 16), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (1600, 16)),
        ((1, 1, 127, 3), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (2, 8)),
        ((1, 1, 64 * 23, 3), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (23, 8)),
        ((1, 1, 64 * 384, 3), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (384, 8)),
        ((1, 1, 64 * 64, 40), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG, (8, 8), (64, 40)),
    ],
)
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_interleaved_to_height_sharded_short_rows_exact(
    device, tensor_shape, dtype, input_mem_config, grid, shard_shape, shard_orientation
):
    _skip_if_grid_smaller(device, grid)
    torch.manual_seed(0)
    output_mem_config = _height_sharded(grid, shard_shape, shard_orientation)
    for _ in range(2):
        torch_input = torch.randn(tensor_shape, dtype=torch.float32)
        ttnn_input = ttnn.from_torch(
            torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_mem_config
        )
        output = ttnn.interleaved_to_sharded(ttnn_input, output_mem_config)
        assert_equal(ttnn.to_torch(ttnn_input), ttnn.to_torch(output))


_HOG_PAGE_BYTES = 16


def _l1_hog_leaving(device, free_bytes_per_bank):
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    pages_per_bank = (view.largest_contiguous_bytes_free_per_bank - free_bytes_per_bank) // _HOG_PAGE_BYTES
    if pages_per_bank <= 0:
        pytest.skip("L1 already has less free space than the test needs")
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape((pages_per_bank * view.num_banks, _HOG_PAGE_BYTES // 2)),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )


def test_interleaved_to_height_sharded_short_rows_with_less_free_l1(device):
    _skip_if_grid_smaller(device, (8, 8))
    torch.manual_seed(0)
    output_mem_config = _height_sharded((8, 8), (64, 24))
    torch_input = torch.randn((1, 1, 64 * 64, 24), dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    expected = ttnn.to_torch(ttnn_input)
    assert_equal(expected, ttnn.to_torch(ttnn.interleaved_to_sharded(ttnn_input, output_mem_config)))
    hog = _l1_hog_leaving(device, 16 * 1024)
    assert_equal(expected, ttnn.to_torch(ttnn.interleaved_to_sharded(ttnn_input, output_mem_config)))
    ttnn.deallocate(hog)
    assert_equal(expected, ttnn.to_torch(ttnn.interleaved_to_sharded(ttnn_input, output_mem_config)))

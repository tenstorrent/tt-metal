# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""ttnn.sharded_to_interleaved must convert ND-sharded tensors, not hand them back unchanged.

An NdShardSpec only carries a legacy 2-D ShardSpec when it folds into one, so ROUND_ROBIN_1D with more
shards than cores, CONTIGUOUS_1D and 3-D shard shapes leave it without a legacy shard_spec while
is_sharded() stays true. The wrapper used to test shard_spec() and returned its input verbatim for
those, which kept the tensor ND_SHARDED in L1 and silently dropped both the requested buffer type and
output dtype.
"""

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_equal


def nd_memory_config(shard_shape, grid, strategy, buffer_type=ttnn.BufferType.L1):
    nd_shard_spec = ttnn.NdShardSpec(
        ttnn.Shape(shard_shape),
        grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=strategy,
    )
    return ttnn.MemoryConfig(buffer_type, nd_shard_spec)


@pytest.mark.parametrize(
    "shape, shard_shape, grid_wh, strategy",
    [
        # ROUND_ROBIN_1D with 8 shards on a 2x2 grid: no 2-D projection exists.
        ([1, 1, 128, 128], [1, 1, 32, 64], (2, 2), ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D),
        # CONTIGUOUS_1D on the same shapes: also never projected.
        ([1, 1, 128, 128], [1, 1, 32, 64], (2, 2), ttnn.ShardDistributionStrategy.CONTIGUOUS_1D),
        # CONTIGUOUS_1D over a 2x1 grid, volume preserved but the split is along batch.
        ([4, 1, 64, 64], [1, 1, 64, 64], (2, 1), ttnn.ShardDistributionStrategy.CONTIGUOUS_1D),
    ],
    ids=["round_robin_8_on_4_cores", "contiguous_1d_8_on_4_cores", "contiguous_1d_batch_split"],
)
@pytest.mark.parametrize("target", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG], ids=["l1", "dram"])
def test_s2i_nd_sharded_converts_to_interleaved(device, shape, shard_shape, grid_wh, strategy, target):
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_wh[0] - 1, grid_wh[1] - 1),
            ),
        }
    )
    input_memory_config = nd_memory_config(shard_shape, grid, strategy)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )
    assert input_tensor.is_sharded(), "test setup must produce an ND-sharded tensor"

    output_tensor = ttnn.sharded_to_interleaved(input_tensor, target)

    assert not output_tensor.is_sharded()
    assert output_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert output_tensor.memory_config().buffer_type == target.buffer_type
    assert output_tensor.dtype == ttnn.bfloat16
    assert_equal(torch_input_tensor, ttnn.to_torch(output_tensor))


def test_s2i_nd_sharded_3d_shard_shape(device):
    # A 3-D shard shape has no 2-D projection.
    shape = [6, 32, 64]
    shard_shape = [3, 32, 32]
    grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 0),
            ),
        }
    )
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=nd_memory_config(shard_shape, grid, ttnn.ShardDistributionStrategy.CONTIGUOUS_1D),
    )
    assert input_tensor.is_sharded()

    output_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    assert not output_tensor.is_sharded()
    assert output_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert output_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_equal(torch_input_tensor, ttnn.to_torch(output_tensor))


def test_s2i_nd_sharded_honours_output_dtype(device):
    shape = [1, 1, 128, 128]
    grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(1, 1),
            ),
        }
    )
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=nd_memory_config([1, 1, 32, 64], grid, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D),
    )

    output_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.float32)

    assert output_tensor.dtype == ttnn.float32
    assert not output_tensor.is_sharded()
    assert output_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert_equal(torch_input_tensor.to(torch.float32), ttnn.to_torch(output_tensor))


def test_s2i_2d_sharded_still_uses_the_device_op(device):
    # Control: a legacy 2-D shard_spec keeps the original path, which the device op validates. A TILE shard
    # must be tile sized in both dims, so the shard is 32 rows of the 64-wide tensor on a 2x1 grid.
    shape = [1, 1, 64, 64]
    grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(1, 0),
            ),
        }
    )
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    input_shard_spec = ttnn.ShardSpec(grid, (32, 64), ttnn.ShardOrientation.ROW_MAJOR)
    input_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec
    )
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_memory_config,
    )

    output_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    assert not output_tensor.is_sharded()
    assert output_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert output_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert_equal(torch_input_tensor, ttnn.to_torch(output_tensor))


def test_s2i_interleaved_input_is_unchanged(device):
    # The early return's real subject: a tensor that is not sharded at all.
    torch_input_tensor = torch.rand([1, 1, 64, 64], dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert not input_tensor.is_sharded()

    output_tensor = ttnn.sharded_to_interleaved(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    assert output_tensor.buffer_address() == input_tensor.buffer_address()
    assert_equal(torch_input_tensor, ttnn.to_torch(output_tensor))

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn

# Define sharding configurations
height_sharded_memory_config = ttnn.create_sharded_memory_config(
    [128, 160],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6)), ttnn.CoreRange((1, 0), (1, 6))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

width_sharded_memory_config = ttnn.create_sharded_memory_config(
    [1792, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 3)), ttnn.CoreRange((1, 0), (1, 3))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    [256, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (3, 6))}),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([4, 7, 64, 128]),
    ],
)
@pytest.mark.parametrize(
    "input_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
@pytest.mark.parametrize(
    "out_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
)
@pytest.mark.parametrize(
    "ttnn_op, dtype",
    [
        (ttnn.abs, ttnn.bfloat16),
        (ttnn.neg, ttnn.bfloat16),
    ],
)
def test_unary_sharded_interleaved(input_shape, input_config, out_config, ttnn_op, dtype, device):
    """Test unary operations with different sharding strategies and configurable thresholds"""
    torch.manual_seed(42)

    # Map ttnn dtype to torch dtype
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    # Create input tensor with range suitable for the operation
    torch_input = torch.randn(input_shape, dtype=torch_dtype)

    # Get golden result from torch
    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(torch_input, device=device)

    # Convert to ttnn with sharded memory config
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_config,
    )

    # Run operation with sharded input and interleaved output
    ttnn_output_interleaved = ttnn_op(ttnn_input, memory_config=out_config)

    # Convert output back to torch
    ttnn_output = ttnn.to_torch(ttnn_output_interleaved)

    # Compare with golden using specified thresholds
    assert torch.equal(ttnn_output, torch_output)

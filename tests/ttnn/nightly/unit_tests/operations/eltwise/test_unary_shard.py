# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp

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
    "sharded_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
    ],
)
@pytest.mark.parametrize(
    "ttnn_op, dtype, atol_threshold, ulp_threshold",
    [
        (ttnn.log_sigmoid, ttnn.bfloat16, 1e-1, 7.0),
    ],
)
def test_unary_sharded_ops(input_shape, sharded_config, ttnn_op, dtype, atol_threshold, ulp_threshold, device):
    """Test unary operations with different sharding strategies and configurable thresholds"""
    torch.manual_seed(2024)

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
        memory_config=sharded_config,
    )

    # Run operation with sharded input
    ttnn_output_sharded = ttnn_op(ttnn_input, memory_config=sharded_config)

    # Convert output back to torch
    ttnn_output = ttnn.to_torch(ttnn_output_sharded)

    # Compare with golden using specified thresholds
    assert torch.allclose(ttnn_output, torch_output, atol=atol_threshold)
    assert_with_ulp(torch_output, ttnn_output, ulp_threshold)


@pytest.mark.parametrize(
    "ttnn_op, dtype, low, high, atol_threshold, ulp_threshold",
    [
        (ttnn.log_sigmoid, ttnn.bfloat16, -87.0, 10.0, 1e-1, 7.0),
    ],
)
def test_unary_exhaustive_bitpatterns(ttnn_op, dtype, low, high, atol_threshold, ulp_threshold, device):
    """Test unary operations with exhaustive bf16 bit patterns within valid range"""
    torch.manual_seed(2024)

    # Map ttnn dtype to torch dtype
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32

    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch_dtype)
    input_tensor_f32 = input_tensor.to(torch.float32)

    # Mask to working range to avoid overflow/underflow
    mask = (input_tensor_f32 >= low) & (input_tensor_f32 <= high)
    input_tensor = input_tensor[mask]

    # Convert to ttnn tensor
    ttnn_input = ttnn.from_torch(
        input_tensor,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get golden result from torch
    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(input_tensor, device=device)

    # Run operation
    ttnn_output = ttnn_op(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    # Compare with golden using specified threshold
    assert_with_ulp(ttnn_output, torch_output, ulp_threshold)
    assert torch.allclose(ttnn_output, torch_output, atol=atol_threshold)

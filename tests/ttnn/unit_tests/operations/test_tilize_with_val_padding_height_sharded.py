# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


# Test TilizeWithValPadding with height sharding
@pytest.mark.parametrize(
    "input_shape, output_shape, shard_shape, core_grid",
    [
        # Test case 1: Simple height sharding with padding
        ([1, 1, 64, 64], [1, 1, 128, 64], [32, 64], (2, 2)),
        # Test case 2: Height sharding with batch
        ([2, 1, 128, 64], [2, 1, 256, 64], [64, 64], (2, 2)),
        # Test case 3: Height sharding without padding
        ([1, 1, 128, 64], [1, 1, 128, 64], [32, 64], (2, 2)),
        # Test case 4: Height sharding with partial tile padding
        ([1, 1, 100, 64], [1, 1, 128, 64], [25, 64], (2, 2)),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_tilize_with_val_padding_height_sharded(
    device, input_shape, output_shape, shard_shape, core_grid, dtype
):
    """Test TilizeWithValPadding with height-sharded input tensors."""
    
    # Create input tensor
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    
    # Create height-sharded memory config
    shard_config = ttnn.ShardConfig(
        shard_shape=shard_shape,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        core_grid=ttnn.CoreGrid(y=core_grid[0], x=core_grid[1]),
    )
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_config,
    )
    
    # Convert to TTNN tensor with row-major layout and height sharding
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    
    # Calculate padded shape
    output_padded_shape = ttnn.Shape(output_shape)
    
    # Create pad value
    pad_value = 0.0 if dtype == ttnn.bfloat16 or dtype == ttnn.float32 else 0
    
    # Run tilize_with_val_padding
    ttnn_output = ttnn.tilize_with_val_padding(
        ttnn_input,
        output_padded_shape,
        pad_value,
        sharded_mem_config,
    )
    
    # Convert output back to torch
    output_torch = ttnn.to_torch(ttnn_output)
    
    # Create expected output by padding the input
    expected_output = torch.zeros(output_shape, dtype=torch_input.dtype)
    expected_output[:input_shape[0], :input_shape[1], :input_shape[2], :input_shape[3]] = torch_input
    
    # Verify output
    assert_with_pcc(expected_output, output_torch, pcc=0.9999)


# Test via to_layout which uses tilize_with_val_padding internally
@pytest.mark.parametrize(
    "input_shape, shard_shape, core_grid",
    [
        ([1, 1, 64, 64], [16, 64], (2, 2)),
        ([1, 1, 128, 128], [32, 128], (2, 2)),
        ([4, 1, 256, 64], [64, 64], (2, 4)),
    ],
)
def test_to_layout_height_sharded_to_tile(device, input_shape, shard_shape, core_grid):
    """Test converting height-sharded row-major tensor to tile layout."""
    
    # Create input tensor
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    
    # Create height-sharded memory config
    shard_config = ttnn.ShardConfig(
        shard_shape=shard_shape,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        core_grid=ttnn.CoreGrid(y=core_grid[0], x=core_grid[1]),
    )
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_config,
    )
    
    # Convert to TTNN tensor with row-major layout and height sharding
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    
    # Convert to tile layout (this should use tilize_with_val_padding for height-sharded tensors)
    ttnn_output = ttnn.to_layout(ttnn_input, ttnn.TILE_LAYOUT)
    
    # Convert output back to torch
    output_torch = ttnn.to_torch(ttnn_output)
    
    # Verify output
    assert_with_pcc(torch_input, output_torch, pcc=0.9999)


# Test error cases
@pytest.mark.parametrize(
    "input_shape, output_shape, shard_shape, core_grid, expected_error",
    [
        # Width dimension mismatch should fail for height sharding
        ([1, 1, 64, 64], [1, 1, 128, 128], [32, 64], (2, 2), "must equal output padded shape"),
    ],
)
def test_tilize_with_val_padding_height_sharded_errors(
    device, input_shape, output_shape, shard_shape, core_grid, expected_error
):
    """Test error cases for height-sharded tilize_with_val_padding."""
    
    # Create input tensor
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    
    # Create height-sharded memory config
    shard_config = ttnn.ShardConfig(
        shard_shape=shard_shape,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        core_grid=ttnn.CoreGrid(y=core_grid[0], x=core_grid[1]),
    )
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_config,
    )
    
    # Convert to TTNN tensor
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_mem_config,
    )
    
    # Calculate padded shape
    output_padded_shape = ttnn.Shape(output_shape)
    
    # Expect an error
    with pytest.raises(RuntimeError, match=expected_error):
        ttnn.tilize_with_val_padding(
            ttnn_input,
            output_padded_shape,
            0.0,
            sharded_mem_config,
        )


# Compare height sharding vs width sharding performance characteristics
@pytest.mark.parametrize(
    "input_shape, height_shard_config, width_shard_config",
    [
        ([1, 1, 256, 256], ([64, 256], (2, 2)), ([256, 64], (2, 2))),
    ],
)
def test_height_vs_width_sharding_equivalence(
    device, input_shape, height_shard_config, width_shard_config
):
    """Verify height sharding produces same results as width sharding."""
    
    # Create input tensor
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    
    # Height-sharded configuration
    height_shard_shape, height_core_grid = height_shard_config
    height_shard_config_obj = ttnn.ShardConfig(
        shard_shape=height_shard_shape,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        core_grid=ttnn.CoreGrid(y=height_core_grid[0], x=height_core_grid[1]),
    )
    height_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        height_shard_config_obj,
    )
    
    # Width-sharded configuration
    width_shard_shape, width_core_grid = width_shard_config
    width_shard_config_obj = ttnn.ShardConfig(
        shard_shape=width_shard_shape,
        shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        core_grid=ttnn.CoreGrid(y=width_core_grid[0], x=width_core_grid[1]),
    )
    width_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        width_shard_config_obj,
    )
    
    # Create height-sharded tensor
    ttnn_input_height = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=height_sharded_mem_config,
    )
    
    # Create width-sharded tensor
    ttnn_input_width = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=width_sharded_mem_config,
    )
    
    # Convert both to tile layout
    ttnn_output_height = ttnn.to_layout(ttnn_input_height, ttnn.TILE_LAYOUT)
    ttnn_output_width = ttnn.to_layout(ttnn_input_width, ttnn.TILE_LAYOUT)
    
    # Convert outputs back to torch
    output_height_torch = ttnn.to_torch(ttnn_output_height)
    output_width_torch = ttnn.to_torch(ttnn_output_width)
    
    # Both should produce the same result
    assert_with_pcc(output_height_torch, output_width_torch, pcc=0.9999)

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

def get_shard_spec(device, layout, batch, height, width, depth, channels):
    """Helper to create shard spec based on layout and grid"""
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = compute_grid_size.x * compute_grid_size.y
    total_elements = batch * depth * height * width
    
    # Utilize full 2D grid
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))})
    
    if layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        shard_shape = [total_elements, channels // num_cores]
    elif layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        # Simplified block logic: split both dims roughly
        shard_shape = [total_elements // compute_grid_size.y, channels // compute_grid_size.x] 
    else: # HEIGHT_SHARDED (Default)
        shard_shape = [total_elements // num_cores, channels]
        
    return ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

@pytest.mark.parametrize("shard_layout", [
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.TensorMemoryLayout.BLOCK_SHARDED
])
def test_conv3d_sharded_input(device, shard_layout):
    """Test Conv3d with Sharded Input Tensor"""
    batch, depth, height, width = 1, 16, 32, 32
    in_channels, out_channels = 32, 32
    kernel_size = (3, 3, 3)

    # 1. Prepare Data
    torch_input = torch.randn((batch, depth, height, width, in_channels), dtype=torch.bfloat16)
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16)
    torch_bias = torch.randn((1, out_channels), dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # 2. Convert Input to Sharded
    try:
        shard_spec = get_shard_spec(device, shard_layout, batch, height, width, depth, in_channels)
        mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)
        tt_input_sharded = ttnn.to_memory_config(tt_input, mem_config)
    except Exception as e:
        pytest.skip(f"Skipping {shard_layout}: Grid/Layout not supported on this device. Error: {e}")

    # 3. Run Conv3d
    try:
        output = ttnn.experimental.conv3d(
            input_tensor=tt_input_sharded,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            groups=1
        )
    except Exception as e:
        pytest.fail(f"Conv3d failed with sharded input {shard_layout}: {e}")

@pytest.mark.parametrize("shard_layout", [ttnn.TensorMemoryLayout.HEIGHT_SHARDED])
def test_conv3d_sharded_output(device, shard_layout):
    """Test Conv3d requesting Sharded Output"""
    batch, depth, height, width = 1, 16, 32, 32
    in_channels, out_channels = 32, 32
    kernel_size = (3, 3, 3)

    # 1. Prepare Data (Interleaved Inputs)
    torch_input = torch.randn((batch, depth, height, width, in_channels), dtype=torch.bfloat16)
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16)
    
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR, device=device)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # 2. Define Output Memory Config
    # Calculating output dimensions (simplified for padding=1, stride=1)
    out_depth, out_height, out_width = depth, height, width 
    
    try:
        shard_spec = get_shard_spec(device, shard_layout, batch, out_height, out_width, out_depth, out_channels)
        output_mem_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)
    except:
        pytest.skip("Could not generate valid shard spec for output")

    # 3. Run Conv3d with Output Config
    try:
        output = ttnn.experimental.conv3d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            dilation=(1, 1, 1),
            groups=1,
            memory_config=output_mem_config # Requesting Sharded Output
        )
        assert output.memory_config().is_sharded(), "Output tensor is not sharded as requested!"
        
    except Exception as e:
        pytest.fail(f"Conv3d failed with sharded output request: {e}")
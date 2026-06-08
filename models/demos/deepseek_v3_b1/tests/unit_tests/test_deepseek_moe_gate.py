# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201, 512])
def test_deepseek_moe_gate(device, batch_size, enable_sigmoid, seed):
    """Test TTNN Deepseek Moe Gate operation on a 16x16 tile"""

    # Tensor dimensions - full 32x32 tile
    input_shape = (batch_size, 8, 32)
    reshaped_input_shape = (batch_size, 16, 16)
    input_shard_shape = (16, 16)
    input_tile = ttnn.Tile(input_shard_shape)
    output_shape = (batch_size, 1, 16)
    output_shard_shape = (1, 16)
    output_tile = ttnn.Tile(output_shard_shape)

    logger.info(f"Testing Deepseek Moe Gate with input shape {input_shape}")
    logger.info(f"Input tile size: {input_tile.tile_shape}")
    logger.info(f"Output tile size: {output_tile.tile_shape}")

    # Create input PyTorch tensor with random values
    torch.manual_seed(seed)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    if not enable_sigmoid:
        torch_input = torch.sigmoid(torch_input)
    torch_bias = torch.randn(input_shape, dtype=torch.bfloat16)
    eps = 1e-20
    scaling_factor = 2.5

    # Compute reference output using PyTorch
    top8_scores, top8_indices = DeepseekMoeGateSingleCore.golden(
        torch_input, torch_bias, eps, scaling_factor, enable_sigmoid
    )

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(
        batch_size,
        ttnn.CoreCoord(grid.x, grid.y),
        row_wise=True,
    )
    # Shard spec: single core
    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Create input values tensor sharded on single core
    reshaped_input = torch.reshape(torch_input, reshaped_input_shape)
    ttnn_input = ttnn.from_torch(
        reshaped_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    reshaped_bias = torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), -2, -1)
    ttnn_bias = ttnn.from_torch(
        reshaped_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    torch_input_indices = torch.arange(reshaped_input_shape[1] * reshaped_input_shape[2], dtype=torch.int32)
    torch_input_indices = torch_input_indices.unsqueeze(0).expand(reshaped_input_shape[0], -1)
    torch_input_indices = torch_input_indices.reshape(reshaped_input_shape)
    torch_input_indices = torch.transpose(torch_input_indices, -2, -1).to(torch.uint16)
    ttnn_input_indices = ttnn.from_torch(
        torch_input_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    # Create output tensor sharded on same core
    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    torch_output_indices = torch.zeros(output_shape, dtype=torch.uint16)
    ttnn_output_indices = ttnn.from_torch(
        torch_output_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    logger.info(
        f"Created tensors sharded on single core with input shard shape {input_shard_shape} and output shard shape {output_shard_shape}"
    )

    # Run Deepseek Moe Gate operation
    logger.info("Running Deepseek Moe Gate operation...")
    ttnn_result, ttnn_result_indices = DeepseekMoeGateSingleCore.op(
        ttnn_input,
        ttnn_bias,
        ttnn_output,
        ttnn_input_indices,
        ttnn_output_indices,
        eps,
        scaling_factor,
        enable_sigmoid,
    )

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)
    output_indices_torch = ttnn.to_torch(ttnn_result_indices)

    output_torch = output_torch[:, 0, :8]
    output_indices_torch = output_indices_torch[:, 0, :8]

    sorted_output_indices_torch, i = torch.sort(output_indices_torch, dim=-1)
    sorted_output_torch = torch.gather(output_torch, dim=-1, index=i)

    top8_indices, i = torch.sort(top8_indices, dim=-1)
    top8_scores = torch.gather(top8_scores, dim=-1, index=i)

    assert torch.equal(sorted_output_indices_torch.to(top8_indices.dtype), top8_indices), "Output indices do not match"
    assert torch.allclose(sorted_output_torch, top8_scores, atol=1e-2, rtol=1e-4), "Output scores do not match"


@pytest.mark.parametrize("start_index", [0x8000], ids=["v_0x8000"])
def test_deepseek_moe_gate_uint16_high_bit(device, start_index):
    """Regression test for #43149: reconfig from bfloat16 to uint16 must not zero values with bit 15 set.

    compute_kernel_hw_startup initialises the unpacker for bfloat16, leaving the
    zero-src flag enabled.  That flag maps the bit pattern 0x8000 (= -0.0f in
    bfloat16) to zero.  When the kernel reconfigures to uint16 without updating
    the flag, 0x8000 (= 32768 as uint16) is silently zeroed.

    This test feeds 256 uint16 indices starting from start_index (>= 0x8000)
    through the gate copy path and asserts they all survive intact.
    """
    input_shape = (1, 8, 32)
    reshaped_input_shape = (1, 16, 16)
    input_shard_shape = (16, 16)
    input_tile = ttnn.Tile(input_shard_shape)
    output_shape = (1, 1, 16)
    output_shard_shape = (1, 16)
    output_tile = ttnn.Tile(output_shard_shape)

    torch_input = torch.full(input_shape, 0.5, dtype=torch.bfloat16)

    # Rig bias so top-8 picks are columns [1,4,7,11,15,19,23,28]; column 1 maps
    # to post-transpose position [1, 0] in the indices tile (index = start_index + 1).
    bias_2d = torch.full(input_shape, -100.0, dtype=torch.bfloat16)
    for col in (1, 4, 7, 11, 15, 19, 23, 28):
        bias_2d[0, 0, col] = 100.0

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(1, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    input_shard_spec = ttnn.ShardSpec(core_grid, input_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    output_shard_spec = ttnn.ShardSpec(core_grid, output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch.reshape(torch_input, reshaped_input_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_cfg,
        tile=input_tile,
    )
    ttnn_bias = ttnn.from_torch(
        torch.transpose(torch.reshape(bias_2d, reshaped_input_shape), -2, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_cfg,
        tile=input_tile,
    )

    # Build uint16 indices: start_index, start_index+1, ..., start_index+255,
    # laid out as (1,16,16) then transposed (same order the kernel expects).
    raw = torch.arange(start_index, start_index + 256, dtype=torch.int32)
    torch_indices = torch.transpose(raw.reshape(reshaped_input_shape), -2, -1).to(torch.uint16)

    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_cfg,
        tile=input_tile,
    )

    ttnn_output = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_cfg,
        tile=output_tile,
    )
    ttnn_output_indices = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_cfg,
        tile=output_tile,
    )

    _, result_indices = DeepseekMoeGateSingleCore.op(
        ttnn_input, ttnn_bias, ttnn_output, ttnn_indices, ttnn_output_indices, 1e-20, 1.0, False
    )

    # The gate outputs top-8 indices in the first 8 slots; remaining slots are padding.
    top8 = ttnn.to_torch(result_indices)[:, 0, :8].to(torch.int32)
    assert top8.numel() == 8, f"Expected 8 top-k indices, got {top8.numel()}"
    assert (top8 >= start_index).all(), (
        f"Detected uint16 high-bit truncation: some index < {start_index:#x} in output.\n"
        f"Got top-8: {top8.flatten().tolist()}"
    )

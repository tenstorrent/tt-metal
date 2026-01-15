# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore


@pytest.mark.parametrize("enable_sigmoid", [True, False])
@pytest.mark.parametrize("seed", [42, 201, 512])
def test_deepseek_moe_gate(device, enable_sigmoid, seed):
    """Test TTNN Deepseek Moe Gate operation on a 16x16 tile"""

    # Tensor dimensions - full 32x32 tile
    input_shape = (8, 32)
    reshaped_input_shape = (16, 16)
    tile = ttnn.Tile([16, 16])

    logger.info(f"Testing Deepseek Moe Gate with input shape {input_shape}")
    logger.info(f"Tile size: {tile.tile_shape}")

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

    # Shard spec: single core
    shard_shape = reshaped_input_shape
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Create input values tensor sharded on single core
    reshaped_input = torch.reshape(torch_input, reshaped_input_shape)
    ttnn_input = ttnn.from_torch(
        reshaped_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    reshaped_bias = torch.transpose(torch.reshape(torch_bias, reshaped_input_shape), 0, 1)
    ttnn_bias = ttnn.from_torch(
        reshaped_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    torch_input_indices = torch.arange(reshaped_input_shape[0] * reshaped_input_shape[1], dtype=torch.int32)
    torch_input_indices = torch_input_indices.reshape(reshaped_input_shape)
    torch_input_indices = torch.transpose(torch_input_indices, 0, 1).to(torch.uint16)

    ttnn_input_indices = ttnn.from_torch(
        torch_input_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    # Create output tensor sharded on same core
    torch_output = torch.zeros(reshaped_input_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    torch_output_indices = torch.zeros(reshaped_input_shape, dtype=torch.uint16)
    ttnn_output_indices = ttnn.from_torch(
        torch_output_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    logger.info(f"Created tensors sharded on single core with shard shape {shard_shape}")

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

    output_torch = output_torch[0, :8]
    output_indices_torch = output_indices_torch[0, :8]

    sorted_output_indices_torch, i = torch.sort(output_indices_torch, dim=-1)
    sorted_output_torch = output_torch[i]

    top8_indices, i = torch.sort(top8_indices, dim=-1)
    top8_scores = top8_scores[i]

    assert torch.equal(sorted_output_indices_torch, top8_indices), "Output indices do not match"
    assert torch.allclose(sorted_output_torch, top8_scores, atol=1e-2, rtol=1e-4), "Output scores do not match"

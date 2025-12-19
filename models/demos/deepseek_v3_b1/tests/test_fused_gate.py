# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Fused Gate Test
Tests fused gate (sigmoid) operation with shape [1, N]
Input and output are sharded on a single core
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.fused_gate.op import FusedGateSingleCore


@pytest.mark.parametrize(
    "width",
    [
        32,
    ],
)
def test_fused_gate(device, width):
    """Test TTNN fused gate operation on a single core"""

    # Tensor dimensions
    # Change to 1x32 once tiny tiles are supported
    input_shape = (32, width)
    output_shape = (32, width * 2)
    tile = ttnn.Tile([32, 32])

    # Validate that width is a valid tile size
    assert width % tile.tile_shape[1] == 0, f"Width {width} must be divisible by tile width {tile.tile_shape[1]}"
    logger.info(f"Testing fused gate with input shape {input_shape}, output shape {output_shape}")
    logger.info(f"Tile size: {tile.tile_shape}")

    # Create input PyTorch tensor
    torch.manual_seed(0)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_bias = torch.randn(input_shape, dtype=torch.bfloat16)

    # Compute reference output using PyTorch
    torch_expected = FusedGateSingleCore.golden(torch_input, torch_bias)

    # Shard spec: single core
    input_shard_shape = input_shape
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    output_shard_shape = output_shape
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Create input tensor sharded on single core
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )

    # Create bias tensor sharded on same core
    ttnn_bias = ttnn.from_torch(
        torch_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )

    # Create output tensor sharded on same core
    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    logger.info(
        f"Created tensors sharded on single core with input shard shape {input_shard_shape}, output shard shape {output_shard_shape}"
    )

    # Run fused gate operation
    logger.info("Running fused gate operation...")
    ttnn_result = FusedGateSingleCore.op(
        ttnn_input,
        ttnn_bias,
        ttnn_output,
    )

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == output_shape, f"Expected shape {output_shape}, got {output_torch.shape}"

    # Split output into sigmoid and sigmoid+bias components
    sigmoid_output = output_torch[:, :width]
    sigmoid_add_bias_output = output_torch[:, width:]

    # Compute expected values separately
    expected_sigmoid = torch.sigmoid(torch_input)
    expected_sigmoid_add_bias = expected_sigmoid + torch_bias

    # Verify sigmoid results
    logger.info("Verifying sigmoid results...")
    sigmoid_max_diff = torch.max(torch.abs(sigmoid_output - expected_sigmoid)).item()
    sigmoid_mean_diff = torch.mean(torch.abs(sigmoid_output - expected_sigmoid)).item()
    logger.info(f"Sigmoid - Max absolute difference: {sigmoid_max_diff}")
    logger.info(f"Sigmoid - Mean absolute difference: {sigmoid_mean_diff}")

    sigmoid_passing, sigmoid_pcc_message = comp_pcc(expected_sigmoid, sigmoid_output, 0.9999)
    logger.info(f"Sigmoid - {sigmoid_pcc_message}")
    assert sigmoid_passing, f"Sigmoid validation failed: {sigmoid_pcc_message}"
    logger.info("✓ Sigmoid test passed!")

    # Verify sigmoid + bias results
    logger.info("Verifying sigmoid + bias results...")
    add_bias_max_diff = torch.max(torch.abs(sigmoid_add_bias_output - expected_sigmoid_add_bias)).item()
    add_bias_mean_diff = torch.mean(torch.abs(sigmoid_add_bias_output - expected_sigmoid_add_bias)).item()
    logger.info(f"Sigmoid + Bias - Max absolute difference: {add_bias_max_diff}")
    logger.info(f"Sigmoid + Bias - Mean absolute difference: {add_bias_mean_diff}")

    add_bias_passing, add_bias_pcc_message = comp_pcc(expected_sigmoid_add_bias, sigmoid_add_bias_output, 0.9999)
    logger.info(f"Sigmoid + Bias - {add_bias_pcc_message}")
    assert add_bias_passing, f"Sigmoid + Bias validation failed: {add_bias_pcc_message}"
    logger.info("✓ Sigmoid + Bias test passed!")

    # Verify combined fused gate results
    logger.info("Verifying combined fused gate results...")
    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.9999)
    logger.info(f"Combined - {pcc_message}")
    assert passing, f"Combined fused gate validation failed: {pcc_message}"

    logger.info("✓ All fused gate tests passed!")

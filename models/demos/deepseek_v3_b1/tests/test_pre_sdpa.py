# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN PreSDPA Test
Tests pre-SDPA fused operation (RMSNorm)
Input, gamma, and output are sharded on a single core
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.pre_sdpa.op import PreSDPA


@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [False])
def test_pre_sdpa(device, epsilon, use_fp32):
    """Test TTNN pre-SDPA fused operation (RMSNorm)"""

    # Input tensor shapes
    shape = (1, 7168)
    matmul_weights_shape = (7168, 1536)

    tile = ttnn.Tile([1, 32])

    # Create input and gamma PyTorch tensors
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape, dtype=torch.bfloat16)
    torch_matmul_weights = torch.randn(matmul_weights_shape, dtype=torch.bfloat16)

    # Shard spec: single core for input, gamma
    shard_shape = shape
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    # Create input tensor sharded on single core
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    # Create gamma tensor sharded on same core
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    # Create matmul weights tensor - width sharded on 6x8 grid (48 cores)
    matmul_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 7))  # x=6, y=8
    num_matmul_cores = 6 * 8  # 48 cores
    matmul_shard_shape = (matmul_weights_shape[0], matmul_weights_shape[1] // num_matmul_cores)  # (7168, 32)
    matmul_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({matmul_grid}),
        matmul_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    matmul_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, matmul_shard_spec)

    ttnn_matmul_weights = ttnn.from_torch(
        torch_matmul_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=matmul_mem_config,
    )

    # Compute reference output using PyTorch
    torch_expected = PreSDPA.golden(torch_input, torch_gamma, num_output_cores=1, epsilon=epsilon)

    # Create output tensor sharded on same core
    torch_output = torch.zeros(shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    logger.info(f"Created tensors sharded on single core with shard shape {shard_shape}")

    # Run pre-SDPA operation
    logger.info("Running pre-SDPA operation...")
    ttnn_result = PreSDPA.op(
        ttnn_input,
        ttnn_gamma,
        ttnn_matmul_weights,
        ttnn_output,
        epsilon=epsilon,
        fp32_dest_acc_en=use_fp32,
    )

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == shape, f"Expected shape {shape}, got {output_torch.shape}"

    # Verify results
    logger.info("Verifying pre-SDPA results...")

    # Check if outputs are close (allowing for numerical precision differences)
    # bfloat16 has limited precision, so we use a relatively loose tolerance
    max_diff = torch.max(torch.abs(output_torch - torch_expected)).item()
    mean_diff = torch.mean(torch.abs(output_torch - torch_expected)).item()

    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info("✓ PreSDPA test passed!")

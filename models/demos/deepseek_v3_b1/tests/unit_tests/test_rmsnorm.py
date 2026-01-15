# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN RMSNorm Test
Tests rmsnorm operation with shape [1, N]
Input, gamma, and output are sharded on a single core
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.rmsnorm.op import RMSNormSingleCore


@pytest.mark.parametrize(
    "width",
    [
        7168,  # input_layernorm, post_attention_layernorm
        1536,  # q_a_layernorm
        512,  # kv_a_layernorm
    ],
)
@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("use_fp32", [True, False])
def test_rmsnorm(device, width, epsilon, use_fp32):
    """Test TTNN rmsnorm operation on a single core"""

    # Tensor dimensions
    shape = (1, width)
    tile = ttnn.Tile([1, 32])

    # Validate that width is a valid tile size
    assert width % tile.tile_shape[1] == 0, f"Width {width} must be divisible by tile width {tile.tile_shape[1]}"
    logger.info(f"Testing rmsnorm with shape {shape}, epsilon={epsilon}")
    logger.info(f"Tile size: {tile.tile_shape}")

    # Create input and gamma PyTorch tensors
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape, dtype=torch.bfloat16)

    # Compute reference output using PyTorch
    torch_expected = RMSNormSingleCore.golden(torch_input, torch_gamma, epsilon=epsilon)

    # Shard spec: single core
    shard_shape = (shape[0], width)
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

    # Create output tensor sharded on same core
    torch_output = torch.zeros((shape[0], width), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        tile=tile,
    )

    logger.info(f"Created tensors sharded on single core with shard shape {shard_shape}")

    # Run rmsnorm operation
    logger.info("Running rmsnorm operation...")
    ttnn_result = RMSNormSingleCore.op(
        ttnn_input,
        ttnn_gamma,
        ttnn_output,
        epsilon=epsilon,
        numel=torch_input.numel(),
        fp32_dest_acc_en=use_fp32,
    )

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)
    output_torch = output_torch[:, :width]

    # Verify output shape
    assert output_torch.shape == shape, f"Expected shape {shape}, got {output_torch.shape}"

    # Verify rmsnorm results
    logger.info("Verifying rmsnorm results...")

    # Check if outputs are close (allowing for numerical precision differences)
    # bfloat16 has limited precision, so we use a relatively loose tolerance
    max_diff = torch.max(torch.abs(output_torch - torch_expected)).item()
    mean_diff = torch.mean(torch.abs(output_torch - torch_expected)).item()

    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.999)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info("✓ RMSNorm test passed!")

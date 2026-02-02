# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for MoE Routed Expert fused operation.

Tests the fused operation:
- Input: [1, 7168] tensor on sender core (outside compute grid)
- Mcast from sender to 8 compute cores
- Each compute core: [1, 7168] @ [7168, 32] -> [1, 32] + sigmoid
- Output: [1, 256] width-sharded across 8 compute cores
"""

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.moe_routed_expert.op import MoeRoutedExpert


def test_moe_routed_expert(device):
    """Test MoE routed expert fused operation"""

    # MoE router: [1, 7168] x [7168, 256] with 8 cores
    M = 1
    K = 7168
    N_per_core = 32
    num_cores = 8
    N = N_per_core * num_cores  # 256 total output width

    # Tile definitions
    tile_1x32 = ttnn.Tile([1, 32])
    tile_32x32 = ttnn.Tile([32, 32])  # For weights

    logger.info(f"Testing MoE routed expert: [{M}, {K}] x [{K}, {N}] with {num_cores} cores")

    # Create input and weights tensors
    torch.manual_seed(0)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)

    # Compute reference output
    torch_expected = MoeRoutedExpert.golden(torch_input.float(), torch_weights.float()).bfloat16()

    # Define core grid for compute (first column, 8 cores)
    compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))])

    # Input tensor: sharded on sender core OUTSIDE the compute grid
    # Same location as pre_sdpa mcast sender: (device_grid_x - 1, 9)
    device_grid_size = device.compute_with_storage_grid_size()
    input_core = ttnn.CoreCoord(device_grid_size.x - 1, 9)
    input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(input_core, input_core)])
    input_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        (M, K),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile_1x32,
    )
    logger.info(f"Created input tensor with shard shape ({M}, {K}) on core ({input_core.x}, {input_core.y})")

    # Weights: width-sharded across 8 cores
    # Each core gets [K, N_per_core] = [7168, 32]
    weights_shard_spec = ttnn.ShardSpec(
        compute_core_grid,
        (K, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights_shard_spec
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=tile_32x32,
    )
    logger.info(f"Created weights tensor with shard shape ({K}, {N_per_core}) on {num_cores} cores")

    # Output tensor: width-sharded across 8 cores
    # Each core produces [1, N_per_core] = [1, 32]
    output_shard_spec = ttnn.ShardSpec(
        compute_core_grid,
        (M, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile_1x32,
    )
    logger.info(f"Created output tensor with shard shape ({M}, {N_per_core}) on {num_cores} cores")

    # Run fused operation
    logger.info("Running MoE routed expert fused operation...")
    ttnn_result = MoeRoutedExpert.op(
        ttnn_input,
        ttnn_weights,
        ttnn_output,
        fp32_dest_acc_en=True,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    # Verify results
    logger.info("Verifying results...")
    pcc_threshold = 0.98  # Slightly lower due to sigmoid approximation
    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info("MoE routed expert fused operation test passed!")

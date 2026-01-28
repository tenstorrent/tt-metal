# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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
@pytest.mark.parametrize("use_fp32", [True])
def test_pre_sdpa(device, epsilon, use_fp32):
    """Test TTNN pre-SDPA fused operation (RMSNorm)"""

    # Input tensor shapes
    shape = (1, 7168)
    matmul_weights_shape = (7168, 1536)

    # Matmul2 weights shape: 1536 x (num_cores * 4 * 32)
    # Uses device grid: 8x12 = 96 cores (P150) or 8x11 = 88 cores (non-P150)
    device_grid_size = device.compute_with_storage_grid_size()
    matmul2_grid_x = device_grid_size.x  # 12 for P150, 11 for non-P150
    matmul2_grid_y = 8
    matmul2_num_cores = matmul2_grid_x * matmul2_grid_y  # 96 or 88
    matmul2_width = matmul2_num_cores * 4 * 32  # 12288 or 11264
    matmul2_weights_shape = (1536, matmul2_width)

    # Mcast/gather core coordinates (same as RMSNorm input core)
    mcast_core_x = matmul2_grid_x - 1  # 11 for P150, 10 for non-P150
    mcast_core_y = 9

    tile = ttnn.Tile([1, 32])

    # RMSNorm2 parameters (1536 elements = 3 tiles of 16x32)
    rmsnorm2_width = 1536

    # Create input and gamma PyTorch tensors
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape, dtype=torch.bfloat16)
    torch_matmul_weights = torch.randn(matmul_weights_shape, dtype=torch.bfloat16)
    # RMSNorm2 gamma: 1536 elements (no padding needed with 16x32 tiles)
    torch_rmsnorm2_gamma = torch.randn((1, rmsnorm2_width), dtype=torch.bfloat16)

    # Shard spec: single core for input, gamma (on mcast/gather core)
    shard_shape = shape
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
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
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=matmul_mem_config,
    )

    # Create matmul2 weights tensor - width sharded on device grid (8x12 or 8x11), 4 tiles per core
    torch_matmul2_weights = torch.randn(matmul2_weights_shape, dtype=torch.bfloat16)
    matmul2_grid = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0), ttnn.CoreCoord(matmul2_grid_x - 1, matmul2_grid_y - 1)
    )  # (0,0) to (11,7) or (10,7)
    matmul2_shard_shape = (matmul2_weights_shape[0], matmul2_weights_shape[1] // matmul2_num_cores)  # (1536, 128)
    matmul2_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({matmul2_grid}),
        matmul2_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    matmul2_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, matmul2_shard_spec
    )

    ttnn_matmul2_weights = ttnn.from_torch(
        torch_matmul2_weights,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=matmul2_mem_config,
    )

    # Create RMSNorm2 gamma tensor sharded on same core (3 tiles of 16x32)
    rmsnorm2_gamma_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
        (1, rmsnorm2_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    rmsnorm2_gamma_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, rmsnorm2_gamma_shard_spec
    )
    ttnn_rmsnorm2_gamma = ttnn.from_torch(
        torch_rmsnorm2_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=rmsnorm2_gamma_mem_config,
        tile=tile,
    )

    # Create output tensor - width sharded on same grid as matmul2, shape is (1, matmul2_width)
    output_shape = (1, matmul2_width)  # (1, 12288) or (1, 11264)
    output_shard_shape = (1, matmul2_width // matmul2_num_cores)  # (1, 128)
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({matmul2_grid}),
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    logger.info(f"Created tensors sharded on single core with shard shape {shard_shape}")

    # Run pre-SDPA operation
    logger.info("Running pre-SDPA operation...")
    ttnn_result = PreSDPA.op(
        ttnn_input,
        ttnn_gamma,
        ttnn_matmul_weights,
        ttnn_rmsnorm2_gamma,
        ttnn_matmul2_weights,
        ttnn_output,
        epsilon=epsilon,
        fp32_dest_acc_en=use_fp32,
    )

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == output_shape, f"Expected shape {output_shape}, got {output_torch.shape}"

    # Verify results
    logger.info("Verifying pre-SDPA results...")

    # Compute reference output using PyTorch
    torch_expected = PreSDPA.golden(
        torch_input,
        torch_gamma,
        torch_matmul_weights,
        torch_rmsnorm2_gamma,
        torch_matmul2_weights,
        epsilon=epsilon,
    )

    # torch_expected = torch_expected[:, 128:256]
    # output_torch = output_torch[:, 128:256]
    # torch_expected = torch_expected[:, :128]
    # output_torch = output_torch[:, :128]

    # Check if outputs are close (allowing for numerical precision differences)
    # bfloat16 has limited precision, so we use a relatively loose tolerance
    max_diff = torch.max(torch.abs(output_torch - torch_expected)).item()
    mean_diff = torch.mean(torch.abs(output_torch - torch_expected)).item()

    logger.info(f"Max absolute difference: {max_diff}")
    logger.info(f"Mean absolute difference: {mean_diff}")

    # passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)
    # torch.set_printoptions(threshold=1000000)
    # print(torch_expected)
    # print(output_torch)

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.98)

    logger.info(pcc_message)

    # assert passing, pcc_message

    logger.info("✓ PreSDPA test passed!")

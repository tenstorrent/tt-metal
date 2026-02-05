# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Post SDPA Fused Op Test

Tests the full post_sdpa fused operation which implements:
- Matmul1: [1, 512] x [512, 128] -> [1, 128] per core on 64 cores (8x8)
- Gather1: Collect to [1, 8192] on gather core (12, 9)
- Mcast: Broadcast [1, 8192] to 117 cores (13x9 rectangular grid)
- Matmul2: [1, 8192] x [8192, 64] -> [1, 64] per core on 112 active cores
- Gather2: Collect to [1, 7168] on gather core (12, 9)

The mcast grid (13x9=117 cores) includes 5 inactive cores (row 8, cols 8-12)
that receive mcast data but skip matmul2 via is_matmul2_core=false.

Full operation: [1, 512] @ [512, 8192] @ [8192, 7168] -> [1, 7168]
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.post_sdpa.op import PostSDPA


@pytest.mark.parametrize(
    "M, K1, intermediate, K2, output_size, in0_dtype, in1_dtype",
    [
        # Main target shape: [1,512] @ [512,8192] @ [8192,7168] -> [1,7168]
        (1, 512, 8192, 8192, 7168, ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_post_sdpa(device, M, K1, intermediate, K2, output_size, in0_dtype, in1_dtype):
    """Test full post_sdpa fused operation"""

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # 1x32 tiles for input/activation
    b_tile = ttnn.Tile([32, 32])  # 32x32 tiles for weights

    # ========================================================================
    # Grid configuration
    # ========================================================================
    # Matmul1 grid: 8x8 = 64 cores
    MATMUL1_GRID_X = 8
    MATMUL1_GRID_Y = 8
    num_matmul1_cores = MATMUL1_GRID_X * MATMUL1_GRID_Y  # 64

    # Mcast grid: 13x9 = 117 cores (rectangular for efficient mcast)
    MCAST_GRID_X = 13
    MCAST_GRID_Y = 9
    num_mcast_cores = MCAST_GRID_X * MCAST_GRID_Y  # 117

    # Active Matmul2 cores: 112 (rows 0-7 full 13 cols + row 8 cols 0-7)
    # Non-rectangular grid: 13*8 + 8 = 104 + 8 = 112
    num_matmul2_cores = 112

    # Per-core dimensions
    n1_per_core = intermediate // num_matmul1_cores  # 8192 / 64 = 128
    n2_per_core = output_size // num_matmul2_cores  # 7168 / 112 = 64

    logger.info(f"Testing full post_sdpa fused op:")
    logger.info(f"  Matmul1: [{M}, {K1}] x [{K1}, {intermediate}] on {num_matmul1_cores} cores")
    logger.info(f"  Mcast: [{M}, {intermediate}] to {num_mcast_cores} cores (13x9 grid)")
    logger.info(f"  Matmul2: [{M}, {K2}] x [{K2}, {output_size}] on {num_matmul2_cores} active cores")
    logger.info(f"  Output: [{M}, {output_size}]")

    # Create core grids
    matmul1_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(MATMUL1_GRID_X - 1, MATMUL1_GRID_Y - 1))]
    )
    # Active matmul2 cores: non-rectangular grid (112 cores)
    # - Rows 0-7: all 13 columns = 104 cores
    # - Row 8: columns 0-7 = 8 cores
    matmul2_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 7)),  # 13x8 = 104 cores
            ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8)),  # 8x1 = 8 cores
        ]
    )
    gather_core = ttnn.CoreCoord(12, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    # ========================================================================
    # Create PyTorch tensors
    # ========================================================================
    torch.manual_seed(0)

    # Input: [1, 512] - replicated to 64 matmul1 cores
    torch_input_single = torch.randn((M, K1), dtype=torch.bfloat16)
    torch_input = torch_input_single.repeat(num_matmul1_cores, 1)  # [64, 512]

    # Weights1: [512, 8192]
    torch_weights1 = torch.randn((K1, intermediate), dtype=torch.bfloat16)

    # Weights2: [8192, 7168]
    torch_weights2 = torch.randn((K2, output_size), dtype=torch.bfloat16)

    # ========================================================================
    # Compute golden reference: input @ weights1 @ weights2
    # ========================================================================
    torch_expected = PostSDPA.golden(
        torch_input_single.float(), torch_weights1.float(), torch_weights2.float()
    ).bfloat16()
    logger.info(f"Golden output shape: {torch_expected.shape}")

    # ========================================================================
    # Create input tensor (height-sharded across matmul1 cores)
    # Each core gets [1, 512]
    # ========================================================================
    input_shard_shape = (M, K1)  # [1, 512] per core
    input_shard_spec = ttnn.ShardSpec(
        matmul1_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=a_tile,
    )
    logger.info(f"Created input tensor: shard {input_shard_shape} on {num_matmul1_cores} cores")

    # ========================================================================
    # Create weights1 tensor (width-sharded across matmul1 cores)
    # Each core gets [512, 128]
    # ========================================================================
    weights1_shard_shape = (K1, n1_per_core)  # [512, 128] per core
    weights1_shard_spec = ttnn.ShardSpec(
        matmul1_grid,
        weights1_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights1_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights1_shard_spec
    )

    ttnn_weights1 = ttnn.from_torch(
        torch_weights1,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights1_mem_config,
        tile=b_tile,
    )
    logger.info(f"Created weights1 tensor: shard {weights1_shard_shape} on {num_matmul1_cores} cores")

    # ========================================================================
    # Create weights2 tensor (width-sharded across 112 active matmul2 cores)
    # Each core gets [8192, 64]
    # ========================================================================
    weights2_shard_shape = (K2, n2_per_core)  # [8192, 64] per core
    weights2_shard_spec = ttnn.ShardSpec(
        matmul2_grid,  # Non-rectangular grid of 112 active cores
        weights2_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights2_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights2_shard_spec
    )

    ttnn_weights2 = ttnn.from_torch(
        torch_weights2,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights2_mem_config,
        tile=b_tile,
    )
    logger.info(f"Created weights2 tensor: shard {weights2_shard_shape} on {num_matmul2_cores} active cores")

    # ========================================================================
    # Create gather1 output tensor (intermediate [1, 8192] on gather core)
    # ========================================================================
    gather1_output_shard_shape = (M, intermediate)  # [1, 8192]
    gather1_output_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        gather1_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gather1_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gather1_output_shard_spec
    )

    torch_gather1_zeros = torch.zeros((M, intermediate), dtype=torch.bfloat16)
    ttnn_gather1_output = ttnn.from_torch(
        torch_gather1_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gather1_output_mem_config,
        tile=a_tile,
    )
    logger.info(f"Created gather1 output tensor: {gather1_output_shard_shape} on gather core")

    # ========================================================================
    # Create final output tensor ([1, 7168] on gather core)
    # ========================================================================
    output_shard_shape = (M, output_size)  # [1, 7168]
    output_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output_zeros = torch.zeros((M, output_size), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=a_tile,
    )
    logger.info(f"Created output tensor: {output_shard_shape} on gather core")

    # ========================================================================
    # Run fused operation
    # ========================================================================
    logger.info("Running full post_sdpa fused operation...")
    ttnn_result = PostSDPA.op(
        ttnn_input,
        ttnn_weights1,
        ttnn_weights2,
        ttnn_gather1_output,
        ttnn_output,
        fp32_dest_acc_en=False,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)
    logger.info(f"Output shape: {output_torch.shape}")

    # ========================================================================
    # Verify results
    # ========================================================================
    # Output should be [1, 7168] on gather core
    expected_shape = (M, output_size)
    assert output_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {output_torch.shape}"

    # Compare with golden reference
    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.99)
    logger.info(f"PCC comparison: {pcc_message}")

    assert passing, f"PCC check failed: {pcc_message}"
    logger.info("✓ Post SDPA full fused op test passed!")

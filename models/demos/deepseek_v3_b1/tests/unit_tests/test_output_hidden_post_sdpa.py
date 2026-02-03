# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Output Hidden Post SDPA Fused Op Test

Tests the output_hidden_post_sdpa fused operation which implements:
- Matmul: [1, 512] x [512, 128] -> [1, 128]
- Distributed across 64 cores (8x8 grid)
- Each core computes [1, 512] x [512, 128] -> [1, 128]
- Gather collects results to core (11, 9) -> [1, 8192]
- Mcast broadcasts result to 8x12 grid (96 cores)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.output_hidden_post_sdpa.op import OutputHiddenPostSDPA


@pytest.mark.parametrize(
    "M, K, N, in0_dtype, in1_dtype",
    [
        # Main target shape
        (1, 512, 8192, ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_output_hidden_post_sdpa(device, M, K, N, in0_dtype, in1_dtype):
    """Test output_hidden_post_sdpa fused operation with distributed matmul + gather + mcast"""

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # 1x32 tiles for input
    b_tile = ttnn.Tile([32, 32])  # 32x32 tiles for weights
    out_tile = ttnn.Tile([M, 32])  # 1x32 tiles for output

    # Grid configuration: 8x8 = 64 cores for matmul
    MATMUL_GRID_X = 8  # columns
    MATMUL_GRID_Y = 8  # rows
    num_matmul_cores = MATMUL_GRID_X * MATMUL_GRID_Y  # 64

    # Mcast grid: 8x12 = 96 cores
    MCAST_GRID_X = 12  # columns
    MCAST_GRID_Y = 8  # rows
    num_mcast_cores = MCAST_GRID_X * MCAST_GRID_Y  # 96

    # Calculate per-core dimensions
    assert N % num_matmul_cores == 0, f"N ({N}) must be divisible by num_matmul_cores ({num_matmul_cores})"
    n_per_core = N // num_matmul_cores  # 8192 / 64 = 128

    # Calculate tiles
    num_tiles_k = K // a_tile.tile_shape[1]  # 512 / 32 = 16
    num_tiles_n_per_core = n_per_core // b_tile.tile_shape[1]  # 128 / 32 = 4

    logger.info(f"Testing output_hidden_post_sdpa fused op with shape [{M}, {K}] x [{K}, {N}]")
    logger.info(f"Matmul grid: {MATMUL_GRID_X}x{MATMUL_GRID_Y} = {num_matmul_cores} cores")
    logger.info(f"Mcast grid: {MCAST_GRID_X}x{MCAST_GRID_Y} = {num_mcast_cores} cores")
    logger.info(f"Per-core: K={num_tiles_k} tiles, N={num_tiles_n_per_core} tiles")

    # Create matmul core grid
    matmul_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(MATMUL_GRID_X - 1, MATMUL_GRID_Y - 1))]
    )

    # Gather receiver core: (11, 9)
    gather_core = ttnn.CoreCoord(11, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    # Mcast grid: 8x12 = 96 cores (x=0-7, y=0-11)
    mcast_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(MCAST_GRID_X - 1, MCAST_GRID_Y - 1))]
    )

    # Create input and weights PyTorch tensors
    torch.manual_seed(0)
    # Input: [1, 512] - same input on all cores for simplicity
    # (In actual usage, there may be 8 unique shards, each replicated to 8 cores)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_input = torch_input.repeat(num_matmul_cores, 1)  # Replicate for 64 cores
    logger.info(f"Input shape: {torch_input.shape}")

    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)

    # Compute reference output using PyTorch
    # Note: Each core gets the same [1, 512] input, so we compute golden reference
    # using only the first row (M=1), not the full [64, 512] tensor
    torch_input_single = torch_input[0:1, :]  # Take first row: [1, 512]
    torch_expected = OutputHiddenPostSDPA.golden(torch_input_single.float(), torch_weights.float()).bfloat16()

    # ========================================================================
    # Create input tensor (height-sharded across matmul cores)
    # 8 unique shards, each replicated to 8 cores
    # Each core gets [1, 512] input
    # ========================================================================
    input_shard_shape = (M, K)  # [1, 512] per core
    input_shard_spec = ttnn.ShardSpec(
        matmul_grid,
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

    logger.info(f"Created input tensor with shard shape {input_shard_shape}")

    # ========================================================================
    # Create weights tensor (width-sharded across matmul cores)
    # Each core gets [512, 128]
    # ========================================================================
    weights_shard_shape = (K, n_per_core)  # [512, 128] per core
    weights_shard_spec = ttnn.ShardSpec(
        matmul_grid,
        weights_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights_shard_spec
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    logger.info(f"Created weights tensor with shard shape {weights_shard_shape}")

    # ========================================================================
    # Create gather output tensor (height-sharded on gather core)
    # This is an intermediate tensor for the gather result
    # ========================================================================
    gather_output_shard_shape = (M, N)  # Full output on gather core: [1, 8192]
    gather_output_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        gather_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gather_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gather_output_shard_spec
    )

    torch_gather_output_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_gather_output = ttnn.from_torch(
        torch_gather_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gather_output_mem_config,
        tile=out_tile,
    )

    logger.info(
        f"Created gather output tensor with shard shape {gather_output_shard_shape} on core ({gather_core.x}, {gather_core.y})"
    )

    # ========================================================================
    # Create mcast output tensor (height-sharded on mcast grid)
    # Each core in mcast grid gets [1, 8192]
    # Total: 96 cores * [1, 8192] = [96, 8192]
    # ========================================================================
    mcast_output_shard_shape = (M, N)  # [1, 8192] per core
    mcast_output_shard_spec = ttnn.ShardSpec(
        mcast_grid,
        mcast_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mcast_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, mcast_output_shard_spec
    )

    # Create tensor with shape [num_mcast_cores, N] to hold all mcast outputs
    torch_mcast_output_zeros = torch.zeros((num_mcast_cores, N), dtype=torch.bfloat16)
    ttnn_mcast_output = ttnn.from_torch(
        torch_mcast_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mcast_output_mem_config,
        tile=out_tile,
    )

    logger.info(f"Created mcast output tensor with shard shape {mcast_output_shard_shape} on {num_mcast_cores} cores")

    # ========================================================================
    # Run output_hidden_post_sdpa fused operation
    # ========================================================================
    logger.info("Running output_hidden_post_sdpa fused operation...")
    ttnn_result = OutputHiddenPostSDPA.op(
        ttnn_input,
        ttnn_weights,
        ttnn_gather_output,
        ttnn_mcast_output,
        fp32_dest_acc_en=False,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape - should be [num_mcast_cores, N] since mcast broadcasts to all cores
    expected_shape = (num_mcast_cores, N)
    assert output_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {output_torch.shape}"

    # Verify results - each mcast core should have the same output as torch_expected
    logger.info("Verifying output_hidden_post_sdpa results...")

    # Compare first core's output with expected
    first_core_output = output_torch[0:1, :]
    passing, pcc_message = comp_pcc(torch_expected, first_core_output, 0.99)
    logger.info(f"First core PCC: {pcc_message}")

    assert passing, f"First core failed: {pcc_message}"

    # Verify all cores received the same data (mcast broadcast)
    for i in range(1, num_mcast_cores):
        core_output = output_torch[i : i + 1, :]
        core_passing, core_pcc = comp_pcc(first_core_output, core_output, 0.9999)
        if not core_passing:
            logger.warning(f"Core {i} differs from core 0: {core_pcc}")
        assert core_passing, f"Core {i} differs from core 0: {core_pcc}"

    logger.info(f"✓ All {num_mcast_cores} mcast cores verified!")
    logger.info("✓ Output hidden post SDPA fused op test passed!")

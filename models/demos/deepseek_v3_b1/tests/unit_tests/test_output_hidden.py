# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Output Hidden Fused Op Test

Tests the output_hidden fused operation which implements:
- Matmul: [1, 8192] x [8192, 6144] -> [1, 6144]
- Distributed across 96 cores (8x12 grid)
- Each core computes [1, 8192] x [8192, 64] -> [1, 64]
- Gather collects results to core (9, 11)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.output_hidden.op import OutputHidden


@pytest.mark.parametrize(
    "M, K, N, in0_dtype, in1_dtype",
    [
        # Main target shape
        (1, 8192, 6144, ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_output_hidden(device, M, K, N, in0_dtype, in1_dtype):
    """Test output_hidden fused operation with distributed matmul + gather"""

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # 1x32 tiles for input
    b_tile = ttnn.Tile([32, 32])  # 32x32 tiles for weights
    out_tile = ttnn.Tile([M, 32])  # 1x32 tiles for output

    # Grid configuration: 8x12 = 96 cores
    GRID_X = 12  # columns
    GRID_Y = 8  # rows
    num_cores = GRID_X * GRID_Y  # 96

    # Calculate per-core dimensions
    assert N % num_cores == 0, f"N ({N}) must be divisible by num_cores ({num_cores})"
    n_per_core = N // num_cores  # 6144 / 96 = 64

    # Calculate tiles
    num_tiles_k = K // a_tile.tile_shape[1]  # 8192 / 32 = 256
    num_tiles_n_per_core = n_per_core // b_tile.tile_shape[1]  # 64 / 32 = 2

    logger.info(f"Testing output_hidden fused op with shape [{M}, {K}] x [{K}, {N}]")
    logger.info(f"Grid: {GRID_X}x{GRID_Y} = {num_cores} cores")
    logger.info(f"Per-core: K={num_tiles_k} tiles, N={num_tiles_n_per_core} tiles")

    # Create matmul core grid
    matmul_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

    # Gather receiver core: (11, 9)
    gather_core = ttnn.CoreCoord(11, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    # Create input and weights PyTorch tensors
    torch.manual_seed(0)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_input = torch_input.repeat(96, 1)
    logger.info(f"Input shape: {torch_input.shape}")
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)

    # Compute reference output using PyTorch
    # Note: Each core gets the same [1, 8192] input, so we compute golden reference
    # using only the first row (M=1), not the full [96, 8192] tensor
    torch_input_single = torch_input[0:1, :]  # Take first row: [1, 8192]
    torch_expected = OutputHidden.golden(torch_input_single.float(), torch_weights.float()).bfloat16()

    # ========================================================================
    # Create input tensor (height-sharded across matmul cores)
    # Each core gets the full [1, 8192] input
    # ========================================================================
    input_shard_shape = (M, K)  # Full input on each core
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
    # Each core gets [8192, 64]
    # ========================================================================
    weights_shard_shape = (K, n_per_core)  # [8192, 64] per core
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
    # Create output tensor (height-sharded on gather core)
    # ========================================================================
    output_shard_shape = (M, N)  # Full output on gather core
    output_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    logger.info(
        f"Created output tensor with shard shape {output_shard_shape} on core ({gather_core.x}, {gather_core.y})"
    )

    # ========================================================================
    # Run output_hidden fused operation
    # ========================================================================
    logger.info("Running output_hidden fused operation...")
    ttnn_result = OutputHidden.op(
        ttnn_input,
        ttnn_weights,
        ttnn_output,
        fp32_dest_acc_en=False,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    # Verify results
    logger.info("Verifying output_hidden results...")

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.99)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info("✓ Output hidden fused op test passed!")

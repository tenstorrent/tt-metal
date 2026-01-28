# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Multi-Core Fused SwiGLU with Mcast Test

Tests multi-core fused SwiGLU with mcast input distribution for DeepSeek v3 shapes:
- W_gate/W_up: [1, 7168] x [7168, 2304] on 4x9=36 cores with 64 per core

Architecture:
  - Input activations: HEIGHT_SHARDED on single sender core
  - W_gate: WIDTH_SHARDED across M cores
  - W_up: WIDTH_SHARDED across SAME M cores
  - Output: WIDTH_SHARDED across same M cores
  - Mcast broadcasts input from sender to all matmul cores

Fusion benefit: All three operations (gate matmul+SiLU, up matmul, eltwise mul)
execute on the same core using local CBs, avoiding cross-core data movement.

Computes: output = SiLU(input @ W_gate) * (input @ W_up)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.mcast_matmul.op import McastSwiGLUMultiCore


@pytest.mark.parametrize(
    "M, K, N, num_cores_x, num_cores_y, in0_dtype, weights_dtype",
    [
        # Small test case: 8 cores (2x4 grid), 32 per core
        (1, 256, 256, 2, 4, ttnn.bfloat16, ttnn.bfloat8_b),
        # Medium test case: 12 cores (3x4 grid), 64 per core
        (1, 512, 768, 3, 4, ttnn.bfloat16, ttnn.bfloat8_b),
        # DeepSeek SwiGLU shape: 36 cores (4x9 grid), 64 per core
        # [1, 7168] x [7168, 2304] with 64 per core = 36 cores
        (1, 7168, 2304, 4, 9, ttnn.bfloat16, ttnn.bfloat8_b),
        # DeepSeek SwiGLU with bfloat4_b weights
        (1, 7168, 2304, 4, 9, ttnn.bfloat16, ttnn.bfloat4_b),
    ],
)
def test_mcast_swiglu_multi_core(device, M, K, N, num_cores_x, num_cores_y, in0_dtype, weights_dtype):
    """Test multi-core fused SwiGLU with mcast input distribution"""

    num_cores = num_cores_x * num_cores_y
    N_per_core = N // num_cores

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # Input: M x 32 tiles
    b_tile = ttnn.Tile([32, 32])  # Weights: 32x32 tiles
    out_tile = ttnn.Tile([M, 32])  # Output: M x 32 tiles

    # Calculate tiles
    num_tiles_k = K // a_tile.tile_shape[1]
    num_tiles_n_per_core = N_per_core // b_tile.tile_shape[1]

    logger.info(f"Testing multi-core fused SwiGLU with mcast")
    logger.info(f"Shape: [{M}, {K}] x [{K}, {N}]")
    logger.info(f"Grid: {num_cores_x}x{num_cores_y} = {num_cores} cores")
    logger.info(f"Per-core: N={N_per_core}, tiles_k={num_tiles_k}, tiles_n={num_tiles_n_per_core}")
    logger.info(f"Dtypes: in0={in0_dtype}, weights={weights_dtype}")

    # Create input and weight tensors
    torch.manual_seed(42)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gate = torch.randn((K, N), dtype=torch.bfloat16)
    torch_up = torch.randn((K, N), dtype=torch.bfloat16)

    # Compute reference output using PyTorch: SiLU(input @ W_gate) * (input @ W_up)
    torch_expected = McastSwiGLUMultiCore.golden(torch_input.float(), torch_gate.float(), torch_up.float()).bfloat16()

    # ========================================================================
    # Input tensor: HEIGHT_SHARDED on single sender core
    # ========================================================================
    sender_core = ttnn.CoreCoord(0, 0)
    sender_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(sender_core, sender_core)})

    input_shard_shape = (M, K)
    input_shard_spec = ttnn.ShardSpec(
        sender_core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_shard_spec,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=a_tile,
    )

    logger.info(f"Created input tensor on core (0,0) with shard shape {input_shard_shape}")

    # ========================================================================
    # Weight tensors: WIDTH_SHARDED across matmul cores (both on same grid)
    # ========================================================================
    # Matmul core grid: starting from (0,0) spanning num_cores_x x num_cores_y
    # Note: sender core (0,0) is also a matmul core (part of the grid)
    matmul_grid_start = ttnn.CoreCoord(0, 0)
    matmul_grid_end = ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)
    matmul_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(matmul_grid_start, matmul_grid_end)})

    weights_shard_shape = (K, N_per_core)
    weights_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        weights_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        weights_shard_spec,
    )

    ttnn_gate = ttnn.from_torch(
        torch_gate,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    ttnn_up = ttnn.from_torch(
        torch_up,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    logger.info(f"Created W_gate and W_up tensors across {num_cores} cores with shard shape {weights_shard_shape}")

    # ========================================================================
    # Output tensor: WIDTH_SHARDED across same matmul cores
    # ========================================================================
    output_shard_shape = (M, N_per_core)
    output_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )

    torch_output_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    logger.info(f"Created output tensor across {num_cores} cores with shard shape {output_shard_shape}")

    # ========================================================================
    # Run multi-core fused SwiGLU with mcast
    # ========================================================================
    logger.info("Running multi-core fused SwiGLU with mcast...")
    ttnn_result = McastSwiGLUMultiCore.op(
        ttnn_input,
        ttnn_gate,
        ttnn_up,
        ttnn_output,
        fp32_dest_acc_en=False,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    # Verify fused SwiGLU results
    logger.info("Verifying fused SwiGLU results...")

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.97)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info("✓ Multi-core fused SwiGLU with mcast test passed!")


@pytest.mark.parametrize(
    "M, K, N, num_cores_x, num_cores_y, in0_dtype, weights_dtype",
    [
        # Test with sender core NOT part of matmul grid
        # Sender at (7, 0), matmul grid at (0,0)-(1,3)
        (1, 256, 256, 2, 4, ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_mcast_swiglu_separate_sender(device, M, K, N, num_cores_x, num_cores_y, in0_dtype, weights_dtype):
    """Test multi-core fused SwiGLU where sender is separate from matmul grid"""

    num_cores = num_cores_x * num_cores_y
    N_per_core = N // num_cores

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    num_tiles_k = K // a_tile.tile_shape[1]

    logger.info(f"Testing multi-core fused SwiGLU with separate sender core")
    logger.info(f"Shape: [{M}, {K}] x [{K}, {N}]")
    logger.info(f"Grid: {num_cores_x}x{num_cores_y} = {num_cores} cores")

    # Create input and weight tensors
    torch.manual_seed(42)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gate = torch.randn((K, N), dtype=torch.bfloat16)
    torch_up = torch.randn((K, N), dtype=torch.bfloat16)

    torch_expected = McastSwiGLUMultiCore.golden(torch_input.float(), torch_gate.float(), torch_up.float()).bfloat16()

    # Sender core at (7, 0) - separate from matmul grid (0,0)-(1,3)
    sender_core = ttnn.CoreCoord(7, 0)
    sender_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(sender_core, sender_core)})

    input_shard_shape = (M, K)
    input_shard_spec = ttnn.ShardSpec(
        sender_core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_shard_spec,
    )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=a_tile,
    )

    logger.info(f"Created input tensor on separate sender core (7,0)")

    # Matmul core grid: (0,0) to (num_cores_x-1, num_cores_y-1)
    matmul_grid_start = ttnn.CoreCoord(0, 0)
    matmul_grid_end = ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)
    matmul_core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(matmul_grid_start, matmul_grid_end)})

    weights_shard_shape = (K, N_per_core)
    weights_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        weights_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        weights_shard_spec,
    )

    ttnn_gate = ttnn.from_torch(
        torch_gate,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    ttnn_up = ttnn.from_torch(
        torch_up,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )

    output_shard_shape = (M, N_per_core)
    output_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )

    torch_output_zeros = torch.zeros((M, N), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    # Run multi-core fused SwiGLU with separate sender
    logger.info("Running multi-core fused SwiGLU with separate sender...")
    ttnn_result = McastSwiGLUMultiCore.op(
        ttnn_input,
        ttnn_gate,
        ttnn_up,
        ttnn_output,
        fp32_dest_acc_en=False,
    )

    output_torch = ttnn.to_torch(ttnn_result)

    assert output_torch.shape == (M, N), f"Expected shape ({M}, {N}), got {output_torch.shape}"

    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.97)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info("✓ Multi-core fused SwiGLU with separate sender test passed!")

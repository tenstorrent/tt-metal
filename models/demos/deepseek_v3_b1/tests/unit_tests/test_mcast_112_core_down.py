# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Down Projection (W_down) 112-Core Unit Test

Tests the W_down [1, K] x [K, N] + add_input -> [1, N] fused operation using:
  {mcast1} -> {mcast2} -> {matmul} -> {add} -> {gather}

on 112 matmul cores within a 13x10 = 130 core mcast grid.

Core Layout:
    D = DRAM worker (8 cores) — receives mcast semaphore, skips matmul & gather
    P = Phantom (9 cores, col 12 rows 0-8) — same as DRAM worker
    M = Mcast sender + Gather receiver at (12, 9)
    R = Matmul core (112 cores)

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_mcast_112_core_down.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj


@pytest.mark.parametrize(
    "M, K, N_per_core, weights_dtype",
    [
        # (1, 256, 32, ttnn.bfloat8_b),  # Small: 112 * 32 = 3584
        (1, 256, 64, ttnn.bfloat8_b),  # Target: 112 * 64 = 7168
        (1, 256, 64, ttnn.bfloat4_b),  # Target with bfloat4 weights
        # (1, 7168, 64, ttnn.bfloat8_b),  # Full K dimension
        # (1, 7168, 64, ttnn.bfloat4_b),  # bfloat4 weights
    ],
)
def test_down_proj_112_core(device, M, K, N_per_core, weights_dtype):
    """Test down projection with mcast1 -> mcast2 -> matmul -> add -> gather on 112 scattered cores"""

    device_grid = device.compute_with_storage_grid_size()
    logger.info(f"Device grid: {device_grid.x}x{device_grid.y}")

    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    N = N_per_core * DownProj.NUM_MATMUL_CORES  # 112 cores

    logger.info("=" * 70)
    logger.info(f"Testing Down Projection: [{M}, {K}] x [{K}, {N}] -> [{M}, {N}]")
    logger.info(f"Matmul cores: {DownProj.NUM_MATMUL_CORES} (scattered)")
    logger.info(f"Per-core output: {N_per_core} columns")
    logger.info(f"Weights dtype: {weights_dtype}")
    logger.info("=" * 70)

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    # Create test data
    torch.manual_seed(42)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_weights = torch.randn((K, N), dtype=torch.bfloat16)
    torch_add_input = torch.randn((M, N), dtype=torch.bfloat16)

    # Compute golden reference
    torch_expected = DownProj.golden(torch_input.float(), torch_weights.float(), torch_add_input.float()).bfloat16()
    logger.info(f"Golden output shape: {torch_expected.shape}")

    # ========================================================================
    # Input tensor: HEIGHT_SHARDED on mcast/gather core (12, 9)
    # ========================================================================
    mcast_gather_core = DownProj.MCAST_GATHER_CORE
    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_gather_core, mcast_gather_core)])

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
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=a_tile,
    )
    logger.info(f"Created input tensor: shard {input_shard_shape} on ({mcast_gather_core.x},{mcast_gather_core.y})")

    # ========================================================================
    # Weights: WIDTH_SHARDED across 112 scattered matmul cores
    # ========================================================================
    matmul_core_grid = DownProj.build_matmul_core_grid()

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

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=b_tile,
    )
    logger.info(f"Created weights tensor: shard {weights_shard_shape} on {DownProj.NUM_MATMUL_CORES} cores")

    # ========================================================================
    # Output: HEIGHT_SHARDED on gather core (12, 9)
    # ========================================================================
    output_shard_shape = (M, N)
    output_shard_spec = ttnn.ShardSpec(
        sender_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
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
    logger.info(f"Created output tensor: shard {output_shard_shape} on ({mcast_gather_core.x},{mcast_gather_core.y})")

    # ========================================================================
    # Add input: HEIGHT_SHARDED on mcast/gather core (12, 9)
    # ========================================================================
    add_input_shard_shape = (M, N)
    add_input_shard_spec = ttnn.ShardSpec(
        sender_core_grid,
        add_input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    add_input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        add_input_shard_spec,
    )

    ttnn_add_input = ttnn.from_torch(
        torch_add_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=add_input_mem_config,
        tile=out_tile,
    )
    logger.info(
        f"Created add_input tensor: shard {add_input_shard_shape} on ({mcast_gather_core.x},{mcast_gather_core.y})"
    )

    # ========================================================================
    # Run fused operation
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Running down projection: mcast1 -> mcast2 -> matmul -> add -> gather ...")

    ttnn_result = DownProj.op(ttnn_input, ttnn_weights, ttnn_output, ttnn_add_input)

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)
    logger.info(f"Output shape: {output_torch.shape}")

    # Verify output shape
    expected_shape = (M, N)
    assert output_torch.shape == expected_shape, f"Expected shape {expected_shape}, got {output_torch.shape}"

    # Compare with golden reference
    passing, pcc_message = comp_pcc(torch_expected, output_torch, 0.97)
    logger.info(f"PCC comparison: {pcc_message}")

    assert passing, f"PCC check failed: {pcc_message}"
    logger.info("=" * 70)
    logger.info("Down projection test PASSED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

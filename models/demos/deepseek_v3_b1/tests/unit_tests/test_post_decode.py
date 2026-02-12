# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Post-Decode Mcast + Matmul Op Test

Sender core (last column, row 9) multicasts input_a [1, 7168] to 101 matmul cores.
Each matmul core holds a weight shard [7168, N_per_core] and computes
[1, 7168] x [7168, N_per_core] -> [1, N_per_core].
Output stays width-sharded across matmul cores.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.post_decode.op import PostDecode


@pytest.mark.parametrize("use_fp32", [True])
def test_post_decode(device, use_fp32):
    """Test mcast + matmul with input_a broadcast from sender to matmul cores.

    Core layout:
        (max_x, 9) = sender core (holds input_a, multicasts to matmul cores)
        (0,0)..(9,9) + (10,0) = 101 matmul cores (hold weight shards)
    Each matmul core computes [1, K] x [K, N_per_core] -> [1, N_per_core].
    Output stays width-sharded across matmul cores.
    """

    M = 1
    K = 7168
    input_shape = (1, 7168)

    num_matmul_cores = 101
    N_per_core = 128  # TODO: change to 160, when matmul supports odd number of tiles
    N_total = N_per_core * num_matmul_cores
    vocab_shape = (7168, N_total)

    output_shape = (1, N_total)

    # Tile dimensions
    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    # Core layout
    mcast_core_x = device.compute_with_storage_grid_size().x - 1  # Last column
    mcast_core_y = 9

    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])

    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )

    num_tiles_k = K // a_tile.tile_shape[1]
    num_tiles_n_per_core = N_per_core // b_tile.tile_shape[1]

    fp32_str = " (fp32 acc)" if use_fp32 else ""
    logger.info(
        f"Testing mcast matmul{fp32_str} with shape [{M}, {K}] x [{K}, {N_total}] "
        f"({num_matmul_cores} cores, {N_per_core} per core), in0=bfloat16, in1=bfloat8_b"
    )
    logger.info(f"Tiles: K={num_tiles_k}, N_per_core={num_tiles_n_per_core}")

    # Create input tensors
    torch.manual_seed(0)
    torch_a = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_b = torch.randn(vocab_shape, dtype=torch.bfloat16)

    # Compute reference output
    torch_expected = PostDecode.golden(torch_a.float(), torch_b.float()).bfloat16()

    # Input A: HEIGHT_SHARDED on sender core
    input_a_shard_spec = ttnn.ShardSpec(
        mcast_core_grid,
        input_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_a_shard_spec
    )
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )

    logger.info(f"Created input A with shard shape ({M}, {K}) on sender core ({mcast_core_x}, {mcast_core_y})")

    # Input B: WIDTH_SHARDED across matmul cores
    input_b_shard_shape = (K, N_per_core)
    input_b_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        input_b_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_b_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_b_shard_spec
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_b_mem_config,
        tile=b_tile,
    )

    logger.info(f"Created input B with shard shape {input_b_shard_shape} on {num_matmul_cores} matmul cores")

    # Output: WIDTH_SHARDED across matmul cores
    output_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        (M, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    logger.info(f"Created output tensor with shard shape ({M}, {N_per_core}) on {num_matmul_cores} matmul cores")

    # Run mcast matmul operation
    logger.info(f"Running mcast matmul{fp32_str} operation...")
    ttnn_result = PostDecode.op(
        ttnn_a,
        ttnn_b,
        ttnn_output,
        fp32_dest_acc_en=use_fp32,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == (M, N_total), f"Expected shape ({M}, {N_total}), got {output_torch.shape}"

    # Verify matmul results
    logger.info(f"Verifying mcast matmul{fp32_str} results...")
    pcc_threshold = 0.99

    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info(f"✓ Mcast matmul{fp32_str} test passed! " f"({num_matmul_cores} cores, [{M}, {K}] x [{K}, {N_total}])")

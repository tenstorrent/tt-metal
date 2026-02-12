# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Post-Decode Mcast + Matmul Op Test
Tests mcast + matmul operation for various DeepSeek v3 shapes:

Sender core multicasts input_a to N matmul cores.
Each matmul core holds a weight shard [K, N_per_core] and computes
[1, K] x [K, N_per_core] -> [1, N_per_core].
Output stays sharded across matmul cores.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.post_decode.op import PostDecode


@pytest.mark.parametrize(
    "M, K, in0_dtype, in1_dtype, fp32_dest_acc_en",
    [
        # Basic mcast matmul: input_a broadcast to multiple matmul cores
        # Each core computes [1, K] x [K, N_per_core] -> [1, N_per_core]
        (1, 7168, ttnn.bfloat16, ttnn.bfloat8_b, False),  # Q down, 4 cores
        (1, 7168, ttnn.bfloat16, ttnn.bfloat8_b, False),  # Q down, 8 cores
        (1, 3584, ttnn.bfloat16, ttnn.bfloat8_b, False),  # Q down split K, 4 cores
        (1, 1536, ttnn.bfloat16, ttnn.bfloat8_b, False),  # Q up, 2 cores
        # MoE with mcast (bfloat4_b weights)
        (1, 7168, ttnn.bfloat16, ttnn.bfloat8_b, False),  # Gate proj, 4 cores
        (1, 2048, ttnn.bfloat16, ttnn.bfloat8_b, False),  # Down proj, 4 cores
        # FP32 accumulation
        (1, 7168, ttnn.bfloat16, ttnn.bfloat8_b, True),  # Router gate, fp32 acc, 4 cores
    ],
)
def test_matmul_mcast(device, M, K, in0_dtype, in1_dtype, fp32_dest_acc_en):
    """Test mcast + matmul with input_a broadcast from sender to matmul cores.

    Core layout:
        (0, 0) = sender core (holds input_a, multicasts to matmul cores)
        (1, 0) .. (num_matmul_cores, 0) = matmul cores (hold weight shards)
    Mcast grid: rectangular region (0,0) to (num_matmul_cores, 0)
    """
    num_matmul_cores = 101
    N_per_core = 128
    N_total = N_per_core * num_matmul_cores

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

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

    fp32_str = " (fp32 acc)" if fp32_dest_acc_en else ""
    logger.info(
        f"Testing mcast matmul{fp32_str} with shape [{M}, {K}] x [{K}, {N_total}] "
        f"({num_matmul_cores} cores, {N_per_core} per core), in0={in0_dtype}, in1={in1_dtype}"
    )
    logger.info(f"Tiles: K={num_tiles_k}, N_per_core={num_tiles_n_per_core}")

    # Create input tensors
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N_total), dtype=torch.bfloat16)

    # Compute reference output
    torch_expected = PostDecode.golden(torch_a.float(), torch_b.float()).bfloat16()

    # Input A: HEIGHT_SHARDED on sender core
    input_a_shard_spec = ttnn.ShardSpec(
        mcast_core_grid,
        (M, K),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_a_shard_spec
    )
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )

    logger.info(f"Created input A with shard shape ({M}, {K}) on sender core (0, 0)")

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
        dtype=in1_dtype,
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
        torch.zeros((M, N_total), dtype=torch.bfloat16),
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
        fp32_dest_acc_en=fp32_dest_acc_en,
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

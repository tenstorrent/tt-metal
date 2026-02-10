# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Matmul Micro Op Test - Single Core
Tests matmul operation for various DeepSeek v3 shapes:

Before SDPA (bfloat16 in0, bfloat8_b in1):
- Q down + K down: [1, 7168] x [7168, 32]
- Q down with split inner dim: [1, 3584] x [3584, 32]
- Q up: [1, 1536] x [1536, 128]
- Q nope: [1, 128] x [128, 512]

SDPA (bfloat16 in0, bfloat8_b in1):
- Q @ K.T: [8, 576] x [576, 256/512]
- S @ V: [8, 256/512] x [256/512, 512]

After SDPA (bfloat16 in0, bfloat8_b in1):
- V out: [1, 512] x [512, 128]
- Out: [1, 8192] x [8192, 64]

MoE (bfloat16 in0, bfloat4_b in1):
- Gate proj + up proj: [1, 7168] x [7168, 32]
- Down proj: [1, 2048] x [2048, 32]
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.matmul.op import Matmul


@pytest.mark.parametrize(
    "M, K, N, in0_dtype, in1_dtype, transpose, fused_activation, fp32_dest_acc_en, core_grid",
    [
        pytest.param(
            1,
            8192,
            64,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            False,
            None,
            False,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 8)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(3, 9)),
                }
            ),
            id="out_proj",
        ),
        pytest.param(
            1,
            896,
            32,
            ttnn.bfloat16,
            ttnn.bfloat4_b,
            False,
            None,
            False,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 7)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(11, 9)),
                }
            ),
            id="gate_proj_up_proj",
        ),
        pytest.param(
            1,
            256,
            64,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            False,
            None,
            False,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 8)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(3, 9)),
                }
            ),
            id="down_proj",
        ),
        pytest.param(
            1,
            3584,
            32,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            False,
            None,
            False,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7)),
                }
            ),
            id="q_a_proj_with_split_inner_dim",
        ),
    ],
)
@pytest.mark.skip_post_commit
def test_matmul_single_core(
    device, M, K, N, in0_dtype, in1_dtype, transpose, fused_activation, fp32_dest_acc_en, core_grid
):
    """Test single-core matmul operation with fully sharded inputs"""

    # Use fused_activation directly (no special suffixes needed now)
    actual_activation = fused_activation

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # Tile height matches M for A
    b_tile = ttnn.Tile([32, 32])  # Standard tile for B
    out_tile = ttnn.Tile([M, 32])  # Tile height matches M for output

    # Calculate tiles
    num_tiles_m = M * core_grid.num_cores() // a_tile.tile_shape[0]
    num_tiles_k = K // a_tile.tile_shape[1]
    num_tiles_n = N * core_grid.num_cores() // b_tile.tile_shape[1]

    activation_str = f"+{fused_activation}" if fused_activation else ""
    fp32_str = " (fp32 acc)" if fp32_dest_acc_en else ""
    transpose_str = " transposed" if transpose else ""
    logger.info(
        f"Testing single-core matmul{activation_str}{fp32_str}{transpose_str} with shape [{M}, {K}] x [{K}, {N}], in0={in0_dtype}, in1={in1_dtype}"
    )
    logger.info(f"Tiles: M={num_tiles_m}, K={num_tiles_k}, N={num_tiles_n}")

    # Create input A and input B PyTorch tensors
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N * core_grid.num_cores()), dtype=torch.bfloat16)
    output_shape = (M, N * core_grid.num_cores())
    if transpose:
        torch_b = torch_b.T

    # Compute reference output using PyTorch
    if transpose:
        torch_expected = Matmul.golden(torch_a.float(), torch_b.T.float(), actual_activation).bfloat16()
    else:
        torch_expected = Matmul.golden(torch_a.float(), torch_b.float(), actual_activation).bfloat16()

    # Create HEIGHT_SHARDED memory config for input A
    # Single core has full 1xK tensor
    input_a_shard_shape = (M, K)
    input_a_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_a_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_a_shard_spec
    )

    # Create input A (height-sharded on single core)
    ttnn_a = ttnn.from_torch(
        torch_a.repeat(core_grid.num_cores(), 1),
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )

    logger.info(f"Created input A with shard shape {input_a_shard_shape}")

    # Create WIDTH_SHARDED memory config for input B
    # Single core has full KxN tensor
    input_b_shard_shape = (K, N)
    if transpose:
        input_b_shard_shape = (N, K)
    input_b_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_b_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_b_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_b_shard_spec
    )

    # Create input B (width-sharded on single core)
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_b_mem_config,
        tile=b_tile,
    )

    logger.info(f"Created input B with shard shape {input_b_shard_shape}")

    # Create WIDTH_SHARDED memory config for output
    # Single core produces full MxN output
    output_shard_shape = (M, N)
    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Create output tensor
    torch_output_zeros = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    logger.info(f"Created output tensor with shard shape {output_shard_shape}")

    # Run matmul operation
    logger.info(f"Running matmul{activation_str}{fp32_str}{transpose_str} operation...")
    ttnn_result = Matmul.op(
        ttnn_a,
        ttnn_b,
        ttnn_output,
        fp32_dest_acc_en=fp32_dest_acc_en,
        transpose=transpose,
        fused_activation=actual_activation,
    )

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == output_shape, f"Expected shape {output_shape}, got {output_torch.shape}"

    # Verify matmul results (slightly lower PCC for fused activations due to approximation)
    logger.info(f"Verifying matmul{activation_str}{fp32_str} results...")
    pcc_threshold = 0.98 if actual_activation else 0.99

    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info(f"✓ Single-core matmul{activation_str}{fp32_str} test passed!")

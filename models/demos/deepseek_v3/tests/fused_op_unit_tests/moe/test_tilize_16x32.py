# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Tilize 16x32 Test - Single Core
Tests tilize operation for 16xN tensors, tiling into 16x32 blocks.

Input: Row-major 16xN tensor
Output: Tiled 16xN tensor (tiled into 16x32 blocks)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.tests.fused_op_unit_tests.moe.tilize_16x32.op import golden, tilize_16x32_kernel


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize(
    "N",
    [
        pytest.param(
            96, marks=pytest.mark.xfail(reason="Expected Failure: N must be divisible by 64")
        ),  # this should fail because it's not divisible by 64 (right now 16xN fast tilize only supports N divisible by 64)
        256,  # 8 tiles - 1 block
        1792,  # Deepseek case 1
        7168,  # Deepseek case 2
    ],
)
def test_tilize_16x32(device, N):
    """
    Test tilize operation: row-major 16xN tensor -> tiled 16x32 blocks.

    Args:
        device: TTNN device
        N: Width dimension (must be divisible by 64)
    """
    torch.manual_seed(42)

    H = 16
    input_shape = (H, N)
    logger.info(f"Testing tilize 16x32 with input shape {input_shape}")

    # Create input tensor in row-major format
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    # Single core
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    # Create HEIGHT_SHARDED memory config for input
    input_shard_shape = (H, N)
    input_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Input tile: 16x32 - represents the logical tile size
    # The data is in row-major format before tilizing
    input_tile = ttnn.Tile((16, 32))

    # Create input tensor - use ROW_MAJOR_LAYOUT since tilize_block expects row-major input
    # The tilize_block kernel will convert from row-major to tiled format
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=input_tile,
    )

    logger.info(f"Created input tensor with shard shape {input_shard_shape}")

    # Create output tensor in tiled format
    output_shard_shape = (H, N)
    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Output tile: 16x32 (tiled format)
    output_tile = ttnn.Tile((16, 32))

    # Create output tensor (will be filled by kernel)
    torch_output_zeros = torch.zeros(input_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    # Run tilize kernel
    ttnn_output = tilize_16x32_kernel(ttnn_input, ttnn_output)

    # Read back output
    tt_out_torch = ttnn.to_torch(ttnn_output)

    # Compute reference using golden function
    ref_output = golden(torch_input)

    # Compare results
    passing, pcc_msg = comp_pcc(ref_output, tt_out_torch, 0.999)
    logger.info(pcc_msg)
    assert passing, f"PCC check failed: {pcc_msg}"

    if passing:
        logger.info("✓ Tilize 16x32 test passed!")

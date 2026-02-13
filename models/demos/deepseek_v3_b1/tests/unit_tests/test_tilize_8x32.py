# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Tilize 8x32 Test - Single Core
Tests tilize operation for 8xN tensors, tiling into 8x32 blocks.

Input: Row-major 8xN tensor
Output: Tiled 8xN tensor (tiled into 8x32 blocks)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.tilize_8x32.op import golden, tilize_8x32_kernel


@pytest.mark.parametrize("N", [256, 64])
def test_tilize_8x32(device, N):
    """
    Test tilize operation: row-major 8xN tensor -> tiled 8x32 blocks.

    Args:
        device: TTNN device
        N: Width dimension (must be divisible by 32)
    """
    torch.manual_seed(42)

    H = 8
    input_shape = (H, N)
    logger.info(f"Testing tilize 8x32 with input shape {input_shape}")

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

    # Input tile: 8x32 - represents the logical tile size
    # The data is in row-major format before tilizing
    input_tile = ttnn.Tile((8, 32))

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

    # Output tile: 8x32 (tiled format)
    output_tile = ttnn.Tile((8, 32))

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
    ttnn_output = tilize_8x32_kernel(ttnn_input, ttnn_output)

    # Read back output
    tt_out_torch = ttnn.to_torch(ttnn_output)

    # Compute reference using golden function
    ref_output = golden(torch_input)

    # Compare results
    passing, pcc_msg = comp_pcc(ref_output, tt_out_torch, 0.999)
    logger.info(pcc_msg)
    assert passing, f"PCC check failed: {pcc_msg}"

    if passing:
        logger.info("✓ Tilize 8x32 test passed!")

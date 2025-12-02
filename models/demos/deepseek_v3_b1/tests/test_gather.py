# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Gather Test
Tests gather operation with shape [1, full_width]
Input is sharded across multiple cores (gather_grid)
Output is sharded on a single core (gather_core)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.gather.op import GatherSingleCore


@pytest.mark.parametrize(
    "width, gather_core, gather_grid, noc",
    [
        (
            32,
            ttnn.CoreCoord(11, 8),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 4),
                ttnn.CoreCoord(11, 7),
            ),
            None,
        ),  # q_a_proj output, if on 48 cores (could also do 6x8 instead of 12x4 grid)
        (
            32,
            ttnn.CoreCoord(11, 8),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(11, 7),
            ),
            None,
        ),  # q_a_proj output, if on 96 cores
        (
            32,
            ttnn.CoreCoord(0, 8),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 8),
                ttnn.CoreCoord(7, 9),
            ),
            1,
        ),  # kv_a_proj output, 16 cores (Gather only a subset for kv_a_layernorm)
        (
            128,
            ttnn.CoreCoord(11, 8),
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 0),
                ttnn.CoreCoord(11, 7),
            ),
            None,
        ),  # v_b_proj output, on 64 cores
    ],
)
def test_gather(device, width, gather_core, gather_grid, noc):
    """Test TTNN gather operation from multiple cores to single core"""
    # Truncate number of columns to 11 for P100 for testing
    if gather_core.x >= device.compute_with_storage_grid_size().x:
        gather_core = ttnn.CoreCoord(device.compute_with_storage_grid_size().x - 1, gather_core.y)
        gather_grid = ttnn.CoreRange(
            gather_grid.start, ttnn.CoreCoord(device.compute_with_storage_grid_size().x - 1, gather_grid.end.y)
        )

    # Tensor dimensions
    height = 1
    tile_height = 1
    tile_width = 32

    logger.info(f"Testing gather with width {width}")
    logger.info(f"Tile size: [{tile_height}, {tile_width}]")

    num_input_cores = gather_grid.grid_size().x * gather_grid.grid_size().y

    full_width = width * num_input_cores

    # Create PyTorch tensor for reference
    torch.manual_seed(0)
    torch_input = torch.randn((height, full_width), dtype=torch.bfloat16)

    # Input: sharded across multiple cores (gather_grid)
    input_shard_shape = (height, width)  # Each core has a shard of width
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({gather_grid}),
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Create input tensor sharded across gather_grid cores
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=ttnn.Tile([tile_height, tile_width]),
    )

    logger.info(f"Created input tensor sharded across {num_input_cores} cores with shard shape {input_shard_shape}")

    # Output: sharded on single core (gather_core)
    output_shard_shape = (height, full_width)  # Full tensor on one core
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    gather_core,
                    gather_core,
                )
            }
        ),
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Create output tensor with the sharded memory config
    # We need to create an empty tensor with the right shape and memory config
    torch_output = torch.zeros((height, full_width), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=ttnn.Tile([tile_height, tile_width]),
    )

    # Run gather operation using generic implementation
    logger.info("Running gather operation...")
    ttnn_result = GatherSingleCore.op(ttnn_input, ttnn_output, gather_core, gather_grid, noc)

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == (
        height,
        full_width,
    ), f"Expected shape ({height}, {full_width}), got {output_torch.shape}"

    # Verify that the output matches the input
    logger.info("Verifying gather results...")
    assert torch.equal(output_torch, torch_input), "Output tensor does not match input tensor"
    logger.info("✓ Gather test passed!")

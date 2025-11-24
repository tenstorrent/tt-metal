"""
TTNN Scatter Test
Tests scatter operation with shape [1, 7168]
Input is sharded on a single core (0,0)
Output is sharded across 96 cores (8x12 grid) with same shard size
"""

import pytest
import torch
from loguru import logger

import ttnn


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

    # Tensor dimensions
    height = 1
    width = 32
    tile_height = 1
    tile_width = 32

    logger.info(f"Testing gather with shape [{height}, {width}]")
    logger.info(f"Tile size: [{tile_height}, {tile_width}]")

    num_input_cores = gather_grid.grid_size().x * gather_grid.grid_size().y

    full_width = width * num_input_cores

    # Create PyTorch tensor for reference
    torch.manual_seed(0)
    torch_input = torch.randn((height, full_width), dtype=torch.bfloat16)

    # Input: sharded on single core (0,0)
    input_shard_shape = (height, width)  # Full tensor on one core
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({gather_grid}),
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Create input tensor sharded on core (0,0)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=ttnn.Tile([tile_height, tile_width]),
    )

    logger.info(f"Created input tensor sharded on core (0,0) with shard shape {input_shard_shape}")

    # Each output core gets the same shard size as input
    output_shard_shape = (height, full_width)  # Same shard size as input
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

    # Run gather operation
    logger.info("Running gather operation...")
    ttnn_result = ttnn.experimental.deepseek_b1.gather(ttnn_input, ttnn_output, noc)

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

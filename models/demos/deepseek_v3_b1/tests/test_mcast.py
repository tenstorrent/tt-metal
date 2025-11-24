"""
TTNN Mcast Test
Tests mcast operation with shape [1, 7168]
Input is sharded on a single core (0,0)
Output is sharded across 96 cores (8x12 grid) with same shard size
"""

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize(
    "width, mcast_core, mcast_grid, noc",
    [
        (
            7168,
            ttnn.CoreCoord(11, 8),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(11, 7),
            ),
            1,
        ),  # q_a_proj input, 96 cores
        (
            7168,
            ttnn.CoreCoord(0, 8),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 8),
                ttnn.CoreCoord(8, 9),
            ),
            1,
        ),  # kv_a_proj input, 18 cores
        (
            1536,
            ttnn.CoreCoord(11, 8),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(11, 7),
            ),
            0,
        ),  # q_b_proj input, 96 cores
        (
            8192,
            ttnn.CoreCoord(11, 8),
            ttnn.CoreRange(
                ttnn.CoreCoord(5, 0),
                ttnn.CoreCoord(11, 7),
            ),
            1,
        ),  # o_proj input (TP 2), 56 cores to be finalized
    ],
)
def test_mcast(device, width, mcast_core, mcast_grid, noc):
    """Test TTNN mcast operation from single core to multiple cores"""

    # Tensor dimensions
    height = 1
    tile_height = 1
    tile_width = 32

    logger.info(f"Testing mcast with shape [{height}, {width}]")
    logger.info(f"Tile size: [{tile_height}, {tile_width}]")

    # Create PyTorch tensor for reference
    torch.manual_seed(0)
    torch_input = torch.randn((height, width), dtype=torch.bfloat16)

    input_shard_shape = (height, width)  # Full tensor on one core
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=ttnn.Tile([tile_height, tile_width]),
    )

    num_output_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

    output_height = height * num_output_cores

    # Each output core gets the same shard size as input
    output_shard_shape = (height, width)  # Same shard size as input
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({mcast_grid}),
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Create output tensor with the sharded memory config
    # We need to create an empty tensor with the right shape and memory config
    torch_output = torch.zeros((output_height, width), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=ttnn.Tile([tile_height, tile_width]),
    )

    # Run mcast operation
    logger.info("Running mcast operation...")
    ttnn_result = ttnn.experimental.deepseek_b1.mcast(ttnn_input, ttnn_output, noc)

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == (
        output_height,
        width,
    ), f"Expected shape ({output_height}, {width}), got {output_torch.shape}"

    # Verify that all output cores have the same data (mcasted from input)
    # Since each core has the full shard, all should have the same data
    logger.info("Verifying mcast results...")

    # The output should match the input since we're mcasting the same data to all cores
    # Each core should have received a copy of the input data
    assert torch.equal(
        output_torch, torch_input.repeat(num_output_cores, 1)
    ), "Output tensor does not match input tensor"
    logger.info("✓ Mcast test passed!")

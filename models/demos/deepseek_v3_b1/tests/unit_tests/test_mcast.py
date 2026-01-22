# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Mcast Test
Tests mcast operation with shape [1, width]
Input is sharded on a single core (mcast_core)
Output is sharded across multiple cores (mcast_grid) with same shard size
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.mcast.op import McastSingleCore


@pytest.mark.parametrize(
    "width, mcast_core, mcast_receivers, mcast_grid, noc",
    [
        (
            7168,
            ttnn.CoreCoord(11, 9),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(8, 9)),
                }
            ),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(11, 9),
            ),
            ttnn.NOC.NOC_1,
        ),  # q_a_proj input + kv_a_proj input, 96 cores + 18 cores (120 cores total)
        (
            1536,
            ttnn.CoreCoord(11, 9),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7))}),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(11, 9),
            ),
            ttnn.NOC.NOC_1,
        ),  # q_b_proj input, 96 cores (120 cores total)
        (
            8192,
            ttnn.CoreCoord(11, 9),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 7)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 9)),
                }
            ),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(11, 9),
            ),
            ttnn.NOC.NOC_1,
        ),  # o_proj input (TP 2), 112 cores (120 cores total)
        (
            1536,
            ttnn.CoreCoord(11, 9),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9))}),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(11, 9),
            ),
            ttnn.NOC.NOC_1,
        ),  # loopback test for testing, not used in model
    ],
)
def test_mcast(device, width, mcast_core, mcast_receivers, mcast_grid, noc):
    """Test TTNN mcast operation from single core to multiple cores"""
    # Truncate number of columns to 11 for P100 for testing
    if mcast_core.x >= device.compute_with_storage_grid_size().x:
        mcast_core = ttnn.CoreCoord(device.compute_with_storage_grid_size().x - 1, mcast_core.y)
        mcast_grid = ttnn.CoreRange(
            mcast_grid.start, ttnn.CoreCoord(device.compute_with_storage_grid_size().x - 1, mcast_grid.end.y)
        )

    # Tensor dimensions
    shape = (1, width)
    tile = ttnn.Tile([1, 32])

    logger.info(f"Testing mcast with shape {shape}")
    logger.info(f"Tile size: {tile.tile_shape}")

    # Create PyTorch tensor for reference
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_shard_shape = shape  # Full tensor on one core
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
        tile=tile,
    )

    num_output_cores = mcast_receivers.num_cores()

    output_shape = (shape[0] * num_output_cores, width)

    # Each output core gets the same shard size as input
    output_shard_shape = shape  # Same shard size as input
    output_shard_spec = ttnn.ShardSpec(
        mcast_receivers,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Create output tensor with the sharded memory config
    # We need to create an empty tensor with the right shape and memory config
    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    # Compute expected output using PyTorch reference
    torch_expected = McastSingleCore.golden(torch_input, num_output_cores)

    # Run mcast operation using generic implementation
    logger.info("Running mcast operation...")
    ttnn_result = McastSingleCore.op(ttnn_input, ttnn_output, mcast_grid, noc)

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == output_shape, f"Expected shape {output_shape}, got {output_torch.shape}"

    # Verify that all output cores have the same data (mcasted from input)
    # Since each core has the full shard, all should have the same data
    logger.info("Verifying mcast results...")

    # The output should match the input since we're mcasting the same data to all cores
    # Each core should have received a copy of the input data
    assert torch.equal(output_torch, torch_expected), "Output tensor does not match expected tensor"
    logger.info("✓ Mcast test passed!")

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
        (
            896,
            ttnn.CoreCoord(12, 9),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 9)),  # 13 cols × 10 rows = 130 cores
                }
            ),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(12, 9),
            ),
            ttnn.NOC.NOC_1,
        ),  # loopback test: 10×13 grid (130 cores), mcast from (12,9)
    ],
)
def test_mcast(device, width, mcast_core, mcast_receivers, mcast_grid, noc):
    """Test TTNN mcast operation from single core to multiple cores"""
    # Truncate number of columns to 11 for P100 for testing
    if mcast_core.x >= device.compute_with_storage_grid_size().x:
        logger.warning(
            f"Truncating mcast_core.x to {device.compute_with_storage_grid_size().x - 1} due to insufficient grid size"
        )
        mcast_core = ttnn.CoreCoord(device.compute_with_storage_grid_size().x - 1, mcast_core.y)
    if mcast_grid.end.x >= device.compute_with_storage_grid_size().x:
        logger.warning(
            f"Truncating mcast_grid.end.x to {device.compute_with_storage_grid_size().x - 1} due to insufficient grid size"
        )
        mcast_grid = ttnn.CoreRange(
            mcast_grid.start, ttnn.CoreCoord(device.compute_with_storage_grid_size().x - 1, mcast_grid.end.y)
        )
        mcast_receivers = ttnn.CoreRangeSet(
            {
                (
                    cr
                    if cr.end.x < device.compute_with_storage_grid_size().x
                    else ttnn.CoreRange(
                        cr.start, ttnn.CoreCoord(device.compute_with_storage_grid_size().x - 1, cr.end.y)
                    )
                )
                for cr in mcast_receivers.ranges()
            }
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


def test_mcast_128_cores(device):
    """
    Test mcast loopback to 128-core grid (10×12 + 8 cores of column 12).

    Grid layout (logical):
        - 12 cols × 10 rows = 120 cores: (0,0) → (11,9)
        - col 12, rows 0-7 = 8 cores: (12,0) → (12,7)
        - Total: 128 receiver cores
        - Mcast core: (12, 8) - outside receiver grid

    ASCII grid:
            Col:  0   1   2   3   4   5   6   7   8   9  10  11  12
                ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
        Row 0:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 1:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 2:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 3:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 4:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 5:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 6:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤ Row 7:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 8:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │   │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 9:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ M │
                └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
        R=Receiver (128), M=Mcast source (12,9)
    """
    width = 7168  # DeepSeek activation width
    mcast_core = ttnn.CoreCoord(12, 9)
    noc = ttnn.NOC.NOC_1

    # Check device grid size
    device_grid = device.compute_with_storage_grid_size()
    if mcast_core.x >= device_grid.x or mcast_core.y >= device_grid.y:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for mcast core (12,9)")

    # Build the 128-core receiver grid
    # Region 1: 12 cols × 10 rows = (0,0) → (11,9) = 120 cores
    # Region 2: col 12, rows 0-7 = 8 cores
    mcast_receivers = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9)),  # 120 cores
            ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7)),  # 8 cores
        }
    )

    # Mcast grid is the bounding box
    mcast_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 9))

    num_receiver_cores = 128

    # Tensor dimensions
    shape = (1, width)
    tile = ttnn.Tile([1, 32])

    logger.info(f"Testing mcast with shape {shape}, {num_receiver_cores} receiver cores")

    # Create PyTorch tensor for reference
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_shard_shape = shape
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

    output_shape = (shape[0] * num_receiver_cores, width)

    output_shard_shape = shape
    output_shard_spec = ttnn.ShardSpec(
        mcast_receivers,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    # Compute expected output
    torch_expected = McastSingleCore.golden(torch_input, num_receiver_cores)

    # Run mcast operation
    logger.info("Running mcast operation...")
    ttnn_result = McastSingleCore.op(ttnn_input, ttnn_output, mcast_grid, noc)

    # Verify
    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == output_shape, f"Expected shape {output_shape}, got {output_torch.shape}"
    assert torch.equal(output_torch, torch_expected), "Output tensor does not match expected tensor"
    logger.info(f"✓ Mcast 128-core loopback test passed!")


def test_mcast_excluding_dram_workers(device):
    """
    Test mcast to 10×12 logical grid minus the 8 DRAM streaming worker cores.

    Grid layout (logical):
        - 12 cols × 10 rows = 120 cores: (0,0) → (11,9)
        - Mcast core: (12, 9) - outside receiver grid
        - Minus: 8 DRAM workers at x=0 and x=7
        - Final: 112 receiver cores

    DRAM workers (logical coords):
        Left (x=0):  (0,0), (0,3), (0,7), (0,9)
        Right (x=7): (7,1), (7,4), (7,6), (7,9)

    ASCII grid:
            Col:  0   1   2   3   4   5   6   7   8   9  10  11  (12)
                ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
        Row 0:  │ D │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 1:  │ R │ R │ R │ R │ R │ R │ R │ D │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 2:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 3:  │ D │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 4:  │ R │ R │ R │ R │ R │ R │ R │ D │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 5:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 6:  │ R │ R │ R │ R │ R │ R │ R │ D │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 7:  │ D │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 8:  │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │ R │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 9:  │ D │ R │ R │ R │ R │ R │ R │ D │ R │ R │ R │ R │  M=(12,9)
                └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
        R=Receiver (112), D=DRAM worker (8), M=Mcast source
    """
    width = 256  # Small tensor for testing
    mcast_core = ttnn.CoreCoord(12, 9)
    noc = ttnn.NOC.NOC_1

    # Check device grid size
    device_grid = device.compute_with_storage_grid_size()
    if mcast_core.x >= device_grid.x or mcast_core.y >= device_grid.y:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for mcast core (12,9)")

    # Get DRAM streaming worker cores to exclude
    dram_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    dram_worker_set = {(c.x, c.y) for c in dram_worker_cores}
    logger.info(f"DRAM worker cores to exclude: {sorted(dram_worker_set)}")

    # Build the 10×12 grid (120 cores), excluding DRAM workers
    receiver_cores = []

    # 12 cols × 10 rows = (0,0) → (11,9)
    for x in range(12):
        for y in range(10):
            if (x, y) not in dram_worker_set:
                receiver_cores.append((x, y))

    logger.info(f"Total receiver cores: {len(receiver_cores)} (120 - {120 - len(receiver_cores)} DRAM workers)")

    # Build CoreRangeSet from individual cores
    mcast_receivers = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in receiver_cores]
    )

    # Mcast grid is the bounding box
    mcast_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 9))

    # Tensor dimensions
    shape = (1, width)
    tile = ttnn.Tile([1, 32])

    logger.info(f"Testing mcast with shape {shape}, {len(receiver_cores)} receiver cores")

    # Create PyTorch tensor for reference
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_shard_shape = shape
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

    num_output_cores = len(receiver_cores)
    output_shape = (shape[0] * num_output_cores, width)

    output_shard_shape = shape
    output_shard_spec = ttnn.ShardSpec(
        mcast_receivers,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    # Compute expected output
    torch_expected = McastSingleCore.golden(torch_input, num_output_cores)

    # Run mcast operation
    logger.info("Running mcast operation...")
    ttnn_result = McastSingleCore.op(ttnn_input, ttnn_output, mcast_grid, noc)

    # Verify
    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == output_shape, f"Expected shape {output_shape}, got {output_torch.shape}"
    assert torch.equal(output_torch, torch_expected), "Output tensor does not match expected tensor"
    logger.info(f"✓ Mcast test passed with {num_output_cores} cores (excluding DRAM workers)!")


def test_mcast_matmul_simple(device):
    """
    Simple test of McastMatmulMultiCore with a small contiguous grid.

    Tests: mcast [1, 256] → 12 cores → matmul with [256, 32] weights → [1, 384] output
    Grid: 4x3 = 12 cores at (0,0)→(3,2), sender at (4,0)
    """
    from models.common.utility_functions import comp_pcc
    from models.demos.deepseek_v3_b1.micro_ops.mcast_matmul.op import McastMatmulMultiCore

    # Small dimensions for fast compilation
    # Note: K dimension must be divisible by tile height (32) for standard tiles
    activation_width = 256  # K dimension
    weight_k = 256
    weight_n_per_core = 32
    num_matmul_cores = 12  # 4x3 grid

    mcast_core = ttnn.CoreCoord(4, 0)  # Sender outside matmul grid
    matmul_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2)),  # 4x3 = 12 cores
        }
    )

    # Tile configurations:
    # - Input [1, K] uses [1, 32] tile: 1 row, K/32 cols of tiles
    # - Weights [K, N] uses [32, 32] tile: K/32 rows, N/32 cols of tiles
    # - Output [1, N] uses [1, 32] tile: 1 row, N/32 cols of tiles
    activation_tile = ttnn.Tile([1, 32])
    weights_tile = ttnn.Tile([32, 32])  # Standard matmul tile for weights
    output_tile = ttnn.Tile([1, 32])

    logger.info(f"Testing McastMatmulMultiCore with simple 4x3 grid")

    # Create input tensor (HEIGHT_SHARDED on sender core)
    torch.manual_seed(42)
    input_shape = (1, activation_width)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
        input_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=activation_tile,
    )

    # Create weights tensor (WIDTH_SHARDED across matmul cores)
    total_weight_n = weight_n_per_core * num_matmul_cores  # 32 × 12 = 384
    weights_shape = (weight_k, total_weight_n)
    torch_weights = torch.randn(weights_shape, dtype=torch.bfloat16)

    weights_shard_shape = (weight_k, weight_n_per_core)
    weights_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        weights_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights_shard_spec
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=weights_tile,
    )

    # Create output tensor (WIDTH_SHARDED across matmul cores)
    output_shape = (input_shape[0], total_weight_n)
    output_shard_shape = (input_shape[0], weight_n_per_core)
    output_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    # Run McastMatmulMultiCore
    logger.info("Running McastMatmulMultiCore...")
    ttnn_result = McastMatmulMultiCore.op(ttnn_input, ttnn_weights, ttnn_output)

    # Verify
    result_torch = ttnn.to_torch(ttnn_result)
    golden = McastMatmulMultiCore.golden(torch_input.float(), torch_weights.float()).bfloat16()

    passing, pcc_msg = comp_pcc(golden, result_torch, 0.99)
    logger.info(f"Output shape: {result_torch.shape}, PCC: {pcc_msg}")
    assert passing, f"McastMatmulMultiCore PCC failed: {pcc_msg}"
    logger.info("✓ test_mcast_matmul_simple passed!")


def test_mcast_matmul_two_ranges(device):
    """
    Test McastMatmulMultiCore with a non-contiguous 2-range grid.

    Grid: Range0 (0,0)→(3,2) = 12 cores + Range1 (4,0)→(4,2) = 3 cores = 15 cores total
    Sender at (5,0)
    """
    from models.common.utility_functions import comp_pcc
    from models.demos.deepseek_v3_b1.micro_ops.mcast_matmul.op import McastMatmulMultiCore

    # Small dimensions for fast compilation
    activation_width = 256
    weight_k = 256
    weight_n_per_core = 32
    num_matmul_cores = 15  # 12 + 3

    mcast_core = ttnn.CoreCoord(5, 0)  # Sender outside matmul grid
    matmul_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 2)),  # 4x3 = 12 cores
            ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 2)),  # 1x3 = 3 cores
        }
    )

    # Tile configurations:
    # - Input [1, K] uses [1, 32] tile
    # - Weights [K, N] uses [32, 32] tile for proper k_num_tiles alignment
    # - Output [1, N] uses [1, 32] tile
    activation_tile = ttnn.Tile([1, 32])
    weights_tile = ttnn.Tile([32, 32])
    output_tile = ttnn.Tile([1, 32])

    logger.info(f"Testing McastMatmulMultiCore with 2-range grid (12 + 3 = 15 cores)")

    # Create input tensor (HEIGHT_SHARDED on sender core)
    torch.manual_seed(42)
    input_shape = (1, activation_width)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
        input_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=activation_tile,
    )

    # Create weights tensor (WIDTH_SHARDED across matmul cores)
    total_weight_n = weight_n_per_core * num_matmul_cores  # 32 × 15 = 480
    weights_shape = (weight_k, total_weight_n)
    torch_weights = torch.randn(weights_shape, dtype=torch.bfloat16)

    weights_shard_shape = (weight_k, weight_n_per_core)
    weights_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        weights_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights_shard_spec
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=weights_tile,
    )

    # Create output tensor (WIDTH_SHARDED across matmul cores)
    output_shape = (input_shape[0], total_weight_n)
    output_shard_shape = (input_shape[0], weight_n_per_core)
    output_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    # Run McastMatmulMultiCore
    logger.info("Running McastMatmulMultiCore with 2 ranges...")
    ttnn_result = McastMatmulMultiCore.op(ttnn_input, ttnn_weights, ttnn_output)

    # Verify
    result_torch = ttnn.to_torch(ttnn_result)
    golden = McastMatmulMultiCore.golden(torch_input.float(), torch_weights.float()).bfloat16()

    passing, pcc_msg = comp_pcc(golden, result_torch, 0.99)
    logger.info(f"Output shape: {result_torch.shape}, PCC: {pcc_msg}")
    assert passing, f"McastMatmulMultiCore PCC failed: {pcc_msg}"
    logger.info("✓ test_mcast_matmul_two_ranges passed!")


def test_mcast_matmul_gather_mul(device):
    """
    Test mcast → local matmul → gather → eltwise multiply pipeline.

    Flow:
        1. Mcast [1, 7168] → 128 cores
        2. Local matmul on each core: [1, 7168] @ [7168, 32] → [1, 32]
        3. Gather A (8 groups of 8 cores) → 8 × [1, 256]
        4. Gather B (8 groups of 8 cores) → 8 × [1, 256]
        5. Eltwise mul: A ⊙ B → 8 × [1, 256]

    Core groupings (flat enumeration, row-major within each region):

    A cores (64 total):
        - (row,col) 0-9 × 0-2 = 30 cores (cols 0-2)
        - (row,col) 0-3 × 3   =  4 cores (col 3, rows 0-3)
        - (row,col) 0-9 × 7-9 = 30 cores (cols 7-9)

    B cores (64 total):
        - (row,col) 4-9 × 3     =  6 cores (col 3, rows 4-9)
        - (row,col) 0-9 × 4-6   = 30 cores (cols 4-6)
        - (row,col) 0-9 × 10-11 = 20 cores (cols 10-11)
        - (row,col) 0-7 × 12    =  8 cores (col 12, rows 0-7)

    ASCII grid:
            Col:  0   1   2   3   4   5   6   7   8   9  10  11  12
                ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
        Row 0:  │ A │ A │ A │ A │ B │ B │ B │ A │ A │ A │ B │ B │ B │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 1:  │ A │ A │ A │ A │ B │ B │ B │ A │ A │ A │ B │ B │ B │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 2:  │ A │ A │ A │ A │ B │ B │ B │ A │ A │ A │ B │ B │ B │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 3:  │ A │ A │ A │ A │ B │ B │ B │ A │ A │ A │ B │ B │ B │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 4:  │ A │ A │ A │ B │ B │ B │ B │ A │ A │ A │ B │ B │ B │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 5:  │ A │ A │ A │ B │ B │ B │ B │ A │ A │ A │ B │ B │ B │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 6:  │ A │ A │ A │ B │ B │ B │ B │ A │ A │ A │ B │ B │ B │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 7:  │ A │ A │ A │ B │ B │ B │ B │ A │ A │ A │ B │ B │ B │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 8:  │ A │ A │ A │ B │ B │ B │ B │ A │ A │ A │ B │ B │   │
                ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
        Row 9:  │ A │ A │ A │ B │ B │ B │ B │ A │ A │ A │ B │ B │ M │
                └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
        A=First gather (64), B=Second gather (64), M=Mcast source (12,9)
    """
    from models.common.utility_functions import comp_pcc
    from models.demos.deepseek_v3_b1.micro_ops.mcast_matmul.op import McastMatmulMultiCore

    # === Configuration ===
    activation_width = 7168
    weight_k = 7168
    weight_n_per_core = 32
    mcast_core = ttnn.CoreCoord(12, 9)
    num_groups = 8
    cores_per_group = 8

    # Check device grid size
    device_grid = device.compute_with_storage_grid_size()
    if mcast_core.x >= device_grid.x or mcast_core.y >= device_grid.y:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for mcast core (12,9)")

    # === Build A and B core lists (row-major order within each region) ===
    # Note: coordinates are (col, row) for ttnn.CoreCoord but we enumerate in (row, col) order

    # A cores: cols 0-2 (all rows) + col 3 (rows 0-3) + cols 7-9 (all rows)
    a_cores = []
    for row in range(10):
        for col in range(3):  # cols 0-2
            a_cores.append((col, row))
    for row in range(4):  # rows 0-3 only
        a_cores.append((3, row))  # col 3
    for row in range(10):
        for col in range(7, 10):  # cols 7-9
            a_cores.append((col, row))

    # B cores: col 3 (rows 4-9) + cols 4-6 (all rows) + cols 10-11 (all rows) + col 12 (rows 0-7)
    b_cores = []
    for row in range(4, 10):  # rows 4-9 only
        b_cores.append((3, row))  # col 3
    for row in range(10):
        for col in range(4, 7):  # cols 4-6
            b_cores.append((col, row))
    for row in range(10):
        for col in range(10, 12):  # cols 10-11
            b_cores.append((col, row))
    for row in range(8):  # rows 0-7 only
        b_cores.append((12, row))  # col 12

    logger.info(f"A cores: {len(a_cores)} (expected 64)")
    logger.info(f"B cores: {len(b_cores)} (expected 64)")
    assert len(a_cores) == 64, f"Expected 64 A cores, got {len(a_cores)}"
    assert len(b_cores) == 64, f"Expected 64 B cores, got {len(b_cores)}"

    # === Split into 8 groups of 8 (flat enumeration) ===
    a_groups = [a_cores[i * cores_per_group : (i + 1) * cores_per_group] for i in range(num_groups)]
    b_groups = [b_cores[i * cores_per_group : (i + 1) * cores_per_group] for i in range(num_groups)]

    logger.info(f"A groups ({cores_per_group} cores each):")
    for i, group in enumerate(a_groups):
        logger.info(f"  Group {i}: {group}")

    logger.info(f"B groups ({cores_per_group} cores each):")
    for i, group in enumerate(b_groups):
        logger.info(f"  Group {i}: {group}")

    # Build CoreRangeSet for matmul cores (128 cores = 10×12 + col 12 rows 0-7)
    matmul_core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9)),  # 120 cores
            ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7)),  # 8 cores
        }
    )
    num_matmul_cores = 128

    # === Create tensors ===
    input_shape = (1, activation_width)

    # Tile configurations:
    # - Input [1, K] uses [1, 32] tile: 1 row, K/32 cols of tiles
    # - Weights [K, N] uses [32, 32] tile for proper k_num_tiles alignment
    # - Output [1, N] uses [1, 32] tile: 1 row, N/32 cols of tiles
    activation_tile = ttnn.Tile([1, 32])
    weights_tile = ttnn.Tile([32, 32])
    output_tile = ttnn.Tile([1, 32])

    gather_width = weight_n_per_core * cores_per_group  # 32 × 8 = 256

    logger.info(f"Testing McastMatmul → Gather → Eltwise Mul pipeline")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(
        f"  Fused mcast+matmul: [{input_shape[0]}, {weight_k}] @ [{weight_k}, {weight_n_per_core * num_matmul_cores}]"
    )
    logger.info(f"  Output per core: [{input_shape[0]}, {weight_n_per_core}]")
    logger.info(f"  Gather: {num_groups} groups × {cores_per_group} cores → {num_groups} × [1, {gather_width}]")

    # Create input tensor (HEIGHT_SHARDED on sender core)
    torch.manual_seed(42)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(mcast_core, mcast_core)}),
        input_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=activation_tile,
    )

    # Create weights tensor (WIDTH_SHARDED across 128 matmul cores)
    # Total weight shape: [K, N] = [7168, 4096]
    # Each core gets [7168, 32]
    total_weight_n = weight_n_per_core * num_matmul_cores  # 32 × 128 = 4096
    weights_shape = (weight_k, total_weight_n)
    torch_weights = torch.randn(weights_shape, dtype=torch.bfloat16)

    weights_shard_shape = (weight_k, weight_n_per_core)  # [7168, 32] per core
    weights_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        weights_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights_shard_spec
    )

    ttnn_weights = ttnn.from_torch(
        torch_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weights_mem_config,
        tile=weights_tile,
    )

    # Create output tensor (WIDTH_SHARDED across 128 matmul cores)
    # Total output shape: [1, 4096]
    # Each core outputs [1, 32]
    output_shape = (input_shape[0], total_weight_n)
    output_shard_shape = (input_shape[0], weight_n_per_core)  # [1, 32] per core
    output_shard_spec = ttnn.ShardSpec(
        matmul_core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=output_tile,
    )

    # === Step 1 & 2: Fused Mcast + Matmul using McastMatmulMultiCore ===
    logger.info("Step 1 & 2: Running fused mcast + matmul (McastMatmulMultiCore)...")

    ttnn_matmul_result = McastMatmulMultiCore.op(ttnn_input, ttnn_weights, ttnn_output)

    # Verify matmul result against PyTorch golden
    matmul_output_torch = ttnn.to_torch(ttnn_matmul_result)
    golden_matmul = McastMatmulMultiCore.golden(torch_input.float(), torch_weights.float()).bfloat16()

    passing, pcc_msg = comp_pcc(golden_matmul, matmul_output_torch, 0.99)
    logger.info(f"  Matmul output shape: {matmul_output_torch.shape}")
    logger.info(f"  Matmul PCC vs golden: {pcc_msg}")
    assert passing, f"Matmul PCC failed: {pcc_msg}"
    logger.info("  ✓ Fused mcast + matmul verified")

    # === Step 3: Gather - collect results from A and B core groups ===
    logger.info("Step 3: Gathering results from A and B groups...")

    # The matmul output is [1, 4096] where each 32-element segment corresponds to one core
    # WIDTH_SHARDED means cores are enumerated by CoreRangeSet order:
    # - Range 0: (0,0)→(11,9) = 120 cores, enumerated row by row (shards 0-119)
    # - Range 1: (12,0)→(12,7) = 8 cores, enumerated row by row (shards 120-127)

    # Build mapping from (col, row) to output column index
    # The CoreRangeSet enumeration order: row-major within each range
    core_to_shard_idx = {}
    shard_idx = 0
    # Range 0: (0,0)→(11,9)
    for y in range(10):
        for x in range(12):
            core_to_shard_idx[(x, y)] = shard_idx
            shard_idx += 1
    # Range 1: (12,0)→(12,7)
    for y in range(8):
        core_to_shard_idx[(12, y)] = shard_idx
        shard_idx += 1

    # Gather A: 64 cores split into 8 groups of 8
    gathered_a = []
    for group_idx in range(num_groups):
        group_cores = a_groups[group_idx]
        group_data_list = []
        for col, row in group_cores:
            idx = core_to_shard_idx[(col, row)]
            start_col = idx * weight_n_per_core
            end_col = start_col + weight_n_per_core
            group_data_list.append(matmul_output_torch[:, start_col:end_col])
        # Concatenate 8 cores × [1, 32] → [1, 256]
        group_data = torch.cat(group_data_list, dim=1)
        gathered_a.append(group_data)

    tensor_a = torch.cat(gathered_a, dim=0)  # [8, 256]
    expected_a_shape = (num_groups, gather_width)
    logger.info(f"  Gathered A shape: {tensor_a.shape} (expected {expected_a_shape})")
    assert tensor_a.shape == expected_a_shape, f"Expected A shape {expected_a_shape}, got {tensor_a.shape}"

    # Gather B: 64 cores split into 8 groups of 8
    gathered_b = []
    for group_idx in range(num_groups):
        group_cores = b_groups[group_idx]
        group_data_list = []
        for col, row in group_cores:
            idx = core_to_shard_idx[(col, row)]
            start_col = idx * weight_n_per_core
            end_col = start_col + weight_n_per_core
            group_data_list.append(matmul_output_torch[:, start_col:end_col])
        # Concatenate 8 cores × [1, 32] → [1, 256]
        group_data = torch.cat(group_data_list, dim=1)
        gathered_b.append(group_data)

    tensor_b = torch.cat(gathered_b, dim=0)  # [8, 256]
    expected_b_shape = (num_groups, gather_width)
    logger.info(f"  Gathered B shape: {tensor_b.shape} (expected {expected_b_shape})")
    assert tensor_b.shape == expected_b_shape, f"Expected B shape {expected_b_shape}, got {tensor_b.shape}"
    logger.info("  ✓ Gather completed")

    # === Step 4: Eltwise multiply A ⊙ B ===
    logger.info("Step 4: Running eltwise multiply A ⊙ B...")

    # Convert to ttnn tensors (use standard 32x32 tiles for [6, 320])
    ttnn_a = ttnn.from_torch(
        tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=ttnn.Tile([32, 32])
    )
    ttnn_b = ttnn.from_torch(
        tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=ttnn.Tile([32, 32])
    )

    # Eltwise multiply
    ttnn_result = ttnn.mul(ttnn_a, ttnn_b)

    # Verify
    result_torch = ttnn.to_torch(ttnn_result)
    golden_result = tensor_a * tensor_b

    passing, pcc_msg = comp_pcc(golden_result, result_torch, 0.99)
    logger.info(f"  Eltwise mul result shape: {result_torch.shape}")
    logger.info(f"  Eltwise mul PCC: {pcc_msg}")
    assert passing, f"Eltwise mul PCC failed: {pcc_msg}"
    logger.info("  ✓ Eltwise multiply verified")

    logger.info("✓ Full pipeline test passed: McastMatmul → Gather → Eltwise Mul")

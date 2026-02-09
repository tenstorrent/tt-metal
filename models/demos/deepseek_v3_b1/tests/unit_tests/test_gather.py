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
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.gather.op import GatherSingleCore

# =============================================================================
# Helper functions
# =============================================================================


def create_input_tensor(device, torch_input, gather_grid, shard_shape, tile):
    """Create a sharded input tensor."""
    input_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({gather_grid}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )


def create_output_tensor(device, output_shape, gather_core, tile, dtype=ttnn.bfloat16):
    """Create a sharded output tensor on the gather core."""
    output_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(gather_core, gather_core)}),
        output_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_output,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )


def create_scattered_input_tensor(device, torch_input, sender_cores, shard_shape, tile, dtype=ttnn.bfloat16):
    """Create a sharded input tensor on scattered (non-rectangular) cores."""
    core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in sender_cores])
    input_shard_spec = ttnn.ShardSpec(
        core_range_set,
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    return ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile,
    )


# =============================================================================
# Basic gather test (rectangular grid)
# =============================================================================


@pytest.mark.parametrize(
    "width_per_core, gather_core, gather_grid, noc",
    [
        (
            32,
            ttnn.CoreCoord(11, 9),
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 4),
                ttnn.CoreCoord(11, 7),
            ),
            None,
        ),  # q_a_proj output, if on 48 cores (could also do 6x8 instead of 12x4 grid)
        (
            32,
            ttnn.CoreCoord(11, 9),
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
            ttnn.NOC.NOC_1,
        ),  # kv_a_proj output, 16 cores (Gather only a subset for kv_a_layernorm)
        (
            128,
            ttnn.CoreCoord(11, 9),
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 0),
                ttnn.CoreCoord(11, 7),
            ),
            None,
        ),  # v_b_proj output, on 64 cores
    ],
)
def test_gather(device, width_per_core, gather_core, gather_grid, noc):
    """Test TTNN gather operation from multiple cores to single core"""
    # Truncate number of columns to 11 for P100 for testing
    if gather_core.x >= device.compute_with_storage_grid_size().x:
        logger.warning(
            f"Truncating gather_core.x to {device.compute_with_storage_grid_size().x - 1} due to insufficient grid size"
        )
        gather_core = ttnn.CoreCoord(device.compute_with_storage_grid_size().x - 1, gather_core.y)
    if gather_grid.end.x >= device.compute_with_storage_grid_size().x:
        logger.warning(
            f"Truncating gather_grid.end.x to {device.compute_with_storage_grid_size().x - 1} due to insufficient grid size"
        )
        gather_grid = ttnn.CoreRange(
            gather_grid.start, ttnn.CoreCoord(device.compute_with_storage_grid_size().x - 1, gather_grid.end.y)
        )

    # Tensor dimensions
    tile = ttnn.Tile([1, 32])

    num_input_cores = gather_grid.grid_size().x * gather_grid.grid_size().y
    full_width = width_per_core * num_input_cores

    shard_shape = (1, width_per_core)  # Each core has a shard of width_per_core
    output_shape = (1, full_width)  # Full tensor on one core

    logger.info(f"Testing gather with shard shape {shard_shape}, output shape {output_shape}")
    logger.info(f"Tile size: {tile.tile_shape}")

    # Create PyTorch tensor for reference
    torch.manual_seed(0)
    torch_input = torch.randn(output_shape, dtype=torch.bfloat16)

    # Compute expected output using PyTorch reference
    torch_expected = GatherSingleCore.golden(torch_input)

    # Create input and output tensors
    ttnn_input = create_input_tensor(device, torch_input, gather_grid, shard_shape, tile)
    logger.info(f"Created input tensor sharded across {num_input_cores} cores with shard shape {shard_shape}")

    ttnn_output = create_output_tensor(device, output_shape, gather_core, tile)

    # Run gather operation
    logger.info("Running gather operation...")
    ttnn_result = GatherSingleCore.op(ttnn_input, ttnn_output, noc)

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == output_shape, f"Expected shape {output_shape}, got {output_torch.shape}"

    # Verify that the output matches the expected
    logger.info("Verifying gather results...")
    assert torch.equal(output_torch, torch_expected), "Output tensor does not match expected tensor"
    logger.info("Gather test passed!")


# =============================================================================
# Scattered cores test (non-rectangular grid)
# =============================================================================


@pytest.mark.parametrize(
    "sender_cores, gather_core, width_per_core, description",
    [
        (
            # Diagonal pattern - non-contiguous cores
            [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2), ttnn.CoreCoord(3, 3)],
            ttnn.CoreCoord(7, 5),
            32,
            "Diagonal scattered cores (4 cores)",
        ),
        (
            # L-shaped pattern
            [
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(0, 1),
                ttnn.CoreCoord(0, 2),
                ttnn.CoreCoord(1, 0),
                ttnn.CoreCoord(2, 0),
            ],
            ttnn.CoreCoord(7, 5),
            32,
            "L-shaped scattered cores (5 cores)",
        ),
        (
            # Sparse checkerboard pattern
            [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0), ttnn.CoreCoord(0, 2), ttnn.CoreCoord(2, 2)],
            ttnn.CoreCoord(5, 5),
            64,
            "Checkerboard scattered cores (4 cores)",
        ),
        (
            # Single scattered core
            [ttnn.CoreCoord(3, 4)],
            ttnn.CoreCoord(7, 7),
            128,
            "Single scattered core",
        ),
    ],
)
def test_gather_scattered_cores(device, sender_cores, gather_core, width_per_core, description):
    """
    Test gather from scattered (non-rectangular) cores using per-core compile-time args.

    This test uses the PerCoreCompileTimeDescriptor feature to specify per-core
    sender indices, enabling gather from arbitrary non-contiguous core patterns.
    """
    # Truncate cores for device grid size
    device_grid_x = device.compute_with_storage_grid_size().x
    device_grid_y = device.compute_with_storage_grid_size().y

    # Filter out cores that don't fit on the device
    valid_sender_cores = [core for core in sender_cores if core.x < device_grid_x and core.y < device_grid_y]

    if len(valid_sender_cores) < len(sender_cores):
        logger.warning(
            f"Reduced sender cores from {len(sender_cores)} to {len(valid_sender_cores)} due to device grid size"
        )

    if len(valid_sender_cores) == 0:
        pytest.skip("No valid sender cores for this device grid size")

    # Truncate gather_core if needed
    if gather_core.x >= device_grid_x or gather_core.y >= device_grid_y:
        gather_core = ttnn.CoreCoord(min(gather_core.x, device_grid_x - 1), min(gather_core.y, device_grid_y - 1))
        logger.warning(f"Truncated gather_core to {gather_core}")

    tile = ttnn.Tile([1, 32])
    num_senders = len(valid_sender_cores)
    full_width = width_per_core * num_senders

    shard_shape = (1, width_per_core)
    output_shape = (1, full_width)

    logger.info(f"Testing {description}: {num_senders} scattered cores, shard={shard_shape}, output={output_shape}")
    logger.info(f"Sender cores: {[(c.x, c.y) for c in valid_sender_cores]}")

    # Create random input
    torch.manual_seed(42)
    torch_input = torch.randn(output_shape, dtype=torch.bfloat16)
    torch_expected = GatherSingleCore.golden(torch_input)

    # Create input tensor on scattered cores
    ttnn_input = create_scattered_input_tensor(device, torch_input, valid_sender_cores, shard_shape, tile)
    logger.info(f"Created input tensor sharded across {num_senders} scattered cores")

    # Create output tensor on gather core
    ttnn_output = create_output_tensor(device, output_shape, gather_core, tile)

    # Run gather operation using the scattered cores variant
    logger.info("Running gather operation on scattered cores...")
    ttnn_result = GatherSingleCore.op_scattered(ttnn_input, ttnn_output, valid_sender_cores, noc=None)

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == output_shape, f"Expected shape {output_shape}, got {output_torch.shape}"

    # Verify that the output matches the expected
    logger.info("Verifying gather results...")
    assert torch.equal(output_torch, torch_expected), f"{description}: Output mismatch"
    logger.info(f"{description} passed!")


# =============================================================================
# Gate/Up parallel gather test (A/B split pattern from DeepSeek MoE)
# =============================================================================


def get_gate_up_core_assignment():
    """
    Returns the A (Gate) and B (Up) core assignments for the 128-core split pattern.

        Col:  0   1   2   3   4   5   6   7   8   9  10  11  12
    Row 0:  | A | A | A | A | B | B | B | A | A | A | B | B | B |
    Row 1:  | A | A | A | A | B | B | B | A | A | A | B | B | B |
    Row 2:  | A | A | A | A | B | B | B | A | A | A | B | B | B |
    Row 3:  | A | A | A | A | B | B | B | A | A | A | B | B | B |
    Row 4:  | A | A | A | B | B | B | B | A | A | A | B | B | B |
    Row 5:  | A | A | A | B | B | B | B | A | A | A | B | B | B |
    Row 6:  | A | A | A | B | B | B | B | A | A | A | B | B | B |
    Row 7:  | A | A | A | B | B | B | B | A | A | A | B | B | B |
    Row 8:  | A | A | A | B | B | B | B | A | A | A | B | B |   |
    Row 9:  | A | A | A | B | B | B | B | A | A | A | B | B | M |

    A = Gate matmul (64 cores)
    B = Up matmul (64 cores)
    M = Mcast/Reduce core (12, 9)
    """
    a_cores = []  # Gate cores
    b_cores = []  # Up cores

    for row in range(10):
        for col in range(13):
            # Skip the M core at (12, 9)
            if col == 12 and row == 9:
                continue
            # Skip (12, 8) - empty in diagram
            if col == 12 and row == 8:
                continue

            # Determine if A or B based on the pattern
            if row <= 3:
                # Rows 0-3: cols 0-3 and 7-9 are A, cols 4-6 and 10-12 are B
                if col <= 3 or (7 <= col <= 9):
                    a_cores.append(ttnn.CoreCoord(col, row))
                else:
                    b_cores.append(ttnn.CoreCoord(col, row))
            else:
                # Rows 4-9: cols 0-2 and 7-9 are A, cols 3-6 and 10-12 are B
                if col <= 2 or (7 <= col <= 9):
                    a_cores.append(ttnn.CoreCoord(col, row))
                else:
                    b_cores.append(ttnn.CoreCoord(col, row))

    return a_cores, b_cores


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_gather_gate_up_parallel_pattern(device, dtype):
    """
    Test two parallel gathers from the Gate/Up A/B split pattern.

    This simulates the DeepSeek MoE scenario where:
    - Gate matmul runs on 64 "A" cores (scattered pattern)
    - Up matmul runs on 64 "B" cores (scattered pattern)
    - Both gather to the M core at (12, 9)
    - The gathers happen in sequence (could be parallel with different semaphores)

    This test verifies that the per-core compile args correctly handle
    the complex scattered pattern where A and B cores are interleaved.

    The dtype parametrization validates tile-based size calculation for BFP formats
    (bfloat8_b, bfloat4_b) which have exponent headers.
    """
    device_grid_x = device.compute_with_storage_grid_size().x
    device_grid_y = device.compute_with_storage_grid_size().y

    # Need at least 13x10 grid for full pattern
    if device_grid_x < 13 or device_grid_y < 10:
        pytest.skip(f"Device grid too small (need 13x10, got {device_grid_x}x{device_grid_y})")

    # Get core assignments
    a_cores, b_cores = get_gate_up_core_assignment()
    m_core = ttnn.CoreCoord(12, 9)  # Mcast/Reduce core

    logger.info("=" * 70)
    logger.info(f"Test: Gate/Up Parallel Gather (A/B Split Pattern) with dtype={dtype}")
    logger.info("=" * 70)
    logger.info(f"A (Gate) cores: {len(a_cores)}")
    logger.info(f"B (Up) cores: {len(b_cores)}")
    logger.info(f"M (Gather) core: ({m_core.x}, {m_core.y})")

    # Verify we have 64 cores each
    assert len(a_cores) == 64, f"Expected 64 A cores, got {len(a_cores)}"
    assert len(b_cores) == 64, f"Expected 64 B cores, got {len(b_cores)}"

    # Configuration
    width_per_core = 32
    tile = ttnn.Tile([1, 32])

    # Gate (A) tensor: 64 cores x 32 width = 2048 total width
    a_total_width = width_per_core * len(a_cores)
    a_shard_shape = (1, width_per_core)
    a_output_shape = (1, a_total_width)

    # Up (B) tensor: 64 cores x 32 width = 2048 total width
    b_total_width = width_per_core * len(b_cores)
    b_shard_shape = (1, width_per_core)
    b_output_shape = (1, b_total_width)

    logger.info(f"Gate shard: {a_shard_shape}, total: {a_output_shape}")
    logger.info(f"Up shard: {b_shard_shape}, total: {b_output_shape}")

    # ========================================================================
    # Create Gate (A) input data
    # ========================================================================
    torch.manual_seed(42)
    torch_a_input = torch.randn(a_output_shape, dtype=torch.bfloat16)
    torch_a_expected = torch_a_input.clone()

    # ========================================================================
    # Create Up (B) input data
    # ========================================================================
    torch.manual_seed(123)
    torch_b_input = torch.randn(b_output_shape, dtype=torch.bfloat16)
    torch_b_expected = torch_b_input.clone()

    # ========================================================================
    # Create sharded tensors on A and B cores
    # ========================================================================
    ttnn_a_input = create_scattered_input_tensor(device, torch_a_input, a_cores, a_shard_shape, tile, dtype)
    ttnn_b_input = create_scattered_input_tensor(device, torch_b_input, b_cores, b_shard_shape, tile, dtype)

    # For BFP formats, get the expected values after quantization (round-trip through dtype)
    if dtype != ttnn.bfloat16:
        torch_a_expected = ttnn.to_torch(ttnn_a_input)
        torch_b_expected = ttnn.to_torch(ttnn_b_input)

    logger.info(f"Created Gate input sharded across {len(a_cores)} A cores with dtype={dtype}")
    logger.info(f"Created Up input sharded across {len(b_cores)} B cores with dtype={dtype}")

    # ========================================================================
    # Create output tensors (both gather to M core)
    # ========================================================================
    ttnn_a_output = create_output_tensor(device, a_output_shape, m_core, tile, dtype)
    ttnn_b_output = create_output_tensor(device, b_output_shape, m_core, tile, dtype)

    # ========================================================================
    # Run Gate (A) gather
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Running Gate (A) gather from 64 scattered cores...")
    ttnn_a_result = GatherSingleCore.op_scattered(ttnn_a_input, ttnn_a_output, a_cores, noc=None)
    output_a_torch = ttnn.to_torch(ttnn_a_result)

    # Verify Gate gather
    # Exact match against round-tripped expected (verifies gather preserved data exactly)
    assert torch.equal(output_a_torch, torch_a_expected), f"Gate (A) gather data mismatch with dtype={dtype}"
    # PCC against original input (verifies dtype quantization is within acceptable range)
    passing, pcc_msg = comp_pcc(torch_a_input, output_a_torch, 0.99)
    logger.info(f"Gate (A) {pcc_msg}")
    assert passing, f"Gate (A) PCC check failed with dtype={dtype}: {pcc_msg}"
    logger.info(f"Gate (A) gather with dtype={dtype}: PASSED")

    # ========================================================================
    # Run Up (B) gather
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Running Up (B) gather from 64 scattered cores...")
    ttnn_b_result = GatherSingleCore.op_scattered(ttnn_b_input, ttnn_b_output, b_cores, noc=None)
    output_b_torch = ttnn.to_torch(ttnn_b_result)

    # Verify Up gather
    # Exact match against round-tripped expected (verifies gather preserved data exactly)
    assert torch.equal(output_b_torch, torch_b_expected), f"Up (B) gather data mismatch with dtype={dtype}"
    # PCC against original input (verifies dtype quantization is within acceptable range)
    passing, pcc_msg = comp_pcc(torch_b_input, output_b_torch, 0.99)
    logger.info(f"Up (B) {pcc_msg}")
    assert passing, f"Up (B) PCC check failed with dtype={dtype}: {pcc_msg}"
    logger.info(f"Up (B) gather with dtype={dtype}: PASSED")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("=" * 70)
    logger.info(f"SUCCESS: Both Gate and Up gathers completed correctly with dtype={dtype}!")
    logger.info(f"  Gate (A): Gathered {a_total_width} elements from {len(a_cores)} scattered cores")
    logger.info(f"  Up (B):   Gathered {b_total_width} elements from {len(b_cores)} scattered cores")
    logger.info("=" * 70)

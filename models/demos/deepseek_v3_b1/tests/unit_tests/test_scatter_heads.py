# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN ScatterHeads Test

Tests scatter_heads operation that scatters data from few input cores to many output cores.
Input: 8 cores, each with shard shape (8, 512)
Output: 64 cores, each with shard shape (1, 512)

Each input core's 8 rows are scattered to 8 different output cores.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.scatter_heads.op import ScatterHeads


@pytest.mark.parametrize(
    "num_input_cores, rows_per_input_core, width",
    [
        (8, 8, 512),  # 8 input cores × 8 rows = 64 output cores, width 512
    ],
)
def test_scatter_heads(device, num_input_cores, rows_per_input_core, width):
    """Test TTNN scatter_heads operation from few cores to many cores using ROW_MAJOR layout"""

    num_output_cores = num_input_cores * rows_per_input_core

    # Check device grid size
    grid_size = device.compute_with_storage_grid_size()
    max_cores = grid_size.x * grid_size.y

    if num_output_cores + num_input_cores > max_cores:
        pytest.skip(f"Not enough cores: need {num_output_cores + num_input_cores}, have {max_cores}")

    logger.info(f"Testing scatter_heads: {num_input_cores} input cores -> {num_output_cores} output cores")
    logger.info(f"Input shard shape: ({rows_per_input_core}, {width})")
    logger.info(f"Output shard shape: (1, {width})")

    # Tensor dimensions
    input_shape = (num_input_cores * rows_per_input_core, width)
    output_shape = (num_output_cores, width)  # Same total shape, different sharding

    # Create PyTorch tensor for reference
    torch.manual_seed(42)
    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)

    # Define input cores (use a column of cores for input)
    # Place input cores at the right edge of the grid to avoid overlap with output
    input_core_x = grid_size.x - 1  # Last column
    input_core_list = [ttnn.CoreCoord(input_core_x, y) for y in range(num_input_cores)]
    input_core_range = ttnn.CoreRange(
        ttnn.CoreCoord(input_core_x, 0),
        ttnn.CoreCoord(input_core_x, num_input_cores - 1),
    )
    input_core_range_set = ttnn.CoreRangeSet({input_core_range})

    # Define output cores (use a rectangular grid, avoiding the input column)
    # We need num_output_cores cores in a grid that doesn't overlap with input
    available_cols = grid_size.x - 1  # Exclude input column
    output_rows = (num_output_cores + available_cols - 1) // available_cols

    # Adjust if we need more rows than available
    if output_rows > grid_size.y:
        pytest.skip(f"Not enough rows for output cores: need {output_rows}, have {grid_size.y}")

    # Build output core range set
    output_core_ranges = []
    remaining_cores = num_output_cores
    for row in range(output_rows):
        cores_in_row = min(remaining_cores, available_cols)
        if cores_in_row > 0:
            output_core_ranges.append(
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, row),
                    ttnn.CoreCoord(cores_in_row - 1, row),
                )
            )
            remaining_cores -= cores_in_row

    output_core_range_set = ttnn.CoreRangeSet(set(output_core_ranges))

    logger.info(f"Input cores: column {input_core_x}, rows 0-{num_input_cores - 1}")
    logger.info(f"Output cores: {output_core_range_set}")

    # Create input tensor memory config (sharded on input cores)
    input_shard_shape = (rows_per_input_core, width)
    input_shard_spec = ttnn.ShardSpec(
        input_core_range_set,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_shard_spec,
    )

    # Create output tensor memory config (sharded on output cores)
    output_shard_shape = (1, width)
    output_shard_spec = ttnn.ShardSpec(
        output_core_range_set,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )

    # Create input tensor on device using ROW_MAJOR layout
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    # Create output tensor on device (pre-allocated) using ROW_MAJOR layout
    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=output_mem_config,
    )

    # Compute expected output using PyTorch reference
    torch_expected = ScatterHeads.golden(torch_input)

    # Run scatter_heads operation (using default mapping)
    logger.info("Running scatter_heads operation...")
    ttnn_result = ScatterHeads.op(
        ttnn_input,
        ttnn_output,
        rows_per_input_core=rows_per_input_core,
    )

    # Convert back to torch for verification
    output_torch = ttnn.to_torch(ttnn_result)

    # Verify output shape
    assert output_torch.shape == output_shape, f"Expected shape {output_shape}, got {output_torch.shape}"

    # Verify data correctness
    logger.info("Verifying scatter_heads results...")

    # Compare with expected output
    if not torch.allclose(output_torch, torch_expected, atol=1e-3, rtol=1e-3):
        diff = (output_torch - torch_expected).abs()
        max_diff = diff.max().item()
        logger.error(f"Max difference: {max_diff}")
        assert False, f"Output tensor does not match expected tensor. Max diff: {max_diff}"

    logger.info("✓ ScatterHeads test passed!")


@pytest.mark.parametrize(
    "num_input_cores, rows_per_input_core, width",
    [
        (8, 8, 512),  # Primary test case matching the spec
    ],
)
def test_scatter_heads_with_custom_mapping(device, num_input_cores, rows_per_input_core, width):
    """
    Test scatter_heads with a user-provided core mapping.

    This test demonstrates passing a custom mapping of which output core
    reads from which input core and which row.

    Layout:
    - Input cores: (0,0) to (7,0) - 8 cores in a row, each with 8 rows of data
    - Output cores: (0,0) to (7,7) - 8x8 grid = 64 cores, each with 1 row of data
    - Mapping: Each output core at (x, y) reads row y from input core at (x, 0)
               This means each column of output cores reads from one input core
    """
    num_output_cores = num_input_cores * rows_per_input_core

    # Check device grid size
    grid_size = device.compute_with_storage_grid_size()

    # We need at least 8x8 grid for this test
    if grid_size.x < num_input_cores or grid_size.y < rows_per_input_core:
        pytest.skip(f"Need at least {num_input_cores}x{rows_per_input_core} grid, have {grid_size.x}x{grid_size.y}")

    logger.info(f"Testing scatter_heads with custom mapping: {num_input_cores} -> {num_output_cores} cores")
    logger.info(f"Input cores: (0,0) to ({num_input_cores - 1},0)")
    logger.info(f"Output cores: (0,0) to ({num_input_cores - 1},{rows_per_input_core - 1})")

    # Create input with unique identifiable values per row
    # Input is organized as: input_core_0_row_0, input_core_0_row_1, ..., input_core_7_row_7
    input_shape = (num_input_cores * rows_per_input_core, width)
    output_shape = (num_output_cores, width)

    torch_input = torch.zeros(input_shape, dtype=torch.bfloat16)
    for i in range(input_shape[0]):
        # Fill each row with its row index as a base value
        torch_input[i, :] = float(i) + torch.arange(width, dtype=torch.bfloat16) * 0.001

    # Input cores: (0,0) to (7,0)
    input_core_list = [ttnn.CoreCoord(x, 0) for x in range(num_input_cores)]
    input_core_range = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_input_cores - 1, 0),
    )
    input_core_range_set = ttnn.CoreRangeSet({input_core_range})

    # Output cores: (0,0) to (7,7) - 8x8 grid
    output_core_range = ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_input_cores - 1, rows_per_input_core - 1),
    )
    output_core_range_set = ttnn.CoreRangeSet({output_core_range})

    # Build output core list in row-major order (y first, then x)
    output_core_list = []
    for y in range(rows_per_input_core):
        for x in range(num_input_cores):
            output_core_list.append(ttnn.CoreCoord(x, y))

    # Build custom core mapping: (output_core, input_core, row_offset)
    # Each output core at (x, y) reads row y from input core at (x, 0)
    core_mapping = []
    for output_core in output_core_list:
        input_core = ttnn.CoreCoord(output_core.x, 0)  # Input core is in row 0, same column
        row_offset = output_core.y  # Row offset is the output core's y coordinate
        core_mapping.append((output_core, input_core, row_offset))

    logger.info(f"Custom core mapping: output (x,y) reads row y from input (x,0)")
    logger.info(f"Example: output (3,5) reads row 5 from input (3,0)")

    # Memory configs
    input_shard_spec = ttnn.ShardSpec(
        input_core_range_set,
        (rows_per_input_core, width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        input_shard_spec,
    )

    output_shard_spec = ttnn.ShardSpec(
        output_core_range_set,
        (1, width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        output_shard_spec,
    )

    # Create tensors using ROW_MAJOR layout
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    torch_output = torch.zeros(output_shape, dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=output_mem_config,
    )

    # Run operation with custom core mapping
    ttnn_result = ScatterHeads.op(
        ttnn_input,
        ttnn_output,
        core_mapping=core_mapping,  # Pass the custom mapping
    )

    output_torch = ttnn.to_torch(ttnn_result)

    # Verify each output core received the correct data based on the custom mapping
    # Output is in row-major order: (0,0), (1,0), ..., (7,0), (0,1), (1,1), ..., (7,7)
    # Input is organized as: input_core_0_rows[0:8], input_core_1_rows[0:8], ..., input_core_7_rows[0:8]
    for i, output_core in enumerate(output_core_list):
        # Output core at (x, y) reads row y from input core x
        input_core_idx = output_core.x
        row_in_input_shard = output_core.y

        # The input tensor is organized as: all rows of input_core_0, then all rows of input_core_1, etc.
        expected_input_row_idx = input_core_idx * rows_per_input_core + row_in_input_shard
        expected_row = torch_input[expected_input_row_idx, :]
        actual_row = output_torch[i, :]

        if not torch.allclose(expected_row, actual_row, atol=1e-2, rtol=1e-2):
            logger.error(f"Output core ({output_core.x}, {output_core.y}) mismatch!")
            logger.error(
                f"Expected input row {expected_input_row_idx} (core {input_core_idx}, row {row_in_input_shard})"
            )
            logger.error(f"Expected first 5 values: {expected_row[:5]}")
            logger.error(f"Actual first 5 values: {actual_row[:5]}")
            assert False, f"Output core ({output_core.x}, {output_core.y}) data integrity check failed"

    logger.info("✓ ScatterHeads with custom mapping test passed!")

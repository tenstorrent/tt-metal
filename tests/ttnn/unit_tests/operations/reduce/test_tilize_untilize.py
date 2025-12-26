# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("height", [32, 64, 128])
@pytest.mark.parametrize("width", [32, 64, 128])
def test_tilize_untilize(device, batch_size, channels, height, width):
    """
    Test tilize_untilize operation which serves as a template for compute operations.

    This operation takes row-major input, tilizes it for compute, then untilizes back
    to row-major output. As an identity operation, output should exactly match input.

    Requirements:
    - Height and width must be multiples of 32 (tile-aligned)
    - Input must be in ROW_MAJOR layout
    - Input must be INTERLEAVED (not sharded)
    """
    torch.manual_seed(0)

    shape = (batch_size, channels, height, width)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.tilize_untilize(input_tensor)

    # Verify output properties
    assert (
        output_tensor.shape == input_tensor.shape
    ), f"Shape mismatch: expected {input_tensor.shape}, got {output_tensor.shape}"
    assert (
        output_tensor.dtype == input_tensor.dtype
    ), f"Dtype mismatch: expected {input_tensor.dtype}, got {output_tensor.dtype}"
    assert (
        output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Layout mismatch: expected ROW_MAJOR_LAYOUT, got {output_tensor.layout}"

    # Convert back to torch for comparison
    torch_output = ttnn.to_torch(output_tensor)

    # Identity operation: output should match input exactly
    # Using PCC=1.0 since this is a lossless transform
    assert_with_pcc(torch_input, torch_output, pcc=0.9999)


@pytest.mark.parametrize(
    "batch_size,channels,height,width,description",
    [
        # Heights that create odd numbers of tile rows to trigger cliff cores
        # nblocks = (height / 32) * batch_size
        (1, 1, 160, 64, "5 tile rows - odd number"),
        (1, 1, 224, 64, "7 tile rows - prime number"),
        (1, 1, 352, 64, "11 tile rows - prime number"),
        (1, 1, 96, 64, "3 tile rows - small prime"),
        # Batch combined with height to create cliff scenarios
        (2, 1, 160, 64, "10 tile rows via batch - even but not power of 2"),
        (3, 1, 64, 64, "6 tile rows via batch"),
        (1, 1, 288, 64, "9 tile rows - 3x3"),
        # Wider tensors with cliff-inducing heights
        (1, 1, 160, 256, "5 tile rows, wide tensor"),
        (1, 1, 224, 128, "7 tile rows, moderate width"),
        # Larger tensors that still have cliffs
        (1, 1, 544, 128, "17 tile rows - prime, larger"),
        (2, 1, 224, 64, "14 tile rows via batch"),
        # Edge case: minimal cliff (1 block difference)
        (1, 1, 64, 64, "2 tile rows - even split baseline"),
        (1, 1, 96, 64, "3 tile rows - 1 row cliff on some core counts"),
    ],
)
def test_tilize_untilize_cliff_cores(device, batch_size, channels, height, width, description):
    """
    Test tilize_untilize with tensor sizes that exercise cliff core handling.

    In multi-core mode, work is distributed using split_blocks_for_tilize():
    - Each core processes nblocks_per_core tile rows
    - The last "cliff" core may process fewer blocks (nblocks_per_core_cliff)

    Example: 7 tile rows on 3 cores:
    - Cores 0, 1: 3 blocks each
    - Core 2 (cliff): 1 block

    This test uses tensor heights that create block counts which don't divide
    evenly across typical core counts, ensuring cliff core logic is exercised.
    """
    torch.manual_seed(42)

    shape = (batch_size, channels, height, width)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.tilize_untilize(input_tensor)

    # Verify output properties
    assert output_tensor.shape == input_tensor.shape, f"Shape mismatch: {description}"
    assert output_tensor.dtype == input_tensor.dtype, f"Dtype mismatch: {description}"
    assert output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, f"Layout mismatch: {description}"

    # Convert back to torch for comparison
    torch_output = ttnn.to_torch(output_tensor)

    # Identity operation: output should match input exactly
    assert_with_pcc(torch_input, torch_output, pcc=0.9999)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("height", [32, 64, 128])
@pytest.mark.parametrize("width", [32, 64, 128])
def test_tilize_untilize_reduce_w_avg(device, batch_size, channels, height, width):
    """
    Test REDUCE_W_AVG operation which reduces along the width dimension.

    Input shape: [batch_size, channels, height, width]
    Output shape: [batch_size, channels, height, 32] - width reduced to TILE_WIDTH

    The reduced value (mean) is stored in column 0; columns 1-31 are zeros.
    """
    torch.manual_seed(0)

    shape = (batch_size, channels, height, width)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.tilize_untilize(input_tensor, op_type=ttnn.TilizeUntilizeOpType.REDUCE_W_AVG)

    # Verify output shape: width reduced to 32 (TILE_WIDTH)
    expected_shape = (batch_size, channels, height, 32)
    assert (
        output_tensor.shape == expected_shape
    ), f"Shape mismatch: expected {expected_shape}, got {output_tensor.shape}"
    assert (
        output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Layout mismatch: expected ROW_MAJOR_LAYOUT, got {output_tensor.layout}"

    # Convert back to torch for comparison
    torch_output = ttnn.to_torch(output_tensor)

    # Compute expected: mean along width dimension
    torch_expected = torch.mean(torch_input, dim=-1, keepdim=True)

    # Extract first column from output (where reduced values are stored)
    torch_output_col0 = torch_output[:, :, :, 0:1]

    # Verify reduced values match expected
    assert_with_pcc(torch_expected, torch_output_col0, pcc=0.999)

    # Verify columns 1-31 are zeros
    assert torch.all(torch_output[:, :, :, 1:] == 0), "Columns 1-31 should be zeros"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("height", [32, 64, 128])
@pytest.mark.parametrize("width", [32, 64, 128])
def test_tilize_untilize_reduce_w_sum(device, batch_size, channels, height, width):
    """
    Test REDUCE_W_SUM operation which reduces along the width dimension.

    Input shape: [batch_size, channels, height, width]
    Output shape: [batch_size, channels, height, 32] - width reduced to TILE_WIDTH

    The reduced value (sum) is stored in column 0; columns 1-31 are zeros.
    """
    torch.manual_seed(0)

    shape = (batch_size, channels, height, width)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.tilize_untilize(input_tensor, op_type=ttnn.TilizeUntilizeOpType.REDUCE_W_SUM)

    # Verify output shape: width reduced to 32 (TILE_WIDTH)
    expected_shape = (batch_size, channels, height, 32)
    assert (
        output_tensor.shape == expected_shape
    ), f"Shape mismatch: expected {expected_shape}, got {output_tensor.shape}"
    assert (
        output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Layout mismatch: expected ROW_MAJOR_LAYOUT, got {output_tensor.layout}"

    # Convert back to torch for comparison
    torch_output = ttnn.to_torch(output_tensor)

    # Compute expected: sum along width dimension
    torch_expected = torch.sum(torch_input, dim=-1, keepdim=True)

    # Extract first column from output (where reduced values are stored)
    torch_output_col0 = torch_output[:, :, :, 0:1]

    # Verify reduced values match expected
    assert_with_pcc(torch_expected, torch_output_col0, pcc=0.999)

    # Verify columns 1-31 are zeros
    assert torch.all(torch_output[:, :, :, 1:] == 0), "Columns 1-31 should be zeros"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("height", [32, 64, 128])
@pytest.mark.parametrize("width", [32, 64, 128])
def test_tilize_untilize_reduce_w_max(device, batch_size, channels, height, width):
    """
    Test REDUCE_W_MAX operation which reduces along the width dimension.

    Input shape: [batch_size, channels, height, width]
    Output shape: [batch_size, channels, height, 32] - width reduced to TILE_WIDTH

    The reduced value (max) is stored in column 0; columns 1-31 are zeros.
    """
    torch.manual_seed(0)

    shape = (batch_size, channels, height, width)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.tilize_untilize(input_tensor, op_type=ttnn.TilizeUntilizeOpType.REDUCE_W_MAX)

    # Verify output shape: width reduced to 32 (TILE_WIDTH)
    expected_shape = (batch_size, channels, height, 32)
    assert (
        output_tensor.shape == expected_shape
    ), f"Shape mismatch: expected {expected_shape}, got {output_tensor.shape}"
    assert (
        output_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"Layout mismatch: expected ROW_MAJOR_LAYOUT, got {output_tensor.layout}"

    # Convert back to torch for comparison
    torch_output = ttnn.to_torch(output_tensor)

    # Compute expected: max along width dimension
    torch_expected = torch.max(torch_input, dim=-1, keepdim=True).values

    # Extract first column from output (where reduced values are stored)
    torch_output_col0 = torch_output[:, :, :, 0:1]

    # Verify reduced values match expected
    assert_with_pcc(torch_expected, torch_output_col0, pcc=0.999)

    # Verify columns 1-31 are zeros
    assert torch.all(torch_output[:, :, :, 1:] == 0), "Columns 1-31 should be zeros"

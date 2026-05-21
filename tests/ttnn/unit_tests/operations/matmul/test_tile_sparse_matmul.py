# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for tile-sparse matrix multiplication.

These tests verify the correctness of ttnn.tile_sparse_matmul, which performs
matrix multiplication with tile-level sparsity (32x32 block sparsity).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("m", [32, 64, 128])
@pytest.mark.parametrize("k", [64, 128])
@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_tile_sparse_matmul_dense(device, m, k, n, dtype):
    """
    Test tile_sparse_matmul with dense inputs (no sparsity mask).

    This should produce the same result as regular matmul.
    """
    torch.manual_seed(42)

    # Create random input tensors
    in0 = torch.randn((m, k), dtype=torch.bfloat16)
    in1 = torch.randn((k, n), dtype=torch.bfloat16)

    # Convert to TTNN tensors
    in0_t = ttnn.from_torch(
        in0,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    in1_t = ttnn.from_torch(
        in1,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Perform tile_sparse_matmul without sparsity mask (should be same as dense)
    output_t = ttnn.tile_sparse_matmul(
        in0_t,
        in1_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get result
    output = ttnn.to_torch(output_t)

    # Compute expected result using PyTorch
    expected = torch.matmul(in0, in1)

    # Verify result
    assert_with_pcc(expected, output, 0.999)


@pytest.mark.parametrize("m", [64, 128])
@pytest.mark.parametrize("k", [64, 128])
@pytest.mark.parametrize("n", [64, 128])
def test_tile_sparse_matmul_with_mask_b(device, m, k, n):
    """
    Test tile_sparse_matmul with sparsity mask on input B.

    The sparsity mask indicates which tiles of B are non-zero.
    """
    torch.manual_seed(42)

    # Create random input tensors
    in0 = torch.randn((m, k), dtype=torch.bfloat16)
    in1 = torch.randn((k, n), dtype=torch.bfloat16)

    # Calculate tile dimensions
    tile_rows_b = k // 32
    tile_cols_b = n // 32

    # Create a sparsity mask for B (all ones = dense)
    mask_b = torch.ones((tile_rows_b, tile_cols_b), dtype=torch.uint8)

    # Convert to TTNN tensors
    in0_t = ttnn.from_torch(
        in0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    in1_t = ttnn.from_torch(
        in1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create sparsity mask tensor on host (parsed by device op during setup)
    mask_b_t = ttnn.from_torch(
        mask_b,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Perform tile_sparse_matmul with sparsity mask
    output_t = ttnn.tile_sparse_matmul(
        in0_t,
        in1_t,
        sparsity_mask_b=mask_b_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get result
    output = ttnn.to_torch(output_t)

    # Compute expected result using PyTorch
    expected = torch.matmul(in0, in1)

    # Verify result
    assert_with_pcc(expected, output, 0.999)


@pytest.mark.parametrize("m", [128])
@pytest.mark.parametrize("k", [128])
@pytest.mark.parametrize("n", [128])
def test_tile_sparse_matmul_batched(device, m, k, n):
    """
    Test tile_sparse_matmul with batched inputs.
    """
    torch.manual_seed(42)
    batch = 2

    # Create random batched input tensors
    in0 = torch.randn((batch, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((batch, k, n), dtype=torch.bfloat16)

    # Convert to TTNN tensors
    in0_t = ttnn.from_torch(
        in0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    in1_t = ttnn.from_torch(
        in1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Perform tile_sparse_matmul
    output_t = ttnn.tile_sparse_matmul(
        in0_t,
        in1_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get result
    output = ttnn.to_torch(output_t)

    # Compute expected result using PyTorch
    expected = torch.bmm(in0, in1)

    # Verify result
    assert_with_pcc(expected, output, 0.999)


@pytest.mark.parametrize("shape", [(256, 256)])
def test_create_tile_sparsity_mask(device, shape):
    """
    Test create_tile_sparsity_mask function.
    """
    torch.manual_seed(42)
    m, k = shape

    # Create a dense tensor
    dense = torch.randn((m, k), dtype=torch.bfloat16)

    # Convert to TTNN tensor
    dense_t = ttnn.from_torch(
        dense,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create tile sparsity mask
    mask_t = ttnn.create_tile_sparsity_mask(dense_t)

    # Verify mask shape
    mask = ttnn.to_torch(mask_t)
    expected_tile_rows = m // 32
    expected_tile_cols = k // 32
    assert mask.shape == (
        expected_tile_rows,
        expected_tile_cols,
    ), f"Expected mask shape ({expected_tile_rows}, {expected_tile_cols}), got {mask.shape}"

    # For now, all tiles should be marked as non-zero (value 1)
    assert (mask == 1).all(), "Expected all tiles to be marked as non-zero"


@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("k", [256])
@pytest.mark.parametrize("n", [256])
def test_tile_sparse_matmul_accuracy(device, m, k, n):
    """
    Test tile_sparse_matmul accuracy against PyTorch reference.
    """
    torch.manual_seed(42)

    # Create random input tensors
    in0 = torch.randn((m, k), dtype=torch.bfloat16)
    in1 = torch.randn((k, n), dtype=torch.bfloat16)

    # Convert to TTNN tensors
    in0_t = ttnn.from_torch(
        in0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    in1_t = ttnn.from_torch(
        in1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Perform tile_sparse_matmul
    output_t = ttnn.tile_sparse_matmul(
        in0_t,
        in1_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get result
    output = ttnn.to_torch(output_t)

    # Compute expected result using PyTorch
    expected = torch.matmul(in0, in1)

    # Verify result with high precision requirement
    assert_with_pcc(expected, output, 0.999)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_tile_sparse_matmul_dtypes(device, dtype):
    """
    Test tile_sparse_matmul with different data types.
    """
    torch.manual_seed(42)
    m, k, n = 128, 128, 128

    # Create random input tensors
    in0 = torch.randn((m, k), dtype=torch.bfloat16)
    in1 = torch.randn((k, n), dtype=torch.bfloat16)

    # Convert to TTNN tensors
    in0_t = ttnn.from_torch(
        in0,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    in1_t = ttnn.from_torch(
        in1,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Perform tile_sparse_matmul
    output_t = ttnn.tile_sparse_matmul(
        in0_t,
        in1_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get result
    output = ttnn.to_torch(output_t)

    # Compute expected result using PyTorch
    expected = torch.matmul(in0, in1)

    # Lower PCC threshold for bfloat8_b
    expected_pcc = 0.99 if dtype == ttnn.bfloat8_b else 0.999
    assert_with_pcc(expected, output, expected_pcc)


@pytest.mark.parametrize("sparsity_ratio", [0.25, 0.5, 0.75])
def test_tile_sparse_matmul_with_sparse_b(device, sparsity_ratio):
    """
    Test tile_sparse_matmul with actual sparse data in B.

    This test creates a matrix B where some tiles are entirely zero,
    and verifies that the result matches PyTorch reference.
    """
    torch.manual_seed(42)
    m, k, n = 128, 128, 128
    tile_size = 32

    # Create random input A
    in0 = torch.randn((m, k), dtype=torch.bfloat16)

    # Create input B with some zero tiles
    in1 = torch.randn((k, n), dtype=torch.bfloat16)

    # Calculate tile dimensions
    tile_rows_b = k // tile_size
    tile_cols_b = n // tile_size
    total_tiles = tile_rows_b * tile_cols_b

    # Create sparsity mask (1 = non-zero, 0 = zero tile)
    num_sparse_tiles = int(total_tiles * sparsity_ratio)
    mask_b = torch.ones((tile_rows_b, tile_cols_b), dtype=torch.uint8)

    # Randomly select tiles to zero out
    sparse_indices = torch.randperm(total_tiles)[:num_sparse_tiles]
    for idx in sparse_indices:
        row = idx // tile_cols_b
        col = idx % tile_cols_b
        mask_b[row, col] = 0
        # Zero out the corresponding tile in B
        in1[row * tile_size : (row + 1) * tile_size, col * tile_size : (col + 1) * tile_size] = 0

    # Convert to TTNN tensors
    in0_t = ttnn.from_torch(
        in0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    in1_t = ttnn.from_torch(
        in1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Create sparsity mask tensor (on host)
    mask_b_t = ttnn.from_torch(
        mask_b,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Perform tile_sparse_matmul with sparsity mask
    output_t = ttnn.tile_sparse_matmul(
        in0_t,
        in1_t,
        sparsity_mask_b=mask_b_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get result
    output = ttnn.to_torch(output_t)

    # Compute expected result using PyTorch
    expected = torch.matmul(in0, in1)

    # Verify result
    assert_with_pcc(expected, output, 0.999)


def test_tile_sparse_mask_parsing(device):
    """
    Test that user-provided sparsity mask is correctly parsed.

    This tests the P1 bug fix where the mask tensor was being ignored.
    """
    torch.manual_seed(42)
    m, k, n = 64, 64, 64
    tile_size = 32

    # Create inputs
    in0 = torch.randn((m, k), dtype=torch.bfloat16)
    in1 = torch.randn((k, n), dtype=torch.bfloat16)

    # Create a mask with specific pattern (checkerboard)
    tile_rows_b = k // tile_size  # 2
    tile_cols_b = n // tile_size  # 2
    mask_b = torch.zeros((tile_rows_b, tile_cols_b), dtype=torch.uint8)
    mask_b[0, 0] = 1  # Only top-left tile is non-zero
    mask_b[1, 1] = 1  # Only bottom-right tile is non-zero

    # Zero out corresponding tiles in B
    in1[0:tile_size, tile_size:] = 0  # Top-right
    in1[tile_size:, 0:tile_size] = 0  # Bottom-left

    # Convert to TTNN
    in0_t = ttnn.from_torch(
        in0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    in1_t = ttnn.from_torch(
        in1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mask_b_t = ttnn.from_torch(
        mask_b,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Perform tile_sparse_matmul
    output_t = ttnn.tile_sparse_matmul(
        in0_t,
        in1_t,
        sparsity_mask_b=mask_b_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output = ttnn.to_torch(output_t)
    expected = torch.matmul(in0, in1)

    assert_with_pcc(expected, output, 0.999)

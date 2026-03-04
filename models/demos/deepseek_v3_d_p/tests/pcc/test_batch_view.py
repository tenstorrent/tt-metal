# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the batch_view operation.

The batch_view operation creates a zero-copy view of a single batch from a 3D tensor [b, M, N] -> [M, N].

Usage:
    pytest models/demos/deepseek_v3_d_p/tests/pcc/test_batch_view.py -vvv
"""

import pytest
import torch
import ttnn
from loguru import logger


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize(
    "b,M,N",
    [
        (4, 512, 1024),  # TILE-aligned: M*N = 524288, divisible by 1024
        (8, 256, 2048),  # TILE-aligned
        (2, 32, 32),  # Minimal tile-aligned case
        (16, 1024, 512),  # Larger case
    ],
)
def test_batch_view_tile(mesh_device, b, M, N):
    """Test batch_view with TILE layout."""
    # Create input tensor [b, M, N]
    torch_input = torch.randn(b, M, N, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"Input shape: {ttnn_input.shape}, layout: {ttnn_input.layout}")

    for batch_idx in range(b):
        # Create view
        view = ttnn.experimental.deepseek.batch_view(ttnn_input, batch_idx)

        # Verify shape
        expected_shape = ttnn.Shape([M, N])
        assert view.shape == expected_shape, f"Expected shape {expected_shape}, got {view.shape}"

        # Verify data matches
        view_torch = ttnn.to_torch(view)
        expected = torch_input[batch_idx]
        assert torch.allclose(view_torch, expected, atol=1e-2), (
            f"Data mismatch at batch_idx={batch_idx}: max diff = {(view_torch - expected).abs().max().item()}"
        )

        logger.info(f"batch_idx={batch_idx} passed: shape={view.shape}")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize(
    "b,M,N",
    [
        (4, 512, 1024),  # Standard case
        (2, 100, 200),  # Non-tile-aligned dimensions
        (8, 33, 77),  # Odd dimensions
    ],
)
def test_batch_view_row_major(mesh_device, b, M, N):
    """Test batch_view with ROW_MAJOR layout."""
    # Create input tensor [b, M, N]
    torch_input = torch.randn(b, M, N, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(f"Input shape: {ttnn_input.shape}, layout: {ttnn_input.layout}")

    for batch_idx in range(b):
        # Create view
        view = ttnn.experimental.deepseek.batch_view(ttnn_input, batch_idx)

        # Verify shape
        expected_shape = ttnn.Shape([M, N])
        assert view.shape == expected_shape, f"Expected shape {expected_shape}, got {view.shape}"

        # Verify data matches
        view_torch = ttnn.to_torch(view)
        expected = torch_input[batch_idx]
        assert torch.allclose(view_torch, expected, atol=1e-3), (
            f"Data mismatch at batch_idx={batch_idx}: max diff = {(view_torch - expected).abs().max().item()}"
        )

        logger.info(f"batch_idx={batch_idx} passed: shape={view.shape}")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_batch_view_invalid_index(mesh_device):
    """Test that invalid batch index throws."""
    torch_input = torch.randn(4, 512, 1024, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(RuntimeError):
        ttnn.experimental.deepseek.batch_view(ttnn_input, 4)  # Out of range

    logger.info("Invalid index test passed: RuntimeError raised as expected")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_batch_view_invalid_rank(mesh_device):
    """Test that non-3D tensor throws."""
    # 2D tensor
    torch_input_2d = torch.randn(512, 1024, dtype=torch.bfloat16)
    ttnn_input_2d = ttnn.from_torch(
        torch_input_2d,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with pytest.raises(RuntimeError):
        ttnn.experimental.deepseek.batch_view(ttnn_input_2d, 0)

    logger.info("Invalid rank test passed: RuntimeError raised as expected")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_batch_view_no_copy(mesh_device):
    """Test that batch_view creates a view (no copy) by verifying addresses are the same base."""
    torch_input = torch.randn(4, 512, 1024, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get the original buffer address
    original_address = ttnn_input.buffer_address()

    logger.info(f"Original address: {original_address}")

    for batch_idx in range(4):
        view = ttnn.experimental.deepseek.batch_view(ttnn_input, batch_idx)

        # The view should have the same base address (offset is handled internally via root_buffer_offset)
        view_address = view.buffer_address()
        logger.info(f"Batch {batch_idx}: view_address={view_address}, original_address={original_address}")

        assert view_address == original_address, (
            f"View address {view_address} should match original address {original_address} "
            f"(offset is handled internally via root_buffer_offset)"
        )

    logger.info("No-copy test passed: view addresses point to same base buffer")


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        ttnn.float32,
    ],
)
def test_batch_view_dtypes(mesh_device, dtype):
    """Test batch_view with different data types."""
    b, M, N = 4, 512, 1024

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch_input = torch.randn(b, M, N, dtype=torch_dtype)

    ttnn_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=dtype,
    )

    view = ttnn.experimental.deepseek.batch_view(ttnn_input, 0)

    assert view.dtype == dtype
    assert view.shape == ttnn.Shape([M, N])

    view_torch = ttnn.to_torch(view)
    expected = torch_input[0]

    # float32 should have exact match, bfloat16 has some tolerance
    atol = 1e-2 if dtype == ttnn.bfloat16 else 1e-5
    assert torch.allclose(view_torch.to(torch_dtype), expected, atol=atol)

    logger.info(f"dtype={dtype} test passed")

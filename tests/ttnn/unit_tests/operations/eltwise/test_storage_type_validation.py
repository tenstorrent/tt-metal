# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import random

# Set fixed random seeds for deterministic test execution
torch.manual_seed(42)


def test_binary_add_storage_type_validation_device_tensors(device):
    """Test that binary add (use_legacy=True) works correctly with device tensors."""
    # Create tensors on device (should work)
    x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y_torch = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Verify these are device tensors
    assert x_tt.storage_type() == ttnn.StorageType.DEVICE
    assert y_tt.storage_type() == ttnn.StorageType.DEVICE

    # This should work without errors
    result = ttnn.add(x_tt, y_tt, use_legacy=True)
    result_torch = ttnn.to_torch(result)

    expected = x_torch + y_torch
    assert torch.allclose(result_torch, expected, atol=1e-5, rtol=1e-5)


def test_binary_ng_add_storage_type_validation_device_tensors(device):
    """Test that binary_ng add (use_legacy=False) works correctly with device tensors."""
    # Create tensors on device (should work)
    x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y_torch = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Verify these are device tensors
    assert x_tt.storage_type() == ttnn.StorageType.DEVICE
    assert y_tt.storage_type() == ttnn.StorageType.DEVICE

    # This should work without errors
    result = ttnn.add(x_tt, y_tt, use_legacy=False)
    result_torch = ttnn.to_torch(result)

    expected = x_torch + y_torch
    assert torch.allclose(result_torch, expected, atol=1e-5, rtol=1e-5)


def test_binary_add_storage_type_validation_host_tensors():
    """Test that binary add (use_legacy=True) raises runtime error when given host tensors."""
    # Create tensors without device (should be host tensors)
    x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y_torch = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    # Verify these are host tensors
    assert x_tt.storage_type() == ttnn.StorageType.HOST
    assert y_tt.storage_type() == ttnn.StorageType.HOST

    # This should raise a runtime error due to our storage type validation
    with pytest.raises(RuntimeError, match="Input tensor A must be on device"):
        ttnn.add(x_tt, y_tt, use_legacy=True)


def test_binary_ng_add_storage_type_validation_host_tensors():
    """Test that binary_ng add (use_legacy=False) raises runtime error when given host tensors."""
    # Create tensors without device (should be host tensors)
    x_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    y_torch = torch.tensor([[0.5, 1.5], [2.5, 3.5]], dtype=torch.float32)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

    # Verify these are host tensors
    assert x_tt.storage_type() == ttnn.StorageType.HOST
    assert y_tt.storage_type() == ttnn.StorageType.HOST

    # This should raise a runtime error due to our storage type validation
    with pytest.raises(RuntimeError, match="Input tensor A must be on device"):
        ttnn.add(x_tt, y_tt, use_legacy=False)

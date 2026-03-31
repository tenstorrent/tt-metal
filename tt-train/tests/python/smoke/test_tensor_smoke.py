# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ttml tensor operations.

These tests verify basic tensor creation and conversion functionality.
"""

import numpy as np
import pytest

import ttml
import ttnn


@pytest.mark.smoke
def test_tensor_from_numpy_float32():
    """Verify tensor creation from float32 numpy array."""
    data = np.random.randn(1, 1, 32, 32).astype(np.float32)
    tensor = ttml.autograd.Tensor.from_numpy(data)

    assert tensor is not None
    result = tensor.to_numpy()
    assert np.allclose(data, result)


@pytest.mark.smoke
def test_tensor_from_numpy_bfloat16():
    """Verify tensor creation with bfloat16 conversion."""
    data = np.random.randn(1, 1, 32, 32).astype(np.float32)
    tensor = ttml.autograd.Tensor.from_numpy(data, new_type=ttnn.DataType.BFLOAT16)

    assert tensor is not None
    result = tensor.to_numpy(ttnn.DataType.FLOAT32)
    assert result.shape == data.shape


@pytest.mark.smoke
def test_tensor_shape():
    """Verify tensor shape reporting."""
    data = np.random.randn(2, 3, 32, 64).astype(np.float32)
    tensor = ttml.autograd.Tensor.from_numpy(data)

    shape = tensor.shape()
    assert shape == [2, 3, 32, 64]


@pytest.mark.smoke
def test_tensor_layout_row_major():
    """Verify tensor creation with ROW_MAJOR layout."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor = ttml.autograd.Tensor.from_numpy(data, layout=ttnn.Layout.ROW_MAJOR)

    assert tensor is not None
    result = tensor.to_numpy()
    assert np.allclose(data, result)


@pytest.mark.smoke
def test_tensor_roundtrip_consistency():
    """Verify numpy -> tensor -> numpy roundtrip preserves data."""
    np.random.seed(42)
    data = np.random.randn(1, 1, 32, 32).astype(np.float32)

    tensor = ttml.autograd.Tensor.from_numpy(data)
    result = tensor.to_numpy()

    assert data.shape == result.shape
    assert np.allclose(data, result, rtol=1e-5, atol=1e-5)

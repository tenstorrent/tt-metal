# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ttml operations.

These tests verify that core operations work correctly on device.
"""

import numpy as np
import pytest

import ttml
import ttnn


@pytest.mark.smoke
@pytest.mark.requires_device
def test_binary_add():
    """Verify basic tensor addition."""
    a = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 32, 32), dtype=np.float32))
    b = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 32, 32), dtype=np.float32))

    c = a + b

    result = c.to_numpy()
    assert np.allclose(result, 2.0)


@pytest.mark.smoke
@pytest.mark.requires_device
def test_binary_sub():
    """Verify basic tensor subtraction."""
    a = ttml.autograd.Tensor.from_numpy(np.full((1, 1, 32, 32), 5.0, dtype=np.float32))
    b = ttml.autograd.Tensor.from_numpy(np.full((1, 1, 32, 32), 3.0, dtype=np.float32))

    c = a - b

    result = c.to_numpy()
    assert np.allclose(result, 2.0)


@pytest.mark.smoke
@pytest.mark.requires_device
def test_binary_mul():
    """Verify basic tensor multiplication."""
    a = ttml.autograd.Tensor.from_numpy(np.full((1, 1, 32, 32), 3.0, dtype=np.float32))
    b = ttml.autograd.Tensor.from_numpy(np.full((1, 1, 32, 32), 4.0, dtype=np.float32))

    c = a * b

    result = c.to_numpy()
    assert np.allclose(result, 12.0)


@pytest.mark.smoke
@pytest.mark.requires_device
def test_scalar_mul():
    """Verify scalar multiplication."""
    a = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 32, 32), dtype=np.float32))

    c = a * 5.0

    result = c.to_numpy()
    assert np.allclose(result, 5.0)


@pytest.mark.smoke
@pytest.mark.requires_device
def test_mse_loss():
    """Verify MSE loss computation."""
    pred = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 32, 32), dtype=np.float32))
    target = ttml.autograd.Tensor.from_numpy(np.zeros((1, 1, 32, 32), dtype=np.float32))

    loss = ttml.ops.loss.mse_loss(pred, target, ttml.ops.ReduceType.MEAN)

    assert loss is not None
    loss_val = float(loss.to_numpy(ttnn.DataType.FLOAT32))
    assert np.isfinite(loss_val)
    assert loss_val > 0


@pytest.mark.smoke
@pytest.mark.requires_device
def test_unary_mean():
    """Verify unary mean operation."""
    data = np.full((1, 1, 32, 32), 4.0, dtype=np.float32)
    tensor = ttml.autograd.Tensor.from_numpy(data)

    mean = ttml.ops.unary.mean(tensor)

    result = float(mean.to_numpy(ttnn.DataType.FLOAT32))
    assert np.isclose(result, 4.0, rtol=0.01)

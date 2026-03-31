# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ttml forward/backward pass and training loop.

These tests verify that end-to-end training workflows complete successfully.
"""

import numpy as np
import pytest

import ttml
import ttnn
from ttml.modules import AbstractModuleBase, Parameter


@pytest.mark.smoke
@pytest.mark.requires_device
def test_simple_forward_pass():
    """Verify a simple forward pass completes."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np, new_type=ttnn.DataType.BFLOAT16))

        def __call__(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    model = SimpleModule()
    model.eval()

    x = ttml.autograd.Tensor.from_numpy(np.random.randn(1, 1, 32, 32).astype(np.float32))
    output = model(x)

    assert output is not None
    result = output.to_numpy(ttnn.DataType.FLOAT32)
    assert result.shape == (1, 1, 32, 32)
    assert np.all(np.isfinite(result))


@pytest.mark.smoke
@pytest.mark.requires_device
def test_backward_pass():
    """Verify backward pass computes gradients."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np, new_type=ttnn.DataType.BFLOAT16))

        def __call__(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    model = SimpleModule()
    model.train()

    x = ttml.autograd.Tensor.from_numpy(np.random.randn(1, 1, 32, 32).astype(np.float32))
    output = model(x)
    loss = ttml.ops.unary.mean(output)

    loss.backward(False)


@pytest.mark.smoke
@pytest.mark.requires_device
def test_single_training_step():
    """Verify a single training step (forward + backward + optimizer step)."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np, new_type=ttnn.DataType.BFLOAT16))

        def __call__(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    model = SimpleModule()
    params = model.parameters()

    config = ttml.optimizers.SGDConfig.make(0.1, 0.0, 0.0, 0.0, False)
    optimizer = ttml.optimizers.SGD(params, config)

    model.train()
    optimizer.zero_grad()

    x = ttml.autograd.Tensor.from_numpy(np.random.randn(1, 1, 32, 32).astype(np.float32))
    target = ttml.autograd.Tensor.from_numpy(np.zeros((1, 1, 32, 32), dtype=np.float32))

    output = model(x)
    loss = ttml.ops.loss.mse_loss(output, target, ttml.ops.ReduceType.MEAN)

    loss_val = float(loss.to_numpy(ttnn.DataType.FLOAT32))
    assert np.isfinite(loss_val)

    loss.backward(False)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    optimizer.step()


@pytest.mark.smoke
@pytest.mark.requires_device
def test_linear_regression_model_forward():
    """Verify LinearRegression model forward pass works."""
    from ttml.models.linear_regression import create_linear_regression_model

    model = create_linear_regression_model(input_features=32, output_features=1)
    model.eval()

    x = ttml.autograd.Tensor.from_numpy(np.random.randn(4, 1, 1, 32).astype(np.float32))
    output = model(x)

    assert output is not None
    result = output.to_numpy(ttnn.DataType.FLOAT32)
    assert result.shape == (4, 1, 1, 1)
    assert np.all(np.isfinite(result))


@pytest.mark.smoke
@pytest.mark.requires_device
def test_linear_regression_training_loop():
    """Verify a minimal training loop with LinearRegression model."""
    from ttml.models.linear_regression import create_linear_regression_model

    np.random.seed(42)

    model = create_linear_regression_model(input_features=32, output_features=1)
    params = model.parameters()

    config = ttml.optimizers.SGDConfig.make(0.01, 0.0, 0.0, 0.0, False)
    optimizer = ttml.optimizers.SGD(params, config)

    model.train()

    batch_size = 4
    n_steps = 3

    for _ in range(n_steps):
        optimizer.zero_grad()

        x = ttml.autograd.Tensor.from_numpy(np.random.randn(batch_size, 1, 1, 32).astype(np.float32))
        target = ttml.autograd.Tensor.from_numpy(np.random.randn(batch_size, 1, 1, 1).astype(np.float32))

        output = model(x)
        loss = ttml.ops.loss.mse_loss(output, target, ttml.ops.ReduceType.MEAN)

        loss_val = float(loss.to_numpy(ttnn.DataType.FLOAT32))
        assert np.isfinite(loss_val), "Loss should be finite"

        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()


@pytest.mark.smoke
@pytest.mark.requires_device
def test_adamw_training_step():
    """Verify training step with AdamW optimizer."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np, new_type=ttnn.DataType.BFLOAT16))

        def __call__(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    model = SimpleModule()
    params = model.parameters()

    config = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.01)
    optimizer = ttml.optimizers.AdamW(params, config)

    model.train()

    for _ in range(2):
        optimizer.zero_grad()

        x = ttml.autograd.Tensor.from_numpy(np.random.randn(1, 1, 32, 32).astype(np.float32))
        target = ttml.autograd.Tensor.from_numpy(np.zeros((1, 1, 32, 32), dtype=np.float32))

        output = model(x)
        loss = ttml.ops.loss.mse_loss(output, target, ttml.ops.ReduceType.MEAN)

        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        optimizer.step()

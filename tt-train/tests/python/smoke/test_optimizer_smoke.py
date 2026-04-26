# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ttml optimizers.

These tests verify that optimizers can be created and perform basic operations.
"""

import numpy as np
import pytest

import ttml
import ttnn
from ttml.modules import AbstractModuleBase, Parameter


@pytest.mark.smoke
def test_sgd_config_creation():
    """Verify SGDConfig can be created."""
    config = ttml.optimizers.SGDConfig.make(lr=0.01, momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=False)

    assert config is not None


@pytest.mark.smoke
def test_adamw_config_creation():
    """Verify AdamWConfig can be created."""
    config = ttml.optimizers.AdamWConfig.make(lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01)

    assert config is not None


@pytest.mark.smoke
def test_sgd_optimizer_creation():
    """Verify SGD optimizer can be created with module parameters."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

        def __call__(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    model = SimpleModule()
    params = model.parameters()

    config = ttml.optimizers.SGDConfig.make(0.01, 0.0, 0.0, 0.0, False)
    optimizer = ttml.optimizers.SGD(params, config)

    assert optimizer is not None


@pytest.mark.smoke
def test_adamw_optimizer_creation():
    """Verify AdamW optimizer can be created with module parameters."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

        def __call__(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    model = SimpleModule()
    params = model.parameters()

    config = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.01)
    optimizer = ttml.optimizers.AdamW(params, config)

    assert optimizer is not None


@pytest.mark.smoke
def test_optimizer_lr_get_set():
    """Verify optimizer learning rate can be get/set."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

        def __call__(self, x):
            return x

    model = SimpleModule()
    params = model.parameters()

    config = ttml.optimizers.AdamWConfig.make(0.001, 0.9, 0.999, 1e-8, 0.0)
    optimizer = ttml.optimizers.AdamW(params, config)

    assert abs(optimizer.get_lr() - 0.001) < 1e-6

    optimizer.set_lr(0.01)
    assert abs(optimizer.get_lr() - 0.01) < 1e-6


@pytest.mark.smoke
@pytest.mark.requires_device
def test_optimizer_zero_grad():
    """Verify optimizer zero_grad works."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np, new_type=ttnn.DataType.BFLOAT16))

        def __call__(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    model = SimpleModule()
    params = model.parameters()

    config = ttml.optimizers.SGDConfig.make(0.01, 0.0, 0.0, 0.0, False)
    optimizer = ttml.optimizers.SGD(params, config)

    optimizer.zero_grad()

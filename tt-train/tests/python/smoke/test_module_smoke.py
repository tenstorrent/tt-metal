# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ttml modules.

These tests verify that module creation, parameter registration, and
train/eval mode switching work correctly.
"""

import numpy as np
import pytest

import ttml
import ttnn
from ttml.modules import AbstractModuleBase, Parameter, RunMode


@pytest.mark.smoke
def test_parameter_creation():
    """Verify Parameter can be created from a tensor."""
    data = np.random.randn(1, 1, 32, 32).astype(np.float32)
    tensor = ttml.autograd.Tensor.from_numpy(data)
    param = Parameter(tensor)

    assert param is not None
    assert param.tensor is not None


@pytest.mark.smoke
def test_simple_module_creation():
    """Verify a simple module can be created with parameter registration."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

        def __call__(self, x):
            return ttml.ops.binary.mul(x, self.weight.tensor)

    model = SimpleModule()

    assert model is not None
    params = model.parameters()
    assert len(params) > 0
    assert any("weight" in k for k in params.keys())


@pytest.mark.smoke
def test_module_train_eval_modes():
    """Verify train/eval mode switching works."""

    class SimpleModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

        def __call__(self, x):
            return x

    model = SimpleModule()

    model.train()
    assert model.get_run_mode() == RunMode.TRAIN

    model.eval()
    assert model.get_run_mode() == RunMode.EVAL


@pytest.mark.smoke
def test_nested_module_parameters():
    """Verify nested modules properly track all parameters."""

    class InnerModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.inner_weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

        def __call__(self, x):
            return x

    class OuterModule(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.random.randn(1, 1, 32, 32).astype(np.float32)
            self.outer_weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))
            self.inner = InnerModule()

        def __call__(self, x):
            return x

    model = OuterModule()
    params = model.parameters()
    param_names = list(params.keys())

    assert any("outer" in k.lower() for k in param_names), "Should have outer_weight"
    assert any("inner" in k.lower() for k in param_names), "Should have inner_weight"


@pytest.mark.smoke
def test_module_list_basic():
    """Verify ModuleList works for containing multiple modules."""
    from ttml.modules import ModuleList

    class SimpleLayer(AbstractModuleBase):
        def __init__(self, idx):
            super().__init__()
            w_np = np.ones((1, 1, 32, 32), dtype=np.float32) * idx
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

        def __call__(self, x):
            return x

    layers = ModuleList([SimpleLayer(i) for i in range(3)])

    assert len(layers) == 3
    assert layers[0] is not None
    assert layers[2] is not None


@pytest.mark.smoke
def test_module_dict_basic():
    """Verify ModuleDict works for named module containers."""
    from ttml.modules import ModuleDict

    class SimpleLayer(AbstractModuleBase):
        def __init__(self):
            super().__init__()
            w_np = np.ones((1, 1, 32, 32), dtype=np.float32)
            self.weight = Parameter(ttml.autograd.Tensor.from_numpy(w_np))

        def __call__(self, x):
            return x

    modules = ModuleDict({"encoder": SimpleLayer(), "decoder": SimpleLayer()})

    assert len(modules) == 2
    assert "encoder" in modules
    assert "decoder" in modules

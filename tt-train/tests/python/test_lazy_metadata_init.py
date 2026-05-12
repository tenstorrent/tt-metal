# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for lazy TensorMetadata parameter initialization."""

import pytest

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter, TensorMetadata
from ttml.modules.parameter import LAZY_PARAMETER_ACCESS_MSG


class _TieModel(AbstractModuleBase):
    """Two linear layers sharing one Parameter (weight tying)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = LinearLayer(32, 32, has_bias=False)
        self.b = LinearLayer(32, 32, has_bias=False)
        self.b.weight = self.a.weight

    def forward(self, x):
        return self.b(self.a(x))


def test_lazy_linear_stores_metadata_no_cpp_params():
    with ttml.lazy_init():
        layer = LinearLayer(64, 32, has_bias=True)

    assert isinstance(layer.weight.peek_tensor(), TensorMetadata)
    assert isinstance(layer.bias.peek_tensor(), TensorMetadata)
    assert len(dict(layer.parameters())) == 0


def test_lazy_access_tensor_raises():
    with ttml.lazy_init():
        layer = LinearLayer(64, 32, has_bias=False)

    with pytest.raises(RuntimeError, match="materialized"):
        _ = layer.weight.tensor


def test_lazy_access_tensor_message_contains_hint():
    with ttml.lazy_init():
        layer = LinearLayer(64, 32, has_bias=False)

    with pytest.raises(RuntimeError) as exc_info:
        _ = layer.weight.tensor
    assert "materialize_module" in str(exc_info.value) or LAZY_PARAMETER_ACCESS_MSG in str(exc_info.value)


def test_inplace_init_on_lazy_parameter_raises():
    with ttml.lazy_init():
        layer = LinearLayer(64, 32, has_bias=False)

    with pytest.raises(RuntimeError, match="materialize_module"):
        ttml.init.uniform_(layer.weight, -0.1, 0.1)


def test_eager_linear_unchanged():
    layer = LinearLayer(64, 32, has_bias=False)
    assert not isinstance(layer.weight.peek_tensor(), TensorMetadata)
    assert len(dict(layer.parameters())) >= 1


@pytest.mark.requires_device
def test_materialize_then_parameters_and_forward():
    ttml.autograd.AutoContext.get_instance().open_device()
    try:
        with ttml.lazy_init():
            layer = LinearLayer(64, 32, has_bias=True)

        assert len(dict(layer.parameters())) == 0
        ttml.materialize_module(layer)
        params = dict(layer.parameters())
        assert len(params) >= 2

        import numpy as np
        import ttnn

        x = ttml.autograd.Tensor.from_numpy(
            np.random.randn(1, 1, 4, 64).astype("float32"),
            layout=ttnn.Layout.TILE,
            new_type=ttnn.DataType.BFLOAT16,
        )
        y = layer(x)
        assert y is not None
    finally:
        ttml.autograd.AutoContext.get_instance().close_device()


@pytest.mark.requires_device
def test_lazy_weight_tying_same_storage():
    ttml.autograd.AutoContext.get_instance().open_device()
    try:
        with ttml.lazy_init():
            m = _TieModel()

        ttml.materialize_module(m)
        w_a = m.a.weight.tensor
        w_b = m.b.weight.tensor
        assert w_a.get_value() is w_b.get_value()
    finally:
        ttml.autograd.AutoContext.get_instance().close_device()


def test_is_lazy_init_enabled_only_inside_context():
    assert not ttml.is_lazy_init_enabled()
    with ttml.lazy_init():
        assert ttml.is_lazy_init_enabled()
    assert not ttml.is_lazy_init_enabled()

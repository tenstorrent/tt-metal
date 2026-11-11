# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import contextlib
import random

import torch
import torchvision

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
    convert_torch_model_to_ttnn_model,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import is_wormhole_b0, is_blackhole


@contextlib.contextmanager
def use_ttnn_model_cache():
    ttnn.CONFIG.enable_model_cache = True
    yield
    ttnn.CONFIG.enable_model_cache = False


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", [None, "linear"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [64])
@pytest.mark.parametrize("k_size", [128])
@pytest.mark.parametrize("n_size", [96])
def test_linear(device, model_name, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, m_size, k_size), dtype=torch.float32)
    torch_model = torch.nn.Linear(k_size, n_size)
    torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        model_name=model_name,
        initialize_model=lambda: torch_model,
        device=device,
    )
    assert len(parameters) == 2

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = input_tensor @ parameters.weight + parameters.bias
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.9997)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [64])
@pytest.mark.parametrize("k_size", [128])
@pytest.mark.parametrize("n_size", [96])
def test_module_with_childen_and_parameters(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    class ModuleWithChildrenAndParameters(torch.nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.child = torch.nn.Linear(in_features, out_features)
            self.parameter = torch.nn.Parameter(torch.rand((out_features)))

        def forward(self, x):
            x = self.child(x)
            x *= self.parameter
            return x

    torch_input_tensor = torch.rand((batch_size, m_size, k_size), dtype=torch.float32)
    torch_model = ModuleWithChildrenAndParameters(k_size, n_size)
    torch_output_tensor = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
    )
    assert "child" in parameters
    assert "weight" in parameters.child
    assert "bias" in parameters.child
    assert "parameter" in parameters

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def functional_ttnn(input_tensor, parameters):
        output = input_tensor @ parameters.child.weight + parameters.child.bias
        output = output * ttnn.to_layout(parameters.parameter, layout=ttnn.TILE_LAYOUT)
        return output

    output_tensor = functional_ttnn(input_tensor, parameters)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.99988)


@pytest.mark.parametrize("with_bias", [True, False])
def test_fold_batch_norm2d_without_affine(with_bias):
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(4, 8, 3, bias=with_bias)
    bn = torch.nn.BatchNorm2d(8, affine=False)
    bn.running_mean = torch.randn(8)
    bn.running_var = torch.rand(8) + 0.5
    conv.eval()
    bn.eval()

    weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)

    torch_input = torch.randn(1, 4, 12, 12)
    expected = bn(conv(torch_input))
    actual = torch.nn.functional.conv2d(torch_input, weight, bias)
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


def test_model_cache_without_git_hash(tmp_path, monkeypatch):
    import ttnn.model_preprocessing as model_preprocessing

    monkeypatch.setattr(model_preprocessing, "git_hash", lambda: None)
    monkeypatch.setattr(ttnn.CONFIG, "model_cache_path", tmp_path)
    created = []

    def initialize_model():
        created.append(1)
        return torch.nn.Sequential(torch.nn.Linear(32, 32))

    with use_ttnn_model_cache():
        first = preprocess_model_parameters(model_name="no_git_model", initialize_model=initialize_model)
        second = preprocess_model_parameters(model_name="no_git_model", initialize_model=initialize_model)

    assert len(first) == len(second) == 1
    assert len(created) == 1

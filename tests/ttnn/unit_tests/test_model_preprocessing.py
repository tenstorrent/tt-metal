# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull


@contextlib.contextmanager
def use_ttnn_model_cache():
    ttnn.CONFIG.enable_model_cache = True
    yield
    ttnn.CONFIG.enable_model_cache = False


@skip_for_wormhole_b0()
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


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("model_name", [None, "conv_relu_conv"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_input_channels", [128])
@pytest.mark.parametrize("input_height", [28])
@pytest.mark.parametrize("input_width", [28])
@pytest.mark.parametrize("num_output_channels", [128])
@pytest.mark.parametrize("kernel_size", [(3, 3)])
@pytest.mark.parametrize("padding", [(1, 1)])
def test_conv_relu_conv(
    device,
    model_name,
    batch_size,
    num_input_channels,
    input_height,
    input_width,
    num_output_channels,
    kernel_size,
    padding,
):
    torch.manual_seed(0)

    class ConvReluConv(torch.nn.Module):
        def __init__(self, num_input_channels, num_output_channels, kernel_size, padding):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(num_input_channels, num_output_channels, kernel_size, padding=padding)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(num_output_channels, num_output_channels, kernel_size, padding=padding)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            return x

    torch_input_tensor = torch.rand((batch_size, num_input_channels, input_height, input_width), dtype=torch.float32)
    torch_model = ConvReluConv(num_input_channels, num_output_channels, kernel_size, padding=padding)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        model_name=model_name,  # Name to use for the cache
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        device=device,
        reader_patterns_cache=reader_patterns_cache,
    )
    assert len(parameters) == 2

    conv1 = parameters.conv1
    conv2 = parameters.conv2

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = conv1.copy_input_to_device(input_tensor)
    output_tensor = conv1(output_tensor)
    output_tensor = conv1.copy_output_from_device(output_tensor)
    output_tensor = ttnn.to_device(output_tensor, device=device)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.relu(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = conv2.copy_input_to_device(output_tensor)
    output_tensor = conv2(output_tensor)
    output_tensor = conv2.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("model_name", [None, "nested_conv_relu_conv"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_input_channels", [128])
@pytest.mark.parametrize("input_height", [28])
@pytest.mark.parametrize("input_width", [28])
@pytest.mark.parametrize("num_output_channels", [128])
@pytest.mark.parametrize("kernel_size", [(3, 3)])
@pytest.mark.parametrize("padding", [(1, 1)])
def test_nested_conv_relu_conv(
    device,
    model_name,
    batch_size,
    num_input_channels,
    input_height,
    input_width,
    num_output_channels,
    kernel_size,
    padding,
):
    torch.manual_seed(0)

    class ConvReluConv(torch.nn.Module):
        def __init__(self, num_input_channels, num_output_channels, kernel_size, padding):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(num_input_channels, num_output_channels, kernel_size, padding=padding)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(num_output_channels, num_output_channels, kernel_size, padding=padding)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            return x

    class Model(torch.nn.Module):
        def __init__(self, num_input_channels, num_output_channels, kernel_size, padding):
            super().__init__()
            self.conv_relu_conv = ConvReluConv(num_input_channels, num_output_channels, kernel_size, padding)

        def forward(self, x):
            x = self.conv_relu_conv(x)
            return x

    torch_input_tensor = torch.rand((batch_size, num_input_channels, input_height, input_width), dtype=torch.float32)
    torch_model = Model(num_input_channels, num_output_channels, kernel_size, padding=padding)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        model_name=model_name,  # Name to use for the cache
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        device=device,
        reader_patterns_cache=reader_patterns_cache,
    )
    assert len(parameters) == 1

    conv1 = parameters.conv_relu_conv.conv1
    conv2 = parameters.conv_relu_conv.conv2

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = conv1.copy_input_to_device(input_tensor)
    output_tensor = conv1(output_tensor)
    output_tensor = conv1.copy_output_from_device(output_tensor)
    output_tensor = ttnn.to_device(output_tensor, device=device)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.relu(output_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = conv2.copy_input_to_device(output_tensor)
    output_tensor = conv2(output_tensor)
    output_tensor = conv2.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("model_name", [None, "conv_relu_linear"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_input_channels", [128])
@pytest.mark.parametrize("input_height", [28])
@pytest.mark.parametrize("input_width", [28])
@pytest.mark.parametrize("num_output_channels", [128])
@pytest.mark.parametrize("kernel_size", [(3, 3)])
@pytest.mark.parametrize("padding", [(1, 1)])
def test_conv_relu_linear(
    device,
    model_name,
    batch_size,
    num_input_channels,
    input_height,
    input_width,
    num_output_channels,
    kernel_size,
    padding,
):
    torch.manual_seed(0)

    class ConvReluLinear(torch.nn.Module):
        def __init__(self, num_input_channels, num_output_channels, kernel_size, padding):
            super().__init__()
            self.conv = torch.nn.Conv2d(num_input_channels, num_output_channels, kernel_size, padding=padding)
            self.relu = torch.nn.ReLU()
            self.linear = torch.nn.Linear(num_output_channels, num_output_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = torch.reshape(x, (-1, num_output_channels))
            x = self.linear(x)
            return x

    torch_input_tensor = torch.rand((batch_size, num_input_channels, input_height, input_width), dtype=torch.float32)
    torch_model = ConvReluLinear(num_input_channels, num_output_channels, kernel_size, padding=padding)
    torch_output_tensor = torch_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        model_name=model_name,  # Name to use for the cache
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        device=device,
        reader_patterns_cache=reader_patterns_cache,
    )
    assert len(parameters) == 2

    conv = parameters.conv
    linear = parameters.linear

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    output_tensor = conv.copy_input_to_device(input_tensor)
    output_tensor = conv(output_tensor)
    output_tensor = conv.copy_output_from_device(output_tensor)
    output_tensor = ttnn.to_device(output_tensor, device=device)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.relu(output_tensor)
    output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.get_fallback_function(ttnn.reshape)(output_tensor, (-1, num_output_channels))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = output_tensor @ linear.weight + linear.bias
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


@skip_for_wormhole_b0()
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

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from ttnn.dot_access import make_dot_access_dict
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [64])
@pytest.mark.parametrize("k_size", [128])
@pytest.mark.parametrize("n_size", [96])
def test_linear(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, m_size, k_size), dtype=torch.float32)
    torch_linear = torch.nn.Linear(k_size, n_size)
    torch_output_tensor = torch_linear(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_linear,
        device=device,
    )
    assert len(parameters) == 2

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = input_tensor @ parameters.weight + parameters.bias
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_input_channels", [128])
@pytest.mark.parametrize("input_height", [28])
@pytest.mark.parametrize("input_width", [28])
@pytest.mark.parametrize("num_output_channels", [128])
@pytest.mark.parametrize("kernel_size", [(3, 3)])
@pytest.mark.parametrize("padding", [(1, 1)])
def test_conv(
    device, batch_size, num_input_channels, input_height, input_width, num_output_channels, kernel_size, padding
):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((batch_size, num_input_channels, input_height, input_width), dtype=torch.float32)
    torch_conv = torch.nn.Conv2d(num_input_channels, num_output_channels, kernel_size, padding=padding)
    torch_output_tensor = torch_conv(torch_input_tensor)

    operation_configs = make_dot_access_dict(
        dict(
            conv=dict(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                kernel_size=kernel_size,
                padding=padding,
                dtype=ttnn.bfloat16,
                device=device,
                use_1d_systolic_array=True,
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                math_fidelity=ttnn.MathFidelity.HiFi4,
                weights_dtype=ttnn.bfloat16,
            )
        )
    )

    def custom_preprocessor(model, name):
        if name == "conv":
            weight = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
            bias = ttnn.from_torch(torch.reshape(model.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)
            conv = ttnn.Conv2D(
                **operation_configs.conv,
                weight=weight,
                bias=bias,
                reader_patterns_cache=None,
                conv_blocking_and_parallelization_config_override=None,
                move_weights_to_device=False,
            )
            return {
                "weight": ttnn.Tensor(conv.conv.weight),
                "bias": ttnn.Tensor(conv.conv.bias),
            }

    parameters = preprocess_model_parameters(
        model_name="conv",  # Name to use for the cache
        initialize_model=lambda: torch_conv,
        custom_preprocessor=custom_preprocessor,
        prefix="conv",  # prefix is not needed if you have an actual CNN model. Just use the names of the torch.nn.Conv2d variables
    )
    assert len(parameters) == 2

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    reader_patterns_cache = {}
    conv = ttnn.Conv2D(
        **operation_configs.conv,
        weight=parameters.weight,
        bias=parameters.bias,
        reader_patterns_cache=reader_patterns_cache,
        using_parameters_cache=True,
    )

    output_tensor = conv.copy_input_to_device(input_tensor)
    output_tensor = conv(output_tensor)
    output_tensor = conv.copy_output_from_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.999)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_input_channels", [128])
@pytest.mark.parametrize("input_height", [28])
@pytest.mark.parametrize("input_width", [28])
@pytest.mark.parametrize("num_output_channels", [128])
@pytest.mark.parametrize("kernel_size", [(3, 3)])
@pytest.mark.parametrize("padding", [(1, 1)])
def test_conv_relu_conv(
    device, batch_size, num_input_channels, input_height, input_width, num_output_channels, kernel_size, padding
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
    torch_conv_relu_conv = ConvReluConv(num_input_channels, num_output_channels, kernel_size, padding=padding)
    torch_output_tensor = torch_conv_relu_conv(torch_input_tensor)

    operation_configs = make_dot_access_dict(
        dict(
            conv1=dict(
                in_channels=num_input_channels,
                out_channels=num_output_channels,
                kernel_size=kernel_size,
                padding=padding,
                dtype=ttnn.bfloat16,
                device=device,
                use_1d_systolic_array=True,
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                math_fidelity=ttnn.MathFidelity.HiFi4,
                weights_dtype=ttnn.bfloat16,
            ),
            conv2=dict(
                in_channels=num_output_channels,
                out_channels=num_output_channels,
                kernel_size=kernel_size,
                padding=padding,
                dtype=ttnn.bfloat16,
                device=device,
                use_1d_systolic_array=True,
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                math_fidelity=ttnn.MathFidelity.HiFi4,
                weights_dtype=ttnn.bfloat16,
            ),
        )
    )

    def custom_preprocessor(model, name):
        if name == "conv1":
            weight = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
            bias = ttnn.from_torch(torch.reshape(model.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)
            conv = ttnn.Conv2D(
                **operation_configs.conv1,
                weight=weight,
                bias=bias,
                reader_patterns_cache=None,
                conv_blocking_and_parallelization_config_override=None,
                move_weights_to_device=False,
            )
            return {
                "weight": ttnn.Tensor(conv.conv.weight),
                "bias": ttnn.Tensor(conv.conv.bias),
            }
        elif name == "conv2":
            weight = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
            bias = ttnn.from_torch(torch.reshape(model.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)
            conv = ttnn.Conv2D(
                **operation_configs.conv2,
                weight=weight,
                bias=bias,
                reader_patterns_cache=None,
                conv_blocking_and_parallelization_config_override=None,
                move_weights_to_device=False,
            )
            return {
                "weight": ttnn.Tensor(conv.conv.weight),
                "bias": ttnn.Tensor(conv.conv.bias),
            }

    parameters = preprocess_model_parameters(
        model_name="conv_relu_conv",  # Name to use for the cache
        initialize_model=lambda: torch_conv_relu_conv,
        custom_preprocessor=custom_preprocessor,
    )
    assert len(parameters) == 3

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

    reader_patterns_cache = {}
    conv1 = ttnn.Conv2D(
        **operation_configs.conv1,
        weight=parameters.conv1.weight,
        bias=parameters.conv1.bias,
        reader_patterns_cache=reader_patterns_cache,
        using_parameters_cache=True,
    )
    conv2 = ttnn.Conv2D(
        **operation_configs.conv2,
        weight=parameters.conv2.weight,
        bias=parameters.conv2.bias,
        reader_patterns_cache=reader_patterns_cache,
        using_parameters_cache=True,
    )

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

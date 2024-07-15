# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "approximate",
    (
        "none",
        "tanh",
    ),
)
def test_bw_binary_bias_gelu(input_shapes, approximate, device):
    in_data_a, input_tensor_a = data_gen_with_range(input_shapes, -100, 100, device, True)
    in_data_b, input_tensor_b = data_gen_with_range(input_shapes, -10, 10, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    in_data = in_data_a + in_data_b

    pyt_y = torch.nn.functional.gelu(in_data, approximate=approximate)

    tt_output_tensor_on_device = ttnn.bias_gelu_bw(grad_tensor, input_tensor_a, input_tensor_b, approximate=approximate)
    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad, in_data.grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "approximate",
    (
        "none",
        "tanh",
    ),
)
@pytest.mark.parametrize(
    "bias",
    (
        4.5,
        16.8,
    ),
)
def test_bw_bias_gelu_unary(input_shapes, approximate, bias, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device, True)
    in_data = in_data + bias

    pyt_y = torch.nn.functional.gelu(in_data, approximate=approximate)

    tt_output_tensor_on_device = ttnn.bias_gelu_bw(grad_tensor, input_tensor, bias, approximate=approximate)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "bias",
    (
        4.5,
        16.8,
    ),
)
def test_bw_bias_gelu_unary_default(input_shapes, bias, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device, True)
    in_data = in_data + bias

    pyt_y = torch.nn.functional.gelu(in_data)

    tt_output_tensor_on_device = ttnn.bias_gelu_bw(grad_tensor, input_tensor, bias)

    in_data.retain_grad()

    pyt_y.backward(gradient=grad_data)

    golden_tensor = [in_data.grad]
    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass

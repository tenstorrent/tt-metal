# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.remainder,
    ],
)
def test_remainder_fp32(device, ttnn_function):
    torch.manual_seed(213919)
    x_torch = torch.rand([2, 3, 64, 64], dtype=torch.float32)
    y_torch = torch.rand([2, 3, 64, 64], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch, device=device)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn.remainder(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_div)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.div_no_nan,
    ],
)
def test_div_no_nan_fp32(device, ttnn_function):
    x_torch = torch.tensor(
        [
            [
                15,
                0,
                2,
                0,
            ]
        ],
        dtype=torch.float32,
    )
    y_torch = torch.tensor(
        [
            [
                10,
                0,
                0,
                2,
            ]
        ],
        dtype=torch.float32,
    )
    # direct div sfpu result before where: tt out in torch TorchTensor([[1.5000,    nan,    inf, 0.0000]])
    # predicate b==0 tt out in torch TorchTensor([[0., 1., 1., 0.]])
    # t1 false tt out in torch TorchTensor([[1.5000,    nan,    nan, 0.0000]]) 0 * nan = nan, 0 * inf = nan ?
    # t2 true  tt out in torch TorchTensor([[0., 0., 0., 0.]])
    # t1 + t2 tt out in torch TorchTensor([[1.5000,    nan,    nan, 0.0000]]) 0 + nan = nan
    # final div_result = tt out in torch TorchTensor([[1.5000,    nan,    nan, 0.0000]])

    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_div = ttnn.div_no_nan(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_div)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.999
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.remainder,
    ],
)
def test_remainder_forge(device, ttnn_function):
    torch.manual_seed(213919)
    input1 = torch.randn(2, 32, 32)
    input2 = torch.randn(2, 32, 32)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output = golden_fn(input1, input2, device=device)

    input1 = ttnn.from_torch(input1, dtype=ttnn.float32)
    input2 = ttnn.from_torch(input2, dtype=ttnn.float32)

    input1 = ttnn.to_device(input1, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input2 = ttnn.to_device(input2, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    input1 = ttnn.to_layout(input1, ttnn.TILE_LAYOUT)
    input2 = ttnn.to_layout(input2, ttnn.TILE_LAYOUT)

    output = ttnn.remainder(input1, input2)

    output = ttnn.to_torch(output)

    status = ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.999
    assert status


def generate_torch_tensor(shape, low, high, step=0.0025, dtype=torch.float32):
    num_elements = torch.prod(torch.tensor(shape))
    values = torch.arange(low, high + step, step, dtype=dtype)

    if values.numel() < num_elements:
        values = values.repeat((num_elements // values.numel()) + 1)
    values = values[:num_elements]
    return values.reshape(shape)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    [[64, 640], [2, 32, 320], [2, 1, 1024, 1024], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 64]],
)
def test_binary_fmod_bf16(
    device,
    input_shapes,
):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -100, 100, step=0.22, dtype=torch.bfloat16)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -80, 120, step=0.11, dtype=torch.bfloat16)
    torch_output_tensor = torch.fmod(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.fmod(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99

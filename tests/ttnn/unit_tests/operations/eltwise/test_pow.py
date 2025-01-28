# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("exponent", [0.5, 2.0, 4])
def test_unary_pow_ttnn(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.pow(input_tensor, exponent, output_tensor=output_tensor, queue_id=cq_id)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    golden_tensor = golden_fn(in_data, exponent)

    comp_pass = compare_pcc([output_tensor], [golden_tensor], pcc=0.9)
    assert comp_pass


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    ([20, 20], [2, 32, 320], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 64]),
)
@pytest.mark.parametrize("input", [10.0, 5.5, -5.0, -2.5, -10, -3, 9.5, -7.25, -6.15])
@pytest.mark.parametrize("exponent", [2.75, 2.5, 1.5, 4, 5.75, 0, -1.5, -2.25, -3, -4.25, -5.5])
# Both input and exponent are -ve and exponent is a non-integer, TT and Torch output = nan
# input = non-zero and exponent = 0, TT and Torch output = 1
# Both input and exponent are 0, TT = 1 and Torch output = 0
def test_binary_pow_scalar_input(input_shapes, input, exponent, device):
    torch_input_tensor_b = torch.full(input_shapes, exponent, dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(input, torch_input_tensor_b)

    cq_id = 0
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.pow(input, input_tensor_b, queue_id=cq_id)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.999)


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
    [[64, 640], [2, 32, 320], [2, 1, 32, 1024], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 128]],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.experimental.pow,
        ttnn.pow,
    ],
)
def test_binary_sfpu_pow(
    device,
    input_shapes,
    ttnn_fn,
):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -30, 30, step=0.0022)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -20, 20)
    golden_fn = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_fn(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    [[64, 640], [2, 32, 320], [2, 1, 1024, 1024], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 64]],
)
def test_binary_sfpu_pow_bf16(
    device,
    input_shapes,
):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -30, 30, step=0.0021, dtype=torch.bfloat16)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -20, 20, dtype=torch.bfloat16)
    torch_output_tensor = torch.pow(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    [[2, 1, 32, 1024], [1, 3, 320, 384], [1, 2, 32, 64, 128], [1, 1, 32, 64]],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.experimental.pow,
        ttnn.pow,
    ],
)
def test_binary_sfpu_pow_pos(
    device,
    input_shapes,
    ttnn_fn,
):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, 0, 30, step=0.0111)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -20, 20)
    golden_fn = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_fn(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    [[2, 1, 32, 1024], [1, 3, 320, 384], [1, 2, 32, 64, 128]],
)
@pytest.mark.parametrize(
    "ttnn_fn",
    [
        ttnn.experimental.pow,
        ttnn.pow,
    ],
)
def test_binary_sfpu_pow_neg(
    device,
    input_shapes,
    ttnn_fn,
):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -30, 0, step=0.0111)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, 0, 10)
    golden_fn = ttnn.get_golden_function(ttnn_fn)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_fn(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.pow,
    ],
)
def test_pow_fp32(device, ttnn_function):
    x_torch = torch.tensor([[0.98828125, 0.47851562, 1.1875]], dtype=torch.float32)
    y_torch = torch.tensor([[0.0751953125, 0.53125, -0.6640625]], dtype=torch.float32)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_pow = ttnn.pow(x_tt, y_tt)
    print("z_tt_pow", z_tt_pow)
    tt_out = ttnn.to_torch(z_tt_pow)
    print("z_torch", z_torch)
    # z_tt_pow ttnn.Tensor([[-0.00057, -0.00164, -0.00074]], shape=Shape([1, 3]), dtype=DataType::FLOAT32, layout=Layout::TILE)
    # z_torch tensor([[0.9991, 0.6760, 0.8922]])
    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.pow,
    ],
)
def test_pow_bf16(device, ttnn_function):
    x_torch = torch.tensor([[0.98828125, 0.478515625, 1.1875]], dtype=torch.bfloat16)
    y_torch = torch.tensor([[0.0751953125, 0.53125, -0.6640625]], dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_pow = ttnn.pow(x_tt, y_tt)
    print("z_tt_pow", z_tt_pow)
    tt_out = ttnn.to_torch(z_tt_pow)
    print("z_torch", z_torch)
    # z_tt_pow ttnn.Tensor([[-0.00056, -0.00163, -0.00074]], shape=Shape([1, 3]), dtype=DataType::BFLOAT16, layout=Layout::TILE)
    # z_torch tensor([[1.0000, 0.6758, 0.8906]], dtype=torch.bfloat16)

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status

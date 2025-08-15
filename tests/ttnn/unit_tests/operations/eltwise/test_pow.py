# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, assert_allclose


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


@pytest.mark.parametrize(
    "input_shapes",
    [[64, 640], [2, 32, 320], [2, 1, 32, 1024], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 128]],
)
def test_binary_sfpu_pow(device, input_shapes):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -30, 30, step=0.0022)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -20, 20)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    [[64, 640], [2, 32, 320], [2, 1, 1024, 1024], [1, 1, 32, 32], [1, 3, 320, 384], [1, 2, 32, 64, 64]],
)
def test_binary_sfpu_pow_bf16(device, input_shapes):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -30, 30, step=0.0021, dtype=torch.bfloat16)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -20, 20, dtype=torch.bfloat16)
    torch_output_tensor = torch.pow(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    [[2, 1, 32, 1024], [1, 3, 320, 384], [1, 2, 32, 64, 128], [1, 1, 32, 64]],
)
def test_binary_sfpu_pow_pos(device, input_shapes):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, 0, 30, step=0.0111)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, -20, 20)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "input_shapes",
    [[2, 1, 32, 1024], [1, 3, 320, 384], [1, 2, 32, 64, 128]],
)
def test_binary_sfpu_pow_neg(device, input_shapes):
    torch_input_tensor_a = generate_torch_tensor(input_shapes, -30, 0, step=0.0111)
    torch_input_tensor_b = generate_torch_tensor(input_shapes, 0, 10)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.99


@pytest.mark.parametrize(
    "dtype_a",
    [
        "float32",
        "bfloat16",
    ],
)
@pytest.mark.parametrize(
    "dtype_b",
    [
        "float32",
        "bfloat16",
    ],
)
def test_binary_pow(device, dtype_a, dtype_b):
    torch_dtype_a = getattr(torch, dtype_a)
    ttnn_dtype_a = getattr(ttnn, dtype_a)
    torch_dtype_b = getattr(torch, dtype_b)
    ttnn_dtype_b = getattr(ttnn, dtype_b)
    x_torch = torch.tensor([[0.98828125, 0.47851562, 1.1875, -1.59375]], dtype=torch_dtype_a)
    y_torch = torch.tensor([[0.0751953125, 0.53125, -0.6640625, 0.1533203125]], dtype=torch_dtype_b)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype_a, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn_dtype_b, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_pow = ttnn.pow(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_pow)
    # output - bfloat16
    # Due to HW limitations for bfloat16 dtype, NaN value gets packed as inf.
    # z_tt_pow ttnn.Tensor([[ 0.99609,  0.67969,  ...,  0.89844,      inf]])
    # z_torch tensor([[1.0000, 0.6758, 0.8906,    nan]], dtype=torch.bfloat16)
    # output - float32
    # z_tt_pow ttnn.Tensor([[ 0.99930,  0.68274,  ...,  0.90147,      nan]])
    # z_torch tensor([[0.9991, 0.6760, 0.8922,    nan]])

    status = ttnn.pearson_correlation_coefficient(z_torch, tt_out) >= 0.99
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        [32, 64],
        [1, 128, 96],
        [5, 3, 64, 128],
    ),
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_sfpu_pow_bug(device, input_shapes, dtype):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    torch_input_tensor_a = torch.randn(input_shapes, dtype=torch_dtype)
    torch_input_tensor_b = torch.randn(input_shapes, dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.999


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_sfpu_accuracy(device, dtype):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    torch_input_tensor_a = torch.tensor([[10.0, 10.0, 9.0, 9.0, 5.0, 100000, 10.0, 10.0, 2.0, 2.0]], dtype=torch_dtype)
    torch_input_tensor_b = torch.tensor([[2.0, 3.0, 2.0, 3.0, 3.0, 1.7984, -1.0, -2.0, 1.0, 10.0]], dtype=torch_dtype)

    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    if dtype == "bfloat16":
        assert_with_ulp(torch_output_tensor, output, 1)
    else:
        assert_allclose(torch_output_tensor, output, rtol=0.005, atol=1e-3)  # Ensures > 99.5% accuracy


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pow_determinism(device, dtype):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    shape = [512, 512]

    torch_a = torch.randn(shape, dtype=torch_dtype)
    torch_b = torch.randn(shape, dtype=torch_dtype)

    # Run the operations twice, and check that results are the same
    # This ensures that, by default, ttnn.pow is deterministic (expected behavior from MLIR)

    ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # First round
    ttnn_result_1 = ttnn.pow(ttnn_a, ttnn_b)
    ttnn_result_1_torch = ttnn.to_torch(ttnn_result_1)

    # Second round
    ttnn_result_2 = ttnn.pow(ttnn_a, ttnn_b)
    ttnn_result_2_torch = ttnn.to_torch(ttnn_result_2)

    mask = torch.isnan(ttnn_result_1_torch) | torch.isnan(ttnn_result_2_torch)
    assert torch.equal(ttnn_result_1_torch[~mask], ttnn_result_2_torch[~mask])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_sfpu_accuracy_pos(device, dtype):
    torch.manual_seed(0)

    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)

    def gen_uniform(shape, dtype, a, b):
        tensor = torch.rand(shape, dtype=dtype)
        tensor = tensor * (b - a) + a
        return tensor

    torch_input_tensor_a = gen_uniform([256, 256], torch_dtype, 1e-9, 1e3)
    torch_input_tensor_b = gen_uniform([256, 256], torch_dtype, 0.125, 4)

    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    if dtype == "bfloat16":
        assert_with_ulp(torch_output_tensor, output, 5)
    else:
        assert_allclose(torch_output_tensor, output, rtol=0.02, atol=1e-5)  # Ensure > 98% accuracy


@pytest.mark.parametrize(
    "input_a, input_b",
    [
        ([32, 64], [32, 64]),
        ([1, 128, 96], [1, 128, 1]),
        ([5, 3, 1, 128], [5, 1, 64, 128]),
        ([2, 1, 1, 1, 1], [2, 1, 2, 64, 128]),
        ([], [128]),
    ],
)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_binary_ng_pow(device, input_a, input_b, dtype):
    torch.manual_seed(0)
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    torch_input_tensor_a = torch.randn(input_a, dtype=torch_dtype)
    torch_input_tensor_b = torch.randn(input_b, dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.pow)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.pow(input_tensor_a, input_tensor_b, use_legacy=None)
    output = ttnn.to_torch(output)

    pcc = ttnn.pearson_correlation_coefficient(torch_output_tensor, output)
    assert pcc >= 0.999

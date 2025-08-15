# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    data_gen_with_range_dtype,
    compare_pcc,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal, assert_with_ulp


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_square_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.square(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.square(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("exponent", [0.5, 2.0])
def test_unary_pow_ttnn(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.pow(input_tensor, exponent, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.pow(in_data, exponent)

    comp_pass = compare_pcc([output_tensor], [golden_tensor], pcc=0.9)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_abs_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.abs(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.abs(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_asin_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.asin(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.asin(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_acos_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.acos(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.acos(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_atan_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.atan(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.atan(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_cos_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.cos(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.cos(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_eqz_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.eqz(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = in_data == 0

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_eqz_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.eqz(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = in_data == 0

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_nez_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.nez(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = in_data != 0

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_gez_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.gez(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = in_data >= 0

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_lez_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.lez(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = in_data <= 0

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_ltz_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.ltz(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = in_data < 0

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_gtz_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.gtz(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = in_data > 0

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("alpha", [1.0, 5.0, 10.0])
def test_unary_elu_ttnn(input_shapes, alpha, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.elu(input_tensor, alpha=alpha, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.nn.functional.elu(in_data, alpha)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("fast_and_approx", [False, True])
def test_unary_erf_ttnn(input_shapes, fast_and_approx, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.erf(input_tensor, fast_and_approximate_mode=fast_and_approx, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.erf(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("fast_and_approx", [False, True])
def test_unary_erfc_ttnn(input_shapes, fast_and_approx, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.erfc(input_tensor, fast_and_approximate_mode=fast_and_approx, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.erfc(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_erfinv_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.erfinv(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.erfinv(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_expm1_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.expm1(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.expm1(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("fast_and_approx", [False, True])
def test_unary_gelu_ttnn(input_shapes, fast_and_approx, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.gelu(input_tensor, fast_and_approximate_mode=fast_and_approx, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.nn.functional.gelu(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("negative_slope", [1.0, 5.0, 10.0, 0.1])
def test_unary_leaky_relu_ttnn(input_shapes, negative_slope, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.leaky_relu(input_tensor, negative_slope=negative_slope, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.nn.functional.leaky_relu(in_data, negative_slope)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
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
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
def test_unary_logical_not_ttnn(input_shapes, torch_dtype, ttnn_dtype, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    in_data = torch.linspace(-100, 100, num_elements, dtype=torch_dtype)
    in_data[::5] = 0  # every 5th element is zero
    in_data = in_data[:num_elements].reshape(input_shapes).nan_to_num(0.0)

    input_tensor = ttnn.from_torch(
        in_data,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.logical_not(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.logical_not)
    golden_tensor = golden_function(in_data, device=device)

    assert torch.equal(output_tensor, golden_tensor.to(torch_dtype))


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_i0_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.i0(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.i0(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_isfinite_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.isfinite(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.isfinite(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_isinf_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.isinf(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.isinf(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_isposinf_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.isposinf(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.isposinf(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_isneginf_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.isneginf(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.isneginf(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_isnan_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.isnan(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.isnan(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_unary_neg_ttnn(input_shapes, device, ttnn_dtype):
    in_data1, input_tensor1 = data_gen_with_range_dtype(input_shapes, -100, 100, device, ttnn_dtype=ttnn_dtype)

    output_tensor = ttnn.neg(input_tensor1)
    golden_function = ttnn.get_golden_function(ttnn.neg)
    golden_tensor = golden_function(in_data1)
    output = ttnn.to_torch(output_tensor)

    if ttnn_dtype == ttnn.int32:
        golden_tensor = golden_tensor.to(torch.int32)
    assert torch.equal(output, golden_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_relu_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.relu(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.relu(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_relu6_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.relu6(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.nn.functional.relu6(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_tan_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.tan(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.tan(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_tanh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.tanh(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.tanh(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_rsqrt_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.rsqrt(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.rsqrt(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_sign_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.sign(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.sign(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_signbit_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.signbit(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.signbit(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_silu_ttnn(input_shapes, device):
    torch.manual_seed(0)
    max_atol = 0.03125
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.silu(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.nn.functional.silu(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass
    atol_delta = torch.max(torch.abs(ttnn.to_torch(output_tensor) - golden_tensor)).item()
    torch.allclose(golden_tensor, ttnn.to_torch(output_tensor), atol=max_atol)
    assert atol_delta <= max_atol, f"Max Atol exceeded: {atol_delta} (allowed: {max_atol})"


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_silu_ttnn_pos_ulp_check(input_shapes, device):
    torch.manual_seed(0)
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 10, device)

    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.silu(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.nn.functional.silu(in_data)

    assert_with_ulp(output_tensor, golden_tensor, ulp_threshold=2)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=0.999)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_log_sigmoid_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.log_sigmoid(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.nn.functional.logsigmoid(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_rsqrt_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.rsqrt(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.rsqrt(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
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
    "approx_mode",
    (False, True),
)
def test_unary_sigmoid_ttnn(input_shapes, device, approx_mode):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.sigmoid(
        input_tensor, vector_mode=4, fast_and_approximate_mode=approx_mode, output_tensor=output_tensor, queue_id=cq_id
    )
    golden_tensor = torch.sigmoid(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_recip_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.reciprocal(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.reciprocal(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_unary_heaviside_ttnn(input_shapes, value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.heaviside(input_tensor, value=value, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.heaviside(in_data, torch.tensor(value, dtype=in_data.dtype))

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_log_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1e-6, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.log(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.log(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_log2_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1e-6, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.log2(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.log2(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_log10_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1e-6, 1, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.log10(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.log10(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_sqrt_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 0, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.sqrt(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.sqrt(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([64, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_bitwise_not(input_shapes, device):
    torch.manual_seed(213919)

    # Generate a uniform range of values across the valid int32 range
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    uniform_values = torch.linspace(-2147483648, 2147483647, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor([0, 1, -1, 2147483647, -2147483648], dtype=torch.int32)
    in_data = torch.cat([uniform_values, corner_cases])

    in_data = in_data[-num_elements:].reshape(input_shapes)

    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.bitwise_not(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.bitwise_not)
    golden_tensor = golden_function(in_data)

    output_tensor = ttnn.to_torch(output_tensor)

    pcc = ttnn.pearson_correlation_coefficient(golden_tensor, output_tensor)
    assert pcc == 1


# Supported range: [-1, 1e7]. log1p(-1) approaches negative infinity. For input beyond 1e7, pcc drops below 0.999.
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([3, 123, 115])),
        (torch.Size([69, 178])),
        (torch.Size([1024])),
        (torch.Size([])),
    ),
)
def test_unary_log1p_ttnn(input_shapes, device):
    if len(input_shapes) == 0:
        torch_input_tensor = torch.rand((), dtype=torch.bfloat16)
    else:
        num_elements = torch.prod(torch.tensor(input_shapes)).item()
        uniform_input_values = torch.linspace(-1, 1e6, num_elements - 1, dtype=torch.bfloat16)
        corner_cases = torch.tensor([0.0], dtype=torch.bfloat16)  # Verifies log(0+1) = 0
        torch_input_tensor = torch.cat([uniform_input_values, corner_cases])
        torch_input_tensor = torch_input_tensor[:num_elements].reshape(input_shapes)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cq_id = 0
    output_tensor = ttnn.log1p(input_tensor, queue_id=cq_id)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_function = ttnn.get_golden_function(ttnn.log1p)
    torch_output_tensor = golden_function(torch_input_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999


@pytest.mark.parametrize("scalar", [1, 2, -10, -25, 15.5, 28.5, -13.5, -29.5, 0, -0, -5, 8, 100, -100])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
        (torch.int32, ttnn.int32),
    ],
)
def test_fill(device, h, w, scalar, torch_dtype, ttnn_dtype):
    torch.manual_seed(0)

    if torch_dtype.is_floating_point:
        torch_input_tensor_a = torch.empty((h, w), dtype=torch_dtype).uniform_(-100, 100)
    else:
        torch_input_tensor_a = torch.randint(low=-100, high=100, size=(h, w), dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.fill)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar, device=device)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.fill(input_tensor_a, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "param",
    {-98.5, -43.7, -8.5, 0.45, 7.7, 58.4, 89.9, float("inf"), float("-inf"), float("nan")},
)
def test_unary_celu(input_shapes, param, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    output_tensor = ttnn.celu(input_tensor, alpha=param)
    golden_function = ttnn.get_golden_function(ttnn.celu)
    golden_tensor = golden_function(in_data, alpha=param)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass

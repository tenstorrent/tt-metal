# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.backward.utility_funcs import data_gen_with_range, compare_pcc
from models.utility_functions import skip_for_grayskull


def run_unary_composite_test(device, input_shapes, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn_function(input_tensor)
    golden_tensor = torch_function(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor], pcc=pcc)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_tanhshrink_ttnn(input_shapes, device):
    run_unary_composite_test(device, input_shapes, ttnn.tanhshrink, torch.nn.functional.tanhshrink)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_acosh_ttnn(input_shapes, device):
    run_unary_composite_test(device, input_shapes, ttnn.acosh, torch.acosh)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_asinh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.asinh(input_tensor)
    golden_tensor = torch.asinh(in_data)

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
def test_unary_atanh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.atanh(input_tensor)
    golden_tensor = torch.atanh(in_data)

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
def test_unary_cbrt_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.cbrt(input_tensor)
    golden_tensor = in_data.sign() * in_data.abs().pow(1.0 / 3.0)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


# range is -9 to 9
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_cosh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -9, 9, device)

    output_tensor = ttnn.cosh(input_tensor)
    golden_tensor = torch.cosh(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


# range limit 1, to 1000
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_digamma_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device)

    output_tensor = ttnn.digamma(input_tensor)
    golden_tensor = torch.digamma(in_data)

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
def test_unary_lgamma_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 0.1, 1e32, device)

    output_tensor = ttnn.lgamma(input_tensor)
    golden_tensor = torch.lgamma(in_data)

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
def test_unary_log1p_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    output_tensor = ttnn.log1p(input_tensor)
    golden_tensor = torch.log1p(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_mish_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.mish(input_tensor)
    golden_tensor = in_data * torch.tanh(torch.nn.functional.softplus(in_data, beta=1.0, threshold=20.0))

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
def test_unary_multigammaln_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.6, 1e32, device)

    output_tensor = ttnn.multigammaln(input_tensor)
    golden_tensor = torch.special.multigammaln(in_data, 4)

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
def test_unary_sinh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -9, 9, device)

    output_tensor = ttnn.sinh(input_tensor)
    golden_tensor = torch.sinh(in_data)

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
def test_unary_softsign_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device)

    output_tensor = ttnn.softsign(input_tensor)
    golden_tensor = torch.nn.functional.softsign(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


# 0.9714 pcc
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_swish_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.swish(input_tensor)
    golden_tensor = torch.nn.functional.silu(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


# trunc is supported only in whB0
@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_trunc_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.trunc(input_tensor)
    golden_tensor = torch.trunc(in_data)

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
def test_unary_hardswish_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.hardswish(input_tensor)
    golden_tensor = torch.nn.functional.hardswish(in_data)

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
def test_unary_hardsigmoid_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.hardsigmoid(input_tensor)
    golden_tensor = torch.nn.functional.hardsigmoid(in_data)

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
def test_unary_hardtanh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    low = -1
    high = 1
    output_tensor = ttnn.hardtanh(input_tensor, low, high)
    golden_tensor = torch.nn.functional.hardtanh(in_data, min_val=low, max_val=high)

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
def test_unary_clip_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    low = -10
    high = 10
    output_tensor = ttnn.clip(input_tensor, low, high)
    golden_tensor = torch.clip(in_data, low, high)

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
def test_unary_clamp_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    low = -10
    high = 10
    output_tensor = ttnn.clamp(input_tensor, low, high)
    golden_tensor = torch.clamp(in_data, low, high)

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
def test_unary_threshold_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    threshold = -10
    value = 10
    output_tensor = ttnn.threshold(input_tensor, threshold, value)
    golden_tensor = torch.threshold(in_data, threshold, value)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],
)
def test_unary_glu_ttnn(input_shapes, dim, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    golden_fn = ttnn.get_golden_function(ttnn.glu)

    output_tensor = ttnn.glu(input_tensor, dim)
    golden_tensor = golden_fn(in_data, dim)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],
)
def test_unary_reglu_ttnn(input_shapes, dim, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    golden_fn = ttnn.get_golden_function(ttnn.reglu)

    output_tensor = ttnn.reglu(input_tensor, dim)
    golden_tensor = golden_fn(in_data, dim)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],
)
def test_unary_geglu_ttnn(input_shapes, dim, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    golden_fn = ttnn.get_golden_function(ttnn.geglu)

    output_tensor = ttnn.geglu(input_tensor, dim)
    golden_tensor = golden_fn(in_data, dim)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 64])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "dim",
    [-1, 3],
)
def test_unary_swiglu_ttnn(input_shapes, dim, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    golden_fn = ttnn.get_golden_function(ttnn.swiglu)

    output_tensor = ttnn.swiglu(input_tensor, dim)
    golden_tensor = golden_fn(in_data, dim)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass

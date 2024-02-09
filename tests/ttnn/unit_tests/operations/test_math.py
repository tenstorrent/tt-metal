# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


def run_math_unary_test(device, h, w, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_i0(device, h, w):
    run_math_unary_test(device, h, w, ttnn.i0, torch.i0, pcc=0.998)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_isfinite(device, h, w):
    run_math_unary_test(device, h, w, ttnn.isfinite, torch.isfinite, pcc=0.993)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_isinf(device, h, w):
    run_math_unary_test(device, h, w, ttnn.isinf, torch.isinf, pcc=0.9997)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_isnan(device, h, w):
    run_math_unary_test(device, h, w, ttnn.isnan, torch.isnan)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_isneginf(device, h, w):
    run_math_unary_test(device, h, w, ttnn.isneginf, torch.isneginf)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_isposinf(device, h, w):
    run_math_unary_test(device, h, w, ttnn.isposinf, torch.isposinf)


@pytest.mark.parametrize("h", [5])
@pytest.mark.parametrize("w", [5])
def test_lgamma(device, h, w):
    run_math_unary_test(device, h, w, ttnn.lgamma, torch.lgamma, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log10(device, h, w):
    run_math_unary_test(device, h, w, ttnn.log10, torch.log10)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log1p(device, h, w):
    run_math_unary_test(device, h, w, ttnn.log1p, torch.log1p, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log2(device, h, w):
    run_math_unary_test(device, h, w, ttnn.log2, torch.log2, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_neg(device, h, w):
    run_math_unary_test(device, h, w, ttnn.neg, torch.neg)


def run_math_unary_test_range(device, h, w, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)
    low = 1.6
    high = 100

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def torch_multigammaln(x, *args, **kwargs):
    result = torch.lgamma(x)
    result += torch.lgamma(x - 0.5)
    result += torch.lgamma(x - 1.0)
    result += torch.lgamma(x - 1.5)
    result += 3.434189657547
    return result


@pytest.mark.parametrize("h", [5])
@pytest.mark.parametrize("w", [5])
def test_multigammaln(device, h, w):
    run_math_unary_test_range(device, h, w, ttnn.multigammaln, torch_multigammaln, pcc=0.999)

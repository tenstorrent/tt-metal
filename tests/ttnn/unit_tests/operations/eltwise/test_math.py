# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random

from loguru import logger

pytestmark = pytest.mark.use_module_device


def run_math_unary_test(device, h, w, ttnn_function, layout=ttnn.TILE_LAYOUT, pcc=0.9999):
    torch.manual_seed(0)

    # Generate random [0; 1] tensor
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    if "digamma" in str(ttnn_function):
        # Scale and shift range to [2; 102] for digamma
        torch_input_tensor = torch_input_tensor * 100.0 + 2.0

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    output_tensor = ttnn_function(input_tensor)
    # Verify output layout matches input layout
    assert output_tensor.layout == layout, f"Output layout {output_tensor.layout} should match input layout {layout}"
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_i0(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.i0, layout=layout, pcc=0.998)


@pytest.mark.parametrize("h", [5])
@pytest.mark.parametrize("w", [5])
def test_lgamma(device, h, w):
    run_math_unary_test(device, h, w, ttnn.lgamma, pcc=0.999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.uint16, ttnn.uint32])
def test_eq(device, h, w, output_dtype):
    torch.manual_seed(0)

    same = 50
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_a[0, 0] = same
    torch_input_tensor_a[0, 1] = same
    torch_input_tensor_a[0, 2] = same

    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b[0, 0] = same
    torch_input_tensor_b[0, 1] = same
    torch_input_tensor_b[0, 2] = same

    golden_function = ttnn.get_golden_function(ttnn.eq)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    output_tensor = ttnn.eq(input_tensor_a, input_tensor_b, dtype=output_dtype)
    assert output_tensor.get_dtype() == output_dtype
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device)) - 1
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)

    # EQ with a preallocated output tensor
    output_tensor_preallocated_bfloat16 = ttnn.ones(
        [h, w], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    output_tensor_preallocated = output_tensor_preallocated_bfloat16
    # There is no good way to create uint16 tensor in ttnn/torch, so we create bfloat16 and typecast to target
    if output_dtype != ttnn.bfloat16:
        output_tensor_preallocated = ttnn.typecast(
            output_tensor_preallocated_bfloat16, output_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.eq(input_tensor_a, input_tensor_b, dtype=output_dtype, output_tensor=output_tensor_preallocated)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    torch_output_tensor_preallocated = ttnn.to_torch(output_tensor_preallocated)
    assert_with_pcc(torch_output_tensor, torch_output_tensor_preallocated, 0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log10(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.log10, layout=layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log1p(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.log1p, layout=layout, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log2(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.log2, layout=layout, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_neg(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.neg, layout=layout)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_abs(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.abs, layout=layout)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rad2deg(device, h, w):
    run_math_unary_test(device, h, w, ttnn.rad2deg)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cbrt(device, h, w):
    run_math_unary_test(device, h, w, ttnn.cbrt, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tril(device, h, w):
    run_math_unary_test(device, h, w, ttnn.tril)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_deg2rad(device, h, w):
    run_math_unary_test(device, h, w, ttnn.deg2rad)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sqrt(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.sqrt, layout=layout)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_digamma(device, h, w):
    run_math_unary_test(device, h, w, ttnn.digamma)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erf(device, h, w):
    run_math_unary_test(device, h, w, ttnn.erf)


@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("w", [3])
def test_erfc(device, h, w):
    run_math_unary_test(device, h, w, ttnn.erfc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erfinv(device, h, w):
    run_math_unary_test(device, h, w, ttnn.erfinv, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_square(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.square, layout=layout)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_exp2(device, h, w):
    run_math_unary_test(device, h, w, ttnn.exp2, pcc=0.98)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_expm1(device, h, w):
    run_math_unary_test(device, h, w, ttnn.expm1, pcc=0.99)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_triu(device, h, w):
    run_math_unary_test(device, h, w, ttnn.triu)


def run_math_unary_test_recip(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    low = -100
    high = 100

    torch_input_tensor = torch.empty((h, w), dtype=torch.bfloat16).uniform_(low, high) + 0.0001
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_math_unary_test_fixed_val(device, h, w, fill_value, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)
    torch_input_tensor = torch.full((h, w), fill_value, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_recip(device, h, w):
    run_math_unary_test_recip(device, h, w, ttnn.reciprocal, pcc=0.999)


def run_math_unary_test_range(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)
    low = 1.6
    high = 100

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [5])
@pytest.mark.parametrize("w", [5])
def test_multigammaln(device, h, w):
    run_math_unary_test_range(device, h, w, ttnn.multigammaln, pcc=0.999)


def run_math_test_polygamma(device, h, w, scalar, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    low = 1
    high = 10

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, scalar)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("scalar", [1, 2, 5, 10])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_polygamma(device, h, w, scalar):
    run_math_test_polygamma(device, h, w, scalar, ttnn.polygamma, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_recip_fixed(device, h, w):
    run_math_unary_test_fixed_val(device, h, w, 0, ttnn.reciprocal, pcc=0.999)

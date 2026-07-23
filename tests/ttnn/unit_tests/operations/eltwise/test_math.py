# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc, assert_with_ulp, assert_allclose
from models.common.utility_functions import torch_random

from loguru import logger

pytestmark = pytest.mark.use_module_device


def run_math_unary_test(
    device,
    h,
    w,
    ttnn_function,
    layout=ttnn.TILE_LAYOUT,
    ulp=1,
    allow_nonfinite=False,
    pcc_check=False,
    pcc=0.9999,
):
    """Run a single-input math op on a random bf16 tensor in [0, 1) and assert vs the torch golden.

    Default ``ulp=1`` covers kernels that are bit-exact in bf16 or accurate to one bf16 ULP over
    [0, 1). Callers override ``ulp`` when the kernel has a larger expected error, or set
    ``pcc_check=True`` with an op-specific ``pcc`` when ULP is not the appropriate tolerance.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    if "digamma" in str(ttnn_function):
        torch_input_tensor = torch_input_tensor * 100.0 + 2.0

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    output_tensor = ttnn_function(input_tensor)
    assert output_tensor.layout == layout, f"Output layout {output_tensor.layout} should match input layout {layout}"
    output_tensor = ttnn.to_torch(output_tensor)

    if pcc_check:
        assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    else:
        assert_with_ulp(torch_output_tensor, output_tensor, ulp, allow_nonfinite=allow_nonfinite)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_i0(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.i0, layout=layout, ulp=1)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lgamma(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.lgamma, layout=layout, pcc_check=True, pcc=0.99)


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
    assert_equal(torch_output_tensor.float(), output_tensor.float())

    # EQ with a preallocated output tensor
    output_tensor_preallocated_bfloat16 = ttnn.ones(
        [h, w], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    output_tensor_preallocated = output_tensor_preallocated_bfloat16
    if output_dtype != ttnn.bfloat16:
        output_tensor_preallocated = ttnn.typecast(
            output_tensor_preallocated_bfloat16, output_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.eq(input_tensor_a, input_tensor_b, dtype=output_dtype, output_tensor=output_tensor_preallocated)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    torch_output_tensor_preallocated = ttnn.to_torch(output_tensor_preallocated)
    assert_equal(torch_output_tensor.float(), torch_output_tensor_preallocated.float())


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log10(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.log10, layout=layout, ulp=2, allow_nonfinite=True)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log1p(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.log1p, layout=layout, ulp=1)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log2(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.log2, layout=layout, ulp=1, allow_nonfinite=True)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_neg(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.neg, layout=layout, ulp=0)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_abs(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.abs, layout=layout, ulp=0)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rad2deg(device, h, w):
    run_math_unary_test(device, h, w, ttnn.rad2deg, ulp=1)


def test_cbrt(device):
    # Exhaustive over every bf16 value.
    # Evaluate the registered golden in fp64 by upcasting the input,
    # in bf16 the non-representable 1/3 was rounding the reference up to 2 ULP short of the true
    # cube root while the kernel was correct. Subnormal bf16 inputs are flushed to zero on device.
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor[torch.isfinite(input_tensor.to(torch.float32))]
    input_tensor = torch.where(
        input_tensor.to(torch.float32).abs() < 2.0**-126, torch.zeros_like(input_tensor), input_tensor
    )

    golden_function = ttnn.get_golden_function(ttnn.cbrt)
    golden = golden_function(input_tensor.to(torch.float64))

    tt_in = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.to_torch(ttnn.cbrt(tt_in))
    assert_with_ulp(golden, result, 1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tril(device, h, w):
    run_math_unary_test(device, h, w, ttnn.tril, ulp=0)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_deg2rad(device, h, w):
    run_math_unary_test(device, h, w, ttnn.deg2rad, ulp=2)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sqrt(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.sqrt, layout=layout, ulp=1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_digamma(device, h, w):
    run_math_unary_test(device, h, w, ttnn.digamma, ulp=1)


def test_digamma_large_x(device):
    """Regression guard for digamma at large x (issue #45520: "behaves bad for x>1000").

    The LUT kernel is fit on [0.01, 102]; beyond it a Bernoulli asymptotic branch
    (ln(x) - 1/2x - 1/12x^2 + ...) restores the (1, inf) support the pre-LUT composite
    op had. ``test_digamma`` only exercises [2, 102], so this covers the LUT->asymptotic
    crossover (102) and several decades past x=1000.
    """
    xs = torch.tensor(
        [[101.0, 102.0, 103.0, 150.0, 500.0, 1000.0, 5000.0, 1e4, 5e4, 1e5, 5e5, 1e6, 1e7, float("inf")]],
        dtype=torch.bfloat16,
    )
    golden = torch.digamma(xs.to(torch.float64)).to(torch.float32)
    input_tensor = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.digamma(input_tensor))
    assert_with_ulp(golden, output_tensor, 2, allow_nonfinite=True)


def test_digamma_small_x(device):
    """Guard the steep near-pole region [0.01, 2): psi has a pole at 0 (psi(x) ~ -1/x),
    the steepest part of the fitted domain. test_digamma only exercises [2, 102].
    Sample avoids the zero-crossing at x~=1.4616 where ULP is ill-defined.
    """
    xs = torch.tensor(
        [[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.75, 1.9, 1.99]],
        dtype=torch.bfloat16,
    )
    golden = torch.digamma(xs.to(torch.float64)).to(torch.float32)
    input_tensor = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.digamma(input_tensor))
    assert_with_ulp(golden, output_tensor, 2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erf(device, h, w):
    run_math_unary_test(device, h, w, ttnn.erf, ulp=1)


@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("w", [3])
def test_erfc(device, h, w):
    run_math_unary_test(device, h, w, ttnn.erfc, ulp=1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_erfinv(device, h, w):
    # ULP=227 exceeds the ULP ≤ 5 policy; fall back to PCC with the original per-test threshold.
    run_math_unary_test(device, h, w, ttnn.erfinv, pcc_check=True, pcc=0.999)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_square(device, h, w, layout):
    run_math_unary_test(device, h, w, ttnn.square, layout=layout, ulp=1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_exp2(device, h, w):
    run_math_unary_test(device, h, w, ttnn.exp2, ulp=1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_expm1(device, h, w):
    run_math_unary_test(device, h, w, ttnn.expm1, ulp=1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_triu(device, h, w):
    run_math_unary_test(device, h, w, ttnn.triu, ulp=0)


def run_math_unary_test_recip(device, h, w, ttnn_function, ulp=1):
    """Reciprocal on random bf16 inputs in ``[-100, 100] + 0.0001`` (non-zero).

    Default ``ulp=1``; ``test_recip`` overrides to 2 to cover an additional bf16 ULP of error from
    the reciprocal hardware approximation near the tails of this range. The ``1/+0 = +inf`` edge
    case is covered separately by ``test_recip_fixed[fill_value=0.0]`` so the test can assert the
    sign of the resulting infinity, which a generic ``allow_nonfinite=True`` ULP check on a random
    tensor cannot do.
    """
    torch.manual_seed(0)

    low = -100
    high = 100

    torch_input_tensor = torch.empty((h, w), dtype=torch.bfloat16).uniform_(low, high) + 0.0001
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(torch_output_tensor, output_tensor, ulp)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_recip(device, h, w):
    class reciprocal_golden_wrapper:
        def __call__(self, input_tensor):
            return ttnn.reciprocal.golden_function(input_tensor, device=device)

    class reciprocal_wrapper:
        def __init__(self):
            self.golden_function = reciprocal_golden_wrapper()

        def __call__(self, input_tensor):
            return ttnn.reciprocal(input_tensor)

    run_math_unary_test_recip(device, h, w, reciprocal_wrapper(), ulp=2)


def run_math_unary_test_range(device, h, w, ttnn_function, ulp=1):
    torch.manual_seed(0)
    low = 1.6
    high = 100

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp)


@pytest.mark.parametrize("h", [5])
@pytest.mark.parametrize("w", [5])
def test_multigammaln(device, h, w):
    run_math_unary_test_range(device, h, w, ttnn.multigammaln, ulp=2)


def run_math_test_polygamma(device, h, w, scalar, ttnn_function, ulp=1):
    torch.manual_seed(0)

    low = 1
    high = 10

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, scalar)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor, scalar)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp)


@pytest.mark.parametrize("scalar", [1, 2, 5, 10])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_polygamma(device, h, w, scalar):
    run_math_test_polygamma(device, h, w, scalar, ttnn.polygamma, ulp=1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("fill_value", [0.0, 0.001, -0.001, 1.0, -1.0])
def test_recip_fixed(device, h, w, fill_value):
    torch.manual_seed(0)
    torch_input_tensor = torch.full((h, w), fill_value, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.reciprocal(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    if fill_value == 0.0:
        # 1/+0 must be +inf on device (the bf16 golden clamps to +max-bf16, which is incorrect).
        # Compare against an exact +inf tensor so that a -inf or +max-bf16 regression is caught.
        expected = torch.full_like(output_tensor, float("inf"))
        assert torch.equal(
            output_tensor, expected
        ), f"reciprocal(+0) should produce +inf, got unique values {output_tensor.unique()}"
    else:
        golden_function = ttnn.get_golden_function(ttnn.reciprocal)
        torch_output_tensor = golden_function(torch_input_tensor, device=device)
        assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
        assert_allclose(torch_output_tensor, output_tensor, atol=1e-2, rtol=1e-2)

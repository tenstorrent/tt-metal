# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from tests.ttnn.utils_for_testing import (
    assert_with_pcc,
    assert_with_ulp,
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
)

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.neg,
    ],
)
def test_neg_fp32(device, ttnn_function):
    x_torch = torch.tensor([[0.00001]], dtype=torch.float32)
    y_torch = -x_torch

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.sin,
    ],
)
def test_sin_fp32(device, ttnn_function):
    x_torch = torch.rand((64, 128), dtype=torch.float32)
    y_torch = torch.sin(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.cos,
    ],
)
def test_cos_fp32(device, ttnn_function):
    x_torch = torch.rand((64, 128), dtype=torch.float32)
    y_torch = torch.cos(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.tan,
    ],
)
def test_tan_fp32(device, ttnn_function):
    x_torch = torch.rand((64, 128), dtype=torch.float32)
    y_torch = torch.tan(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.relu,
    ],
)
def test_relu_fp32(device, ttnn_function):
    x_torch = torch.rand((64, 128), dtype=torch.float32)
    y_torch = torch.relu(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)

    tt_out = ttnn.to_torch(y_tt)
    status = torch.allclose(y_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


def run_unary_fp32_test_with_ulp(device, ttnn_function, torch_function, max_ulp, pcc_check=False, pcc=0.9999):
    all_bf16_values = generate_all_bfloat16_bitpatterns(torch.float32)

    # Flush subnormal inputs
    # Hardware does not handle subnormal values and will flush these values to 0.0 (known behavior)
    # For testing, we set these values to 0.0 beforehand so that golden function also gets 0.0
    x_torch = flush_subnormal_values_to_zero(all_bf16_values)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    y_tt = ttnn_function(x_tt)
    y_torch = torch_function(x_torch)

    # Compare results
    tt_out = ttnn.to_torch(y_tt)

    # If function is expected to return subnormal value, then hardware is expected to flush it to 0.0
    # Thus, we flush golden output to 0.0 as well to verify this behavior
    y_torch = flush_subnormal_values_to_zero(y_torch)

    if pcc_check:
        # PCC masks non-finite entries; verify NaN positions, Inf positions, and Inf signs match
        # exactly so a regression that replaces golden NaNs with ±Inf (or flips sign) is still caught.
        g_isnan, d_isnan = torch.isnan(y_torch), torch.isnan(tt_out)
        g_isinf, d_isinf = torch.isinf(y_torch), torch.isinf(tt_out)
        torch.testing.assert_close(g_isnan, d_isnan, msg="NaN positions differ between golden and device")
        torch.testing.assert_close(g_isinf, d_isinf, msg="Inf positions differ between golden and device")
        inf_both = g_isinf & d_isinf
        if inf_both.any():
            torch.testing.assert_close(
                torch.signbit(y_torch[inf_both]),
                torch.signbit(tt_out[inf_both]),
                msg="Inf signs differ between golden and device",
            )
        finite_mask = torch.isfinite(y_torch) & torch.isfinite(tt_out)
        assert_with_pcc(y_torch[finite_mask], tt_out[finite_mask], pcc)
    else:
        assert_with_ulp(y_torch, tt_out, max_ulp, allow_nonfinite=True)


def test_atan_fp32(device):
    run_unary_fp32_test_with_ulp(device, ttnn.atan, torch.atan, max_ulp=3)


def test_asin_fp32(device):
    run_unary_fp32_test_with_ulp(device, ttnn.asin, torch.asin, max_ulp=100, pcc_check=True)


def test_acos_fp32(device):
    run_unary_fp32_test_with_ulp(device, ttnn.acos, torch.acos, max_ulp=100, pcc_check=True)


def test_sinh_fp32_all_bfloat16_bitpatterns(device):
    # Full-tensor torch.sinh overflows at +/-89.0 even though the rounded fp32 result is finite.
    # Use a float64 golden rounded back to fp32 to test the kernel against the representable result.
    run_unary_fp32_test_with_ulp(device, ttnn.sinh, lambda x: torch.sinh(x.double()).float(), max_ulp=3)


def test_cosh_fp32_all_bfloat16_bitpatterns(device):
    run_unary_fp32_test_with_ulp(
        device,
        ttnn.cosh,
        lambda x: torch.cosh(x.double()).float(),
        max_ulp=1,
    )


def run_unary_test(device, h, w, ttnn_function, ulp=1, allow_nonfinite=False, pcc_check=False, pcc=0.9999):
    """Run a single-input fp32 unary op on a random tensor in [0, 1) and assert vs the torch golden.

    Default ``ulp=1`` covers kernels accurate to one float32 ULP. Callers override ``ulp`` when the
    kernel has a larger expected error, or set ``pcc_check=True`` with an op-specific ``pcc`` when
    ULP is not the appropriate tolerance.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.float32)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    if pcc_check:
        assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    else:
        assert_with_ulp(torch_output_tensor, output_tensor, ulp, allow_nonfinite=allow_nonfinite)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_exp(device, h, w):
    run_unary_test(device, h, w, ttnn.exp, ulp=1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tanh(device, h, w):
    run_unary_test(device, h, w, ttnn.tanh, pcc_check=True, pcc=0.993)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gelu(device, h, w):
    run_unary_test(device, h, w, ttnn.gelu, pcc_check=True, pcc=0.9996)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rsqrt(device, h, w):
    run_unary_test(device, h, w, ttnn.rsqrt, ulp=2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_silu(device, h, w):
    run_unary_test(device, h, w, ttnn.silu, ulp=3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log(device, h, w):
    run_unary_test(device, h, w, ttnn.log, ulp=3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sinh(device, h, w):
    run_unary_test(device, h, w, ttnn.sinh, ulp=3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cosh(device, h, w):
    run_unary_test(device, h, w, ttnn.cosh, ulp=1)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_acosh(device, h, w):
    run_unary_test(device, h, w, ttnn.acosh, ulp=1, allow_nonfinite=True)


@pytest.mark.skip("The current version doesn’t work with float32, but this will be fixed in issue #231689.")
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atanh(device, h, w):
    run_unary_test(device, h, w, ttnn.atanh, ulp=2)

# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np

from tests.ttnn.utils_for_testing import (
    assert_with_pcc,
    assert_with_ulp,
    assert_allclose,
)

pytestmark = pytest.mark.use_module_device


def _bfloat16_neighbour(value, steps):
    """Return ``value`` moved ``steps`` bfloat16 ULP away from zero (``steps`` may be negative).

    Adding to the raw bit pattern moves magnitude the right way for either sign, but only away
    from zero: stepping down from ``+0.0`` gives ``0xFFFF`` and from ``-0.0`` wraps past the int16
    floor to ``0x7FFF``, both NaN. Reject zero rather than feed NaN into a caller's stimulus.
    """
    assert value != 0, "bfloat16 neighbour is undefined at zero; the bit step would produce NaN"
    bits = torch.tensor(value, dtype=torch.bfloat16).view(torch.int16)
    return (bits + steps).view(torch.bfloat16).item()


def run_softplus_boundary_test(device, beta, threshold, pcc=0.99):
    """``beta * x == threshold`` must evaluate softplus, not the linear fallback.

    torch reverts to the linear function only when ``beta * x > threshold`` holds strictly, so the
    boundary point itself still takes the softplus path. The reference is computed from torch
    directly rather than through ``ttnn.get_golden_function(ttnn.softplus)``, which drops ``beta``
    and ``threshold`` and would otherwise compare against the defaults.

    ``beta`` and ``threshold`` are powers of two so that ``threshold / beta`` is exact in bfloat16
    and ``beta * x`` reproduces ``threshold`` exactly, which is what puts an element on the
    boundary at all.
    """
    boundary = threshold / beta
    assert boundary == torch.tensor(boundary, dtype=torch.bfloat16).item(), "boundary must be exact in bf16"

    # One bf16 step either side pins the split: below and on the boundary are softplus, above is linear.
    values = [_bfloat16_neighbour(boundary, steps=-1), boundary, _bfloat16_neighbour(boundary, steps=1)]
    stride = len(values)
    torch_input_tensor = torch.tensor(values, dtype=torch.bfloat16).repeat(64, 32)

    torch_output_tensor = torch.nn.functional.softplus(torch_input_tensor, beta=beta, threshold=threshold)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.softplus(input_tensor, beta=beta, threshold=threshold)
    output_tensor = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)))

    # The regression assertion is tolerance-free and per-element. softplus(x) is strictly greater
    # than x, so on the boundary any lane equal to its input took the linear branch. Comparing
    # elementwise rather than with torch.equal is what makes this catch an error confined to a
    # subset of lanes, faces or unrolled iterations; a whole-tensor equality check would pass as
    # long as any single lane differed.
    on_boundary = output_tensor[:, 1::stride]
    assert torch.all(
        on_boundary > torch_input_tensor[:, 1::stride]
    ), f"softplus took the linear branch at beta*x == threshold (beta={beta}, threshold={threshold})"

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


# `threshold` must stay at roughly 4 or below. The boundary residual is log1p(exp(-threshold))/beta,
# which falls under half a bfloat16 output ULP by threshold=5, and past SOFTPLUS_POLY_BOUNDARY = 5.0f
# the kernel drops the residual entirely. A correct softplus then returns the input bit for bit and
# the strict-inequality assertion below would fail spuriously. Note test_softplus_op in
# test_unary_category6_bfloat16.py uses thresholds up to 20; neither can be reused here.
@pytest.mark.parametrize("beta, threshold", [(2, 1), (1, 2), (0.5, 1), (4, 1), (1, 0.5)])
def test_softplus_threshold_boundary(device, beta, threshold):
    run_softplus_boundary_test(device, beta, threshold)


def test_tanhshrink_ulp(device):
    """ULP regression guard for the dedicated tanhshrink SFPU op (issue #45520).

    tanhshrink(x) = x - tanh(x) ~= x^3/3 for small |x|, where the subtractive form
    cancels in bf16 (the original kernel returned 0 -> Max ULP ~254) even though the
    true value is a normal bf16 number. torch's golden cancels there too, so use an
    mpmath reference. Points span the cancellation region, the |x|~1 crossover, and
    saturation. The dedicated op measures Max ULP = 1; gate at 2. (The exhaustive
    bfloat16 coverage in test_unary_category2_bfloat16.py::test_tanhshrink stays on
    PCC because its torch golden cancels near zero the same way bf16 hardware does.)
    """
    from mpmath import mp, tanh as mp_tanh

    mp.prec = 200
    xs = torch.tensor(
        [
            [
                0.0,
                1e-4,
                1e-3,
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                0.9,
                1.0,
                1.1,
                2.0,
                5.0,
                50.0,
                100.0,
                -1e-3,
                -0.05,
                -0.25,
                -0.9,
                -1.0,
                -1.1,
                -5.0,
                -50.0,
            ]
        ],
        dtype=torch.bfloat16,
    )
    golden = torch.tensor(
        [[float(mp.mpf(v) - mp_tanh(mp.mpf(v))) for v in xs.flatten().tolist()]],
        dtype=torch.float32,
    )

    input_tensor = ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.tanhshrink(input_tensor))

    assert_with_ulp(expected_result=golden, actual_result=output_tensor, ulp_threshold=2)


def torch_prelu(x, *args, weight, **kwargs):
    result = torch.nn.functional.prelu(x, torch.tensor(weight, dtype=x.dtype))
    return result


def run_activation_test_scalarB(device, h, w, scalar, ttnn_function, ulp=2):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=ulp)


@pytest.mark.parametrize("alpha", [1, 2.5, 5.0, -1, -5, 0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat4_b)],
)
def test_scalarB_celu(device, h, w, alpha, torch_dtype, ttnn_dtype):
    if alpha == 0:
        pytest.skip("alpha=0 is not supported")

    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch_dtype)

    golden_function = ttnn.get_golden_function(ttnn.celu)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    if ttnn_dtype == ttnn.bfloat4_b:
        torch_input_tensor_a = ttnn.to_torch(input_tensor_a)

    torch_output_tensor = golden_function(torch_input_tensor_a, alpha=alpha)

    output_tensor = ttnn.celu(input_tensor_a, alpha=alpha)
    output_tensor = ttnn.to_torch(output_tensor)
    if ttnn_dtype == ttnn.bfloat4_b:
        assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
    else:
        assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=2)


@pytest.mark.parametrize("alpha", [-1.0, -4.0])
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16)],
)
def test_scalarB_celu_negative_alpha_negative_input(device, alpha, torch_dtype, ttnn_dtype):
    """celu(x, alpha) = alpha * (exp(x / alpha) - 1) for x < 0. With alpha < 0 the exp
    argument x / alpha is positive and unbounded; torch overflows to -inf once it passes
    ln(FLT_MAX) ~ 88.72, and the device must agree (it used to wrap the 2^k exponent of
    its exp reconstruction and return +|alpha| instead of -inf).

    x / alpha in [0, 80] exercises the finite path, [90, 200] the overflow path. The gap
    (80, 90) is left out on purpose: there exp(x / alpha) is within a factor of two of
    FLT_MAX and the device's 2^k reconstruction cannot represent it (a separate accuracy
    edge, not the sign wrap). Power-of-two alphas keep x / alpha and alpha * (...) exact on
    both host and device so the ULP bound measures the kernel, not the host reciprocal.
    """
    if device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar celu kernel is unchanged: no upper clamp / nonfinite pass-through yet")
    h, w = 64, 128
    ratio = torch.cat([torch.linspace(0.0, 80.0, h * w // 2), torch.linspace(90.0, 200.0, h * w // 2)])
    torch_input_tensor_a = (ratio * alpha).reshape(h, w).to(torch_dtype)  # x = alpha * ratio <= 0

    golden_function = ttnn.get_golden_function(ttnn.celu)
    torch_output_tensor = golden_function(torch_input_tensor_a, alpha=alpha)
    overflow = torch.isinf(torch_output_tensor)
    assert overflow.sum() == h * w // 2, "the x / alpha >= 90 half must overflow to -inf in the golden"

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.celu(input_tensor_a, alpha=alpha))

    assert torch.equal(
        output_tensor[overflow], torch_output_tensor[overflow]
    ), f"x / alpha past ln(FLT_MAX) must give -inf, device gave {output_tensor[overflow].unique().tolist()}"
    assert_with_ulp(
        expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=2, allow_nonfinite=True
    )


@pytest.mark.parametrize("alpha", [1.0, -1.0])
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16)],
)
def test_scalarB_celu_nonfinite_input(device, alpha, torch_dtype, ttnn_dtype):
    """+-NaN and +-inf inputs for either sign of alpha. Both NaN signs matter: the kernel's
    x >= 0 branch is a sign test, so a sign-set NaN takes the exp branch, where a swap-based
    clamp of the exp argument would turn it into a number.

    Expectations beyond torch's, all pre-existing and documented in the ISA:
    - bfloat16 output goes through SFPSTOCHRND, which turns +NaN into +inf and -NaN into
      -inf, so for bfloat16 a NaN input must give an infinity with the NaN's sign.
    - For alpha > 0 the sign-set NaN element is not checked: on that (unchanged) branch the
      NaN sign of the x / alpha product is unspecified on Wormhole, so the helper's own
      clamp may rank it below -87 and return alpha * (exp(-87) - 1) instead of NaN.
    """
    if device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar celu kernel is unchanged: no upper clamp / nonfinite pass-through yet")
    h, w = 32, 32
    if torch_dtype == torch.float32:
        nan_bits = torch.tensor([0x7FC00000, -0x00400000], dtype=torch.int32)  # +NaN, -NaN (0xFFC00000)
    else:
        nan_bits = torch.tensor([0x7FC0, -0x0040], dtype=torch.int16)  # +NaN, -NaN (0xFFC0)
    specials = [*nan_bits.view(torch_dtype), torch.tensor(float("inf")), torch.tensor(float("-inf"))]
    x = torch.linspace(-4.0, 4.0, h * w, dtype=torch.float32).to(torch_dtype)
    for i, v in enumerate(specials * 8):  # every face and every SFPU lane group sees each special
        x[i * 29] = v
    torch_input_tensor_a = x.reshape(h, w)
    int_dtype = torch.int32 if torch_dtype == torch.float32 else torch.int16
    nan_in = torch.isnan(torch_input_tensor_a)
    nan_in_signset = nan_in & (torch_input_tensor_a.view(int_dtype) < 0)
    assert nan_in_signset.any() and (nan_in & ~nan_in_signset).any(), "the input must contain NaNs of both signs"

    golden_function = ttnn.get_golden_function(ttnn.celu)
    torch_output_tensor = golden_function(torch_input_tensor_a, alpha=alpha)
    assert torch.equal(torch.isnan(torch_output_tensor), nan_in)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(ttnn.celu(input_tensor_a, alpha=alpha))

    checked = nan_in if alpha < 0 else (nan_in & ~nan_in_signset)
    if torch_dtype == torch.float32:
        assert torch.isnan(
            output_tensor[checked]
        ).all(), f"NaN in must be NaN out, device gave {output_tensor[checked].unique().tolist()}"
    else:
        assert torch.isinf(output_tensor[checked]).all() and torch.equal(
            output_tensor[checked] < 0, nan_in_signset[checked]
        ), f"bfloat16 NaN in must be inf with the NaN's sign, device gave {output_tensor[checked].unique().tolist()}"
    rest = ~nan_in
    assert not torch.isnan(output_tensor[rest]).any(), "NaN produced from a non-NaN input"
    inf_out = torch.isinf(torch_output_tensor)
    assert torch.equal(output_tensor[inf_out], torch_output_tensor[inf_out]), "infinite outputs differ from torch"
    assert_with_ulp(
        expected_result=torch_output_tensor[rest & ~inf_out],
        actual_result=output_tensor[rest & ~inf_out],
        ulp_threshold=2,
    )


@pytest.mark.parametrize("weight", [-0.5, 1.0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_prelu(device, h, w, weight):
    torch.manual_seed(0)
    ttnn_function = ttnn.prelu
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_prelu(torch_input_tensor_a, weight=weight)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, weight)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=2)


@pytest.mark.parametrize("ttnn_dtype, torch_dtype", [(ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)])
def test_mish_golden_verification(ttnn_dtype, torch_dtype, device):
    input_data = torch.tensor(
        [
            [-1.1258, -1.1524, -0.2506, 1.5863, 0.9463, -0.8437],
            [-0.6136, 0.0316, -0.4927, -1.2341, 1.8197, -0.5515],
            [-0.5692, 0.9200, 1.1108, -0.9565, 0.0335, 0.7101],
        ],
        dtype=torch_dtype,
    )
    golden_function = torch.nn.functional.mish
    golden_output = golden_function(input_data)

    input_tensor = ttnn.from_torch(
        input_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, device=device
    )
    output_tensor = ttnn.mish(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(golden_output, output_tensor, pcc=0.99)


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
    ],
)
@pytest.mark.parametrize("alpha_p, alpha_n", [(0.8, 0.8), (0.3, 0.1), (0.5, 1.0), (1.0, 0.5)])
def test_xielu(alpha_p, alpha_n, dtype, device):
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    torch.manual_seed(0)
    torch_input = torch.randn([32, 32], dtype=torch_dtype)
    golden_fn = ttnn.get_golden_function(ttnn.xielu)
    torch_output = golden_fn(torch_input, alpha_p=alpha_p, alpha_n=alpha_n)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.xielu(ttnn_input, alpha_p=alpha_p, alpha_n=alpha_n)
    ttnn_output = ttnn.to_torch(ttnn_output)

    if dtype == "float32":
        assert_allclose(torch_output, ttnn_output, rtol=6e-05, atol=1e-06)
    else:
        assert_with_ulp(expected_result=torch_output, actual_result=ttnn_output, ulp_threshold=1)


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
        "bfloat16",
    ],
)
@pytest.mark.parametrize("alpha_p, alpha_n", [(0.8, 0.8), (0.3, 0.1)])
def test_xielu_large_negative(alpha_p, alpha_n, dtype, device):
    """xielu's large-negative branch scales by 2**k via setexp.

    Without an exponent-underflow guard the scaled exponent goes non-positive and
    setexp wraps it into a large positive exponent, so the op returned values around
    1e38, Inf or NaN for inputs near -89 where the true result is about 26.  The
    default test above draws from torch.randn, so it never reaches this region.
    """
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    torch_input = torch.arange(-95.0, -80.0, 0.25, dtype=torch_dtype).reshape(1, -1)
    golden_fn = ttnn.get_golden_function(ttnn.xielu)
    torch_output = golden_fn(torch_input, alpha_p=alpha_p, alpha_n=alpha_n)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.xielu(ttnn_input, alpha_p=alpha_p, alpha_n=alpha_n)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert torch.isfinite(ttnn_output).all(), "xielu returned a non-finite value for a finite input"

    if dtype == "float32":
        assert_allclose(torch_output, ttnn_output, rtol=6e-05, atol=1e-06)
    else:
        assert_with_ulp(expected_result=torch_output, actual_result=ttnn_output, ulp_threshold=1)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_xielu_below_underflow(dtype, device):
    torch_dtype = getattr(torch, dtype)
    ttnn_dtype = getattr(ttnn, dtype)
    near = torch.arange(-200.0, -95.0, 0.5)
    far = -torch.logspace(5, 7, 64)
    torch_input = torch.cat([near, far]).to(torch_dtype).reshape(1, -1)
    golden_fn = ttnn.get_golden_function(ttnn.xielu)
    torch_output = golden_fn(torch_input, alpha_p=0.8, alpha_n=0.5)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn.xielu(ttnn_input, alpha_p=0.8, alpha_n=0.5)
    ttnn_output = ttnn.to_torch(ttnn_output)

    if dtype == "float32":
        assert_allclose(torch_output, ttnn_output, rtol=6e-05, atol=1e-06)
    else:
        assert_with_ulp(expected_result=torch_output, actual_result=ttnn_output, ulp_threshold=1)

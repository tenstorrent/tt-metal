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


def run_activation_unary_test(device, h, w, ttnn_function, ulp=2, pcc_check=False, pcc=0.99):
    """Run a single-input activation on a torch-random bf16 tensor in [-1, 1) and assert vs golden.

    Default ``ulp=2`` covers kernels with up to ~1 ULP error plus the additional ULP from bf16
    input quantization. Callers override ``ulp`` when the kernel has a different expected error,
    or set ``pcc_check=True`` with an op-specific ``pcc`` when ULP is not the appropriate tolerance.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if pcc_check:
        assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    else:
        assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=ulp)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardtanh(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardtanh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sigmoid_accurate(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sigmoid_accurate)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sign(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sign)


def run_activation_softplus_test(device, h, w, beta, threshold, ttnn_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, beta=beta, threshold=threshold)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output_tensor = ttnn_function(input_tensor_a, beta=beta, threshold=threshold, queue_id=0)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("beta", [-1, 0.5, 1, 2])
@pytest.mark.parametrize("threshold", [-20, 5, 10, 20, 40])
def test_softplus(device, h, w, beta, threshold):
    run_activation_softplus_test(device, h, w, beta, threshold, ttnn.softplus)


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
# the strict-inequality assertion below would fail spuriously. Note the neighbouring test_softplus
# uses thresholds up to 40 and the ttnn default is 20; neither can be reused here.
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


def run_activation_test_elu(device, h, w, scalar, ttnn_function, ulp=2):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, alpha=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, alpha=scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=ulp)


def run_activation_test_leaky_relu(device, h, w, scalar, ttnn_function, ulp=2):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, negative_slope=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, negative_slope=scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=ulp)


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


def run_activation_test_scalarB_key(device, h, w, value, ttnn_function, ulp=2):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, value=value)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, value)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=ulp)


@pytest.mark.parametrize("scalar", [-0.5, 0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_elu(device, h, w, scalar):
    run_activation_test_elu(device, h, w, scalar, ttnn.elu)


@pytest.mark.parametrize("alpha", [1, 2.5, 5.0, -1, -5, 0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype",
    [(torch.float32, ttnn.float32), (torch.bfloat16, ttnn.bfloat16), (torch.bfloat16, ttnn.bfloat4_b)],
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


@pytest.mark.parametrize("scalar", [0.5, 1.0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_hardshrink(device, h, w, scalar):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.hardshrink)
    torch_output_tensor = golden_function(torch_input_tensor_a, lambd=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.hardshrink(input_tensor_a, lambd=scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=2)


@pytest.mark.parametrize("value", [0.88])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_heaviside(device, h, w, value):
    run_activation_test_scalarB_key(device, h, w, value, ttnn.heaviside)


@pytest.mark.parametrize("scalar", [-0.5, 0, 0.1, 0.01, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_leaky_relu(device, h, w, scalar):
    run_activation_test_leaky_relu(device, h, w, scalar, ttnn.leaky_relu)


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


@pytest.mark.parametrize("scalar", [0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_softshrink(device, h, w, scalar):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.softshrink)
    torch_output_tensor = golden_function(torch_input_tensor_a, lambd=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.softshrink(input_tensor_a, lambd=scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=2)


def run_activation_test_scalarBC_key(device, h, w, scalar1, scalar2, ttnn_function, ulp=2):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)

    torch_output_tensor = golden_function(torch_input_tensor_a, scalar1, scalar2)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=ulp)


@pytest.mark.parametrize("min", [-0.5, -0.1, -5.5])
@pytest.mark.parametrize("max", [0.5, 1.5, 27.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarBC_clip(device, h, w, min, max):
    run_activation_test_scalarBC_key(device, h, w, min, max, ttnn.clip)


def run_activation_test_threshold(device, h, w, value, threshold, ttnn_function, ulp=1):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)

    torch_output_tensor = golden_function(torch_input_tensor_a, value=value, threshold=threshold)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, threshold, value)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    # threshold is a piecewise-exact op; use ULP=1 to absorb bf16 rounding of non-representable scalars.
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=ulp)


@pytest.mark.parametrize("value", [-0.5, -0.1, -5.5])
@pytest.mark.parametrize("threshold", [-0.5, 1.5, 27.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_threshold(device, h, w, value, threshold):
    run_activation_test_threshold(device, h, w, value, threshold, ttnn.threshold)


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
        "bfloat16",
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


@pytest.mark.parametrize(
    "shapes",
    [
        (3, 4, 64, 32),
        (128, 128),
    ],
)
def test_lgamma_fp32(device, shapes):
    torch.manual_seed(42)
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.float32

    x_torch = torch.empty(shapes, dtype=torch_dtype).uniform_(-5, 5)
    z_torch = torch.lgamma(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.lgamma(x_tt)

    tt_out = ttnn.to_torch(z_tt)

    assert_with_pcc(z_torch, tt_out, 0.999)

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
    generate_all_bfloat16_bitpatterns,
    flush_subnormal_values_to_zero,
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
        assert_with_ulp(torch_output_tensor, output_tensor, ulp)


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
def test_hardswish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardswish)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log_sigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.log_sigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_mish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.mish, ulp=3)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_relu6(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.relu6)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gelu(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.gelu)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high, atol, rtol",
    [
        (-13, 0, 1e-2, 1e-2),  # Negative saturation region
        (0, 3, 1e-2, 1e-2),  # Positive transition region
        (3, 6, 1e-3, 1e-3),  # Positive saturation region
    ],
)
def test_gelu_accurate_allclose(input_shapes, low, high, atol, rtol, device):
    """Test GELU accuracy using allclose for different input regions matching analysis ranges"""
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.gelu)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.gelu(tt_in)
    result = ttnn.to_torch(tt_result)
    # Use allclose with range-specific tolerances
    assert_allclose(result, golden, atol=atol, rtol=rtol)


def test_gelu_bfloat16_accuracy(device):
    """Exhaustive bf16 accuracy test: all positive normal bfloat16 bit-patterns (0x0100–0x7F7F).

    Every positive finite normal bf16 value is swept through ttnn.gelu and compared
    against a float32 reference (torch.nn.functional.gelu upcast).  The requirement
    is ≤ 10 ULP for every tested input.

    Excluded categories:
    - Subnormal inputs (x < 2^-126): hardware may flush subnormals to zero.
    - NaN / +inf: handled by dedicated special-value tests.
    - 128 positive normals whose exponent field = 1 (x ∈ [2^-126, 2^-125)):
        gelu(x) ≈ x/2 falls in [2^-127, 2^-126) which is subnormal in fp32.
        TT hardware DAZ/FTZ flushes this intermediate value to 0, so hardware
        returns 0 while torch (no FTZ) returns a tiny bf16 subnormal or rounds
        up to 2^-126 — up to 128 ULP error (e.g. 0x00FF → 128 ULP).
        Bit-patterns: 0x0080–0x00FF.
    """
    # generate_all_bfloat16_bitpatterns returns (256, 256) — tile-layout compatible with no padding waste.
    all_bf16_2d = generate_all_bfloat16_bitpatterns(torch.bfloat16)
    all_bf16 = all_bf16_2d.flatten()

    idx = torch.arange(0, 2**16, dtype=torch.int32)
    exp_field = (idx >> 7) & 0xFF
    is_negative = idx >= 0x8000

    tiny = torch.finfo(torch.bfloat16).tiny
    is_special = torch.isnan(all_bf16) | torch.isinf(all_bf16)
    is_subnormal = (all_bf16.abs() > 0) & (all_bf16.abs() < tiny)
    # exp=1 normals: gelu output is fp32-subnormal → hardware FTZ → 0; up to 128 ULP
    is_exp1_normal = exp_field == 1

    test_mask = ~is_negative & ~is_special & ~is_subnormal & ~is_exp1_normal

    tt_in = ttnn.from_torch(
        all_bf16_2d,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.gelu)
    golden = golden_function(all_bf16, device=device)

    result = ttnn.to_torch(ttnn.gelu(tt_in)).flatten()

    check_mask = test_mask & torch.isfinite(golden) & torch.isfinite(result)
    assert_with_ulp(golden[check_mask], result[check_mask], ulp_threshold=10)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardsigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardsigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sign(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sign)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_softsign(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.softsign)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_swish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.swish)


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


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tanhshrink(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.tanhshrink, pcc_check=True)


def test_tanhshrink_ulp(device):
    """ULP regression guard for the dedicated tanhshrink SFPU op (issue #45520).

    tanhshrink(x) = x - tanh(x) ~= x^3/3 for small |x|, where the subtractive form
    cancels in bf16 (the original kernel returned 0 -> Max ULP ~254) even though the
    true value is a normal bf16 number. torch's golden cancels there too, so use an
    mpmath reference. Points span the cancellation region, the |x|~1 crossover, and
    saturation. The dedicated op measures Max ULP = 1; gate at 2. (test_tanhshrink
    above stays on PCC because its torch golden cancels near zero.)
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

    assert_with_ulp(golden, output_tensor, ulp_threshold=2)


def run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn_function, ulp=2, pcc_check=False, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16).unsqueeze(0)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if pcc_check:
        assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    else:
        assert_with_ulp(torch_output_tensor, output_tensor, ulp)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1, 3])
def test_glu(device, batch_size, h, w, dim):
    run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn.glu)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1, 3])
def test_reglu(device, batch_size, h, w, dim):
    run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn.reglu)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1, 3])
def test_swiglu(device, batch_size, h, w, dim):
    run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn.swiglu)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("dim", [-1, 3])
def test_geglu(device, batch_size, h, w, dim):
    run_activation_unary_test_glu(device, batch_size, h, w, dim, ttnn.geglu, pcc_check=True)


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
    assert_with_ulp(torch_output_tensor, output_tensor, ulp)


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
    assert_with_ulp(torch_output_tensor, output_tensor, ulp)


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
    assert_with_ulp(torch_output_tensor, output_tensor, ulp)


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
    assert_with_ulp(torch_output_tensor, output_tensor, ulp)


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
        assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=2)


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
    assert_with_ulp(torch_output_tensor, output_tensor, 2)


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
    assert_with_ulp(torch_output_tensor, output_tensor, 2)


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
    assert_with_ulp(torch_output_tensor, output_tensor, 2)


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
    assert_with_ulp(torch_output_tensor, output_tensor, ulp)


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
    assert_with_ulp(torch_output_tensor, output_tensor, ulp)


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
        assert_with_ulp(torch_output, ttnn_output, 1)


def test_lgamma_bfloat16(device):
    high = 1000
    low = -1000

    # All 2^16 bfloat16 bit patterns (256x256), flattened for masking
    input_tensor = generate_all_bfloat16_bitpatterns(torch.bfloat16).flatten()
    input_tensor = flush_subnormal_values_to_zero(input_tensor)
    input_tensor_f32 = input_tensor.to(torch.float32)

    in_range = (input_tensor_f32 >= low) & (input_tensor_f32 <= high)
    is_non_positive_int = (input_tensor_f32 <= 0) & (input_tensor_f32 == torch.floor(input_tensor_f32))
    mask = in_range & ~is_non_positive_int  # exclude lgamma poles at 0,-1,-2,...

    input_tensor = input_tensor[mask]

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.lgamma)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.lgamma(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_pcc(golden, result, 0.999)


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

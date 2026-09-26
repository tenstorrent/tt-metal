# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_equal,
    assert_with_ulp,
    assert_div_by_zero_outputs,
    flush_subnormal_values_to_zero,
    ulp_distance,
)
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_binary_grid,
    pairwise_inputs,
    to_tt_tensor,
)

pytestmark = pytest.mark.use_module_device

"""
Category 5: division family + power

 1. ttnn.div         - Division, with rounding_mode None / "trunc" / "floor"
 2. ttnn.remainder   - Floating-point remainder (sign follows the divisor)
 3. ttnn.fmod        - Floating-point remainder (sign follows the dividend)
 4. ttnn.div_no_nan  - Division that returns 0 (instead of NaN/inf) when the divisor is 0
 5. ttnn.floor_div   - floor(a / b)
 6. ttnn.pow         - tensor ** exponent (int / float / tensor)

div/div_no_nan/floor_div are checked exact against the device's own bfloat16
quotient (see _bf16_quotient). divide's rounding_mode=None also covers
fast_and_approximate_mode=True (the FPU path), which is genuinely distinct
from the accurate/SFPU path the trunc/floor branches force. remainder/fmod
use a magnitude-matched grid since a - b * trunc(a / b) loses precision once
|a / b| is large. pow uses a few ULP of slack and a positive-only base for
non-integer exponents (bfloat16 packs NaN as +inf -- see test_binary_pow in
test_pow.py).

Each op also gets a tensor-scalar variant, checked looser (ULP or atol)
since the scalar overloads don't take the same binary_ng path.
"""


def _bf16_quotient(tt_a, tt_b, input_a, input_b):
    """The bfloat16 quotient ttnn.divide itself computes before rounding_mode
    is applied: subnormal results flush to zero, and |b| >= 2**126 underflows
    the reciprocal to zero too (detected from the device's own plain divide,
    since the cutoff isn't a clean power of two).
    """
    raw_quotient = flush_subnormal_values_to_zero((input_a.float() / input_b.float()).clone())
    bf16_quotient = raw_quotient.to(torch.bfloat16).float()

    plain = ttnn.to_torch(ttnn.divide(tt_a, tt_b, fast_and_approximate_mode=False)).float()
    recip_underflow = (
        (plain == 0) & (bf16_quotient != 0) & (input_b.abs() >= 2.0**126) & torch.isfinite(bf16_quotient)
    )
    # Sanity check on the correction itself, not just the final result: if the
    # device ever stopped underflowing here (or started underflowing
    # elsewhere), this golden would silently follow it and the test would
    # keep passing. Measured ~14,340 lanes on the full pairwise_inputs grid
    # when this was written; a bound this loose only catches the path going
    # fully cold, but that's still strictly better than no check at all.
    assert recip_underflow.sum() > 0, "expected at least one |b| >= 2**126 reciprocal-underflow lane in this grid"
    return torch.where(recip_underflow, torch.zeros_like(bf16_quotient), bf16_quotient)


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("fast_and_approximate_mode", [True, False])
def test_divide_round_modes(device, fast_and_approximate_mode, rounding_mode):
    """ttnn.divide's rounding_mode over the stratified grid.

    fast_and_approximate_mode only has an effect when rounding_mode is None:
    ExecuteDiv's trunc/floor branch suppresses it on bfloat16 (workaround for
    #43209) after computing the quotient, but forwards it unmodified when
    rounding_mode is None, so that combination genuinely takes the FPU path
    and needs its own tolerance and corrections below.
    """
    input_a, input_b = pairwise_inputs(include_zero=False)
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)
    result = ttnn.to_torch(
        ttnn.divide(tt_a, tt_b, rounding_mode=rounding_mode, fast_and_approximate_mode=fast_and_approximate_mode)
    ).float()

    if fast_and_approximate_mode and rounding_mode is None:
        # FPU-specific: reciprocal underflow at |b| >= 2**126 has to be
        # detected from this test's own (fast) result rather than reused from
        # _bf16_quotient's accurate-mode baseline -- the FPU and SFPU
        # reciprocals don't underflow at exactly the same boundary (see
        # a=127.5, b=2**126 exactly: accurate mode keeps it, fast doesn't).
        raw_quotient = flush_subnormal_values_to_zero((input_a.float() / input_b.float()).clone())
        bf16_quotient = raw_quotient.to(torch.bfloat16).float()
        recip_underflow = (result == 0) & (bf16_quotient != 0) & (input_b.abs() >= 2.0**126)
        golden = torch.where(recip_underflow, torch.zeros_like(bf16_quotient), bf16_quotient)

        golden_bf16 = golden.to(torch.bfloat16)
        result_bf16 = result.to(torch.bfloat16)
        both_finite = torch.isfinite(golden_bf16) & torch.isfinite(result_bf16)
        # FPU underflow band: LoFi dest rounding can exceed 2 ULP near the
        # smallest normals (matches category1's test_divide FPU exception).
        above_2_ulp = both_finite & (golden_bf16.abs() < (2.0**-120)) & (ulp_distance(golden_bf16, result_bf16) > 2)
        result = torch.where(above_2_ulp, golden, result)
        # FPU overflow-to-zero and saturate-instead-of-overflow, same as category1.
        overflow_to_zero = torch.isinf(golden) & (result == 0)
        result = torch.where(overflow_to_zero, golden, result)
        bf16_max = torch.finfo(torch.bfloat16).max
        allowed_sat = (
            torch.isinf(golden) & (result.abs() == bf16_max) & (torch.signbit(golden) == torch.signbit(result))
        )
        result = torch.where(allowed_sat, golden, result)

        assert_with_ulp(
            expected_result=golden.to(torch.bfloat16),
            actual_result=result.to(torch.bfloat16),
            ulp_threshold=2,
            allow_nonfinite=True,
        )
        return

    bf16_quotient = _bf16_quotient(tt_a, tt_b, input_a, input_b)
    if rounding_mode == "trunc":
        golden = torch.trunc(bf16_quotient)
    elif rounding_mode == "floor":
        golden = torch.floor(bf16_quotient)
    else:
        golden = bf16_quotient
    assert_equal(golden, result)


def test_divide_output_tensor(device):
    """ttnn.divide with a preallocated output_tensor= writes the same result
    as the plain call (moved from test_binary_composite.py's
    test_binary_div_ttnn_opt, which used PCC over random data instead).
    """
    input_a, input_b = pairwise_inputs(include_zero=False)
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)
    output_tensor = to_tt_tensor(torch.zeros_like(input_a), device)

    ttnn.divide(tt_a, tt_b, output_tensor=output_tensor)
    result = ttnn.to_torch(output_tensor).float()

    golden = _bf16_quotient(tt_a, tt_b, input_a, input_b)
    assert_equal(golden, result)


_DIV_SCALAR_VALUES = [-5.1, 0.0, 10.9]


@pytest.mark.parametrize("scalar", _DIV_SCALAR_VALUES)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
@pytest.mark.parametrize("fast_and_approximate_mode", [True, False])
def test_divide_scalar(device, fast_and_approximate_mode, rounding_mode, scalar):
    """Tensor-scalar ttnn.divide over the stratified grid. This overload
    multiplies by a host-computed reciprocal instead of dividing directly, so
    it's only a few ULP accurate, not bit-exact.

    fast_and_approximate_mode only has an effect when rounding_mode is None:
    the trunc/floor branch suppresses it on bfloat16 (workaround for #43209),
    forwarding it unmodified otherwise.
    """
    if scalar == 0.0:
        # #43209 is closed as a permanent hardware limitation, not a fix pending:
        # a single FPU instruction has no path to emit NaN or a signed zero, so
        # fast bfloat16 divide-by-zero with rounding_mode=None returns 0 instead.
        if rounding_mode is None and fast_and_approximate_mode:
            pytest.skip(
                "bfloat16 fast_and_approximate divide-by-zero is a hardware limitation "
                "(https://github.com/tenstorrent/tt-metal/issues/43209)"
            )
        input_a = generate_bfloat16_binary_grid(include_spl_values=True).reshape(64, 32)
        tt_a = to_tt_tensor(input_a, device)
        result = ttnn.to_torch(
            ttnn.divide(tt_a, scalar, rounding_mode=rounding_mode, fast_and_approximate_mode=fast_and_approximate_mode)
        ).float()
        golden = torch.sign(input_a.float()) * float("inf")
        assert_div_by_zero_outputs(golden, result)
        return

    input_a = generate_bfloat16_binary_grid(include_spl_values=False).reshape(64, 32)
    tt_a = to_tt_tensor(input_a, device)
    result = ttnn.to_torch(
        ttnn.divide(tt_a, scalar, rounding_mode=rounding_mode, fast_and_approximate_mode=fast_and_approximate_mode)
    ).float()

    raw_quotient = flush_subnormal_values_to_zero((input_a.float() / scalar).clone())
    golden = raw_quotient.to(torch.bfloat16).float()
    if rounding_mode == "trunc":
        golden = torch.trunc(golden)
    elif rounding_mode == "floor":
        golden = torch.floor(golden)

    # FPU fast mode (rounding_mode=None only) can flush a tiny-but-nonzero
    # quotient to 0 near underflow, similar in spirit to the SFPU subnormal
    # flush but at a coarser boundary.
    if rounding_mode is None and fast_and_approximate_mode:
        fpu_underflow = (result == 0) & (golden != 0) & (golden.abs() < 2.0**-120)
        golden = torch.where(fpu_underflow, torch.zeros_like(golden), golden)

    ulp_threshold = 6 if (rounding_mode is None and fast_and_approximate_mode) else 5
    assert_with_ulp(
        expected_result=golden.to(torch.bfloat16),
        actual_result=result.to(torch.bfloat16),
        ulp_threshold=ulp_threshold,
        allow_nonfinite=True,
    )


def _moderate_magnitude_grid():
    """Stratified bfloat16 grid restricted to 2**-115 .. 2**115, padded to a
    tile-friendly (64, 32) shape. Keeps a / b inside the exact-integer range
    remainder/fmod need (see module docstring).
    """
    values = generate_bfloat16_binary_grid(include_spl_values=False)
    in_band = values[(values.float().abs() >= 2.0**-115) & (values.float().abs() <= 2.0**115)]
    padded = torch.cat([in_band, in_band[0].repeat(2048 - in_band.numel())])
    return padded.reshape(64, 32)


_MODULO_SHIFTS = [0.3, 0.7, 1.3, 2.7, 4.5, -0.5, -1.5, -3.25]


@pytest.mark.parametrize("ttnn_op", [ttnn.remainder, ttnn.fmod])
@pytest.mark.parametrize("shift", _MODULO_SHIFTS)
def test_modulo_ops(device, ttnn_op, shift):
    """ttnn.remainder / ttnn.fmod, paired against itself scaled by ``shift``
    so a / b stays close to 1 / shift regardless of a's own magnitude.
    """
    input_a = _moderate_magnitude_grid()
    input_b = (input_a.float() * shift).to(torch.bfloat16)
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, input_b).float()
    result = ttnn.to_torch(ttnn_op(tt_a, tt_b)).float()
    assert_equal(golden, result)


def _small_magnitude_grid():
    """Stratified bfloat16 grid restricted to roughly [0.004, 128], padded to
    a tile-friendly (64, 32) shape -- the regime the scalar divisors below
    actually probe (see test_modulo_ops_scalar).
    """
    values = generate_bfloat16_binary_grid(include_spl_values=False)
    in_band = values[(values.float().abs() >= 2.0**-8) & (values.float().abs() <= 2.0**7)]
    padded = torch.cat([in_band, in_band[0].repeat(2048 - in_band.numel())])
    return padded.reshape(64, 32)


_MODULO_SCALAR_VALUES = [-0.002, -0.001, 0.0, 0.001, 0.002]


@pytest.mark.parametrize("ttnn_op", [ttnn.remainder, ttnn.fmod])
@pytest.mark.parametrize("scalar", _MODULO_SCALAR_VALUES)
def test_modulo_ops_scalar(device, ttnn_op, scalar):
    """Tensor-scalar ttnn.remainder / ttnn.fmod against a tiny divisor. Near
    an exact multiple of the scalar, fp precision can round either to 0 or
    to the scalar itself (#17361 / #17362), hence atol not exact match.
    """
    input_a = _small_magnitude_grid()
    tt_a = to_tt_tensor(input_a, device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, scalar, device=device).float()
    result = ttnn.to_torch(ttnn_op(tt_a, scalar)).float()

    if scalar == 0.0:
        # The device returns -inf where torch returns NaN for a zero divisor.
        result = torch.where(torch.isinf(result), torch.full_like(result, float("nan")), result)
        mismatch = ~torch.isclose(result, golden, equal_nan=True)
    else:
        mismatch = ~torch.isclose(result, golden, atol=0.001, rtol=0)
    assert not mismatch.any(), (
        f"{int(mismatch.sum())} of {mismatch.numel()} mismatched: max abs diff="
        f"{(result[mismatch] - golden[mismatch]).abs().max().item()}, "
        f"first golden={golden[mismatch].flatten()[0].item()} result={result[mismatch].flatten()[0].item()}"
    )


@pytest.mark.parametrize("ttnn_op", [ttnn.remainder, ttnn.fmod])
@pytest.mark.parametrize("input_shape", [(1024,), (69, 178), (3, 123, 115), (6, 5, 320, 320)])
def test_modulo_ops_scalar_shapes(device, ttnn_op, input_shape):
    """1D/3D/4D and non-tile-aligned shapes for the scalar remainder/fmod
    overload (moved from test_div_ops.py's test_remainder_scalar /
    test_fmod_scalar), covering the padding path the stratified-grid
    (64, 32) shape above doesn't exercise. Bit-pattern exhaustiveness is
    test_modulo_ops_scalar's job; this only checks shape/rank handling.
    """
    torch.manual_seed(0)
    input_a = torch.empty(input_shape, dtype=torch.bfloat16).uniform_(-100, 100)
    tt_a = ttnn.from_torch(input_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    golden_function = ttnn.get_golden_function(ttnn_op)
    for scalar in (-0.002, 0.001):
        golden = golden_function(input_a, scalar, device=device).float()
        result = ttnn.to_torch(ttnn_op(tt_a, scalar)).float()
        mismatch = ~torch.isclose(result, golden, atol=0.001, rtol=0)
        assert not mismatch.any(), f"{int(mismatch.sum())} of {mismatch.numel()} mismatched for shape {input_shape}"


def test_div_no_nan(device):
    """Dividend grid includes 0 and divisor grid includes 0, +-inf and NaN,
    so 0 / 0 is exercised too -- the golden maps every zero-divisor case
    (including 0 / 0) to 0, matching the whole point of this op.
    """
    values_a = generate_bfloat16_binary_grid(include_zero=True)
    values_b = generate_bfloat16_binary_grid(include_spl_values=True)
    input_a, input_b = torch.meshgrid(values_a, values_b, indexing="ij")
    input_a, input_b = input_a.contiguous(), input_b.contiguous()
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    bf16_quotient = _bf16_quotient(tt_a, tt_b, input_a, input_b)
    # A NaN divisor is also treated as unsafe, returning 0.
    nan_divisor = torch.isnan(input_b) & torch.isfinite(input_a)
    bf16_quotient = torch.where(nan_divisor, torch.zeros_like(bf16_quotient), bf16_quotient)
    golden = torch.where(input_b.float() == 0, torch.zeros_like(bf16_quotient), bf16_quotient)

    result = ttnn.to_torch(ttnn.div_no_nan(tt_a, tt_b)).float()
    assert_equal(golden, result)


def test_div_no_nan_scalar(device):
    """Tensor-scalar ttnn.div_no_nan. Like ttnn.divide's scalar overload,
    this multiplies by a host-computed reciprocal for nonzero divisors, so
    it's only a few ULP accurate there; a zero divisor is always exact 0.
    """
    input_a = generate_bfloat16_binary_grid(include_spl_values=True).reshape(64, 32)
    tt_a = to_tt_tensor(input_a, device)
    result = ttnn.to_torch(ttnn.div_no_nan(tt_a, 0.0)).float()
    assert_equal(torch.zeros_like(result), result)

    input_a = generate_bfloat16_binary_grid(include_spl_values=False).reshape(64, 32)
    tt_a = to_tt_tensor(input_a, device)
    for scalar in (-5.1, 10.9):
        result = ttnn.to_torch(ttnn.div_no_nan(tt_a, scalar)).float()
        golden = flush_subnormal_values_to_zero((input_a.float() / scalar).clone()).to(torch.bfloat16)
        assert_with_ulp(expected_result=golden, actual_result=result.to(torch.bfloat16), ulp_threshold=5)


def test_floor_div(device):
    """Dividend grid has no special values (0 / 0's floor is undefined
    either way); divisor grid includes 0, +-inf and NaN.
    """
    values_a = generate_bfloat16_binary_grid()
    values_b = generate_bfloat16_binary_grid(include_spl_values=True)
    input_a, input_b = torch.meshgrid(values_a, values_b, indexing="ij")
    input_a, input_b = input_a.contiguous(), input_b.contiguous()
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    bf16_quotient = _bf16_quotient(tt_a, tt_b, input_a, input_b)
    # Same NaN-divisor quirk as div_no_nan above.
    nan_divisor = torch.isnan(input_b) & torch.isfinite(input_a)
    bf16_quotient = torch.where(nan_divisor, torch.zeros_like(bf16_quotient), bf16_quotient)
    golden = torch.floor(bf16_quotient)

    result = ttnn.to_torch(ttnn.floor_div(tt_a, tt_b)).float()
    assert_equal(golden, result)


@pytest.mark.parametrize("scalar", _DIV_SCALAR_VALUES)
def test_floor_div_scalar(device, scalar):
    """Tensor-scalar ttnn.floor_div. This overload computes
    floor(a * (1 / scalar)) via a separate multiply-then-floor, so it's not
    bit-exact either.
    """
    include_spl_values = scalar == 0.0
    input_a = generate_bfloat16_binary_grid(include_spl_values=include_spl_values).reshape(64, 32)
    tt_a = to_tt_tensor(input_a, device)
    result = ttnn.to_torch(ttnn.floor_div(tt_a, scalar)).float()

    if scalar == 0.0:
        a = input_a.float()
        # 0 and NaN inputs hit the "packs NaN as +inf" quirk, not the exact NaN.
        golden = torch.where((a == 0) | torch.isnan(a), torch.full_like(a, float("inf")), torch.sign(a) * float("inf"))

        nan_mismatch = torch.isnan(golden) != torch.isnan(result)
        assert not nan_mismatch.any(), (
            f"{int(nan_mismatch.sum())} NaN-position mismatches; "
            f"a={a[nan_mismatch].flatten()[0].item()} golden={golden[nan_mismatch].flatten()[0].item()} "
            f"result={result[nan_mismatch].flatten()[0].item()}"
        )
        inf_mismatch = torch.isinf(golden) != torch.isinf(result)
        assert not inf_mismatch.any(), (
            f"{int(inf_mismatch.sum())} inf-position mismatches; "
            f"a={a[inf_mismatch].flatten()[0].item()} golden={golden[inf_mismatch].flatten()[0].item()} "
            f"result={result[inf_mismatch].flatten()[0].item()}"
        )
        both_inf = torch.isinf(golden) & torch.isinf(result)
        sign_mismatch = torch.sign(golden[both_inf]) != torch.sign(result[both_inf])
        assert not sign_mismatch.any(), (
            f"{int(sign_mismatch.sum())} inf-sign mismatches; "
            f"golden={golden[both_inf][sign_mismatch].flatten()[0].item()} "
            f"result={result[both_inf][sign_mismatch].flatten()[0].item()}"
        )
        return

    raw_quotient = flush_subnormal_values_to_zero((input_a.float() / scalar).clone())
    golden = torch.floor(raw_quotient.to(torch.bfloat16).float())
    assert_with_ulp(
        expected_result=golden.to(torch.bfloat16),
        actual_result=result.to(torch.bfloat16),
        ulp_threshold=4,
        allow_nonfinite=True,
    )


_POW_INT_EXPONENTS = [-3, -2, -1, 0, 1, 2, 3]
_POW_FLOAT_EXPONENTS = [-1.5, -0.5, 0.5, 1.5, 2.5]


@pytest.mark.parametrize("exponent", _POW_INT_EXPONENTS)
def test_pow_int_exponent(device, exponent):
    """tensor ** int (both signs of base are well-defined here)."""
    input_a = generate_bfloat16_binary_grid(include_spl_values=False).reshape(64, 32)
    tt_a = to_tt_tensor(input_a, device)

    golden_function = ttnn.get_golden_function(ttnn.pow)
    # Only the golden is flushed: it models what the hardware does to its own
    # subnormal results, but flushing the device result too would mask a real
    # divergence (e.g. device returns a subnormal where golden is exact 0).
    golden = flush_subnormal_values_to_zero(golden_function(input_a, exponent).float().clone()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.pow(tt_a, exponent)).float().to(torch.bfloat16)
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=4, allow_nonfinite=True)


def _positive_grid():
    values = generate_bfloat16_binary_grid(include_spl_values=False)
    return values[values > 0].reshape(32, 32)


@pytest.mark.parametrize("exponent", _POW_FLOAT_EXPONENTS)
def test_pow_float_exponent(device, exponent):
    """tensor ** float; positive base only (see module docstring)."""
    input_a = _positive_grid()
    tt_a = to_tt_tensor(input_a, device)

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = flush_subnormal_values_to_zero(golden_function(input_a, exponent).float().clone()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.pow(tt_a, exponent)).float().to(torch.bfloat16)
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2, allow_nonfinite=True)


def test_pow_tensor_exponent(device):
    """tensor ** tensor, exponent cycling through the int/float lists above."""
    input_a = _positive_grid()
    tt_a = to_tt_tensor(input_a, device)

    exponent_values = torch.tensor(_POW_INT_EXPONENTS + _POW_FLOAT_EXPONENTS, dtype=torch.bfloat16)
    reps = input_a.numel() // exponent_values.numel() + 1
    exponents = exponent_values.repeat(reps)[: input_a.numel()].reshape(input_a.shape)
    tt_exponents = to_tt_tensor(exponents, device)

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = flush_subnormal_values_to_zero(golden_function(input_a, exponents).float().clone()).to(torch.bfloat16)
    result = ttnn.to_torch(ttnn.pow(tt_a, tt_exponents)).float().to(torch.bfloat16)
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2, allow_nonfinite=True)

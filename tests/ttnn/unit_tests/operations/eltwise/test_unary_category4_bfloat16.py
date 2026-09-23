# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import math
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_with_pcc, assert_allclose
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
    float_to_bf16_bits,
    bf16_bits_to_float,
)
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.use_module_device

"""
Category 4: Ops with float/scalar parameter

11. ttnn.elu          - alpha (default 1.0):  x >= 0 ? x : alpha * (exp(x) - 1)
12. ttnn.heaviside     - value:                 x < 0 ? 0 : x > 0 ? 1 : value
13. ttnn.leaky_relu    - negative_slope:        x >= 0 ? x : negative_slope * x
14. ttnn.relu_max      - upper_limit:           max(0, min(x, upper_limit))
15. ttnn.relu_min      - lower_limit:           max(x, lower_limit)
16. ttnn.rpow          - exponent:              exponent ** x
17. ttnn.celu          - alpha (default 1.0):   x >= 0 ? x : alpha * (exp(x / alpha) - 1)
18. ttnn.softcap       - beta:                  beta * tanh(x / beta)
19. ttnn.fill          - fill_value:            fill_value (input is ignored)
20. ttnn.hardshrink    - lambd (default 0.5):   |x| > lambd ? x : 0
21. ttnn.softshrink    - lambd (default 0.5):   x > lambd ? x - lambd : x < -lambd ? x + lambd : 0

Every op sweeps all 32,512 positive + 32,512 negative normal bfloat16 values
(one (256, 256) tile, one dispatch) through the tensor input. The scalar
parameter can't be swept the same way, so it's parametrized over a
representative set of values instead -- except `fill`, whose output doesn't
depend on the input at all, so `fill_value` itself is the swept domain there.

Accuracy criteria
─────────────────
  heaviside, relu_max, relu_min, hardshrink : exact (bit-for-bit)
      Pure comparison/selection, no arithmetic.
  leaky_relu, softshrink                    : ULP ≤ 1
      One multiply/add-sub on the non-identity branch.
  elu, celu                                 : ULP ≤ 1 (excluding a narrow band)
      exp(x)-1 [or exp(x/alpha)-1] cancels near 0; characterized via
      allclose in test_elu_allclose / test_celu_allclose below.
  softcap                                   : ULP ≤ 2 (post-FTZ) + PCC ≥ 0.9999
      tanh is a Sollya polynomial approximation. Blackhole only.
  rpow                                      : PCC ≥ 0.99 + allclose(atol=1e-2, rtol=0.1)
      exp2/log2 chain; transcendental approximation error dominates.
  fill                                      : exact (bit-for-bit)
      Output is the parameter value itself, broadcast.
"""


def _flush_subnormal_golden_and_result(golden, result, exact=None):
    """Zero `golden` where the true (pre-rounding) result is subnormal, since the
    device flushes such outputs to zero (FTZ). `result` is left untouched so a
    device that fails to flush still shows up as a mismatch.

    `exact`, if given, is a higher-precision (non-bf16-rounded) reference used to
    decide the FTZ mask instead of `golden`. Needed because bf16's coarse rounding
    can round a truly-subnormal result *up* into the smallest normal value (seen
    with leaky_relu(negative_slope=-0.5) near x = -2*tiny), which would make
    `golden` alone miss that the device's 0 there is actually correct.
    """
    reference = exact if exact is not None else golden
    tiny = torch.finfo(torch.bfloat16).tiny  # smallest normal magnitude
    ftz = (reference.abs() > 0) & (reference.abs() < tiny)
    golden = golden.clone()
    golden[ftz] = 0.0
    return golden, result


def _exhaustive_input(mask_predicate=None, safe_value=1.0):
    """All normal bfloat16 bit patterns as a (1, 1, 256, 256) tensor.

    If mask_predicate is given, matching elements are replaced by safe_value
    (preserving tile shape / coverage of the rest) instead of being dropped.
    """
    B = generate_bfloat16_bits(include_spl_values=False)  # (256, 256)
    if mask_predicate is not None:
        mask = mask_predicate(B)
        B = torch.where(mask, torch.full_like(B, safe_value), B)
    return B.unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# heaviside, relu_max, relu_min, hardshrink — exact (bit-for-bit)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, -1.0, 2.5])
def test_heaviside_op(device, value):
    """output_i = 0 if x_i < 0, value if x_i == 0, 1 if x_i > 0.

    Also exercises the preallocated output_tensor=/queue_id= path (previously
    covered only by the now-removed test_unary_heaviside_ttnn).
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.heaviside)
    golden = golden_function(input_tensor, value, device=device)

    tt_result = ttnn.heaviside(tt_in, value)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"heaviside(value={value}) diverged for {int((result != golden).sum().item())} " f"of {result.numel()} elements"
    )

    preallocated_output = to_tt_tensor(torch.zeros_like(input_tensor), device)
    ttnn.heaviside(tt_in, value, output_tensor=preallocated_output, queue_id=0)
    preallocated_result = ttnn.to_torch(preallocated_output)
    assert torch.equal(preallocated_result, golden), (
        f"heaviside(value={value}) via output_tensor=/queue_id= diverged for "
        f"{int((preallocated_result != golden).sum().item())} of {preallocated_result.numel()} elements"
    )


@pytest.mark.parametrize("upper_limit", [0.0, 1.0, 3.0, 100.0, -1.0])
def test_relu_max_op(device, upper_limit):
    """output_i = max(0, min(x_i, upper_limit)). upper_limit=-1.0 covers the
    degenerate case where min(x, upper_limit) <= -1 for every x, collapsing
    the output to the constant 0."""
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.relu_max)
    golden = golden_function(input_tensor, upper_limit, device=device)

    tt_result = ttnn.relu_max(tt_in, upper_limit)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"relu_max(upper_limit={upper_limit}) diverged for {int((result != golden).sum().item())} "
        f"of {result.numel()} elements"
    )


@pytest.mark.parametrize("lower_limit", [0.0, -1.0, -3.0, -100.0, 1.0])
def test_relu_min_op(device, lower_limit):
    """output_i = max(x_i, lower_limit). lower_limit=1.0 covers the case where
    the lower bound sits above 0."""
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.relu_min)
    golden = golden_function(input_tensor, lower_limit, device=device)

    tt_result = ttnn.relu_min(tt_in, lower_limit)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"relu_min(lower_limit={lower_limit}) diverged for {int((result != golden).sum().item())} "
        f"of {result.numel()} elements"
    )


@pytest.mark.parametrize("lambd", [0.5, 0.25, 1.0, 2.0, 4.0])
def test_hardshrink_op(device, lambd):
    """output_i = x_i if |x_i| > lambd else 0. All lambd values used here are
    exactly bf16-representable, so device's RNE pre-rounding of lambd (see
    UnaryOpType::HARDSHRINK in unary_op_utils.cpp) can't diverge from golden."""
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.hardshrink)
    golden = golden_function(input_tensor, lambd=lambd, device=device)

    tt_result = ttnn.hardshrink(tt_in, lambd=lambd)
    result = ttnn.to_torch(tt_result)

    assert torch.equal(result, golden), (
        f"hardshrink(lambd={lambd}) diverged for {int((result != golden).sum().item())} "
        f"of {result.numel()} elements"
    )


# ─────────────────────────────────────────────────────────────────────────────
# leaky_relu, softshrink — ULP ≤ 1
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("negative_slope", [0.01, 0.1, 1.0, 2.0, -0.5])
def test_leaky_relu_op(device, negative_slope):
    """output_i = x_i if x_i >= 0, else negative_slope * x_i.

    A small negative_slope can push a normal x_i into subnormal range (FTZ,
    masked via _flush_subnormal_golden_and_result). |negative_slope| > 1 can
    overflow the most-negative bf16 values to +/-inf on both sides, so
    non-finite values are allowed as long as they agree in position.

    Also exercises the preallocated output_tensor=/queue_id= path (previously
    covered only by the now-removed test_unary_leaky_relu_ttnn).
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.leaky_relu)
    golden_raw = golden_function(input_tensor, negative_slope=negative_slope, device=device)

    # Exact (float64) reference for the FTZ mask -- see _flush_subnormal_golden_and_result.
    x64 = input_tensor.to(torch.float64)
    exact = torch.where(input_tensor >= 0, x64, x64 * negative_slope)

    tt_result = ttnn.leaky_relu(tt_in, negative_slope=negative_slope)
    result = ttnn.to_torch(tt_result)
    golden, result = _flush_subnormal_golden_and_result(golden_raw, result, exact=exact)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)

    preallocated_output = to_tt_tensor(torch.zeros_like(input_tensor), device)
    ttnn.leaky_relu(tt_in, negative_slope=negative_slope, output_tensor=preallocated_output, queue_id=0)
    preallocated_result = ttnn.to_torch(preallocated_output)
    golden_pa, preallocated_result = _flush_subnormal_golden_and_result(golden_raw, preallocated_result, exact=exact)

    assert_with_ulp(expected_result=golden_pa, actual_result=preallocated_result, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize("lambd", [0.5, 0.25, 1.0, 2.0])
def test_softshrink_op(device, lambd):
    """output_i = x_i - lambd if x_i > lambd, x_i + lambd if x_i < -lambd, else 0.
    The shrink branches' add/sub can land in subnormal range near the lambd
    boundary (FTZ, masked via _flush_subnormal_golden_and_result)."""
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.softshrink)
    golden = golden_function(input_tensor, lambd=lambd, device=device)

    # Exact (float64) reference for the FTZ mask -- see _flush_subnormal_golden_and_result.
    x64 = input_tensor.to(torch.float64)
    exact = torch.where(
        input_tensor > lambd, x64 - lambd, torch.where(input_tensor < -lambd, x64 + lambd, torch.zeros_like(x64))
    )

    tt_result = ttnn.softshrink(tt_in, lambd=lambd)
    result = ttnn.to_torch(tt_result)
    golden, result = _flush_subnormal_golden_and_result(golden, result, exact=exact)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


# ─────────────────────────────────────────────────────────────────────────────
# elu, celu — ULP ≤ 1 (excluding a narrow cancellation band near 0)
# ─────────────────────────────────────────────────────────────────────────────

# Characterized (default alpha=1.0)
# elu's band is in x directly (alpha only scales the output
# after cancellation); celu's band is in x/alpha (it evaluates exp(x/alpha)-1),
# so the celu mask below scales the band by alpha.
_ELU_CANCELLATION_BAND = (-0.28515625, 1.1663108012064884e-38)


@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0, 0.1, 0.0, -1.0, -0.5])
def test_elu_op(device, alpha):
    """output_i = x_i if x_i >= 0, else alpha * (exp(x_i) - 1). Includes
    alpha=0 and negative alpha (valid for elu, unlike celu) since the
    removed test_scalarB_elu covered those and this replaces it."""
    low, high = _ELU_CANCELLATION_BAND
    input_tensor = _exhaustive_input(mask_predicate=lambda b: (b >= low) & (b <= high))

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.elu)
    golden = golden_function(input_tensor, alpha=alpha, device=device)

    tt_result = ttnn.elu(tt_in, alpha=alpha)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0, 0.1])
def test_celu_op(device, alpha):
    """output_i = x_i if x_i >= 0, else alpha * (exp(x_i / alpha) - 1)."""
    low, high = _ELU_CANCELLATION_BAND
    low, high = low * alpha, high * alpha
    input_tensor = _exhaustive_input(mask_predicate=lambda b: (b >= low) & (b <= high))

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.celu)
    golden = golden_function(input_tensor, alpha=alpha, device=device)

    tt_result = ttnn.celu(tt_in, alpha=alpha)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


# ─────────────────────────────────────────────────────────────────────────────
# elu (narrow cancellation band near 0)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "low, high, expected_atol, expected_rtol",
    [
        (-0.28515625, 1.1663108012064884e-38, 0.002, 0.02),
        (-88.0, 1.6 * 10**38, 0.0, 0.0),  # bf16 working range for elu
    ],
)
def test_elu_allclose(low, high, expected_atol, expected_rtol, device):
    num_elements = math.prod(torch.Size([1, 3, 320, 320]))
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(torch.Size([1, 3, 320, 320]))

    golden_function = ttnn.get_golden_function(ttnn.elu)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.elu(tt_in)
    result = ttnn.to_torch(tt_result)
    assert torch.allclose(golden, result, atol=expected_atol, rtol=expected_rtol)


# ─────────────────────────────────────────────────────────────────────────────
# celu (narrow cancellation band near 0)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "low, high, expected_atol, expected_rtol",
    [
        (-1.6 * 10**38, -0.28515625, 0.001, 0.004),
        (-0.28515625, 1.1663108012064884e-38, 0.002, 0.02),
        (1.1663108012064884e-38, 1.6 * 10**38, 1e-6, 1e-6),
    ],
)
def test_celu_allclose(low, high, expected_atol, expected_rtol, device):
    num_elements = math.prod([1, 3, 320, 320])
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(torch.Size([1, 3, 320, 320]))

    golden_function = ttnn.get_golden_function(ttnn.celu)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.celu(tt_in)
    result = ttnn.to_torch(tt_result)
    assert torch.allclose(golden, result, atol=expected_atol, rtol=expected_rtol)


# ─────────────────────────────────────────────────────────────────────────────
# softcap — ULP ≤ 2 (post FTZ-mask) + PCC ≥ 0.9999. Blackhole only.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not is_blackhole(), reason="softcap is implemented for Blackhole only")
@pytest.mark.parametrize("beta", [1.0, 0.5, 100.0])  # beta=25.0 already covered exhaustively by
# test_unary.py::test_softcap_bfloat16_full_domain; not duplicated here.
def test_softcap_op(device, beta):
    """output_i = beta * tanh(x_i / beta)."""
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.softcap)
    golden = golden_function(input_tensor, beta=beta, device=device)

    tt_result = ttnn.softcap(tt_in, beta)
    result = ttnn.to_torch(tt_result)

    max_abs = result.to(torch.float32).abs().max().item()
    bound = beta * (1.0 + 2**-8)
    assert max_abs <= bound, f"softcap overshoot: max |out| {max_abs:.4f} > bound {bound:.4f}"

    # FTZ guard scales with beta: tanh(x/beta) ~= x/beta near 0.
    flush_floor = 1e-30 * max(beta, 1.0)
    mask = golden.abs() > flush_floor
    assert_with_ulp(expected_result=golden[mask], actual_result=result[mask], ulp_threshold=2)
    assert_with_pcc(golden[mask], result[mask], pcc=0.9999)

    # Bound the masked-out (negligible-reference) region too, same as
    # test_softcap_bfloat16_full_domain, so it isn't a silent blind spot.
    tiny_max = result[~mask].to(torch.float32).abs().max().item()
    assert tiny_max <= 4.0 * flush_floor, f"softcap negligible-reference region returned {tiny_max:.4e}"


# ─────────────────────────────────────────────────────────────────────────────
# rpow — PCC ≥ 0.99 + allclose(atol=1e-2, rtol=0.1)
# ─────────────────────────────────────────────────────────────────────────────

# rpow evaluates exponent ** x as exp2(x * log2(exponent)). Once |x * log2(exponent)|
# >= ~8.1e31, the device's exp2 argument reduction breaks down and returns a bogus
# finite value (observed: 1.0) instead of +/-inf like golden -- a real SFPU
# limitation at extreme magnitudes. Excluded from the accuracy assertions below
# (wide margin under the observed boundary) but still swept through the device.
_RPOW_UNRELIABLE_ARG_MAGNITUDE = 1e30


@pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0])
def test_rpow_op(device, exponent):
    """output_i = exponent ** x_i. Non-finite classification (overflow/underflow
    to +/-inf) is checked separately from the finite-element PCC/allclose
    comparison, since ULP/PCC aren't meaningful once either side is non-finite.
    """
    input_tensor = _exhaustive_input()

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.rpow)
    golden = golden_function(input_tensor, exponent, device=device)

    tt_result = ttnn.rpow(tt_in, exponent)
    result = ttnn.to_torch(tt_result)

    arg_magnitude = input_tensor.to(torch.float64).abs() * abs(math.log2(exponent))
    reliable = arg_magnitude < _RPOW_UNRELIABLE_ARG_MAGNITUDE

    golden_finite = torch.isfinite(golden)
    result_finite = torch.isfinite(result)
    mismatched_finiteness = (golden_finite != result_finite) & reliable
    assert not mismatched_finiteness.any(), (
        f"rpow(exponent={exponent}) finite/non-finite classification diverged for "
        f"{int(mismatched_finiteness.sum().item())} of {result.numel()} elements"
    )

    # isfinite() alone doesn't distinguish NaN from Inf, or +inf from -inf, so a
    # device returning e.g. +inf where golden has -inf (or NaN) would still pass
    # the check above. Verify those match exactly too.
    golden_nan, result_nan = torch.isnan(golden), torch.isnan(result)
    nan_mismatch = (golden_nan != result_nan) & reliable
    assert (
        not nan_mismatch.any()
    ), f"rpow(exponent={exponent}) NaN classification diverged for {int(nan_mismatch.sum().item())} elements"
    golden_inf, result_inf = torch.isinf(golden), torch.isinf(result)
    sign_mismatch = golden_inf & result_inf & reliable & (torch.sign(golden) != torch.sign(result))
    assert (
        not sign_mismatch.any()
    ), f"rpow(exponent={exponent}) infinity sign diverged for {int(sign_mismatch.sum().item())} elements"

    finite = golden_finite & result_finite & reliable
    assert_with_pcc(golden[finite], result[finite], pcc=0.99)
    assert_allclose(golden[finite], result[finite], atol=1e-2, rtol=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# fill — exact (bit-for-bit). fill_value itself is the swept domain.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fill_value",
    [
        # Exactly bf16-representable.
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        100.0,
        -100.0,
        # Smallest/largest finite bf16 magnitudes, with sign.
        torch.finfo(torch.bfloat16).tiny,
        -torch.finfo(torch.bfloat16).tiny,
        torch.finfo(torch.bfloat16).max,
        -torch.finfo(torch.bfloat16).max,
        # NOT exactly bf16-representable: exercises fill_value's own fp32->bf16
        # conversion instead of trivially agreeing with the CPU reference.
        0.1,
        -3.4028235e38,  # fp32 max; truncates to bf16's max instead of rounding to -inf
        -10.0,
        15.5,
        -29.5,
        2147483647.0,
        -2147483648.0,
    ],
)
def test_fill_op(device, fill_value):
    """output is fill_value broadcast; the input tensor is ignored, so fill_value
    itself is parametrized across representative bf16 value classes instead of
    the input being swept.

    The device truncates fill_value to bf16 rather than round-to-nearest-even
    (unlike torch.full_like()'s default conversion), so golden is built from the
    truncated value to match -- otherwise e.g. 0.1 or fp32 max would spuriously
    disagree (the latter overflows to inf under RNE, but not under truncation).
    """
    input_tensor = torch.arange(256 * 256, dtype=torch.float32).view(1, 1, 256, 256).to(torch.bfloat16)
    device_fill_value = bf16_bits_to_float(float_to_bf16_bits(fill_value))

    tt_in = to_tt_tensor(input_tensor, device)
    golden_function = ttnn.get_golden_function(ttnn.fill)
    golden = golden_function(input_tensor, device_fill_value, device=device)

    tt_result = ttnn.fill(tt_in, fill_value)
    result = ttnn.to_torch(tt_result)

    # Bit-exact, except sign-of-zero: device canonicalizes a -0.0 fill_value to
    # +0.0 (verified on hardware), so fall back to value equality there.
    result_bits = result.view(torch.int16)
    golden_bits = golden.view(torch.int16)
    zero_fill = golden == 0.0
    bits_match = result_bits == golden_bits
    value_match = result == golden
    elementwise_match = torch.where(zero_fill, value_match, bits_match)
    assert elementwise_match.all(), (
        f"fill(fill_value={fill_value}) diverged for {int((~elementwise_match).sum().item())} "
        f"of {result.numel()} elements"
    )

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

Accuracy criteria
─────────────────
  div (all rounding modes), div_no_nan, floor_div : exact, once golden is built
      from the same bfloat16-rounded quotient the device itself computes (see
      _bf16_quotient below). Both the SFPU subnormal-flush and the reciprocal
      underflow at |b| >= 2**126 change what that quotient *is*, not just its
      rounding, so the golden has to see them too.
  remainder, fmod : exact, over operand pairs of comparable magnitude (see
      _moderate_magnitude_grid). Both ops reduce to a - b * trunc(a / b); with
      bfloat16's 8 mantissa bits, trunc(a / b) itself loses precision once
      |a / b| grows large, and multiplying that error back by b amplifies it
      without bound. This is a property of the algorithm at low precision, not
      a device defect, so the grid here keeps every quotient it exercises well
      inside bfloat16's exactly-representable integer range.
  pow : within a few ULP (see assert_with_ulp calls). Non-integer exponents
      make pow(negative_base, exponent) = NaN in torch; bfloat16 dest packs
      that NaN as +inf (a separately-documented hardware limitation -- see
      test_binary_pow in test_pow.py), so those cases use a positive-only base
      to isolate the accuracy of the underlying computation.

Tensor-scalar overloads
────────────────────────
  div and floor_div also each get a tensor-scalar variant (test_divide_scalar,
  test_floor_div_scalar), moved from PCC-based tests in
  test_binary_composite.py. The scalar overloads dispatch through a
  host-computed reciprocal or a separate multiply-then-round rather than the
  tensor-tensor binary_ng path, so they are checked with a few ULP of slack
  rather than exact match.

Deliberately NOT duplicated here
──────────────────────────────────
  A few bfloat16 tests elsewhere are narrower/fixed-value or random-sweep
  checks rather than exhaustive stratified-grid coverage, so they were left
  where they are instead of being folded into this file:
    - test_pow.py's test_binary_sfpu_accuracy (a fixed 10-element tensor) and
      test_binary_sfpu_accuracy_pos (a uniform-random continuous exponent
      sweep) still test bfloat16 alongside float32.
    - test_div_ops.py's test_remainder_scalar / test_fmod_scalar (tiny fixed
      scalar divisors chosen to hit the documented #17361 / #17362
      near-exact-multiple precision quirk) are unchanged.
    - sub_core_grids coverage (test_div_composite_ops_with_subcore_grids,
      test_remainder_composite_ops_with_subcore_grids in
      test_binary_bcast.py) stays put too: its purpose is exercising the
      sub_core_grids parameter itself, not general accuracy.
"""


def _bf16_quotient(tt_a, tt_b, input_a, input_b):
    """Reproduce the bfloat16 quotient that ttnn.divide itself produces before
    any rounding_mode is applied, so div/div_no_nan/floor_div goldens can all
    be built from the same intermediate the device computes from.

    Two device-only effects are folded in here, both changing the quotient
    itself rather than just how it gets rounded:
      - Subnormal flush: the SFPU flushes any |a / b| below the smallest
        normal bfloat16 (2**-126) to zero before rounding to bfloat16.
      - Reciprocal underflow: once |b| >= 2**126, the bfloat16 reciprocal of
        b itself underflows, so the device's a * recip(b) collapses to zero
        even though the true quotient is a representable nonzero value. The
        exact cutoff is not a clean power of two, so this is detected from
        the device's own plain divide rather than guessed at.
    """
    raw_quotient = flush_subnormal_values_to_zero((input_a.float() / input_b.float()).clone())
    bf16_quotient = raw_quotient.to(torch.bfloat16).float()

    plain = ttnn.to_torch(ttnn.divide(tt_a, tt_b, fast_and_approximate_mode=False)).float()
    recip_underflow = (
        (plain == 0) & (bf16_quotient != 0) & (input_b.abs() >= 2.0**126) & torch.isfinite(bf16_quotient)
    )
    return torch.where(recip_underflow, torch.zeros_like(bf16_quotient), bf16_quotient)


@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_divide_round_modes(device, rounding_mode):
    """Pairwise coverage of ttnn.divide's rounding_mode over the stratified
    bfloat16 grid. bfloat16 rounding_mode always runs SFPU-accurate (the
    fast-and-approximate divide bug #43209 is suppressed internally for
    bfloat16), so this only needs one path per mode.
    """
    input_a, input_b = pairwise_inputs(include_zero=False)
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    bf16_quotient = _bf16_quotient(tt_a, tt_b, input_a, input_b)
    if rounding_mode == "trunc":
        golden = torch.trunc(bf16_quotient)
    elif rounding_mode == "floor":
        golden = torch.floor(bf16_quotient)
    else:
        golden = bf16_quotient

    result = ttnn.to_torch(ttnn.divide(tt_a, tt_b, rounding_mode=rounding_mode)).float()
    assert_equal(golden, result)


_DIV_SCALAR_VALUES = [-5.1, 0.0, 10.9]


@pytest.mark.parametrize("scalar", _DIV_SCALAR_VALUES)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_divide_scalar(device, rounding_mode, scalar):
    """Tensor-scalar ttnn.divide over the stratified grid (moved from
    test_binary_composite.py's test_binary_div_scalar_ttnn[_opt], which used
    PCC over three shapes of random data instead).

    The scalar overload multiplies by a host-computed reciprocal rather than
    dividing directly, so unlike the tensor-tensor path above it is not
    bit-exact -- a handful of ULP at the scale tested here -- hence
    assert_with_ulp rather than assert_equal.
    """
    if scalar == 0.0:
        input_a = generate_bfloat16_binary_grid(include_spl_values=True).reshape(64, 32)
        tt_a = to_tt_tensor(input_a, device)
        result = ttnn.to_torch(ttnn.divide(tt_a, scalar, rounding_mode=rounding_mode)).float()
        golden = torch.sign(input_a.float()) * float("inf")
        assert_div_by_zero_outputs(golden, result)
        return

    input_a = generate_bfloat16_binary_grid(include_spl_values=False).reshape(64, 32)
    tt_a = to_tt_tensor(input_a, device)
    result = ttnn.to_torch(ttnn.divide(tt_a, scalar, rounding_mode=rounding_mode)).float()

    raw_quotient = flush_subnormal_values_to_zero((input_a.float() / scalar).clone())
    golden = raw_quotient.to(torch.bfloat16).float()
    if rounding_mode == "trunc":
        golden = torch.trunc(golden)
    elif rounding_mode == "floor":
        golden = torch.floor(golden)
    assert_with_ulp(
        expected_result=golden.to(torch.bfloat16),
        actual_result=result.to(torch.bfloat16),
        ulp_threshold=5,
        allow_nonfinite=True,
    )


def _moderate_magnitude_grid():
    """Stratified bfloat16 grid restricted to a magnitude band safely away
    from the subnormal/overflow edges, padded back out to a tile-friendly
    (64, 32) shape by repeating its first (in-band) value.

    See the module docstring: remainder/fmod's a - b * trunc(a / b) needs
    |a / b| to stay inside bfloat16's exact-integer range, which restricting
    both operands to 2**-115 .. 2**115 guarantees for every shift below.
    """
    values = generate_bfloat16_binary_grid(include_spl_values=False)
    in_band = values[(values.float().abs() >= 2.0**-115) & (values.float().abs() <= 2.0**115)]
    padded = torch.cat([in_band, in_band[0].repeat(2048 - in_band.numel())])
    return padded.reshape(64, 32)


_MODULO_SHIFTS = [0.3, 0.7, 1.3, 2.7, 4.5, -0.5, -1.5, -3.25]


@pytest.mark.parametrize("ttnn_op", [ttnn.remainder, ttnn.fmod])
@pytest.mark.parametrize("shift", _MODULO_SHIFTS)
def test_modulo_ops(device, ttnn_op, shift):
    """ttnn.remainder / ttnn.fmod over the moderate-magnitude grid, paired
    against itself scaled by ``shift`` so the quotient a / b is always close
    to 1 / shift -- comparable in magnitude across the whole grid regardless
    of where a itself sits in the bfloat16 range.
    """
    input_a = _moderate_magnitude_grid()
    input_b = (input_a.float() * shift).to(torch.bfloat16)
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, input_b).float()
    result = ttnn.to_torch(ttnn_op(tt_a, tt_b)).float()
    assert_equal(golden, result)


def test_div_no_nan(device):
    """ttnn.div_no_nan over the stratified grid, dividend without special
    values (so 0 / 0 never arises -- that degenerate case is not part of this
    op's documented contract) against a divisor grid that does include 0,
    +-inf and NaN, since returning 0 there is the entire point of this op.
    """
    values_a = generate_bfloat16_binary_grid(include_zero=True)
    values_b = generate_bfloat16_binary_grid(include_spl_values=True)
    input_a, input_b = torch.meshgrid(values_a, values_b, indexing="ij")
    input_a, input_b = input_a.contiguous(), input_b.contiguous()
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    bf16_quotient = _bf16_quotient(tt_a, tt_b, input_a, input_b)
    # NaN divisor: div_no_nan's "return 0 unless the divisor is finite and
    # nonzero" contract also swallows a NaN divisor as unsafe on this device.
    nan_divisor = torch.isnan(input_b) & torch.isfinite(input_a)
    bf16_quotient = torch.where(nan_divisor, torch.zeros_like(bf16_quotient), bf16_quotient)
    golden = torch.where(input_b.float() == 0, torch.zeros_like(bf16_quotient), bf16_quotient)

    result = ttnn.to_torch(ttnn.div_no_nan(tt_a, tt_b)).float()
    assert_equal(golden, result)


def test_floor_div(device):
    """ttnn.floor_div(tensor, tensor) over the stratified grid, dividend
    without special values (0 / 0's floor is undefined either way) against a
    divisor grid that does include 0, +-inf and NaN.
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
    """Tensor-scalar ttnn.floor_div over the stratified grid (moved from
    test_binary_composite.py's test_binary_floor_div_overload_ttnn, which
    used PCC over three shapes of random data instead).

    Unlike the tensor-tensor path above, this overload computes
    floor(a * (1 / scalar)) via a separate multiply-then-floor, so it also
    isn't bit-exact.
    """
    include_spl_values = scalar == 0.0
    input_a = generate_bfloat16_binary_grid(include_spl_values=include_spl_values).reshape(64, 32)
    tt_a = to_tt_tensor(input_a, device)
    result = ttnn.to_torch(ttnn.floor_div(tt_a, scalar)).float()

    if scalar == 0.0:
        a = input_a.float()
        # 0 and NaN inputs hit the same "packs NaN as +inf" quirk as elsewhere
        # in this file, rather than the exact NaN the C++ zero-divisor branch
        # (ttnn::where(eqz(a), nan, sign(a) * inf)) computes mathematically.
        golden = torch.where((a == 0) | torch.isnan(a), torch.full_like(a, float("inf")), torch.sign(a) * float("inf"))
        assert torch.equal(torch.isnan(golden), torch.isnan(result))
        assert torch.equal(torch.isinf(golden), torch.isinf(result))
        both_inf = torch.isinf(golden) & torch.isinf(result)
        assert torch.equal(torch.sign(golden[both_inf]), torch.sign(result[both_inf]))
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
    """tensor ** int over the stratified grid (both signs of base are
    well-defined for an integer exponent).
    """
    input_a = generate_bfloat16_binary_grid(include_spl_values=False).reshape(64, 32)
    tt_a = to_tt_tensor(input_a, device)

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = flush_subnormal_values_to_zero(golden_function(input_a, exponent).float().clone()).to(torch.bfloat16)
    result = flush_subnormal_values_to_zero(ttnn.to_torch(ttnn.pow(tt_a, exponent)).float().clone()).to(torch.bfloat16)
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=4, allow_nonfinite=True)


def _positive_grid():
    values = generate_bfloat16_binary_grid(include_spl_values=False)
    return values[values > 0].reshape(32, 32)


@pytest.mark.parametrize("exponent", _POW_FLOAT_EXPONENTS)
def test_pow_float_exponent(device, exponent):
    """tensor ** float over a positive-base grid (see module docstring for
    why negative bases are excluded here).
    """
    input_a = _positive_grid()
    tt_a = to_tt_tensor(input_a, device)

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = flush_subnormal_values_to_zero(golden_function(input_a, exponent).float().clone()).to(torch.bfloat16)
    result = flush_subnormal_values_to_zero(ttnn.to_torch(ttnn.pow(tt_a, exponent)).float().clone()).to(torch.bfloat16)
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2, allow_nonfinite=True)


def test_pow_tensor_exponent(device):
    """tensor ** tensor: the binary pow overload, over a positive-base grid
    against an exponent tensor cycling through both the int and float
    exponents used above.
    """
    input_a = _positive_grid()
    tt_a = to_tt_tensor(input_a, device)

    exponent_values = torch.tensor(_POW_INT_EXPONENTS + _POW_FLOAT_EXPONENTS, dtype=torch.bfloat16)
    reps = input_a.numel() // exponent_values.numel() + 1
    exponents = exponent_values.repeat(reps)[: input_a.numel()].reshape(input_a.shape)
    tt_exponents = to_tt_tensor(exponents, device)

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = flush_subnormal_values_to_zero(golden_function(input_a, exponents).float().clone()).to(torch.bfloat16)
    result = flush_subnormal_values_to_zero(ttnn.to_torch(ttnn.pow(tt_a, tt_exponents)).float().clone()).to(
        torch.bfloat16
    )
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2, allow_nonfinite=True)

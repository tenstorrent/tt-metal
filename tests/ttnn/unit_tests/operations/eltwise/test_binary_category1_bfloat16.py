# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, ulp_distance, flush_subnormal_values_to_zero
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    binary_grid_values,
    pairwise_from_values,
    pairwise_inputs,
    flush_to_zero,
    run_binary,
)

pytestmark = pytest.mark.use_module_device

"""
Category 1: basic_binary_arithmetic + corresponding inplace op

    1. ttnn.add              - Addition
    2. ttnn.sub              - Subtraction
    3. ttnn.multiply         - Multiplication
    4. ttnn.divide           - Division
    5. ttnn.add_              - Addition inplace
    6. ttnn.sub_              - Subtraction inplace
    7. ttnn.multiply_         - Multiplication inplace
    8. ttnn.divide_           - Division inplace
    9. ttnn.rsub              - Subtraction (b - a)
   10. ttnn.rsub_             - Subtraction inplace (b - a)
   11. ttnn.addalpha          - a + alpha * b
   12. ttnn.subalpha          - a - alpha * b
"""


@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.sub, ttnn.rsub, ttnn.add_, ttnn.sub_, ttnn.rsub_])
@pytest.mark.parametrize("fast_and_approximate_mode, ulp_threshold", [(True, 1), (False, 0)])
def test_addlike_ops(device, ttnn_op, fast_and_approximate_mode, ulp_threshold):
    """Pairwise coverage of binary ops over the stratified bfloat16 grid.
    Values selected from the range between 3.3895e+38 and -3.3895e+38
    EX: FPU overflow: 3.3895 + 6.6201 = 1.0009e+39 → torch → max; FPU → +inf
    """
    input_a, input_b = pairwise_inputs(include_zero=True)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, fast_and_approximate_mode=fast_and_approximate_mode)

    # Device flushes subnormal sums to zero.
    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    # FPU add (fast_and_approximate_mode=True, the default) can overflow to ±inf
    # where IEEE RNE stays at ±max bf16. SFPU add (False) rounds to max and does
    # not need this exception.
    #
    # Example (mantissa 1111111 on both operands, same sign):
    #   a =  3.3895313892515355e+38   (0x7F7F, max bf16, 1.9921875 × 2^127)
    #   b =  6.620178494631905e+35    (0x7AFF,             1.9921875 × 2^118)
    #   exact a+b is still below the midpoint between max and overflow
    #   (halfway is max + 2^119; b is 255/256 of that step), so torch RNE
    #   returns max (0x7F7F). FPU add overflows to +inf.
    #   The swapped order (b+a) and both-negative pair (0xFF7F + 0xFAFF →
    #   torch -max, device -inf) are the same case.
    #
    # Do not flush all infs to zero — that would hide this discrepancy.
    # Apply mask for positions where golden is already ±max and device is ±inf.
    if fast_and_approximate_mode:
        bf16_max = torch.finfo(torch.bfloat16).max
        allowed_overflow = (
            (golden.abs() == bf16_max) & torch.isinf(result) & (torch.signbit(golden) == torch.signbit(result))
        )
        result = torch.where(allowed_overflow, golden, result)

    # a + b can overflow to ±inf for large-magnitude pairs (e.g. 2^127 + 2^127).
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_threshold, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.multiply, ttnn.multiply_])
@pytest.mark.parametrize("fast_and_approximate_mode, ulp_threshold", [(True, 2), (False, 0)])
def test_multiply(device, ttnn_op, fast_and_approximate_mode, ulp_threshold):
    """Pairwise coverage of ttnn.multiply over the stratified bfloat16 grid.
    FPU: 2 ULP is accepted. ULP > 2 in the underflow band is masked
    (e.g. 2.342e-38 × 16.125 → 8 ULP). Overflow-to-zero: golden ±inf vs
    device +0 rewritten to match (2^{18} × 2^{127}).
    """
    input_a, input_b = pairwise_inputs(include_zero=True)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, fast_and_approximate_mode=fast_and_approximate_mode)

    # SFPU mul flush includes min-normal 2^{-126} to 0.
    if not fast_and_approximate_mode:
        result = flush_to_zero(result)
        golden = flush_to_zero(golden)
    else:
        result = flush_subnormal_values_to_zero(result)
        golden = flush_subnormal_values_to_zero(golden)

    # FPU mul (fast_and_approximate_mode=True) does not match IEEE RNE at the
    # ends of the product range. SFPU (False) does, so these masks are FPU-only.
    # 2 ULP is accepted via ulp_threshold; only ULP > 2 is special-cased.
    #
    # 1) Underflow ULP > 2: |a*b| in [2^{-126}, 2^{-121}] (≈ 1.18e-38 … 7.46e-37),
    #    golden exp −126…−121. LoFi dest rounding can exceed 2 ULP here (max 255).
    #    2 ULP pairs are left for assert_with_ulp, e.g.
    #      12 × 9.477e-38 → torch 1.140e-36, FPU 1.128e-36 (2 ULP).
    #    Example ULP > 2:
    #      2.342e-38 × 16.125 → torch 3.762e-37, FPU 3.644e-37 (8 ULP).
    #
    # 2) Overflow-to-zero: when ea+eb ≳ 145, torch saturates to ±inf but FPU
    #    dest packing returns +0 (sign is dropped).
    #    Example: 2^{18} × 2^{127} = 2^{145} (0x4880 × 0x7F00) → torch +inf, FPU +0.
    #    Treat golden ±inf vs device +0 as allowed. Do not flush all infs.
    if fast_and_approximate_mode:
        golden_bf16 = golden.to(torch.bfloat16)
        result_bf16 = result.to(torch.bfloat16)
        both_finite = torch.isfinite(golden_bf16) & torch.isfinite(result_bf16)
        above_2_ulp = both_finite & (golden_bf16.abs() < (2.0**-120)) & (ulp_distance(golden_bf16, result_bf16) > 2)
        result = torch.where(above_2_ulp, golden, result)

        overflow_to_zero = torch.isinf(golden) & (result == 0)
        result = torch.where(overflow_to_zero, golden, result)
    # a * b overflows to ±inf when |a| * |b| exceeds max bf16.
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_threshold, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.divide, ttnn.divide_])
@pytest.mark.parametrize("fast_and_approximate_mode, ulp_threshold", [(True, 2), (False, 0)])
def test_divide(device, ttnn_op, fast_and_approximate_mode, ulp_threshold):
    """Pairwise coverage of ttnn.divide over the stratified bfloat16 grid.
    FPU: 2 ULP is accepted. ULP > 2 in the underflow band is masked
    (e.g. 2.342e-38 / 0.06226 → 8 ULP). Device +0 vs golden finite/inf
    (recip flush / overflow-to-zero) is rewritten, as is FPU ±max vs
    torch ±inf when ea−eb = 128 (7.96875 / 2.342e-38).
    """
    input_a, input_b = pairwise_inputs(include_zero=False)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, fast_and_approximate_mode=fast_and_approximate_mode)

    # SFPU div flush includes min-normal 2^{-126} to 0 (same as SFPU mul).
    if not fast_and_approximate_mode:
        result = flush_to_zero(result)
        golden = flush_to_zero(golden)
    else:
        result = flush_subnormal_values_to_zero(result)
        golden = flush_subnormal_values_to_zero(golden)

    # Reciprocal flush: device computes a * recip(b). When recip(b) underflows
    # to 0, the quotient is +0 even if torch is a finite value (up to ~4) or
    # ±inf. SFPU only hits this for |b| ≳ 2^126 ie. |b| ≥ 8.507e37.
    # Example finite: max / 1.992×2^126 ≈ 3.984 → torch 3.984, device +0.
    # a = 3.39e38 / b = 1.69e38 ; torch = 2.0 and device = 0.
    # Gate on |b| so that an all-zero regression on normal divisors is caught .
    recip_flush = (result == 0) & (golden != 0) & (input_b.abs() >= 2.0**126)
    result = torch.where(recip_flush, golden, result)

    # FPU-only dest behavior. SFPU matches IEEE RNE at 0 ULP after the recip
    # flush above.
    if fast_and_approximate_mode:
        # 1) Underflow ULP > 2: |a/b| in [2^{-126}, 2^{-121}] (≈ 1.18e-38 … 7.44e-37).
        #    LoFi dest rounding can exceed 2 ULP here (max 8 ULP on this grid).
        #    2 ULP pairs are left for assert_with_ulp.
        #    Example ULP > 2: 2.342e-38 / 0.06226 → torch 3.762e-37, FPU 3.644e-37 (8 ULP).
        golden_bf16 = golden.to(torch.bfloat16)
        result_bf16 = result.to(torch.bfloat16)
        both_finite = torch.isfinite(golden_bf16) & torch.isfinite(result_bf16)
        above_2_ulp = both_finite & (golden_bf16.abs() < (2.0**-120)) & (ulp_distance(golden_bf16, result_bf16) > 2)
        result = torch.where(above_2_ulp, golden, result)

        # 2) FPU overflow-to-zero: when ea−eb ≳ 145, FPU dest packing returns
        #    +0 while torch overflows to ±inf (sign is dropped).
        #    Example: 2^19 / 2^{-126} = 2^145 → torch +inf, FPU +0.
        overflow_to_zero = torch.isinf(golden) & (result == 0)
        result = torch.where(overflow_to_zero, golden, result)

        # 3) Overflow fence the other way from add: ea−eb = 128 overflows IEEE
        #    to ±inf, but FPU saturates at ±max bf16 (same sign).
        #    Example: 7.96875 / 2.342e-38 → torch +inf, FPU +max (0x7F7F).
        bf16_max = torch.finfo(torch.bfloat16).max
        allowed_sat = (
            torch.isinf(golden) & (result.abs() == bf16_max) & (torch.signbit(golden) == torch.signbit(result))
        )
        result = torch.where(allowed_sat, golden, result)

    # a / b overflows to ±inf when |a| / |b| exceeds max bf16.
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_threshold, allow_nonfinite=True)


# |alpha| == 2 scales b by an exponent shift. 2 * 2^126 = 2^127, still finite;
# the next binade (b ≥ 2^127) makes that scale overflow, and the FPU sum then
# disagrees with torch by hundreds of ULP (max + 2*(-max) → torch -max, device -2^120).
_ADDALPHA_LARGE_SCALE_BOUND = 2.0**126


@pytest.mark.parametrize("ttnn_op", [ttnn.addalpha, ttnn.subalpha])
@pytest.mark.parametrize("alpha", [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
def test_addalpha_subalpha(device, ttnn_op, alpha):
    """Pairwise coverage of ttnn.addalpha / ttnn.subalpha: a ± alpha * b.

    The rhs is scaled by an SFPU mul, then added or subtracted on the FPU.
    Neither op exposes fast_and_approximate_mode; the add/sub is always the
    FPU kernel, so the threshold is the FPU add threshold (1 ULP).

    alpha 0 and ±1 use the full grid. |alpha| == 0.5 does too, except where
    the scale of b underflows. |alpha| == 2 keeps both operands in
    [-2^126, 2^126] so the scale stays finite.
    """
    if abs(alpha) > 1:
        values = binary_grid_values(
            low=-_ADDALPHA_LARGE_SCALE_BOUND, high=_ADDALPHA_LARGE_SCALE_BOUND, include_zero=True
        )
        input_a, input_b = pairwise_from_values(values)
    else:
        input_a, input_b = pairwise_inputs(include_zero=True)
    golden, result = run_binary(device, ttnn_op, input_a, input_b, golden_kwargs={"alpha": alpha})

    # Device flushes subnormal sums to zero, same as add/sub.
    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    # SFPU scale underflow: 0 < |alpha| < 1 can flush alpha*b to 0 while b is
    # still a normal, so the device returns a. Torch multiplies first and the
    # sum can be a subnormal that flushes to 0.
    #   a = -1.249e-38, b = 1.175e-38, alpha = 0.5
    #   torch a + 0.5*b ≈ -6.6e-39 → 0; device 0.5*b → 0, then a + 0 = a.
    # alpha == 0 is excluded: |b| * 0 is below the bound for every b, so the
    # mask would replace the whole grid with the golden and the identity
    # (a ± 0) would never reach the ULP check.
    if 0.0 < abs(alpha) < 1:
        scale_underflow = (input_b != 0) & (input_b.abs().to(torch.float32) * abs(alpha) < 2.0**-126)
        assert scale_underflow.any(), "expected the SFPU scale of b to underflow on this grid"
        result = torch.where(scale_underflow, golden, result)

    # Same FPU overflow-to-inf exception as test_addlike_ops. Present for
    # |alpha| <= 1 on the full grid (alpha = 0 never overflows). The |alpha| == 2
    # sweep is bounded below that fence.
    if alpha != 0.0 and abs(alpha) <= 1:
        bf16_max = torch.finfo(torch.bfloat16).max
        allowed_overflow = (
            (golden.abs() == bf16_max) & torch.isinf(result) & (torch.signbit(golden) == torch.signbit(result))
        )
        assert allowed_overflow.any(), "expected the FPU overflow-to-inf pairs in this sweep"
        result = torch.where(allowed_overflow, golden, result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)

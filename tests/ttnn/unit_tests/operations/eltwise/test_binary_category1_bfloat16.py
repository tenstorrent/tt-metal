# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, ulp_distance, flush_subnormal_values_to_zero
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_binary_grid,
    flush_to_zero,
    to_tt_tensor,
)

pytestmark = pytest.mark.use_module_device

"""
Category 1: basic_binary_arithmetic

 1. ttnn.add              - Addition
 2. ttnn.sub              - Subtraction
 3. ttnn.mul              - Multiplication
 4. ttnn.div              - Division
"""


def _pairwise_inputs(include_spl_values=False):
    """Outer product of the 2048-value binary grid: A[i, j] = v[i], B[i, j] = v[j]."""
    values = generate_bfloat16_binary_grid(include_spl_values=include_spl_values)
    a, b = torch.meshgrid(values, values, indexing="ij")
    return a.contiguous(), b.contiguous()


@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.sub, ttnn.rsub])
@pytest.mark.parametrize("fast_and_approximate_mode, ulp_threshold", [(True, 1), (False, 0)])
def test_addlike_ops(device, ttnn_op, fast_and_approximate_mode, ulp_threshold):
    """Pairwise coverage of binary ops over the stratified bfloat16 grid.
    Values selected from the range between 3.3895e+38 and -3.3895e+38
    EX: FPU overflow: 3.3895 + 6.6201 = 1.0009e+39 → torch → max; FPU → +inf
    """
    input_a, input_b = _pairwise_inputs()

    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, input_b, device=device, fast_and_approximate_mode=fast_and_approximate_mode)

    tt_result = ttnn_op(tt_a, tt_b, fast_and_approximate_mode=fast_and_approximate_mode)
    result = ttnn.to_torch(tt_result)

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


@pytest.mark.parametrize("fast_and_approximate_mode, ulp_threshold", [(True, 2), (False, 0)])
def test_mul(device, fast_and_approximate_mode, ulp_threshold):
    """Pairwise coverage of ttnn.mul over the stratified bfloat16 grid.
    FPU: 2 ULP is accepted. ULP > 2 in the underflow band is masked
    (e.g. 2.342e-38 × 16.125 → 8 ULP). Overflow-to-zero: golden ±inf vs
    device +0 rewritten to match (2^{18} × 2^{127}).
    """
    input_a, input_b = _pairwise_inputs()
    ttnn_op = ttnn.mul

    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, input_b, device=device, fast_and_approximate_mode=fast_and_approximate_mode)

    tt_result = ttnn_op(tt_a, tt_b, fast_and_approximate_mode=fast_and_approximate_mode)
    result = ttnn.to_torch(tt_result)

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


@pytest.mark.parametrize("fast_and_approximate_mode, ulp_threshold", [(True, 2), (False, 0)])
def test_div(device, fast_and_approximate_mode, ulp_threshold):
    """Pairwise coverage of ttnn.div over the stratified bfloat16 grid.
    FPU: 2 ULP is accepted. ULP > 2 in the underflow band is masked
    (e.g. 2.342e-38 / 0.06226 → 8 ULP). Device +0 vs golden finite/inf
    (recip flush / overflow-to-zero) is rewritten, as is FPU ±max vs
    torch ±inf when ea−eb = 128 (7.96875 / 2.342e-38).
    """
    input_a, input_b = _pairwise_inputs()
    # Replace ±0 to avoid dividing by zero.
    input_b = torch.where(input_b == 0, torch.ones_like(input_b), input_b)
    ttnn_op = ttnn.divide

    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, input_b, device=device, fast_and_approximate_mode=fast_and_approximate_mode)

    tt_result = ttnn_op(tt_a, tt_b, fast_and_approximate_mode=fast_and_approximate_mode)
    result = ttnn.to_torch(tt_result)

    # Device flushes subnormal quotients to zero.
    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    # Reciprocal flush: device computes a * recip(b). When recip(b) underflows
    # to 0, the quotient is +0 even if torch is a finite value (up to ~4) or
    # ±inf. SFPU only hits this for |b| ≳ 2^126; FPU also overflows to +0 when
    # ea−eb ≳ 145 (e.g. 2^19 / 2^{-126} = 2^145 → torch +inf, FPU +0).
    # Example finite: max / 1.992×2^126 ≈ 3.984 → torch 3.984, device +0.
    zero_mismatch = (result == 0) & (golden != 0)
    result = torch.where(zero_mismatch, golden, result)

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

        # 2) Overflow fence the other way from add: ea−eb = 128 overflows IEEE
        #    to ±inf, but FPU saturates at ±max bf16 (same sign).
        #    Example: 7.96875 / 2.342e-38 → torch +inf, FPU +max (0x7F7F).
        bf16_max = torch.finfo(torch.bfloat16).max
        allowed_sat = (
            torch.isinf(golden) & (result.abs() == bf16_max) & (torch.signbit(golden) == torch.signbit(result))
        )
        result = torch.where(allowed_sat, golden, result)

    # a / b overflows to ±inf when |a| / |b| exceeds max bf16.
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=ulp_threshold, allow_nonfinite=True)

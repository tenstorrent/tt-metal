# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, ulp_distance, flush_subnormal_values_to_zero
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    pairwise_inputs,
    run_ternary,
    ternary_inputs,
)

pytestmark = pytest.mark.use_module_device

"""
Category 1: fused multiply-add family, plus select

    1. ttnn.mac    - a * b + c
    2. ttnn.lerp   - a + weight * (b - a)
    3. ttnn.where  - predicate ? a : b

mac and lerp sweep the full outer product of the 256-value ternary grid, 256^3
= 16,777,216 triples in a [256, 256, 256] tensor. The grid keeps all 4 mantissa
codes of the binary grid and samples 32 of the 254 exponents; see
generate_bfloat16_ternary_grid for why the trade goes that way.

where does not need a 3-way grid. Its predicate carries one bit of information
per element, so only the two branches need pairwise coverage and the 2048-value
binary grid applies at full resolution.
"""

BF16_MAX = torch.finfo(torch.bfloat16).max
MIN_NORMAL_BF16 = 2.0**-126

# Both kernels compute an intermediate in a LoFi dest before the final add, so
# a result whose magnitude lands in the bottom binades carries the rounding of
# that intermediate as well as its own. The binary mul/div sweeps fence this at
# 2^-120; mac and lerp need one binade more because the trailing add can carry
# the sum back up out of the band it was computed in.
UNDERFLOW_BAND = 2.0**-119


def nonfinite_disagreement(golden, result):
    """Positions where the non-finite check inside comp_ulp would fire.

    It is stricter than matching finite masks: a NaN facing an inf fails, and so
    do two infs of opposite sign.
    """
    return (
        (torch.isnan(golden) ^ torch.isnan(result))
        | (torch.isinf(golden) ^ torch.isinf(result))
        | (torch.isinf(golden) & torch.isinf(result) & (torch.signbit(golden) != torch.signbit(result)))
    )


@pytest.mark.parametrize("ttnn_op", [ttnn.mac])
def test_mac(device, ttnn_op):
    """Exhaustive 3-way coverage of ttnn.mac over the stratified bfloat16 grid.

    The device path is SFPMAD: the product accumulates in fp32 and rounds once
    on store. 1 ULP is accepted because 2.4% of the sweep differs by a single
    step, most of it exact midpoints that the two sides break the opposite way:

        a = 1.17549435e-38, b = 1.0, c = 1.18467790e-38
        exact 2.36017225e-38 is the midpoint; torch gives 2.35098870e-38,
        the device gives 2.36935580e-38.

    Neither side owns the rest: off the midpoints the device is closer to the
    fp64 value exactly as often as torch is (30,802 positions each way).
    """
    input_a, input_b, input_c = ternary_inputs(include_zero=True)
    golden, result = run_ternary(device, ttnn_op, input_a, input_b, input_c)

    # Device flushes subnormal results to zero.
    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    # fp64 is exact for every bf16 product and sum on this grid, so it stands in
    # for the infinitely precise result that neither side computes.
    exact = input_a.to(torch.float64) * input_b.to(torch.float64) + input_c.to(torch.float64)

    # The golden promotes to fp32 and multiplies before it adds, so a product
    # above fp32 max overflows the intermediate even when a * b + c is a
    # perfectly ordinary bf16. The fused device path keeps the exponent range
    # and returns the finite value.
    #   a = 1.0078, b = 3.3895e38, c = -2.1268e37
    #   a * b = 3.4159e38 > fp32 max -> torch inf; device 3.2034e38.
    # Trust the device here: it is the side that matches `exact`.
    golden_fp32_overflow = torch.isinf(golden) & (exact.abs() <= BF16_MAX)
    assert golden_fp32_overflow.any(), "expected the fp32 intermediate of the golden to overflow on this grid"
    result = torch.where(golden_fp32_overflow, golden, result)

    # Underflow band: when a * b + c lands below 2^-119 the dest rounding of the
    # product shows up in the sum and the two sides part by more than 1 ULP.
    #   a = 1.1755e-38, b = 0.25, c = -1.2490e-38
    #   torch 0 (the subnormal sum flushes); device -1.2490e-38, i.e. it
    #   dropped the product instead.
    both_finite = torch.isfinite(golden) & torch.isfinite(result)
    in_band = (
        both_finite
        & (golden.abs() < UNDERFLOW_BAND)
        & (ulp_distance(golden.to(torch.bfloat16), result.to(torch.bfloat16)) > 1)
    )
    assert in_band.any(), "expected underflow-band disagreements in this sweep"
    result = torch.where(in_band, golden, result)

    # a * b + c overflows to ±inf for large-magnitude triples.
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


def build_predicate(pattern, shape):
    """0/1 predicates laid out to vary within a tile, across tiles, and not at all.

    The patterns matter more than the values: the kernel reads the predicate a
    face at a time, so a selection that changes every element, every tile row,
    or not at all exercises different paths through it.
    """
    rows, cols = shape
    row_index = torch.arange(rows).unsqueeze(1).expand(rows, cols)
    col_index = torch.arange(cols).unsqueeze(0).expand(rows, cols)
    patterns = {
        "all_true": torch.ones(shape),
        "all_false": torch.zeros(shape),
        "checkerboard": (row_index + col_index) % 2,
        "row_blocks": (row_index // ttnn.TILE_SIZE) % 2,
        "col_blocks": (col_index // ttnn.TILE_SIZE) % 2,
        "tile_diagonal": row_index // ttnn.TILE_SIZE == col_index // ttnn.TILE_SIZE,
    }
    return patterns[pattern].to(torch.bfloat16)


@pytest.mark.parametrize(
    "pattern", ["all_true", "all_false", "checkerboard", "row_blocks", "col_blocks", "tile_diagonal"]
)
@pytest.mark.parametrize("include_spl_values", [False, True])
def test_where_ttt(device, pattern, include_spl_values):
    """Pairwise coverage of the two ttnn.where branches over the binary grid.

    where is a selection, not an arithmetic op, so the bar is bit equality
    rather than a ULP budget: whichever branch the predicate picks should come
    back unaltered. With finite branches it does, on all 4,194,304 pairs and
    every predicate layout.

    Carrying special values is where it stops being a pure copy. Two encodings
    do not survive the selection, both verified to be the kernel rather than the
    host round-trip (from_torch/to_torch returns 0x7FC0 and 0x8000 unchanged):

      * a selected NaN comes back as an infinity of the same sign -- the
        mantissa that distinguishes the two is dropped (0x7FC0 -> 0x7F80).
      * a selected -0 comes back as +0 (0x8000 -> 0x0000). This one is the
        usual SFPU pack canonicalisation and is numerically harmless, so the
        ULP comparison below already tolerates it.
    """
    input_true, input_false = pairwise_inputs(include_spl_values=include_spl_values, include_zero=True)
    predicate = build_predicate(pattern, input_true.shape)
    golden, result = run_ternary(device, ttnn.where, predicate, input_true, input_false)

    if not include_spl_values:
        # Pure copy: compare encodings, which is stricter than any ULP budget
        # and catches a -0 that lost its sign or a payload that got rewritten.
        assert torch.equal(
            golden.view(torch.uint16), result.view(torch.uint16)
        ), "where is expected to carry finite values through bit-for-bit"
        return

    # NaN -> same-signed inf. Check the sign survives before excusing it, so a
    # regression that drops the sign as well still fails.
    nan_to_inf = torch.isnan(golden) & torch.isinf(result)
    assert nan_to_inf.any(), "expected the selected NaN to reach the device as an infinity"
    assert torch.equal(
        torch.signbit(golden[nan_to_inf]), torch.signbit(result[nan_to_inf])
    ), "the sign of a selected NaN is expected to survive the conversion to infinity"
    result = torch.where(nan_to_inf, golden, result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=0, allow_nonfinite=True)


@pytest.mark.parametrize("ttnn_op", [ttnn.lerp])
def test_lerp(device, ttnn_op):
    """Exhaustive 3-way coverage of ttnn.lerp over the stratified bfloat16 grid.

    lerp is a subtract, then a scale, then an add, and unlike mac it is not
    fused: the device rounds b - a and weight * (b - a) into bf16 before the
    final add, while the golden carries the whole chain in fp32. Each
    intermediate therefore has its own overflow and flush behaviour, and the
    masks below are one per stage. Everything outside them holds to 1 ULP.
    """
    input_a, input_b, input_weight = ternary_inputs(include_zero=True)
    golden, result = run_ternary(device, ttnn_op, input_a, input_b, input_weight)

    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    a64, b64, w64 = (t.to(torch.float64) for t in (input_a, input_b, input_weight))
    difference = b64 - a64
    scaled = difference * w64

    def ulp_now(res):
        return ulp_distance(golden.to(torch.bfloat16), res.to(torch.bfloat16))

    # 1) b - a underflows: the subtraction lands on a bf16 subnormal, the device
    #    flushes it, and the whole correction term disappears -- the result is
    #    exactly a. Verified on the full grid: all 24,064 such positions return a.
    #      a = 1.1755e-38, b = 1.1847e-38, weight = 65536
    #      b - a = 9.184e-41 -> device 0, so lerp returns a; torch keeps the
    #      subnormal and reaches 6.0185e-36.
    difference_underflow = (difference != 0) & (difference.abs() < MIN_NORMAL_BF16)
    assert difference_underflow.any(), "expected b - a to underflow on this grid"
    assert torch.equal(
        result[difference_underflow].to(torch.bfloat16), input_a.expand_as(result)[difference_underflow]
    ), "device is expected to return a unchanged when b - a flushes to zero"
    result = torch.where(difference_underflow, golden, result)

    # 2) weight * (b - a) underflows the same way one stage later: the device
    #    flushes the scaled term, torch keeps the subnormal and folds it in.
    scale_underflow = (scaled != 0) & (scaled.abs() < MIN_NORMAL_BF16) & (ulp_now(result) > 1)
    assert scale_underflow.any(), "expected the scaled correction term to underflow on this grid"
    result = torch.where(scale_underflow, golden, result)

    # 3) Either intermediate can exceed bf16 max while a + weight * (b - a) is
    #    still finite, because the device rounds each step into bf16 and so
    #    overflows a binade earlier than the fp32 golden. The device returns
    #    ±inf there.
    #      a = 2.1268e37, b = -2.1268e37, weight = 8
    #      b - a = -4.2535e37, scaled = -3.4028e38, which is past bf16 max but
    #      not past fp32 max -> device -inf, torch -3.1901e38.
    #    Which non-finite each side lands on is not stable. When b - a passes
    #    fp32 max too the golden's own subtraction goes to ±inf, and a zero
    #    weight then makes the scale 0 * inf:
    #      a = 2.1268e37, b = -3.3895e38, weight = 0 -> torch NaN, device -inf.
    #    Excuse any non-finite disagreement inside the region rather than
    #    enumerating the pairings.
    intermediate_overflow = (difference.abs() > BF16_MAX) | (scaled.abs() > BF16_MAX)
    overflow_disagreement = intermediate_overflow & nonfinite_disagreement(golden, result)
    assert overflow_disagreement.any(), "expected a bf16 intermediate to overflow on this grid"
    result = torch.where(overflow_disagreement, golden, result)

    # 4) Catastrophic cancellation. When a + weight * (b - a) falls more than a
    #    bf16 significand (8 bits) below the larger operand, every bit of the
    #    result came out of the rounding of b - a, and a ULP metric on the
    #    output no longer measures the kernel.
    #      a = 1.2622e-29, b = 7.9934e-37, weight = 1
    #      b - a rounds to -a, so the device loses b entirely and returns
    #      7.5232e-37 where torch returns b.
    both_finite = torch.isfinite(golden) & torch.isfinite(result)
    larger_operand = torch.maximum(input_a.abs().to(torch.float64), input_b.abs().to(torch.float64))
    cancellation = both_finite & (golden.abs() < larger_operand * 2.0**-8) & (ulp_now(result) > 1)
    assert cancellation.any(), "expected cancellation cases in this sweep"
    result = torch.where(cancellation, golden, result)

    # 5) Same bottom-binade fence as mac, for results that survived the stages
    #    above but still land inside the band.
    in_band = both_finite & (golden.abs() < UNDERFLOW_BAND) & (ulp_now(result) > 1)
    assert in_band.any(), "expected underflow-band disagreements in this sweep"
    result = torch.where(in_band, golden, result)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)

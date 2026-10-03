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

    1. ttnn.mac      - a * b + c
    2. ttnn.lerp     - a + weight * (b - a)
    3. ttnn.where    - predicate ? a : b
    4. ttnn.addcmul  - a + value * b * c
    5. ttnn.addcdiv  - a + value * b / c

Every op but where sweeps the full outer product of the 256-value ternary grid,
256^3 = 16,777,216 triples in a [256, 256, 256] tensor. The grid keeps all 4
mantissa codes of the binary grid and samples 32 of the 254 exponents; see
generate_bfloat16_ternary_grid for why the trade goes that way.

where does not need a 3-way grid. Its predicate carries one bit of information
per element, so only the two branches need pairwise coverage and the 2048-value
binary grid applies at full resolution.
"""

BF16_MAX = torch.finfo(torch.bfloat16).max
MIN_NORMAL_BF16 = 2.0**-126

# These kernels compute an intermediate in a LoFi dest before the final add, so
# a result whose magnitude lands in the bottom binades carries the rounding of
# that intermediate as well as its own. The binary mul/div sweeps fence this at
# 2^-120; the ternary ops need one binade more because the trailing add can
# carry the sum back up out of the band it was computed in.
UNDERFLOW_BAND = 2.0**-119

# A result this far below the operands that produced it is made entirely of the
# rounding of the intermediate, so a ULP metric on the output stops measuring
# the kernel. 2^-8 is the bf16 significand.
CANCELLATION_RATIO = 2.0**-8


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

    Ranges excused before the assertion, counted on this grid (16,777,216
    triples). `out` is a * b + c; being fused, mac needs only two of the five
    exceptions lerp does:

      #  Stage                              Range that fires it      Positions
      1  golden fp32 intermediate overflow  |a * b| > fp32 max           5,200  0.031%
      2  bottom-binade rounding             |out| < 2^-119               3,804  0.023%
                                                                total    9,004  0.054%

    Stage 1 is the loosest of the two: only 2,080 of its 5,200 positions
    actually disagree, the rest being triples where both sides already return
    the same infinity and the rewrite is a no-op. Stage 2 is gated on a real
    disagreement, so all 3,804 of its positions are ones the assertion would
    otherwise reject.
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

    Ranges excused before the assertion, counted on this grid (16,777,216
    triples). `out` is a + weight * (b - a); bf16 max is 3.3895e38:

      #  Stage                        Range that fires it               Positions
      1  b - a underflows             |b - a| < 2^-126                 24,064  0.143%
      2  weight * (b - a) underflows  |weight * (b - a)| < 2^-126       2,342  0.014%
      3  an intermediate overflows    either intermediate > bf16 max   13,604  0.081%
      4  catastrophic cancellation    |out| < max(|a|, |b|) * 2^-8     27,018  0.161%
      5  bottom-binade rounding       |out| < 2^-119                       53 <0.001%
                                                                total  67,081  0.400%

    Only stage 1 excuses its whole range, and it still asserts the stronger
    property that the device returns a there. The others fire only where the
    two sides really disagree -- stages 2, 4 and 5 by more than 1 ULP, stage 3
    on a non-finite -- so each removes far less than its range would suggest.
    The remaining 99.6% is checked at 1 ULP, and 1,068 positions use it.
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


def apply_exceptions(golden, result, stages, *, require_nonempty):
    """Rewrite ``result`` to the golden wherever a documented exception holds.

    ``stages`` are ``(reason, build_mask)`` pairs applied in order; each
    ``build_mask`` receives the result as rewritten so far, so a stage can gate
    itself on whether the two sides still disagree.
    """
    for reason, build_mask in stages:
        mask = build_mask(result)
        if require_nonempty:
            assert mask.any(), reason
        result = torch.where(mask, golden, result)
    return result


def ulp_against(golden):
    """ULP distance from the golden, measured on the result as rewritten so far."""
    return lambda result: ulp_distance(golden.to(torch.bfloat16), result.to(torch.bfloat16))


def intermediate_overflow_exception(golden, scaled_b, term):
    """The device rounds each intermediate into bf16 and so overflows a binade
    earlier than the fp32 golden, after which the two land on different
    non-finites: value = 2, b = 1.7014e38, c = 0 gives the device +inf, since
    2 * b overflowed before it ever met c, and torch NaN, since its own
    overflow met the zero c as inf * 0.
    """
    region = (scaled_b.abs() > BF16_MAX) | (term.abs() > BF16_MAX)
    return ("expected an intermediate to overflow bf16", lambda result: region & nonfinite_disagreement(golden, result))


def intermediate_underflow_exception(golden, scaled_b, term):
    """Mirror image of the overflow: an intermediate that lands on a bf16
    subnormal is flushed by the device and kept by the golden, so the device
    drops the whole correction and returns a.

        value = 0.5, a = b = 1.1755e-38, c = 65536
        0.5 * b = 5.877e-39 -> device 0, so the result is a; torch keeps the
        subnormal and reaches 3.8519e-34.
    """
    region = ((scaled_b != 0) & (scaled_b.abs() < MIN_NORMAL_BF16)) | ((term != 0) & (term.abs() < MIN_NORMAL_BF16))
    ulp = ulp_against(golden)
    return ("expected an intermediate to underflow", lambda result: region & (ulp(result) > 1))


def cancellation_exception(golden, input_a, term):
    """a + term lands far below both, so every bit of the output came from the
    rounding of term and a ULP metric on it stops measuring the kernel."""
    larger_operand = torch.maximum(input_a.abs().to(torch.float64), term.abs())
    ulp = ulp_against(golden)
    return (
        "expected cancellation cases in this sweep",
        lambda result: torch.isfinite(golden)
        & torch.isfinite(result)
        & (golden.abs() < larger_operand * CANCELLATION_RATIO)
        & (ulp(result) > 1),
    )


def underflow_band_exception(golden):
    """The same bottom-binade fence mac and lerp use."""
    ulp = ulp_against(golden)
    return (
        "expected underflow-band disagreements in this sweep",
        lambda result: torch.isfinite(golden)
        & torch.isfinite(result)
        & (golden.abs() < UNDERFLOW_BAND)
        & (ulp(result) > 1),
    )


@pytest.mark.parametrize("value", [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0])
def test_addcmul(device, value):
    """Exhaustive 3-way coverage of ttnn.addcmul: a + value * b * c.

    The evaluation order is observable, and it is not the order the signature
    suggests: both sides form value * b first and only then multiply by c. The
    grid pins this down at value = 0.5, a = b = 1.1755e-38, c = 65536. Scaling
    first makes 0.5 * b subnormal, which the device flushes, so it returns a;
    had the device formed b * c first the product would have been a normal
    7.7037e-34 and no flush could happen.

    Ranges excused before the assertion, counted on this grid (16,777,216
    triples). `term` is value * b * c and `out` is a + term. The two signs of a
    value give identical counts, and value = 0 excuses nothing at all:

      #  Stage                       Range that fires it                  |v|=1   |v|=0.5     |v|=2
      1  an intermediate overflows   |value*b| or |term| > bf16 max       3,120     2,080     5,688
      2  an intermediate underflows  |value*b| or |term| < 2^-126         3,616   104,960        72
      3  bottom-binade rounding      |out| < 2^-119                         188       268       160
                                                                total     6,924   107,308     5,920
                                                                         0.041%    0.640%    0.035%

    Everything else holds to 1 ULP, the same SFPMAD tie-breaking budget as mac.

    Unlike addcdiv there is no cancellation exception, and that is measured
    rather than assumed: across all six non-zero values tested, not one triple
    has a + value * b * c collapsing far enough below its operands to disagree
    by more than 1 ULP. The multiply keeps enough of the correction term for
    the trailing add to stay accurate; the divide does not.
    """
    input_a, input_b, input_c = ternary_inputs(include_zero=True)
    golden, result = run_ternary(device, ttnn.addcmul, input_a, input_b, input_c, golden_kwargs={"value": value})

    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    scaled_b = input_b.to(torch.float64) * value
    term = scaled_b * input_c.to(torch.float64)

    # value = 0 makes the correction identically zero, so none of the
    # exceptions can fire and addcmul has to reproduce a exactly.
    if value == 0.0:
        assert torch.equal(golden, result), "a + 0 * b * c is expected to return a unchanged"
        return

    stages = [
        intermediate_overflow_exception(golden, scaled_b, term),
        intermediate_underflow_exception(golden, scaled_b, term),
        underflow_band_exception(golden),
    ]
    result = apply_exceptions(golden, result, stages, require_nonempty=True)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


# recip(b) underflows to zero for divisors this large, so the device returns a
# quotient of +0 where torch still has a finite value. Same fence as the binary
# divide sweep.
RECIPROCAL_FLUSH_DIVISOR = 2.0**126


@pytest.mark.parametrize("value", [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0])
def test_addcdiv(device, value):
    """Exhaustive 3-way coverage of ttnn.addcdiv: a + value * b / c.

    Zero divisors are excluded, as in the binary divide sweep: c = 0 makes the
    quotient non-finite for every b and the sweep would be dominated by it.

    addcdiv carries two exceptions the multiply does not. The device divides
    by multiplying with recip(c), and that reciprocal underflows to zero once
    |c| >= 2^126, so the correction term vanishes and the result collapses to
    a; that one stage is most of the cost below. It also needs a cancellation
    fence, which addcmul measurably does not.

    Ranges excused before the assertion, counted on this grid (16,777,216
    triples). `term` is value * b / c and `out` is a + term. The two signs of a
    value give identical counts, and value = 0 excuses nothing at all:

      #  Stage                       Range that fires it                   |v|=1   |v|=0.5     |v|=2
      1  an intermediate overflows   |value*b| or |term| > bf16 max        1,056       528     1,320
      2  recip(c) flushes to zero    |c| >= 2^126                        171,840   169,324   156,972
      3  an intermediate underflows  |value*b| or |term| < 2^-126         11,000   108,292     4,868
      4  catastrophic cancellation   |out| < max(|a|, |term|) * 2^-8       1,968     1,872     1,824
      5  bottom-binade rounding      |out| < 2^-119                           12        12        12
                                                                 total   185,876   280,028   164,996
                                                                          1.108%    1.669%    0.983%

    Everything else holds to 1 ULP.
    """
    input_a, input_b, input_c = ternary_inputs(include_zero=False)
    golden, result = run_ternary(device, ttnn.addcdiv, input_a, input_b, input_c, golden_kwargs={"value": value})

    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    scaled_b = input_b.to(torch.float64) * value
    term = scaled_b / input_c.to(torch.float64)

    if value == 0.0:
        assert torch.equal(golden, result), "a + 0 * b / c is expected to return a unchanged"
        return

    reciprocal_flush = input_c.abs().to(torch.float64) >= RECIPROCAL_FLUSH_DIVISOR
    ulp = ulp_against(golden)
    stages = [
        intermediate_overflow_exception(golden, scaled_b, term),
        ("expected recip(c) to flush to zero on this grid", lambda result: reciprocal_flush & (ulp(result) > 1)),
        intermediate_underflow_exception(golden, scaled_b, term),
        cancellation_exception(golden, input_a, term),
        underflow_band_exception(golden),
    ]
    result = apply_exceptions(golden, result, stages, require_nonempty=True)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)

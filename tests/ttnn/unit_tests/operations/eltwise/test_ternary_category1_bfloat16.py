# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import (
    assert_allclose,
    assert_with_ulp,
    ulp_distance,
    flush_subnormal_values_to_zero,
)
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    SMALLEST_NORMAL_BF16,
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

Each test documents the ranges it excuses in a table, and the names in its
Stage column are the ones apply_exceptions puts in a failure message, so a
failing run points at the row that explains it.
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

# addcdiv divides by multiplying with recip(c), and that reciprocal underflows
# to zero for divisors this large, so the device returns a quotient of +0 where
# torch still has a finite value. Same fence as the binary divide sweep.
RECIPROCAL_FLUSH_DIVISOR = 2.0**126

# Inside the band a ULP metric is useless but an absolute one is not, so the
# band is checked with assert_allclose the way the binary sweeps check theirs
# (test_binary_category4_bfloat16.py). Each bound is the measured worst case on
# this grid rounded up to a power of two; the band itself reaches 128 smallest
# normals, so none of these is a vacuous ceiling.
MAC_BAND_ATOL = 2 * SMALLEST_NORMAL_BF16
LERP_BAND_ATOL = 2 * SMALLEST_NORMAL_BF16
ADDCMUL_BAND_ATOL = 16 * SMALLEST_NORMAL_BF16
ADDCDIV_BAND_ATOL = 32 * SMALLEST_NORMAL_BF16


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


def apply_exceptions(golden, result, stages):
    """Rewrite ``result`` to the golden wherever a documented exception holds.

    ``stages`` are ``(name, build_mask)`` pairs applied in order. Each
    ``build_mask`` receives the result as rewritten so far, so a stage can gate
    itself on whether the two sides still disagree, and asserts whatever the
    device is documented to do over the range it is about to excuse -- an exact
    value, or an absolute bound.

    Order is part of the specification rather than an accident of the list: a
    stage only holds once the ones before it have taken their ranges out. That
    is also why a failure is re-raised carrying the stage name, since otherwise
    a bound that fails on one of fourteen parametrizations says only "mismatch".

    A stage that matches nothing is a hole in the sweep rather than a saving,
    so an empty mask fails too.
    """
    for name, build_mask in stages:
        try:
            mask = build_mask(result)
            assert mask.any(), "matched nothing, so this exception has gone stale"
        except AssertionError as error:
            raise AssertionError(f"exception stage '{name}': {error}") from error
        result = torch.where(mask, golden, result)
    return result


def ulp_against(golden):
    """ULP distance from the golden, measured on the result as rewritten so far."""
    return lambda result: ulp_distance(golden.to(torch.bfloat16), result.to(torch.bfloat16))


def golden_fp32_overflow_exception(golden, exact):
    """The golden computes in fp32, so an intermediate above fp32 max takes it
    to an infinity the device never reaches. ``exact`` is the fp64 value, which
    is what decides that the device is the side to trust."""
    return ("golden fp32 intermediate overflow", lambda result: torch.isinf(golden) & (exact.abs() <= BF16_MAX))


def intermediate_overflow_exception(golden, first, second):
    """The device rounds each intermediate into bf16 and so overflows a binade
    earlier than the fp32 golden, after which the two land on different
    non-finites: value = 2, b = 1.7014e38, c = 0 gives the device +inf, since
    2 * b overflowed before it ever met c, and torch NaN, since its own
    overflow met the zero c as inf * 0. Which pairing comes up is not stable,
    so any non-finite disagreement inside the region is excused rather than
    the pairings being enumerated.
    """
    region = (first.abs() > BF16_MAX) | (second.abs() > BF16_MAX)
    return ("intermediate overflow", lambda result: region & nonfinite_disagreement(golden, result))


def collapses_to_a_exception(golden, input_a, region, name, detail, *, only_where_disagreeing=True):
    """A stage whose device behaviour is fully determined: something upstream
    flushes to zero, the whole correction term vanishes, and the result is a.

    That is asserted rather than assumed, so a regression landing on any other
    finite value here still fails.

    ``only_where_disagreeing`` narrows the range to where the two sides have
    actually parted. The flushes that happen inside a correction term need it,
    because the band stage has to run first: inside the band the output itself
    is subnormal and the device flushes the *sum*, a different behaviour with a
    different expected value. A flush of the very first intermediate does not,
    since nothing downstream can bring the correction back.
    """
    ulp = ulp_against(golden)

    def build_mask(result):
        mask = region.expand_as(result)
        if only_where_disagreeing:
            mask = mask & (ulp(result) > 1)
        assert torch.equal(result[mask].to(torch.bfloat16), input_a.expand_as(result)[mask]), detail
        return mask

    return (name, build_mask)


def difference_underflow_exception(golden, input_a, difference):
    """lerp subtracts before it scales, so a b - a that lands on a bf16
    subnormal is flushed away before the weight ever sees it and lerp returns a
    for every weight."""
    return collapses_to_a_exception(
        golden,
        input_a,
        (difference != 0) & (difference.abs() < MIN_NORMAL_BF16),
        "b - a underflows",
        "device is expected to return a unchanged when b - a flushes to zero",
        only_where_disagreeing=False,
    )


def scaled_underflow_exception(golden, input_a, scaled_b):
    """value * b landing on a bf16 subnormal is flushed by the device and kept
    by the golden, so the device drops the correction and returns a.

        value = 0.5, a = b = 1.1755e-38, c = 65536
        0.5 * b = 5.877e-39 -> device 0, so the result is a; torch keeps the
        subnormal and reaches 3.8519e-34.
    """
    return collapses_to_a_exception(
        golden,
        input_a,
        (scaled_b != 0) & (scaled_b.abs() < MIN_NORMAL_BF16),
        "value * b underflows",
        "device is expected to return a unchanged when value * b flushes to zero",
    )


def reciprocal_flush_exception(golden, input_a, input_c):
    """The device divides by multiplying with recip(c), and that reciprocal
    underflows to zero once |c| >= 2^126, so the quotient vanishes and the
    result is a. Same fence as the binary divide sweep."""
    return collapses_to_a_exception(
        golden,
        input_a,
        input_c.abs().to(torch.float64) >= RECIPROCAL_FLUSH_DIVISOR,
        "recip(c) flushes to zero",
        "device is expected to return a unchanged when recip(c) flushes to zero",
    )


def cancellation_exception(golden, *operands):
    """The output lands more than a bf16 significand below the largest of the
    ``operands`` that produced it, so every bit of it came out of the rounding
    of an intermediate. This is the one range with no bound to offer: the
    output is far from the golden in absolute terms as well as relative ones,
    by construction."""
    larger_operand = operands[0].abs().to(torch.float64)
    for operand in operands[1:]:
        larger_operand = torch.maximum(larger_operand, operand.abs().to(torch.float64))
    ulp = ulp_against(golden)
    return (
        "catastrophic cancellation",
        lambda result: torch.isfinite(golden)
        & torch.isfinite(result)
        & (golden.abs() < larger_operand * CANCELLATION_RATIO)
        & (ulp(result) > 1),
    )


def underflow_band_exception(golden, atol):
    """The bottom-binade fence. A ULP metric is useless in here but an absolute
    one is not, so the slice is bounded by ``atol`` before it is rewritten --
    otherwise any finite magnitude would pass. Run it after the cancellation
    stage: a cancelled result can land in the band too, and that population
    genuinely has no absolute bound."""
    ulp = ulp_against(golden)

    def build_mask(result):
        mask = torch.isfinite(golden) & torch.isfinite(result) & (golden.abs() < UNDERFLOW_BAND) & (ulp(result) > 1)
        assert_allclose(expected_result=golden[mask], actual_result=result[mask], rtol=0, atol=atol)
        return mask

    return ("bottom-binade rounding", build_mask)


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
    otherwise reject, and it is bounded absolutely rather than waived: the
    worst case on this grid is 1.06 smallest normals.
    """
    input_a, input_b, input_c = ternary_inputs(include_zero=True)
    golden, result = run_ternary(device, ttnn_op, input_a, input_b, input_c)

    # Device flushes subnormal results to zero.
    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    # fp64 is exact for every bf16 product and sum on this grid, so it stands in
    # for the infinitely precise result that neither side computes.
    exact = input_a.to(torch.float64) * input_b.to(torch.float64) + input_c.to(torch.float64)

    stages = [
        # The golden promotes to fp32 and multiplies before it adds, so a
        # product above fp32 max overflows the intermediate even when a * b + c
        # is a perfectly ordinary bf16. The fused device path keeps the exponent
        # range and returns the finite value.
        #   a = 1.0078, b = 3.3895e38, c = -2.1268e37
        #   a * b = 3.4159e38 > fp32 max -> torch inf; device 3.2034e38.
        # Trust the device here: it is the side that matches `exact`.
        golden_fp32_overflow_exception(golden, exact),
        # When a * b + c lands below 2^-119 the dest rounding of the product
        # shows up in the sum and the two sides part by more than 1 ULP.
        #   a = 1.1755e-38, b = 0.25, c = -1.2490e-38
        #   torch 0 (the subnormal sum flushes); device -1.2490e-38, i.e. it
        #   dropped the product instead.
        underflow_band_exception(golden, MAC_BAND_ATOL),
    ]
    result = apply_exceptions(golden, result, stages)

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
        # Pure copy: compare encodings, which is stricter than any ULP budget.
        # This grid is +0 and finite normals only -- -0 and NaN live in the
        # include_spl_values branch below -- so what this catches is a selected
        # value whose bits came back altered at all, however slightly.
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
    final add, while the golden carries the whole chain in fp32, so each
    intermediate brings its own overflow and flush behaviour. Everything
    outside the masks below holds to 1 ULP.

    Ranges excused before the assertion, counted on this grid (16,777,216
    triples). `out` is a + weight * (b - a); bf16 max is 3.3895e38:

      #  Stage                        Range that fires it               Positions
      1  b - a underflows             |b - a| < 2^-126                 24,064  0.143%
      2  intermediate overflow        either intermediate > bf16 max   13,604  0.081%
      3  catastrophic cancellation    |out| < max(|a|, |b|) * 2^-8     28,898  0.172%
      4  bottom-binade rounding       |out| < 2^-119                      515  0.003%
                                                                total  67,081  0.400%

    Only stage 1 excuses its whole range, and it still asserts the stronger
    property that the device returns a there. The others fire only where the
    two sides really disagree -- stages 3 and 4 by more than 1 ULP, stage 2 on
    a non-finite -- so each removes far less than its range would suggest, and
    stage 4 is bounded absolutely before it is waived (worst case on this grid
    is 1.00 smallest normals). Stage 3 is the one range with no bound to give:
    the output there is built entirely out of the rounding of b - a, so it is
    far from the golden in both relative and absolute terms by construction.
    The remaining 99.6% is checked at 1 ULP, and 1,068 positions use it.
    """
    input_a, input_b, input_weight = ternary_inputs(include_zero=True)
    golden, result = run_ternary(device, ttnn_op, input_a, input_b, input_weight)

    result = flush_subnormal_values_to_zero(result)
    golden = flush_subnormal_values_to_zero(golden)

    a64, b64, w64 = (t.to(torch.float64) for t in (input_a, input_b, input_weight))
    difference = b64 - a64
    scaled = difference * w64

    stages = [
        # The subtraction lands on a bf16 subnormal, the device flushes it, and
        # the whole correction term disappears -- the result is exactly a.
        #   a = 1.1755e-38, b = 1.1847e-38, weight = 65536
        #   b - a = 9.184e-41 -> device 0, so lerp returns a; torch keeps the
        #   subnormal and reaches 6.0185e-36.
        difference_underflow_exception(golden, input_a, difference),
        # Either intermediate can exceed bf16 max while a + weight * (b - a) is
        # still finite, because the device rounds each step into bf16 and so
        # overflows a binade earlier than the fp32 golden.
        #   a = 2.1268e37, b = -2.1268e37, weight = 8
        #   b - a = -4.2535e37, scaled = -3.4028e38, which is past bf16 max but
        #   not past fp32 max -> device -inf, torch -3.1901e38.
        # Which non-finite each side lands on is not stable. When b - a passes
        # fp32 max too the golden's own subtraction goes to ±inf, and a zero
        # weight then makes the scale 0 * inf:
        #   a = 2.1268e37, b = -3.3895e38, weight = 0 -> torch NaN, device -inf.
        intermediate_overflow_exception(golden, difference, scaled),
        # When a + weight * (b - a) falls more than a bf16 significand below the
        # larger operand, every bit of the result came out of the rounding of
        # b - a.
        #   a = 1.2622e-29, b = 7.9934e-37, weight = 1
        #   b - a rounds to -a, so the device loses b entirely and returns
        #   7.5232e-37 where torch returns b.
        cancellation_exception(golden, input_a, input_b),
        # Same bottom-binade fence as mac, for results that survived the stages
        # above but still land inside the band. weight * (b - a) landing on a
        # subnormal needs no stage of its own: measured on the full grid, every
        # such position that disagrees by more than 1 ULP also lands in here.
        underflow_band_exception(golden, LERP_BAND_ATOL),
    ]
    result = apply_exceptions(golden, result, stages)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


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

      #  Stage                      Range that fires it                  |v|=1   |v|=0.5     |v|=2
      1  intermediate overflow      |value*b| or |term| > bf16 max       3,120     2,080     5,688
      2  catastrophic cancellation  |out| < max(|a|, |term|) * 2^-8        384     2,248        40
      3  bottom-binade rounding     |out| < 2^-119                       3,420    18,104       192
      4  value * b underflows       |value * b| < 2^-126                     -    84,876         -
                                                              total      6,924   107,308     5,920
                                                                        0.041%    0.640%    0.035%

    Everything else holds to 1 ULP, the same SFPMAD tie-breaking budget as mac.

    Stage 4 is the only one that excuses a range outright, and it asserts in
    exchange that the device returns a across the whole of it. It exists only
    for |value| < 1: the grid carries no subnormal inputs, so scaling by 1 or 2
    cannot produce a subnormal and the stage would be empty.

    Stage 3 is bounded absolutely rather than waived; the worst case on this
    grid is 16 smallest normals, at value = 0.5 where the device flushed
    value * b and the golden kept it. Stage 2 has to run first, because a
    cancelled result can land in the band as well and that population has no
    absolute bound to give -- it reaches 1.99 at value = 0.5.
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
        cancellation_exception(golden, input_a, term),
        underflow_band_exception(golden, ADDCMUL_BAND_ATOL),
    ]
    if abs(value) < 1.0:
        stages.append(scaled_underflow_exception(golden, input_a, scaled_b))
    result = apply_exceptions(golden, result, stages)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


@pytest.mark.parametrize("value", [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0])
def test_addcdiv(device, value):
    """Exhaustive 3-way coverage of ttnn.addcdiv: a + value * b / c.

    Zero divisors are excluded, as in the binary divide sweep: c = 0 makes the
    quotient non-finite for every b and the sweep would be dominated by it.

    addcdiv carries one exception the multiply does not. The device divides by
    multiplying with recip(c), and that reciprocal underflows to zero once
    |c| >= 2^126, so the correction term vanishes and the result collapses to
    a; that one stage is most of the cost below.

    Ranges excused before the assertion, counted on this grid (16,777,216
    triples). `term` is value * b / c and `out` is a + term. The two signs of a
    value give identical counts, and value = 0 excuses nothing at all:

      #  Stage                      Range that fires it                   |v|=1   |v|=0.5     |v|=2
      1  intermediate overflow      |value*b| or |term| > bf16 max        1,056       528     1,320
      2  catastrophic cancellation  |out| < max(|a|, |term|) * 2^-8       5,792     7,232     4,852
      3  bottom-binade rounding     |out| < 2^-119                       30,660    41,036    26,556
      4  recip(c) flushes to zero   |c| >= 2^126                        148,368   147,304   132,268
      5  value * b underflows       |value * b| < 2^-126                      -    83,928         -
                                                                total   185,876   280,028   164,996
                                                                         1.108%    1.669%    0.983%

    Everything else holds to 1 ULP. Stages 4 and 5 excuse their ranges outright
    and assert in exchange that the device returns a over all of them. Stage 3
    is bounded absolutely instead (worst case on this grid is 31.75 smallest
    normals). Stage 2 is the only range with nothing to assert, and it has to
    run first: a cancelled result can land in the band or in the recip-flush
    range, and there it is neither close to the golden nor equal to a -- the
    absolute error reaches 7.6e30.
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

    # The band runs ahead of the two collapse-to-a stages for the same reason
    # cancellation runs ahead of the band: a flushed correction whose output
    # still lands in the bottom binades does not leave the result equal to a,
    # because there the device flushes the sum rather than the term. Taking the
    # band out first is what lets those two stages assert an exact value.
    stages = [
        intermediate_overflow_exception(golden, scaled_b, term),
        cancellation_exception(golden, input_a, term),
        underflow_band_exception(golden, ADDCDIV_BAND_ATOL),
        reciprocal_flush_exception(golden, input_a, input_c),
    ]
    if abs(value) < 1.0:
        stages.append(scaled_underflow_exception(golden, input_a, scaled_b))
    result = apply_exceptions(golden, result, stages)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1, allow_nonfinite=True)

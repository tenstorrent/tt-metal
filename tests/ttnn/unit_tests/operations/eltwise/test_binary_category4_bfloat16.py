# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose, assert_equal, assert_with_ulp
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    binary_grid_values,
    pairwise_from_values,
    run_binary,
    to_tt_tensor,
    MAX_BF16,
    SMALLEST_NORMAL_BF16,
)

pytestmark = pytest.mark.use_module_device

"""
Category 4: composite binary math

 1. ttnn.ldexp / ldexp_                        - a * 2**b
 2. ttnn.logaddexp / logaddexp_                - log(exp(a) + exp(b))
 3. ttnn.logaddexp2 / logaddexp2_              - log2(2**a + 2**b)
 4. ttnn.squared_difference / squared_difference_ - (a - b)**2
 5. ttnn.xlogy                                 - a * log(b)
 6. ttnn.hypot                                 - sqrt(a**2 + b**2)
 7. ttnn.bias_gelu / bias_gelu_                - gelu(a + b)
 8. ttnn.prelu                                 - a if a >= 0 else a * w

Working range of the accuracy sweep. "Full grid" is the stratified bfloat16
grid: finite values and +0, no -0, inf, or NaN. In-place ops share the range.

  op                    a                              b or weight
  ────────────────────  ─────────────────────────────  ──────────────────────────
  ldexp                 full grid                      [-126, 127]
  logaddexp             [-80, 80]                      [-80, 80]
  logaddexp2            [-120, 120]                    [-120, 120]
  squared_difference    [-1e19, 1e19]                  [-1e19, 1e19]
  xlogy                 [-1e30, 1e30]                  positive normals, (0, max]
  hypot                 ±[2^-62, 2^62], or 0           same as a
  bias_gelu             [-100, 100]                    [-100, 100]
  prelu                 full grid                      full grid, one per channel

  Outside that range the chain has already flushed or overflowed, and a
  separate test pins it: ldexp for b < -126 and b > 127, hypot on the full
  grid, xlogy for b <= 0. The others stop at the bound.

Most of these are lowered to a pre-op / binary / post-op chain rather than a
single SFPU instruction (see OpConfig::OpConfig in binary_ng_utils.cpp), so
they inherit the domain of their pre-op. ldexp cannot see a scale factor its
EXP2 pre-op cannot represent; logaddexp cannot see a sum its EXP pre-op
overflowed; hypot cannot see an operand its SQUARE pre-op flushed to zero. The
sweeps below are therefore restricted to the domain where the chain is defined,
and each op's boundary behavior is pinned by its own test instead of being
folded into the accuracy threshold. The two exceptions are xlogy, which is one
SFPU instruction, and prelu, which is a where() over a multiply; they are here
because they carry the same absolute-error profile as the chains.

Accuracy criteria
─────────────────
  ldexp              : ULP <= 3 for b in [-126, 127]; see test_ldexp
  logaddexp          : ULP <= 1 for |golden| >= 1, else abs err <= 2^-6
  logaddexp2         : ULP <= 2 for |golden| >= 1, else abs err <= 2^-5
  squared_difference : ULP <= 3; see test_squared_difference
  xlogy              : ULP <= 2 outside log's zero crossing, see test_xlogy
  hypot              : ULP <= 1 where neither square under/overflows
  bias_gelu          : bitwise accurate gelu(add(a, b)); allclose(rtol=.05, atol=.05)
  prelu              : exact, outside the product's underflow band

The grids carry +0 but no -0/inf/NaN: these chains do not propagate non-finite
values the way torch does, and inconsistently so — ldexp(inf, -1) comes back as
max bfloat16 rather than inf, while ldexp(inf, inf) comes back as 0. Pinning
that down is a separate contract from measuring accuracy. Ops are swept
together with their in-place spelling, which shares the kernel and must
therefore share the thresholds. Tensor-scalar overloads — every op here
except hypot — are covered by test_tensor_scalar at those same thresholds.
"""

# Below this magnitude a bfloat16 result has passed through the dest's LoFi
# rounding and subnormal flush, which costs far more than the ULP thresholds
# above; category 1 uses the same fence for multiply and divide. These
# elements are checked with an absolute tolerance instead. bf16 in / bf16 out
# leaves fp32 dest accumulation off, so the intermediate is rounded to
# bfloat16 in dest rather than kept in fp32.
UNDERFLOW_BAND = 2.0**-120

# The exponents EXP2 can turn into a bfloat16 scale factor: outside
# [-126, 127] the pre-op has already under/overflowed before the multiply
# sees it.
LDEXP_MIN_EXPONENT = -126.0
LDEXP_MAX_EXPONENT = 127.0

# ldexp and squared_difference both round an intermediate to bfloat16 in dest
# before the next op (fp32 dest accumulation is off for bf16). The golden
# rounds that same intermediate — 2**b, or a - b — to bfloat16 on its side.
# The third ULP is those two roundings, not an fp32 value the golden lacks.
# Against an fp32 golden both hold at 2.
COMPOSITE_INTERMEDIATE_ULP = 3

# SFPU log's error is absolute, not relative, near its zero crossing: it
# returns 4.4e-4 for log(1) and is off by up to 7.4e-3 across [0.25, 4]. xlogy
# scales that straight into the product, so the band is bounded by |a| * this
# instead of by ULP. The bound is that 7.4e-3 plus the two bfloat16 roundings
# the comparison adds — the device's output and the golden's, each up to
# 1.386/256 of |a| in this band. Measured worst case is 0.0147.
XLOGY_LOG_ATOL = 0.02
XLOGY_NEAR_ONE_LOW = 0.25
XLOGY_NEAR_ONE_HIGH = 4.0

# sqrt(a**2 + b**2) squares both operands before the add. An operand below
# 2^-63 squares to a subnormal that hardware flushes to zero, and one above
# 2^63 overflows the square. One binade of margin on each side.
HYPOT_MIN_OPERAND = 2.0**-62
HYPOT_MAX_OPERAND = 2.0**62


def _assert_nonfinite_match(golden, result, desc):
    """Non-finite results must match the golden, including the sign of inf.

    A finiteness-only check would let -inf regress to +inf and still pass the
    finite-only accuracy checks that follow. Matching NaNs compare equal.
    """
    golden_finite = torch.isfinite(golden)
    result_finite = torch.isfinite(result)
    position_mismatch = golden_finite != result_finite
    assert not position_mismatch.any(), (
        f"{desc}: {int(position_mismatch.sum())} elements disagree on finiteness "
        "outside the documented saturation cases"
    )

    nonfinite = ~golden_finite
    if not nonfinite.any():
        return
    golden_nf, result_nf = golden[nonfinite], result[nonfinite]
    value_mismatch = (torch.isnan(golden_nf) != torch.isnan(result_nf)) | (
        torch.isinf(golden_nf) & (torch.signbit(golden_nf) != torch.signbit(result_nf))
    )
    assert (
        not value_mismatch.any()
    ), f"{desc}: {int(value_mismatch.sum())} non-finite elements differ from the golden, including the sign of inf"


# ─────────────────────────────────────────────────────────────────────────────
# ldexp — a * 2**b, lowered to EXP2(b) followed by an FPU multiply
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", [ttnn.ldexp, ttnn.ldexp_])
def test_ldexp(device, ttnn_op):
    """a over the full grid, b over the exponents EXP2 can represent.

    The third ULP is a bfloat16 rounding of the scale. Dest accumulation is
    off, so EXP2's result is rounded to bfloat16 before the multiply, and
    torch rounds 2**b to bfloat16 on the golden's side. For a = 5.995e-36 and
    b = -0.9961 that rounds 2**b from 0.50272 to 0.5, so the golden halves a
    exactly (2.9975e-36) where the device returns 3.0328e-36, and the true
    3.0056e-36 sits between them. Against an fp32 golden the sweep holds at
    2 ULP.

    Both saturation directions are the FPU multiply's, already documented for
    ttnn.multiply in category 1:

    1) Overflow-to-zero: once the product's exponent runs past ~145 the FPU
       dest packs +0 while torch saturates to ±inf (1716 pairs here, e.g.
       a = 1.511e23 = 2^77 with b = 68).
    2) Overflow-to-inf: the FPU overflows where IEEE RNE still returns ±max
       bf16, because the exact product sits below the midpoint between max and
       the next binade (8 pairs, all a = ±max bf16 with b ~= 0.0039, i.e.
       a * 1.0027).

    Below UNDERFLOW_BAND the product is only accurate to an absolute tolerance:
    the worst case is a = ±2.342e-38 with b ~= -0.001, where the true result is
    still the same normal value but the device packs +0 (2.64e-38 of error).
    """
    input_a, input_b = pairwise_from_values(
        binary_grid_values(), binary_grid_values(LDEXP_MIN_EXPONENT, LDEXP_MAX_EXPONENT)
    )
    golden, result = run_binary(device, ttnn_op, input_a, input_b)

    overflow_to_zero = torch.isinf(golden) & (result == 0)
    # An empty mask rewrites nothing. The infs then match on sign and the finite
    # slice drops them, so this saturation could disappear without a failure.
    assert overflow_to_zero.any(), "expected the FPU overflow-to-zero pairs in this sweep"
    result = torch.where(overflow_to_zero, golden, result)
    overflow_to_inf = (
        (golden.abs() == MAX_BF16) & torch.isinf(result) & (torch.signbit(golden) == torch.signbit(result))
    )
    assert overflow_to_inf.any(), "expected the FPU overflow-to-inf pairs in this sweep"
    result = torch.where(overflow_to_inf, golden, result)
    _assert_nonfinite_match(golden, result, ttnn_op.__name__)

    finite = torch.isfinite(golden)
    underflow = finite & (golden.abs() < UNDERFLOW_BAND)
    assert underflow.any(), "expected the underflow band to be non-empty for this sweep"

    assert_with_ulp(
        expected_result=golden[finite & ~underflow],
        actual_result=result[finite & ~underflow],
        ulp_threshold=COMPOSITE_INTERMEDIATE_ULP,
    )
    assert_allclose(
        expected_result=golden[underflow], actual_result=result[underflow], rtol=0, atol=4 * SMALLEST_NORMAL_BF16
    )


def test_ldexp_exponent_underflow(device):
    """b < -126 is outside EXP2's range, so the pre-op delivers 0 to the
    multiply and ldexp returns exactly 0 for every a — including the 3251
    pairs whose true result is a normal value as large as 1.4 (a = ±max bf16
    with b = -127.5). This is a property of the lowering, not of the inputs,
    which is why test_ldexp stops at -126 rather than masking it."""
    input_a, input_b = pairwise_from_values(
        binary_grid_values(), binary_grid_values(high=LDEXP_MIN_EXPONENT - 0.5, include_zero=False)
    )
    result = ttnn.to_torch(ttnn.ldexp(to_tt_tensor(input_a, device), to_tt_tensor(input_b, device)))

    golden = torch.ldexp(input_a, input_b)
    assert (golden != 0).any(), "expected some representable results in the EXP2 underflow region"
    assert_equal(torch.zeros_like(result), result)


def test_ldexp_exponent_overflow(device):
    """b > 127, the half of EXP2's range test_ldexp does not cover.

    EXP2's bf16 path clamps the biased exponent at 255, so every b >= 128
    delivers the same scale as b == 128. The product is ldexp(a, 128), not
    ldexp(a, b): 2^-126 * 2^128 comes back as 4, while a standalone exp2(128)
    is already +inf. 127.5 is the one grid value above 127 whose scale is
    still the real 2^b.

    Overflow-to-zero is the same FPU multiply case test_ldexp masks (golden
    ±inf, device +0). Everything else, including the sign of the infs that
    remain, matches that saturated golden exactly.
    """
    input_a, input_b = pairwise_from_values(
        binary_grid_values(), binary_grid_values(low=LDEXP_MAX_EXPONENT + 0.5, include_zero=False)
    )
    result = ttnn.to_torch(ttnn.ldexp(to_tt_tensor(input_a, device), to_tt_tensor(input_b, device)))

    assert (input_b >= 128).any() and (input_b < 128).any(), "expected both the saturated and the 127.5 scales"
    saturated_b = torch.where(input_b >= 128, torch.full_like(input_b, 128.0), input_b)
    golden = torch.ldexp(input_a.double(), saturated_b.double()).to(torch.bfloat16)

    overflow_to_zero = torch.isinf(golden) & (result == 0)
    assert overflow_to_zero.any(), "expected the FPU overflow-to-zero pairs in this sweep"
    result = torch.where(overflow_to_zero, golden, result)
    _assert_nonfinite_match(golden, result, "ldexp exponent overflow")

    finite = torch.isfinite(golden)
    assert finite.any(), "expected some products to stay finite under the saturated scale"
    assert_with_ulp(expected_result=golden[finite], actual_result=result[finite], ulp_threshold=0)


# ─────────────────────────────────────────────────────────────────────────────
# logaddexp / logaddexp2 — EXP/EXP2 on both operands, add, then LOG/LOG2
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ttnn_op, low, high, ulp_threshold, small_atol",
    [
        (ttnn.logaddexp, -80.0, 80.0, 1, 2.0**-6),
        (ttnn.logaddexp_, -80.0, 80.0, 1, 2.0**-6),
        (ttnn.logaddexp2, -120.0, 120.0, 2, 2.0**-5),
        (ttnn.logaddexp2_, -120.0, 120.0, 2, 2.0**-5),
    ],
    ids=["logaddexp", "logaddexp_", "logaddexp2", "logaddexp2_"],
)
def test_logaddexp_ops(device, ttnn_op, low, high, ulp_threshold, small_atol):
    """Sweeps stop short of the exponential's overflow (exp at ~88.7, exp2 at
    128) so that the sum the LOG post-op receives is a real number.

    The error is absolute rather than relative, which is why the contract has
    two bands. The post-op's input carries the relative error of a bfloat16
    add (~2^-8), and d(log s) = ds/s turns that into a fixed ~2^-8 of absolute
    error in the output no matter how large the output is. For |golden| >= 1
    that is under 1 ULP; for |golden| < 1 it is the whole answer, and the two
    operands cancel outright in the worst cases — logaddexp(-2.125, -0.133) is
    -4.9e-3 and the device returns 0. Measured worst absolute error is
    8.79e-3 for logaddexp and 1.31e-2 for logaddexp2.
    """
    values = binary_grid_values(low, high)
    input_a, input_b = pairwise_from_values(values)
    golden, result = run_binary(device, ttnn_op, input_a, input_b)

    _assert_nonfinite_match(golden, result, ttnn_op.__name__)
    large = golden.abs() >= 1.0
    assert large.any() and (~large).any(), "expected both the ULP and the cancellation band to be non-empty"

    assert_with_ulp(expected_result=golden[large], actual_result=result[large], ulp_threshold=ulp_threshold)
    assert_allclose(expected_result=golden[~large], actual_result=result[~large], rtol=0, atol=small_atol)


# ─────────────────────────────────────────────────────────────────────────────
# squared_difference — subtract, then a SQUARE post-op
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", [ttnn.squared_difference, ttnn.squared_difference_])
def test_squared_difference(device, ttnn_op):
    """Operands are capped at 1e19, which keeps (a - b)**2 inside bfloat16 for
    all but the couple of dozen widest-spread pairs; those saturate to +inf on
    both sides and drop out with the finite mask.

    The third ULP is the bfloat16 rounding of a - b. Dest accumulation is off,
    so the subtract rounds to bfloat16 before the square, and the golden rounds
    a - b to bfloat16 on its side. a = 3.375e-21, b = -9.216e-19 is the
    clearest case — the exact square is 8.555e-37, the device's 8.640e-37 is
    the square of a - b rounded up one bfloat16 step, and the golden's
    twice-rounded 8.464e-37 sits on the other side. Against an fp32 golden
    the whole sweep is within 2 ULP.
    """
    values = binary_grid_values(-1e19, 1e19)
    input_a, input_b = pairwise_from_values(values)
    golden, result = run_binary(device, ttnn_op, input_a, input_b)

    _assert_nonfinite_match(golden, result, ttnn_op.__name__)
    finite = torch.isfinite(golden)
    underflow = finite & (golden.abs() < UNDERFLOW_BAND)
    assert underflow.any(), "expected the underflow band to be non-empty for this sweep"

    assert_with_ulp(
        expected_result=golden[finite & ~underflow],
        actual_result=result[finite & ~underflow],
        ulp_threshold=COMPOSITE_INTERMEDIATE_ULP,
    )
    assert_allclose(
        expected_result=golden[underflow], actual_result=result[underflow], rtol=0, atol=2 * SMALLEST_NORMAL_BF16
    )


# ─────────────────────────────────────────────────────────────────────────────
# xlogy — a * log(b), a single SFPU instruction but with log's error profile
# ─────────────────────────────────────────────────────────────────────────────


def test_xlogy(device):
    """b over the positive normals, a capped at 1e30 so a * log(b) (|log b| <=
    87.3) stays inside bfloat16. b <= 0 is test_xlogy_nonpositive.

    Away from b = 1 the sweep is within 2 ULP. Around it, log's error stops
    being relative: the kernel returns 4.4e-4 for log(1) and is off by up to
    7.4e-3 across [0.25, 4], so xlogy(6.73e29, 1.0) is 2.96e26 where the true
    answer is 0. That error enters the product scaled by |a| and nothing else,
    which is what the band's bound measures. The band is wider than log's
    actual trouble spot (|log b| < 1, i.e. b in [1/e, e]) so that it does not
    sit right at the edge of the region it describes.

    The SMALLEST_NORMAL_BF16 term covers the other end of the same band, where
    the product itself underflows: a = 1.18e-38 with b = 0.375 has a true
    result of 1.16e-38 and the device packs 0.
    """
    input_a, input_b = pairwise_from_values(
        binary_grid_values(-1e30, 1e30), binary_grid_values(SMALLEST_NORMAL_BF16, MAX_BF16)
    )
    golden, result = run_binary(device, ttnn.xlogy, input_a, input_b)

    _assert_nonfinite_match(golden, result, "xlogy")
    near_one = (input_b >= XLOGY_NEAR_ONE_LOW) & (input_b <= XLOGY_NEAR_ONE_HIGH)
    assert near_one.any(), "expected log's zero crossing to be covered by this sweep"

    assert_with_ulp(expected_result=golden[~near_one], actual_result=result[~near_one], ulp_threshold=2)

    error = (golden[near_one].float() - result[near_one].float()).abs()
    tolerance = XLOGY_LOG_ATOL * input_a[near_one].abs().float() + SMALLEST_NORMAL_BF16
    failed = error > tolerance
    if failed.any():
        worst = int(torch.argmax(error - tolerance))
        raise AssertionError(
            f"xlogy: {int(failed.sum())} elements near log's zero crossing exceed "
            f"|a| * {XLOGY_LOG_ATOL}; worst is a={input_a[near_one][worst]}, b={input_b[near_one][worst]}, "
            f"golden={golden[near_one][worst]}, actual={result[near_one][worst]}"
        )


def _assert_xlogy_class(result, mask, *, allow_nan, allow_pos_inf, allow_neg_inf, desc):
    """Fail with a class count. assert_equal reports Max ATOL nan for any inf, because inf - inf is nan."""
    values = result[mask]
    ok = torch.zeros(values.shape, dtype=torch.bool)
    if allow_nan:
        ok |= torch.isnan(values)
    if allow_pos_inf:
        ok |= torch.isinf(values) & ~torch.signbit(values)
    if allow_neg_inf:
        ok |= torch.isinf(values) & torch.signbit(values)
    if bool(ok.all()):
        return
    bad = ~ok
    sample = values[bad].reshape(-1)[0]
    raise AssertionError(
        f"{desc}: {int(bad.sum())} of {values.numel()} elements are outside "
        f"nan={allow_nan}, +inf={allow_pos_inf}, -inf={allow_neg_inf}; "
        f"sample={sample.item()!r} "
        f"(nan={int(torch.isnan(values).sum())}, "
        f"+inf={int((torch.isinf(values) & ~torch.signbit(values)).sum())}, "
        f"-inf={int((torch.isinf(values) & torch.signbit(values)).sum())}, "
        f"finite={int(torch.isfinite(values).sum())})"
    )


def test_xlogy_nonpositive(device):
    """b <= 0, which the accuracy sweep drops.

    b < 0 takes the kernel's NaN path. Torch returns NaN (and 0 when a == 0).
    A bfloat16 destination reads that NaN back as +inf on Wormhole and as NaN
    on Blackhole, so both are accepted and a finite value or a -inf is not.

    b == 0 matches torch's signed inf: +a gives -inf and -a gives +inf, because
    log(0) is -inf. a == 0 is the exception. Torch returns 0, and 0 * log(0)
    is the same NaN. Wormhole reads it back as -inf and Blackhole as +inf, so
    any non-finite value is accepted and a finite one is not.
    """
    input_a, input_b = pairwise_from_values(binary_grid_values(), binary_grid_values(high=0.0))
    result = ttnn.to_torch(ttnn.xlogy(to_tt_tensor(input_a, device), to_tt_tensor(input_b, device)))

    negative_b = input_b < 0
    zero_b = input_b == 0
    assert negative_b.any() and zero_b.any(), "expected both negative b and zero"
    assert (zero_b & (input_a == 0)).any() and (zero_b & (input_a > 0)).any() and (zero_b & (input_a < 0)).any()

    _assert_xlogy_class(result, negative_b, allow_nan=True, allow_pos_inf=True, allow_neg_inf=False, desc="xlogy b < 0")
    _assert_xlogy_class(
        result,
        zero_b & (input_a > 0),
        allow_nan=False,
        allow_pos_inf=False,
        allow_neg_inf=True,
        desc="xlogy b == 0, a > 0",
    )
    _assert_xlogy_class(
        result,
        zero_b & (input_a < 0),
        allow_nan=False,
        allow_pos_inf=True,
        allow_neg_inf=False,
        desc="xlogy b == 0, a < 0",
    )
    _assert_xlogy_class(
        result,
        zero_b & (input_a == 0),
        allow_nan=True,
        allow_pos_inf=True,
        allow_neg_inf=True,
        desc="xlogy b == 0, a == 0",
    )


# ─────────────────────────────────────────────────────────────────────────────
# hypot — SQUARE on both operands, add, then a SQRT post-op
# ─────────────────────────────────────────────────────────────────────────────


def test_hypot(device):
    """Operands restricted to the magnitudes whose square stays a normal value,
    [2^-62, 2^62] plus 0. Both squares, the add and the sqrt are exact enough
    there that the sweep is within 1 ULP."""
    values = binary_grid_values(-HYPOT_MAX_OPERAND, HYPOT_MAX_OPERAND, min_magnitude=HYPOT_MIN_OPERAND)
    input_a, input_b = pairwise_from_values(values)
    golden, result = run_binary(device, ttnn.hypot, input_a, input_b)

    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_hypot_outside_square_range(device):
    """Over the full grid hypot is not sqrt(a**2 + b**2) but sqrt of whatever
    the SQUARE pre-ops left after flushing, and this pins that to 1 ULP.

    An operand below 2^-63 squares to a subnormal that is flushed, so it drops
    out of the sum entirely: hypot(2^-126, b) returns |b| exactly, and
    hypot(2^-126, 2^-126) is 0 rather than 1.66e-38. An operand above 2^63
    overflows the square, so the sum is +inf and so is the result, well before
    |hypot| itself would overflow bfloat16.
    """
    input_a, input_b = pairwise_from_values(binary_grid_values())
    result = ttnn.to_torch(ttnn.hypot(to_tt_tensor(input_a, device), to_tt_tensor(input_b, device)))

    squares = [x.float() * x.float() for x in (input_a, input_b)]
    squares = [torch.where(s < SMALLEST_NORMAL_BF16, torch.zeros_like(s), s) for s in squares]
    expected = torch.sqrt(squares[0] + squares[1]).to(torch.bfloat16)

    assert (expected == 0).any() and torch.isinf(expected).any(), "expected both boundaries in this sweep"
    _assert_nonfinite_match(expected, result, "hypot")
    assert_with_ulp(expected_result=expected, actual_result=result, ulp_threshold=1, allow_nonfinite=True)


# ─────────────────────────────────────────────────────────────────────────────
# bias_gelu — add, then a GELU post-op
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ttnn_op", [ttnn.bias_gelu, ttnn.bias_gelu_])
def test_bias_gelu(device, ttnn_op):
    """a, b over [-100, 100]. fast_and_approximate_mode defaults to False, so
    the post-op is the accurate gelu, the same default as ttnn.gelu. The
    comparison is torch.nn.functional.gelu of the sum at rtol = atol = 0.05.
    """
    values = binary_grid_values(-100.0, 100.0)
    input_a, input_b = pairwise_from_values(values)
    golden, result = run_binary(device, ttnn_op, input_a, input_b)

    _assert_nonfinite_match(golden, result, ttnn_op.__name__)
    assert_allclose(expected_result=golden, actual_result=result, rtol=0.05, atol=0.05)


def test_bias_gelu_matches_gelu_of_sum(device):
    """bias_gelu must be bitwise identical to accurate gelu over an FPU add,
    the chain its default compiles to. fast_and_approximate_mode defaults to
    False, matching ttnn.gelu. For bfloat16 the add stays on the FPU either way, so ttnn.add's
    own default is the matching sum, so bias_gelu is the same as gelu(add(a, b)).
    """
    values = binary_grid_values(-100.0, 100.0)
    input_a, input_b = pairwise_from_values(values)

    tt_a, tt_b = to_tt_tensor(input_a, device), to_tt_tensor(input_b, device)
    result = ttnn.to_torch(ttnn.bias_gelu(tt_a, tt_b))
    expected = ttnn.to_torch(ttnn.gelu(ttnn.add(tt_a, tt_b), fast_and_approximate_mode=False))

    assert_equal(expected, result)


# ─────────────────────────────────────────────────────────────────────────────
# prelu — where(a < 0, a * w, a), with one weight per channel
# ─────────────────────────────────────────────────────────────────────────────


def test_prelu(device):
    """prelu takes one weight per channel rather than a second full tensor, so
    the pairing is built into the shape: a is the grid's outer product and the
    weight is one grid value per channel, which makes column j of the result
    prelu(grid[i], grid[j]) and still covers every (value, weight) pair.

    prelu selects between a and a * w. Positive a, including the small ones,
    takes the identity and is exact. The absolute tolerance is only the
    negative branch, whose product underflows (a = -1.175e-38 with w = 0.996
    gives 0 instead of -1.175e-38). Products that overflow bfloat16 come back
    as signed inf, matching the golden, so they stay in the non-finite check
    rather than an overflow-to-zero mask.
    """
    weight = binary_grid_values()
    input_a, _ = pairwise_from_values(weight)

    golden = ttnn.get_golden_function(ttnn.prelu)(input_a, weight)
    result = ttnn.to_torch(ttnn.prelu(to_tt_tensor(input_a, device), to_tt_tensor(weight, device)))

    _assert_nonfinite_match(golden, result, "prelu")
    underflow = (input_a < 0) & (golden.abs() < UNDERFLOW_BAND)
    assert underflow.any(), "expected the product's underflow band to be non-empty for this sweep"
    assert (
        (input_a > 0) & (golden.abs() < UNDERFLOW_BAND)
    ).any(), "expected small positive inputs, which take the identity branch"

    assert_with_ulp(
        expected_result=golden[~underflow], actual_result=result[~underflow], ulp_threshold=0, allow_nonfinite=True
    )
    assert_allclose(
        expected_result=golden[underflow], actual_result=result[underflow], rtol=0, atol=SMALLEST_NORMAL_BF16
    )


# ─────────────────────────────────────────────────────────────────────────────
# tensor-scalar — the other overload of the same kernels
# ─────────────────────────────────────────────────────────────────────────────


def _square_inputs(values):
    """Repeat the value set across rows and columns so the scalar path crosses tiles.

    A 1-D tensor in tile layout sits in a single tile band. Category 3's
    scalar test uses the same expansion.
    """
    return values.unsqueeze(1).expand(values.numel(), values.numel()).contiguous()


def _run_scalar(device, ttnn_op, values, scalar):
    input_a = _square_inputs(values)
    tt_a = to_tt_tensor(input_a, device)
    scalar_tensor = torch.tensor(scalar, dtype=input_a.dtype)
    golden = ttnn.get_golden_function(ttnn_op)(input_a, scalar_tensor).to(torch.bfloat16)
    if ttnn_op.__name__.endswith("_"):
        ttnn_op(tt_a, scalar)
        result = ttnn.to_torch(tt_a)
    else:
        result = ttnn.to_torch(ttnn_op(tt_a, scalar))
    return input_a, golden, result


def _check_ldexp_scalar(golden, result, desc, scalar):
    # scalar == 127 is the only value in this sweep whose product overflows the
    # FPU multiply (golden ±inf, device +0). scalar == -126 and scalar == 1 do
    # not, and none of the three hit overflow-to-inf (golden at ±max bf16,
    # device ±inf). Those empty masks are not rewritten: a rewrite would hide
    # a saturation this sweep does not have.
    if scalar == 127.0:
        overflow_to_zero = torch.isinf(golden) & (result == 0)
        assert overflow_to_zero.any(), f"{desc}: expected the FPU overflow-to-zero pairs in this sweep"
        result = torch.where(overflow_to_zero, golden, result)
    _assert_nonfinite_match(golden, result, desc)
    finite = torch.isfinite(golden)
    underflow = finite & (golden.abs() < UNDERFLOW_BAND)
    kept = finite & ~underflow
    assert kept.any(), f"{desc}: expected some results outside the underflow band"
    assert underflow.any(), f"{desc}: expected the underflow band to be non-empty"
    assert_with_ulp(expected_result=golden[kept], actual_result=result[kept], ulp_threshold=COMPOSITE_INTERMEDIATE_ULP)
    assert_allclose(
        expected_result=golden[underflow], actual_result=result[underflow], rtol=0, atol=4 * SMALLEST_NORMAL_BF16
    )


def _check_logaddexp_scalar(golden, result, desc, ulp_threshold, small_atol):
    _assert_nonfinite_match(golden, result, desc)
    large = golden.abs() >= 1.0
    if large.any():
        assert_with_ulp(expected_result=golden[large], actual_result=result[large], ulp_threshold=ulp_threshold)
    if (~large).any():
        assert_allclose(expected_result=golden[~large], actual_result=result[~large], rtol=0, atol=small_atol)


def _check_squared_difference_scalar(golden, result, desc):
    _assert_nonfinite_match(golden, result, desc)
    finite = torch.isfinite(golden)
    underflow = finite & (golden.abs() < UNDERFLOW_BAND)
    kept = finite & ~underflow
    assert kept.any(), f"{desc}: expected some results outside the underflow band"
    assert_with_ulp(expected_result=golden[kept], actual_result=result[kept], ulp_threshold=COMPOSITE_INTERMEDIATE_ULP)
    if underflow.any():
        assert_allclose(
            expected_result=golden[underflow], actual_result=result[underflow], rtol=0, atol=2 * SMALLEST_NORMAL_BF16
        )


def _check_xlogy_scalar(input_a, golden, result, scalar):
    _assert_nonfinite_match(golden, result, "xlogy")
    near_one = XLOGY_NEAR_ONE_LOW <= scalar <= XLOGY_NEAR_ONE_HIGH
    if near_one:
        error = (golden.float() - result.float()).abs()
        tolerance = XLOGY_LOG_ATOL * input_a.abs().float() + SMALLEST_NORMAL_BF16
        assert not (error > tolerance).any(), "xlogy scalar near log's zero crossing"
    else:
        assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=2)


def _check_prelu_scalar(input_a, golden, result):
    _assert_nonfinite_match(golden, result, "prelu")
    underflow = (input_a < 0) & (golden.abs() < UNDERFLOW_BAND)
    assert_with_ulp(
        expected_result=golden[~underflow], actual_result=result[~underflow], ulp_threshold=0, allow_nonfinite=True
    )
    if underflow.any():
        assert_allclose(
            expected_result=golden[underflow], actual_result=result[underflow], rtol=0, atol=SMALLEST_NORMAL_BF16
        )


@pytest.mark.parametrize(
    "ttnn_op, kind, scalar",
    [
        (ttnn.ldexp, "ldexp", -126.0),
        (ttnn.ldexp, "ldexp", 127.0),
        (ttnn.ldexp_, "ldexp", 1.0),
        (ttnn.logaddexp, "logaddexp", -80.0),
        (ttnn.logaddexp, "logaddexp", 0.0),
        (ttnn.logaddexp_, "logaddexp", 80.0),
        (ttnn.logaddexp2, "logaddexp2", -120.0),
        (ttnn.logaddexp2, "logaddexp2", 0.0),
        (ttnn.logaddexp2_, "logaddexp2", 120.0),
        (ttnn.squared_difference, "squared_difference", -1.0e19),
        (ttnn.squared_difference, "squared_difference", 1.0),
        (ttnn.squared_difference_, "squared_difference", 0.0),
        (ttnn.bias_gelu, "bias_gelu", -100.0),
        (ttnn.bias_gelu, "bias_gelu", 0.0),
        (ttnn.bias_gelu_, "bias_gelu", 100.0),
        (ttnn.xlogy, "xlogy", 1.0),
        (ttnn.xlogy, "xlogy", 16.0),
        (ttnn.prelu, "prelu", 0.5),
        (ttnn.prelu, "prelu", -1.0),
        (ttnn.prelu, "prelu", 2.0),
    ],
)
def test_tensor_scalar(device, ttnn_op, kind, scalar):
    """Tensor-scalar coverage of the category-4 overloads.

    The value set is the same domain as the tensor-tensor sweep, expanded to a
    square so the broadcast crosses tiles. Thresholds match that sweep. hypot
    has no scalar overload.
    """
    domains = {
        "ldexp": binary_grid_values(),
        "logaddexp": binary_grid_values(-80.0, 80.0),
        "logaddexp2": binary_grid_values(-120.0, 120.0),
        "squared_difference": binary_grid_values(-1.0e19, 1.0e19),
        "bias_gelu": binary_grid_values(-100.0, 100.0),
        "xlogy": binary_grid_values(-1.0e30, 1.0e30),
        "prelu": binary_grid_values(),
    }
    input_a, golden, result = _run_scalar(device, ttnn_op, domains[kind], scalar)
    desc = f"{ttnn_op.__name__}(tensor, {scalar})"
    if kind == "ldexp":
        _check_ldexp_scalar(golden, result, desc, scalar)
    elif kind == "logaddexp":
        _check_logaddexp_scalar(golden, result, desc, ulp_threshold=1, small_atol=2.0**-6)
    elif kind == "logaddexp2":
        _check_logaddexp_scalar(golden, result, desc, ulp_threshold=2, small_atol=2.0**-5)
    elif kind == "squared_difference":
        _check_squared_difference_scalar(golden, result, desc)
    elif kind == "bias_gelu":
        _assert_nonfinite_match(golden, result, desc)
        assert_allclose(expected_result=golden, actual_result=result, rtol=0.05, atol=0.05)
    elif kind == "xlogy":
        _check_xlogy_scalar(input_a, golden, result, scalar)
    else:
        _check_prelu_scalar(input_a, golden, result)

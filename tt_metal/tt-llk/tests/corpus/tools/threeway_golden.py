#!/usr/bin/env python3
"""Host-side true-math golden and relative-ULP leg for the 2^32 streamer.

This turns the laneMK/laneMQ 2-way (sem-vs-hand *equivalence*) galaxy sweep into a
3-WAY check that also asks: is each certified leg *correct* vs the true-math oracle,
not merely equal-to-the-expert? The golden is computed HOST-SIDE (CPU torch) for the
same raw uint32 inputs a chunk streamed to the device, so it rides along for free on
the same device pass with NO 16GB retention: per chunk we fold a per-leg running
per-input-class max ULP, an out-of-tolerance counter, and the FIRST
out-of-tolerance witness.  Admission compares the candidate to the hand leg on
the same oracle and class population; it does not claim an absolute ULP budget.
This relative numeric gate is not a proof that a compiler transformation
preserves C++/LLK semantics; ordinary semantic equivalence is a separate gate.

Design (faithful reuse, NOT a reinvention):
  * The op->true-math map and every dispatch constant are lifted verbatim from the
    authoritative in-repo golden ``helpers.golden_generators.UnarySFPUGolden`` (the
    exact oracle the harness itself grades sem AND hand against). The selftest
    cross-checks this vectorized golden BYTE-FOR-BYTE against that scalar golden.
  * The output-format pipeline (bf16 input truncation for the Float32/dest_acc=No
    dst, bf16 output rounding, NaN->+inf, FTZ below 2^-126) mirrors
    ``UnarySFPUGolden.__call__`` for the exact config the laneMK streamer nodes use
    (InputOutputFormat(Float32, Float32), DestAccumulation.No -> 16-bit bf16 dst).
  * The ULP metric is the bf16 bit-distance from ``extract_accuracy.compute_ulp_bitdistance``
    (tt-polynomial-fitter); vendored here (galaxy consume venv has no ttpoly on path)
    and asserted identical to the fitter's own function in the selftest.
  * The accuracy contract is the harness's own ``passed_test`` tolerance: the Float32
    default (atol=0.05, rtol=0.05) or the op's ``CUSTOM_TOLERANCES`` override.

Honesty: an out-of-tolerance witness at an out-of-DOMAIN input (erfinv |x|>=1, or a
non-finite input) is NOT a bug -- the witness record carries the input's classification
so the report can say "licensed vs bug" precisely. Ops with no honest torch reference
are marked checkable=False with a reason rather than faked.

Input classes describe the bf16 value the SFPU receives *after* truncating the
raw fp32 stream input. They are disjoint and exhaustive. The tolerance/ULP
metric intentionally treats +0 and -0 as equal; that policy is not a bit-exact
certificate and the input partition still keeps the signed zeros separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ulp_admission import (
    BINARY_CLASS_NAMES,
    IEEE_BF16_CLASSES,
    UNARY_DOMAIN_PARTITION_OPS,
    format_class_ulp,
    unary_class_names,
)

try:
    import torch
except Exception:  # pragma: no cover - torch is always present in the harness env
    torch = None


# ─────────────────────────────────────────────────────────────────────────────
# ULP bit-distance (bf16) — vendored verbatim from tt-polynomial-fitter
# extract_accuracy.compute_ulp_bitdistance (the 'bf16' branch) + _ordered_float_bits.
# "how many representable bf16 values apart are the reference and the device output."
# The selftest asserts this equals the fitter's own function element-for-element.
# ─────────────────────────────────────────────────────────────────────────────
def _ordered_bf16_bits(bits16: np.ndarray) -> np.ndarray:
    """Map bf16 bit patterns (as int32 in [0,0xFFFF]) to a monotone signed ordering."""
    b = np.asarray(bits16, dtype=np.uint64)
    sign = np.uint64(0x8000)
    vmask = np.uint64(0xFFFF)
    mag = sign - np.uint64(1)
    b = np.where((b & mag) == 0, np.uint64(0), b)  # +0 == -0
    neg = (b & sign) != 0
    ordered = np.where(neg, (~b) & vmask, b | sign)
    return ordered.astype(np.int64)


def _to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even a value (any precision) into a 16-bit bf16 code."""
    x32 = np.asarray(x, dtype=np.float64).astype(np.float32)
    bits = x32.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + bias) & np.uint32(0xFFFF0000)
    return (rounded >> np.uint32(16)).astype(np.int32)


def bf16_bitdistance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Integer bf16 ULP distance |ord(RN_bf16(true)) - ord(RN_bf16(pred))| per element."""
    ref = _ordered_bf16_bits(_to_bf16_bits(y_true))
    approx = _ordered_bf16_bits(_to_bf16_bits(y_pred))
    return np.abs(ref - approx).astype(np.float64)


def numeric_comparison(
    golden: np.ndarray,
    device: np.ndarray,
    atol: float,
    rtol: float,
    nan_source: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return policy-defined bf16 ULP and within-tolerance masks.

    Relative/absolute tolerance applies only to two finite values. Matching
    NaNs (payload/sign agnostic) and same-sign infinities are accepted and
    assigned ULP 0 because their payload/lattice distance is not a meaningful
    numeric error. Every other pair involving a non-finite value is rejected
    and assigned the maximum bf16 sentinel distance, 65535.

    CONVERTED-NaN INFINITY SIGN POLICY. `nan_source` marks the elements whose golden
    infinity is a PACKER CONVERSION OF A NaN rather than a computed overflow -- i.e.
    the pre-conversion high-precision value was NaN. There, two infinities of
    OPPOSITE sign are also accepted.

    This module already declares a NaN's payload and sign to be not a numeric
    quantity (see both_nan above, and the signed-zero policy). convert_nan_to_inf
    preserves that sign bit, so what the device reports is whichever sign bit the
    instruction sequence happened to leave in Dest -- and that is not determined by
    any host math library either. numpy's fp64 (-inf)*0, torch's fp32 (-inf)*0 and
    the SFPU all produce a NaN and none of them agrees with the others about its
    sign: silu(-inf) is -inf to numpy, +inf to torch, and +inf on silicon, over all
    65536 patterns of a row that is otherwise CLEAN. Grading that bit would fabricate
    a defect for ~35 ops and certify nothing, so it is declared here, once, rather
    than excused per op.

    What this does NOT relax, deliberately:
      * a COMPUTED OVERFLOW. An overflow produces an infinity directly, so
        `nan_source` is false there and expm1cw returning -inf where +inf is correct
        at x = 3.3e38 is still a defect.
      * a golden that is a determinate FINITE value. gelu(-inf) is 0 by the limit
        (see _gelu_exact), not NaN, so the policy never reaches it.
      * NaN NON-PROPAGATION. A device that returns an ordinary FINITE value where
        the golden says NaN is still out of tolerance -- that is the interesting half
        of the NaN question, and CorrectnessAccumulator counts it separately.
    """
    g = np.asarray(golden, dtype=np.float32).astype(np.float64)
    d = np.asarray(device, dtype=np.float32).astype(np.float64)
    both_finite = np.isfinite(g) & np.isfinite(d)
    both_nan = np.isnan(g) & np.isnan(d)
    both_inf = np.isinf(g) & np.isinf(d) & (np.signbit(g) == np.signbit(d))
    if nan_source is not None:
        both_inf = both_inf | (
            np.isinf(g) & np.isinf(d) & np.asarray(nan_source, dtype=bool)
        )
    with np.errstate(all="ignore"):
        close = both_finite & (np.abs(d - g) <= (atol + rtol * np.abs(g)))
    matching_special = both_nan | both_inf
    within = close | matching_special
    raw_ulp = bf16_bitdistance(g, d)
    special_pair = ~both_finite
    ulp = np.where(
        matching_special,
        0.0,
        np.where(special_pair, 65535.0, raw_ulp),
    )
    return ulp.astype(np.float64), within


def _fold_class_ulps(
    aggregate: dict[str, tuple[int, float]],
    ulp: np.ndarray,
    classes: dict[str, np.ndarray],
) -> None:
    """Fold per-input-class population and max ULP without retaining samples."""
    for name, mask in classes.items():
        count = int(np.count_nonzero(mask))
        if not count:
            continue
        values = ulp[mask]
        finite = values[np.isfinite(values)]
        maximum = float(np.max(finite)) if finite.size else 0.0
        old_count, old_maximum = aggregate.get(name, (0, 0.0))
        aggregate[name] = (old_count + count, max(old_maximum, maximum))


def _ieee_bf16_masks(values: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return disjoint IEEE bf16-special masks and the remaining normal values."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32) >> np.uint32(16)
    sign = (bits & np.uint32(0x8000)) != 0
    exponent = bits & np.uint32(0x7F80)
    fraction = bits & np.uint32(0x007F)
    exp_all_ones = exponent == np.uint32(0x7F80)
    exp_zero = exponent == 0
    zero = exp_zero & (fraction == 0)
    subnormal = exp_zero & (fraction != 0)
    masks = {
        "nan": exp_all_ones & (fraction != 0),
        "pos_inf": exp_all_ones & (fraction == 0) & ~sign,
        "neg_inf": exp_all_ones & (fraction == 0) & sign,
        "pos_zero": zero & ~sign,
        "neg_zero": zero & sign,
        "pos_subnormal": subnormal & ~sign,
        "neg_subnormal": subnormal & sign,
    }
    if set(masks) != IEEE_BF16_CLASSES:
        raise AssertionError("IEEE bf16 class vocabulary drift")
    special = np.zeros(bits.shape, dtype=bool)
    for mask in masks.values():
        special |= mask
    return masks, ~special


def unary_input_classes(
    values: np.ndarray, domain: Optional[tuple]
) -> dict[str, np.ndarray]:
    """Partition post-truncation unary inputs into exhaustive semantic classes."""
    x = np.asarray(values, dtype=np.float32)
    ieee, normal = _ieee_bf16_masks(x)
    classes = {f"{name}_input": mask for name, mask in ieee.items()}
    if domain is None:
        classes["in_domain_finite_normal"] = normal
    else:
        lo, hi = domain
        lower_boundary = normal & (x == lo)
        upper_boundary = normal & (x == hi)
        boundary = lower_boundary | upper_boundary
        in_domain = normal & (x > lo) & (x < hi)
        classes["domain_lower_boundary"] = lower_boundary
        classes["domain_upper_boundary"] = upper_boundary
        classes["in_domain_finite_normal"] = in_domain
        classes["out_of_domain_finite_normal"] = normal & ~boundary & ~in_domain
    membership = sum(
        (mask.astype(np.uint8) for mask in classes.values()),
        np.zeros(x.shape, dtype=np.uint8),
    )
    if np.any(membership != 1):
        raise AssertionError("unary IEEE/domain classes are not disjoint and exhaustive")
    if set(classes) != unary_class_names(domain is not None):
        raise AssertionError("unary IEEE/domain class vocabulary drift")
    return classes


def binary_input_classes(
    base: np.ndarray, exponent: np.ndarray
) -> dict[str, np.ndarray]:
    """Partition pow pairs without a Cartesian explosion.

    Base IEEE specials take precedence. For a normal base, exponent specials
    are split next; pairs with two normals are finally split by base sign.
    """
    b = np.asarray(base, dtype=np.float32)
    e = np.asarray(exponent, dtype=np.float32)
    base_ieee, base_normal = _ieee_bf16_masks(b)
    exp_ieee, exp_normal = _ieee_bf16_masks(e)
    classes = {f"base_{name}": mask for name, mask in base_ieee.items()}
    classes.update(
        {
            f"normal_base_exp_{name}": base_normal & mask
            for name, mask in exp_ieee.items()
        }
    )
    classes["pos_normal_base_normal_exp"] = (
        base_normal & exp_normal & ~np.signbit(b)
    )
    classes["neg_normal_base_normal_exp"] = (
        base_normal & exp_normal & np.signbit(b)
    )
    membership = sum(
        (mask.astype(np.uint8) for mask in classes.values()),
        np.zeros(b.shape, dtype=np.uint8),
    )
    if np.any(membership != 1):
        raise AssertionError("binary IEEE classes are not disjoint and exhaustive")
    if set(classes) != BINARY_CLASS_NAMES:
        raise AssertionError("binary IEEE class vocabulary drift")
    return classes


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch constants — lifted verbatim from sfpu_dispatch_constants.py /
# UnarySFPUGolden (kept local so the streamer needs no harness import at runtime;
# the selftest pins them against the live harness values).
# ─────────────────────────────────────────────────────────────────────────────
RPOW_BASE = 2.0
FMOD_DIVISOR = 2.0
REMAINDER_DIVISOR = 2.0
UNARY_POWER_EXP = 2.0
RDIV_VALUE = 2.0
CLAMP_MIN = -1.0
CLAMP_MAX = 1.0
SOFTSHRINK_LAMBDA = 0.5
HARDSHRINK_LAMBDA = 0.5
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0
XIELU_ALPHA_P = 1.0
XIELU_ALPHA_N = 1.0
XIELU_BETA = 0.5
PRELU_SLOPE = 0.25
POLYGAMMA_ORDER = 1
# laneMU (16-bit band mode) additions, lifted from sfpu_dispatch_constants.py.
THRESHOLD_T = 5.0          # torch.nn.functional.threshold(x, t, v)
THRESHOLD_V = 10.0
RELU_MAX_THRESHOLD = 5.0   # relu(min(x, thr))
UNARY_MAX_MIN_VALUE = 0.0  # max(x, v) / min(x, v)
FILL_CONST_VALUE = 5.0     # UnarySFPUGolden.__call__'s fill_const_value default
CELU_ALPHA = 1.0
ELU_ALPHA = 1.0
# blaze / coverage vehicle constants, lifted from test_sfpu_blaze.py and
# test_sfpu_coverage.py (the same literals their BLAZE_PARAMS / scalar_bits
# template args carry into the kernel).
CSILU_LIMIT = 2.0
CSILU_ALPHA = 1.702  # the GPT-OSS SwiGLU alpha
SITU_BETA = 8.0
SOFTCAP_CAP = 30.0   # Gemma-style final-logit cap
SILU_SCALE = 0.5
ADD_RSQRT_EPS = 0.5
SMOOTHSTEP_EDGE0 = -0.5
SMOOTHSTEP_INV_DELTA = 1.0
# The corpus `binopscalar` row is ScalarAdd at test_sfpu_binop_scalar's
# _PRESUBMIT_SCALAR; the `sdpa` row is exp(x * scale) with the bf16 scale bits
# 16256 = 0x3F80 = 1.0 its node id pins.
BINOP_SCALAR_ADD = 2.0
SDPA_EXP_SCALE = 1.0

BF16_TINY = 2.0**-126  # FTZ threshold for Float16_b / Float32 (finfo.tiny)


def input_ftz(x: np.ndarray) -> np.ndarray:
    """The operand the SFPU actually computes on: subnormals flushed to a signed zero.

    INPUT flush-to-zero is part of the hardware's contract and both oracles now model
    it. bf16 and fp32 share an 8-bit exponent field, so a subnormal bf16 in a 16-bit
    Dest is a subnormal fp32 to SFPLOAD, and the SFPU datapath flushes it -- the value
    the kernel computes on is +-0, never 9.18e-41. A golden that evaluates the op at
    the exact subnormal grades the device against a value it never saw.

    Ordinary stimuli never produce a subnormal, so the gap was invisible until a band
    enumerated all 65536 bf16 patterns. There it makes every steep-at-zero op look
    broken at the subnormals, and the SILICON is right every time:

      sqrt / sqrt-fresh   127/127 graded misses, at the 127 NEGATIVE subnormals:
                          sqrt(-9.18e-41) is NaN -> +inf to the old golden and 0.0 on
                          silicon, because sqrt(-0.0) is -0.0.
      rsqrt-fresh         127/127.
      log / log-fresh     254 of 510, at all 254 subnormals: log(9.18e-41) = -92.0 to
                          the old golden, -inf on silicon, because log(+0) = -inf.
      ceil-fresh          127 of 254, at the 127 POSITIVE subnormals: ceil of a tiny
                          positive is 1.0 to the old golden and 0.0 on silicon.
      recip / recip-ilv2  95, and the count is the proof: 1/x overflows bf16 for the
                          small subnormals (both oracles then say +inf and agree), and
                          is finite only for x >= 1/FLT_MAX = 2^-128.25, i.e. the 96
                          largest of the 127 positive subnormal codes. 95 of those 96
                          miss; the 96th is the boundary code where the flushed and
                          unflushed answers round to the same bf16.

    ``CorrectnessAccumulator`` keeps ``n_out_ftz_explained`` as the FALSIFICATION probe
    for this model rather than as an excuse for a miss: it now counts the residual
    out-of-tolerance patterns at a subnormal operand that the UNFLUSHED golden would
    have explained. A nonzero count is evidence against input FTZ for that op, not a
    licence.

    SCOPE, and it is not universal. The flush lives in the FP ALU, not in SFPLOAD: the
    bit-precise in-repo model flushes a denormal MANTISSA inside the FMA
    (corpus/tools/formal_equiv.py:392-403, ``m = B.ite(B.eq(e, c(0)), c(0), m)``),
    while its SFPLOAD/SFPSTORE applies ``denormals_as_zeros`` only on the STORE side,
    which is the OUTPUT FTZ ``format_golden_*`` already models. A body that never puts
    the operand through the FP ALU therefore does NOT flush it, and INPUT_FTZ_EXEMPT
    names those bodies.
    """
    xf = np.asarray(x, dtype=np.float32)
    sub = (np.abs(xf.astype(np.float64)) < BF16_TINY) & (xf != 0)
    return np.where(sub, np.copysign(np.float32(0.0), xf), xf).astype(np.float32)


# Bodies that never put the operand through the FP ALU, so the input flush cannot
# reach them. Verified at source, one by one, not assumed:
#
#   sign       ckernel_sfpu_sign.h:  v_if (v < 0) -1 / v_elseif (_sfpu_is_fp16_zero_(v))
#              0 / else 1, and _sfpu_is_fp16_zero_ is an exact `v == 0.0F`
#              (ckernel_sfpu_is_fp16_zero.h) -- a subnormal is NOT zero to it, so
#              sign(9.18e-41) is 1.0 on silicon and flushing would make the golden
#              say 0.0. This is the op that proves the model is not universal.
#   heaviside  ckernel_sfpu_heaviside.h: v_if (v < 0) 0 / v_elseif (v > 0) 1 / else s.
#              Three compares, no arithmetic; heaviside(9.18e-41) is 1.0, not 0.5.
#   abs        SFPABS; negative: a sign XOR; identity/copydest: a move.
#   min / max  SFPSWAP (a sign-magnitude compare, see below), and hardtanh / clamp /
#              relu / threshold, which are min/max and selects over the raw operand.
#   fill       ignores the operand entirely.
#   the predicates (signbit, isinf/isnan/isfinite, the comparisons, logical_not) and
#              the integer bodies, which are bit tests.
#
# For every one of these except sign and heaviside the distinction is unobservable
# anyway: op(subnormal) and op(+-0) round to the same bf16 once the OUTPUT flush
# applies. They are listed all the same, so the scope is a statement rather than an
# accident.
INPUT_FTZ_EXEMPT = frozenset(
    {
        "sign",
        "signbit",
        "heaviside",
        "heaviside-fresh",
        "abs",
        "absint32",
        "negative",
        "identity",
        "copydest-fresh",
        "fill",
        "fill-fresh",
        "unarymaxmin-max",
        "unarymaxmin-min",
        "relu",
        "threshold",
        "threshold-fresh",
        "threshold-fitted",
        "hardtanh",
        "hardtanh-fresh",
        "clamp",
        "clamp-fresh",
        "logicalnot",
        "isinfisnan",
        "unarycomp",
        "unarycomp-fresh",
        "comp",
        "eqz-fresh",
        "bitwisenot",
        "unaryshift",
        "unaryshift-fresh",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# The SFPU's float COMPARE is a sign-magnitude total order, and min/max inherit it.
#
# An SFPU vFloat compare is not an IEEE comparison: it ranks operands by their
# sign-magnitude bit pattern, which places a NaN BEYOND the infinity of its own
# sign. So max/min are neither NaN-propagating (what torch does) nor IEEE-754
# minNum/maxNum (return the non-NaN operand). They return whichever operand wins
# that order, and a NaN wins on the positive side and loses on the negative side.
#
# Four independent exhaustive rows agree with this and with nothing else, and the
# MISS COUNTS are what discriminate -- 127 versus 254 out of the 254 NaN patterns:
#
#   unarymaxmin-max  max(x, 0.0)   127 misses, witness x = -NaN, device 0.0
#                                  (x = +NaN gave +inf and MATCHED the old golden)
#   unarymaxmin-min  min(x, 0.0)   254 misses, witness x = +NaN, device 0.0
#   minmax-max       max(a, b)     127 per base stratum, witness b = -NaN, device = a
#   minmax-min       min(a, b)     254 per base stratum, witness b = +NaN, device = a
#
# minNum/maxNum would give 254 for all four. NaN propagation would give 127 for all
# four. The observed 127/254/127/254 split is exactly the sign-magnitude order.
# ─────────────────────────────────────────────────────────────────────────────
def _sign_magnitude_rank(x: np.ndarray) -> np.ndarray:
    """Monotone integer rank of an fp32 value under the SFPU's compare order."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    neg = (b & np.uint64(0x80000000)) != 0
    return np.where(
        neg, (~b) & np.uint64(0xFFFFFFFF), b | np.uint64(0x80000000)
    ).astype(np.int64)


def sfpu_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """max under the SFPU compare order (fp64 out, NaN ranked beyond its own inf)."""
    af = np.asarray(a, dtype=np.float32)
    bf = np.broadcast_to(np.asarray(b, dtype=np.float32), af.shape)
    return np.where(
        _sign_magnitude_rank(af) >= _sign_magnitude_rank(bf), af, bf
    ).astype(np.float64)


def sfpu_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """min under the SFPU compare order (fp64 out, NaN ranked beyond its own inf)."""
    af = np.asarray(a, dtype=np.float32)
    bf = np.broadcast_to(np.asarray(b, dtype=np.float32), af.shape)
    return np.where(
        _sign_magnitude_rank(af) <= _sign_magnitude_rank(bf), af, bf
    ).astype(np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# Vectorized true-math op bodies. Each takes an fp32 numpy array (the value the SFPU
# actually sees after bf16 truncation) and returns an fp64 numpy array (high precision,
# pre-output-rounding). These mirror UnarySFPUGolden's per-element methods exactly.
# ─────────────────────────────────────────────────────────────────────────────
def _t(x):
    return torch.from_numpy(np.asarray(x, dtype=np.float64))


def _np(t):
    return t.detach().numpy().astype(np.float64)


def _erf(x):
    return _np(torch.erf(_t(x)))


def _erfc(x):
    return _np(torch.erfc(_t(x)))


def _erfinv(x):
    return _np(torch.erfinv(_t(x)))


def _gelu_exact(x):
    # GeluAppx golden == exact (erf) gelu, matching UnarySFPUGolden._gelu.
    #
    # gelu(x) = x*Phi(x) evaluates to (-inf)*0 = NaN at x = -inf in any finite
    # precision, but the LIMIT is 0 and 0 is what the device returns. All four gelu
    # rows (gelu, gelu-fresh, gelu-fitted, gelu-licensed) report the same graded
    # witness 0x0000ff80 -- the -inf bf16 pattern, class neg_inf_input -- with
    # device 0.0 against golden +inf, and for gelu-fitted that is its ONLY graded
    # miss. The device is right; the golden was reading a 0*inf artefact.
    y = _np(torch.nn.functional.gelu(_t(x)))
    xf = np.asarray(x, dtype=np.float64)
    return np.where(np.isinf(xf) & (xf < 0), 0.0, y)


def _sigmoid(x):
    return _np(torch.sigmoid(_t(x)))


def _hardtanh(x):
    return _np(torch.clamp(_t(x), CLAMP_MIN, CLAMP_MAX))


def _rpow(x):
    return _np(torch.pow(torch.tensor(RPOW_BASE, dtype=torch.float64), _t(x)))


def _fmod(x):
    return _np(torch.fmod(_t(x), torch.tensor(FMOD_DIVISOR, dtype=torch.float64)))


def _remainder(x):
    return _np(
        torch.remainder(_t(x), torch.tensor(REMAINDER_DIVISOR, dtype=torch.float64))
    )


def _unary_power(x):
    return _np(torch.pow(_t(x), UNARY_POWER_EXP))


def _rdiv(x):
    return RDIV_VALUE / np.asarray(x, dtype=np.float64)


def _add1(x):
    return np.asarray(x, dtype=np.float64) + 1.0


def _sign(x):
    return _np(torch.sign(_t(x)))


def _signbit(x):
    xf = np.asarray(x, dtype=np.float32)
    return (np.signbit(xf)).astype(np.float64)


def _heaviside(x):
    xf = np.asarray(x, dtype=np.float64)
    return np.where(xf < 0.0, 0.0, np.where(xf > 0.0, 1.0, 0.5))


def _cbrt(x):
    return _np(torch.sign(_t(x)) * torch.abs(_t(x)).pow(1.0 / 3.0))


def _expm1(x):
    return _np(torch.expm1(_t(x)))


def _softplus(x):
    return _np(
        torch.nn.functional.softplus(
            _t(x), beta=SOFTPLUS_BETA, threshold=SOFTPLUS_THRESHOLD
        )
    )


def _softsign(x):
    return _np(torch.nn.functional.softsign(_t(x)))


def _softshrink(x):
    return _np(torch.nn.functional.softshrink(_t(x), lambd=SOFTSHRINK_LAMBDA))


def _hardshrink(x):
    return _np(torch.nn.functional.hardshrink(_t(x), lambd=HARDSHRINK_LAMBDA))


def _hardmish(x):
    return _np(_t(x) * torch.clamp(0.5 * _t(x) + 1.0, 0.0, 1.0))


def _xielu(x):
    xf = np.asarray(x, dtype=np.float64)
    beta_x = XIELU_BETA * xf
    pos = XIELU_ALPHA_P * xf * xf + beta_x
    neg = XIELU_ALPHA_N * (np.expm1(xf) - xf) + beta_x
    return np.where(xf > 0.0, pos, neg)


def _tanh_derivative_lut(x):
    # The LICENSED LUT contract (UnarySFPUGolden._tanh_derivative_lut): 1 - t^2 where t
    # is the raw 3-region SFPLUT, NOT accurate tanh. This is the kernel's own design
    # contract (the header documents catastrophic cancellation past |x|~3.4).
    a = np.abs(np.asarray(x, dtype=np.float64))
    t = np.where(a < 1.0, 0.90625 * a, np.where(a < 2.0, 0.09375 * a + 0.8125, 1.0))
    return 1.0 - t * t


def _tanh_derivative_true(x):
    # The TRUE math tanh'(x) = sech^2(x) = 1 - tanh^2, computed stably as 1/cosh^2.
    return _np(1.0 / torch.cosh(_t(x)) ** 2)


# laneMT corpus extension. Same rule as every body above: one expression, lifted
# from the UnarySFPUGolden method of the same name, vectorized. The selftest
# proves each one bit-for-bit against that scalar oracle -- nothing here is
# trusted because it looks right.
def _clamp(x):
    # UnarySFPUGolden._clamp with the dispatch min/max: identical expression to
    # _hardtanh, kept as its own name because they are two separate ops/rows.
    return _np(torch.clamp(_t(x), CLAMP_MIN, CLAMP_MAX))


def _identity(x):
    return np.asarray(x, dtype=np.float64)


def _prelu(x):
    xf = np.asarray(x, dtype=np.float64)
    return np.where(xf >= 0.0, xf, PRELU_SLOPE * xf)


def _mish(x):
    return _np(torch.nn.functional.mish(_t(x)))


def _selu(x):
    return _np(torch.nn.functional.selu(_t(x)))


def _sqrt(x):
    return _np(torch.sqrt(_t(x)))


def _rsqrt(x):
    return _np(torch.rsqrt(_t(x)))


# torch.special.i0/i1 return NaN at +-inf (the Cephes Chebyshev evaluation
# overflows), but the limits are I0(+-inf) = +inf and I1(+-inf) = +-inf, and the
# device returns exactly those. Grading the device against NaN fails a correct
# kernel at fp32 DEST and is invisible at a 16-bit DEST only because the packer
# turns the NaN golden into an inf. State the limits.
def _i0(x):
    t = _t(x)
    return _np(torch.where(torch.isinf(t), torch.abs(t), torch.special.i0(t)))


def _i1(x):
    t = _t(x)
    return _np(torch.where(torch.isinf(t), t, torch.special.i1(t)))


def _digamma(x):
    return _np(torch.digamma(_t(x)))


def _lgamma(x):
    # STAGE semantics, not composite semantics. calculate_lgamma_stirling is stage 1
    # of a three-tile composite and returns the DOCUMENTED intermediate
    # lgamma(z), z = (x < 0.5) ? 1-x : x. That is stated at
    # tt_metal/hw/inc/api/compute/eltwise_unary/lgamma.h:24, repeated in the closing
    # comment of the function itself, and completed by lgamma_adjusted_tile in
    # ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/lgamma_kernel.cpp.
    # The LLK node (test_sfpu_unary.py MathOperation.Lgamma) calls ONLY the stage, so
    # grading it against lgamma(x) on the negative axis grades it against a contract
    # it never made. Note this function has NO poles: for x >= 0.5 lgamma(x) is
    # finite, and for x < 0.5 the argument z = 1-x > 0.5 is too -- so the whole real
    # line is graded, which is STRICTER than the old pole-excluding treatment.
    xf = np.asarray(x, dtype=np.float64)
    z = np.where(xf < 0.5, 1.0 - xf, xf)
    return _np(torch.lgamma(_t(z)))


def _polygamma(x):
    return _np(torch.polygamma(POLYGAMMA_ORDER, _t(x)))


# ─────────────────────────────────────────────────────────────────────────────
# laneMU: the bodies the 16-bit band mode makes reachable. Same rule as every
# body above -- ONE expression, lifted from the UnarySFPUGolden method of the
# same name with its dispatch constants, vectorized in fp64. Nothing here is
# trusted because it reads correctly: selftest_threeway_golden proves each one
# against that scalar oracle over the WHOLE 65536-pattern bf16 space, which is
# exactly the population the device leg covers.
# ─────────────────────────────────────────────────────────────────────────────
def _abs(x):
    # SFPABS, float mod, is NOT mathematical abs at a NaN. The bit-precise in-repo
    # model (corpus/tools/formal_equiv.py:1299-1309) clears the sign bit only for
    # encodings <= 0xFF800000, so the 127 NEGATIVE-NaN patterns 0xFF800001..0xFFFFFFFF
    # pass through UNCHANGED (-inf itself, 0xFF800000, IS cleared to +inf). The kernel
    # is one SFPABS (tt_llk_blackhole/.../ckernel_sfpu_abs.h:20-22 -> sfpi::abs ->
    # __builtin_rvtt_sfpabs with SFPABS_MOD1_FLOAT, sfpi 7.69.0 sfpi_lib.h:280-282),
    # so the negative NaN reaches the packer with its sign intact and packs to -inf.
    # np.abs clears it; that difference is the whole of this row's 127 graded misses.
    xf = np.asarray(x, dtype=np.float64)
    return np.where(np.isnan(xf), xf, np.abs(xf))


def _neg(x):
    return -np.asarray(x, dtype=np.float64)


def _ceil(x):
    # UnarySFPUGolden._ceil: math.ceil(x) if finite else x. np.ceil already
    # returns inf/nan unchanged, so the guard is implicit.
    return np.ceil(np.asarray(x, dtype=np.float64))


def _square(x):
    # UnarySFPUGolden._square: x*x, with handle_infinite_numbers -> the value
    # itself for an exponent-B dst (both Float32 and Float16_b are exponent-B).
    xf = np.asarray(x, dtype=np.float64)
    return xf * xf


def _tanh(x):
    return _np(torch.tanh(_t(x)))


def _tanhshrink(x):
    # UnarySFPUGolden._tanhshrink: x - tanh(x).
    xf = np.asarray(x, dtype=np.float64)
    return xf - np.tanh(xf)


def _exp(x):
    return _np(torch.exp(_t(x)))


def _exp2(x):
    return _np(torch.exp2(_t(x)))


def _log(x):
    return _np(torch.log(_t(x)))


def _log1p(x):
    return _np(torch.log1p(_t(x)))


def _acosh(x):
    return _np(torch.acosh(_t(x)))


def _reciprocal(x):
    return _np(torch.reciprocal(_t(x)))


def _silu(x):
    return _np(torch.nn.functional.silu(_t(x)))


def _celu(x):
    return _np(torch.nn.functional.celu(_t(x), alpha=CELU_ALPHA))


def _elu(x):
    return _np(torch.nn.functional.elu(_t(x), alpha=ELU_ALPHA))


def _hardsigmoid(x):
    return _np(torch.nn.functional.hardsigmoid(_t(x)))


def _threshold(x):
    # UnarySFPUGolden._threshold with the dispatch t/v: x if x > t else v.
    return _np(
        torch.nn.functional.threshold(_t(x), THRESHOLD_T, THRESHOLD_V)
    )


def _relu_max(x):
    # UnarySFPUGolden._relu_max: relu(min(x, RELU_MAX_THRESHOLD)).
    xf = np.asarray(x, dtype=np.float64)
    return np.maximum(0.0, np.minimum(xf, RELU_MAX_THRESHOLD))


def _unary_max(x):
    # UnarySFPUGolden._unary_max: max(x, UNARY_MAX_MIN_VALUE) under the SFPU compare
    # order, not np.maximum -- see sfpu_max.
    return sfpu_max(x, np.float32(UNARY_MAX_MIN_VALUE))


def _unary_min(x):
    return sfpu_min(x, np.float32(UNARY_MAX_MIN_VALUE))


def _fill(x):
    # UnarySFPUGolden._fill: the output is the fill constant for every input.
    return np.full(np.shape(x), FILL_CONST_VALUE, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# laneMU: the POINTWISE blaze / coverage vehicle bodies. Lifted from the
# `_g_*` / local `golden` closures of test_sfpu_blaze.py and
# test_sfpu_coverage.py, whose signatures are `(a, _b)` -- operand B is unused,
# which is why these rows are one-operand rows and can ride a raw band at all.
#
# Only POINTWISE bodies are here. The vehicles also carry rows whose output
# element is a function of other elements or of the element's index; those are
# named in _NOT_POINTWISE below with the specific reason instead of being given a
# golden that would mis-grade them.
# ─────────────────────────────────────────────────────────────────────────────
def _blaze_csilu_gate(x):
    xf = np.minimum(np.asarray(x, dtype=np.float64), CSILU_LIMIT)
    return _np(_t(xf) * torch.sigmoid(_t(CSILU_ALPHA * xf)))


def _blaze_csilu_up(x):
    return np.clip(np.asarray(x, dtype=np.float64), -CSILU_LIMIT, CSILU_LIMIT) + 1.0


def _blaze_csilu_clamped(x):
    return np.clip(np.asarray(x, dtype=np.float64), -CSILU_LIMIT, CSILU_LIMIT)


def _blaze_situ_gate(x):
    xf = np.asarray(x, dtype=np.float64)
    return _np(SITU_BETA * torch.tanh(_t(xf / SITU_BETA)) * torch.sigmoid(_t(xf)))


def _blaze_scaledtanh(x):
    return _np(SITU_BETA * torch.tanh(_t(np.asarray(x, dtype=np.float64) / SITU_BETA)))


def _blaze_logitsoftcap(x):
    return _np(SOFTCAP_CAP * torch.tanh(_t(x)))


def _blaze_siluscaled(x):
    xf = SILU_SCALE * np.asarray(x, dtype=np.float64)
    return _np(SILU_SCALE * (_t(xf) * torch.sigmoid(_t(xf))))


def _blaze_addrsqrt(x):
    return _np(torch.rsqrt(_t(np.asarray(x, dtype=np.float64) + ADD_RSQRT_EPS)))


def _blaze_sdpaexp(x):
    return _np(torch.exp(_t(x)))


def _binop_scalar_add(x):
    # ScalarBinopGolden for MathOperation.ScalarAdd: out = x + s.
    return np.asarray(x, dtype=np.float64) + BINOP_SCALAR_ADD


def _sdpa_exp_unclamped(x):
    # SdpaExpUnclampedGolden: exp(x * scale), scale decoded from bf16 bits.
    return _np(torch.exp(_t(np.asarray(x, dtype=np.float64) * SDPA_EXP_SCALE)))


def _cov_smoothstep(x):
    t = np.clip(
        (np.asarray(x, dtype=np.float64) - SMOOTHSTEP_EDGE0) * SMOOTHSTEP_INV_DELTA,
        0.0,
        1.0,
    )
    return t * t * (3.0 - 2.0 * t)


# ─────────────────────────────────────────────────────────────────────────────
# Output-format pipeline for the streamer's Float32/Float32, dest_acc=No config
# (dst_format = Float16_b). Mirrors UnarySFPUGolden.__call__ for that case.
# ─────────────────────────────────────────────────────────────────────────────
def _bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """Raw bf16 bit patterns (uint16) -> fp32 values (shift into the fp32 high half)."""
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def bf16_truncate(u32: np.ndarray) -> np.ndarray:
    """Input as the SFPU sees it: Float32 operand truncated to bf16 (& 0xFFFF0000)."""
    return (u32.astype(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def _round_bf16_as_f32(hp: np.ndarray) -> np.ndarray:
    """RNE round an fp64 array to bf16, returned as fp32 values (torch bfloat16 cast)."""
    if torch is not None:
        t = (
            torch.from_numpy(np.asarray(hp, dtype=np.float32))
            .to(torch.bfloat16)
            .to(torch.float32)
        )
        return t.numpy()
    bits = np.asarray(hp, dtype=np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + bias) & np.uint32(0xFFFF0000)).view(np.float32)


def format_golden_f32_acc(hp: np.ndarray) -> np.ndarray:
    """hp (fp64 true math) -> the reference for a 32-bit DEST with an fp32 output.

    Mirrors UnarySFPUGolden.__call__ for (input Float16_b, output Float32,
    dest_acc=Yes): dst_format is Float32, so there is NO bf16 rounding of the
    result, and `match (Float32, Float32)` PRESERVES NaN rather than converting it
    to +inf. `_apply_ftz` still applies -- and at the SAME 2^-126, because
    _FTZ_THRESHOLD keys on the output format and fp32's smallest NORMAL is
    2^-126, exactly bf16's.

    That last point is the whole reason this pipeline exists as a separate
    function and also the reason it does not rescue a subnormal: widening Dest
    from 16 to 32 bits does not lower the flush threshold, because bf16 and fp32
    share an 8-bit exponent field. See SIGMOID_SUBNORMAL_NOTE.
    """
    y = np.asarray(hp, dtype=np.float64).astype(np.float32)
    y = np.where(np.abs(y.astype(np.float64)) < BF16_TINY, np.float32(0.0), y)  # FTZ
    return y.astype(np.float32)


def format_golden_f32_noacc(hp: np.ndarray) -> np.ndarray:
    """hp (fp64 true math) -> the fp32-container reference the device should output.

    bf16-round -> NaN->SIGN-PRESERVED inf (dst=Float16_b, data=Float32 falls in
    __call__'s default convert_nan_to_inf case) -> FTZ below 2^-126. Returns fp32.

    NaN->inf PRESERVES THE SIGN BIT. The packer does not synthesize a +inf; it
    converts the NaN that is in Dest and the sign bit rides through. `copydest-fresh`
    settles this with no arithmetic in the way: it is a pure identity move, and at all
    127 negative-NaN bf16 patterns the device returns -inf where the old golden said
    +inf -- 127 of 127 of its graded misses, and nothing else in that row can account
    for them.

    The sign is read from `hp`, NOT from the rounded value, because torch's bfloat16
    cast CANONICALIZES every NaN to 0xFFFF (a negative NaN): reading the sign after
    rounding would turn every NaN into -inf, including a positive one. `negative` is
    the row that would catch that mistake -- it misses at exactly the 127 POSITIVE NaN
    patterns, because SFPU negation flips a NaN's sign bit too.
    """
    hp64 = np.asarray(hp, dtype=np.float64)
    y = _round_bf16_as_f32(hp).astype(np.float32)
    nan = np.isnan(hp64) | np.isnan(y)
    # convert_nan_to_inf, sign-preserving: copysign reads the NaN's own sign bit.
    y = np.where(
        nan, np.copysign(np.float64(np.inf), hp64), y.astype(np.float64)
    ).astype(np.float32)
    y = np.where(np.abs(y.astype(np.float64)) < BF16_TINY, np.float32(0.0), y)  # FTZ
    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Op registry
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GoldenSpec:
    op: str  # streamer op key (matches ops31.tsv / idmap)
    math: Optional[Callable[[np.ndarray], np.ndarray]]  # fp32-in -> fp64 true math
    kind: str = "f32_noacc"  # 'f32_noacc' | 'int32' | 'unsupported'
    atol: float = 0.05  # harness passed_test Float32 default
    rtol: float = 0.05
    domain: Optional[tuple] = (
        None  # (lo, hi) finite in-domain interval, else None=all reals
    )
    note: str = ""
    checkable: bool = True
    dst_acc: bool = False  # True => 32-bit DEST + fp32 output (format_golden_f32_acc)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """The op at the operand the SFPU actually receives: input-FTZ then math.

        Every caller must go through this rather than `spec.math` directly, or it
        grades the device against a subnormal the datapath never had. See input_ftz,
        and INPUT_FTZ_EXEMPT for the compare-only bodies the flush cannot reach.
        """
        if self.op in INPUT_FTZ_EXEMPT:
            return self.math(np.asarray(x, dtype=np.float32))
        return self.math(input_ftz(x))


# Divergent priority ops (the correctness question actually matters for these).
_DIVERGENT = [
    GoldenSpec("erf-fresh", _erf, note="torch.erf; all reals"),
    GoldenSpec("erfc-fresh", _erfc, note="torch.erfc; all reals"),
    GoldenSpec(
        "erfinv-fresh",
        _erfinv,
        domain=(-1.0, 1.0),
        note="torch.erfinv; DOMAIN (-1,1) — |x|>=1 is out-of-domain, +-inf/nan expected",
    ),
    GoldenSpec(
        "geluappx-fresh",
        _gelu_exact,
        atol=0.13,
        rtol=0.05,
        note="exact (erf) gelu; GeluAppx is a LICENSED 6-segment LUT approx (CUSTOM_TOLERANCES 0.13/0.05)",
    ),
    GoldenSpec(
        "sigmoidlut-fresh",
        _sigmoid,
        atol=0.05,
        rtol=0.05,
        note="torch.sigmoid; production is a LUT6 approx (contract 0.05/0.05)",
    ),
    GoldenSpec(
        "tanhderivlut-fresh",
        _tanh_derivative_lut,
        note=(
            "LICENSED LUT contract 1-t_lut^2 (NOT accurate sech^2); see tanhderiv-true "
            "column for the true-math distance the LUT intentionally trades away"
        ),
    ),
    GoldenSpec(
        "hardtanh-fresh",
        _hardtanh,
        note="clamp(x,-1,1); EXACT piecewise-linear (expect 0 ULP)",
    ),
    GoldenSpec("rpow", _rpow, note="2.0**x; inf/nan inputs -> inf/nan (band7)"),
    GoldenSpec(
        "fmod-fresh", _fmod, note="fmod(x,2.0); inf/nan inputs special (band15)"
    ),
]

# Bit-exact ops (sem==hand over all 2^32 — one leg vs torch confirms 'correct AND
# matches expert'). All Float32/dest_acc=No unless noted.
_BITEXACT = [
    GoldenSpec("add1", _add1, note="x+1; EXACT (expect 0 ULP)"),
    GoldenSpec("sign", _sign, note="sign(x); EXACT"),
    GoldenSpec("signbit", _signbit, note="signbit(x)->0/1; EXACT"),
    GoldenSpec("heaviside-fresh", _heaviside, note="0/0.5/1; EXACT"),
    GoldenSpec("cbrt-fresh", _cbrt, note="sign(x)|x|^(1/3)"),
    GoldenSpec("expm1cw-fresh", _expm1, note="expm1(x)"),
    GoldenSpec("softplus-fresh", _softplus, note="softplus beta=1 thr=20"),
    GoldenSpec("softsign-fresh", _softsign, note="x/(1+|x|)"),
    GoldenSpec("softshrink-fresh", _softshrink, note="softshrink lambda=0.5"),
    GoldenSpec("hardshrink-fresh", _hardshrink, note="hardshrink lambda=0.5"),
    GoldenSpec("hardmish-fresh", _hardmish, note="x*clamp(0.5x+1,0,1)"),
    GoldenSpec("xielu-fresh", _xielu, note="xielu beta=0.5 alpha_p=alpha_n=1"),
    GoldenSpec("rdiv", _rdiv, note="2.0/x"),
    GoldenSpec("remainder-fresh", _remainder, note="remainder(x,2.0)"),
    GoldenSpec("unarypower-fresh", _unary_power, note="x**2"),
]

# ─────────────────────────────────────────────────────────────────────────────
# laneMT corpus extension (2026-09-30): the remaining 32-bit unary rows of
# sweep_2x2_ops.tsv, so `--golden <op>` is live for every op the streamer can
# actually drive instead of only the 31 first-wave ones.
#
# Most of these are `kind=semantic` rows: ONE node, the production body, no
# separate hand leg. The equivalence half of the leg is vacuous for them; the
# ULP-vs-golden half is the entire reason they are here.
#
# `domain` stays the MATHEMATICAL domain, as everywhere above -- not the fit
# range. Setting erf's domain to its [-3, 3] fit range would have suppressed the
# real erf sign inversion found at x = 11. The kernels' claimed accuracy ranges
# are separate, reporting-only data: see CLAIMED_ACCURACY_DOMAIN below.
# ─────────────────────────────────────────────────────────────────────────────
_CORPUS_UNARY = [
    # -- bounded / squashing: an out-of-range input must saturate to a finite
    #    bound. Both the erf and the sigmoid-LUT defects are this class.
    GoldenSpec("erf", _erf, note="torch.erf; all reals; production generic-sweep row"),
    GoldenSpec("erfc", _erfc, note="torch.erfc; all reals"),
    GoldenSpec(
        "erfinv",
        _erfinv,
        domain=(-1.0, 1.0),
        note="torch.erfinv; DOMAIN (-1,1) -- |x|>=1 is out-of-domain, +-inf/nan expected",
    ),
    GoldenSpec("sigmoid", _sigmoid, note="torch.sigmoid; typed production body"),
    GoldenSpec("sigmoid-fresh", _sigmoid, note="torch.sigmoid; fresh_cpp arm"),
    GoldenSpec("softsign", _softsign, note="x/(1+|x|); bounded on (-1,1)"),
    GoldenSpec(
        "tanhderivative",
        _tanh_derivative_true,
        note="TRUE sech^2; the fitted tanh_bw row, not the LUT row",
    ),
    GoldenSpec(
        "tanhderivative-lut",
        _tanh_derivative_lut,
        note=(
            "LICENSED LUT contract 1-t_lut^2 (NOT accurate sech^2) -- the golden IS "
            "the production table, so a more accurate kernel would fail it"
        ),
    ),
    GoldenSpec("heaviside", _heaviside, note="0/0.5/1; EXACT"),
    # -- fitted polynomial / series: the erf mechanism (a fit evaluated where it
    #    has no meaning, with a clamp turning garbage into a plausible constant).
    # The gamma family gets NO domain. It is defined on every real that is not a
    # pole, and the poles are exactly the non-positive INTEGERS -- see
    # GAMMA_POLE_OPS / at_gamma_pole below. Declaring domain=(0, inf) here would
    # have been wrong in a way that matters: it would license the whole negative
    # axis, and digamma(-1.5) = +0.703157 is an ordinary defined value where the
    # kernel returns -10.2929, a sign flip. A pole is excluded pointwise, not by
    # excluding a half-line.
    GoldenSpec(
        "digamma",
        _digamma,
        note="torch.digamma; all non-pole reals; poles at the non-positive integers",
    ),
    GoldenSpec(
        "digamma-fresh",
        _digamma,
        note="torch.digamma; fresh_cpp arm; same pole set",
    ),
    GoldenSpec(
        "lgamma",
        _lgamma,
        note="STAGE contract lgamma(z), z=(x<0.5)?1-x:x (lgamma.h:24) -- the LLK node "
        "is calculate_lgamma_stirling, stage 1 of the 3-tile composite; finite on all "
        "reals, so no pole exclusion",
    ),
    GoldenSpec(
        "polygamma",
        _polygamma,
        note=f"torch.polygamma(n={POLYGAMMA_ORDER}, x) trigamma; poles at the non-positive integers",
    ),
    GoldenSpec("i0", _i0, note="torch.special.i0; kernel poly valid |x| <= 3.75"),
    GoldenSpec("i1", _i1, note="torch.special.i1; kernel poly valid |x| <= 3.75"),
    GoldenSpec("i1-fresh", _i1, note="torch.special.i1; fresh_cpp arm"),
    GoldenSpec("expm1", _expm1, note="torch.expm1"),
    GoldenSpec("expm1-fresh", _expm1, note="torch.expm1; fresh_cpp arm"),
    GoldenSpec("expm1cw", _expm1, note="torch.expm1; the CW refit row"),
    GoldenSpec("cbrt", _cbrt, note="sign(x)|x|^(1/3)"),
    GoldenSpec("mish", _mish, note="x*tanh(softplus(x))"),
    GoldenSpec("selu", _selu, note="selu with the default scale/alpha"),
    GoldenSpec("softplus", _softplus, note="softplus beta=1 thr=20"),
    GoldenSpec("xielu", _xielu, note="xielu beta=0.5 alpha_p=alpha_n=1"),
    # -- setexp/addexp exponent-field writes: the rpow mechanism, where the 8-bit
    #    field WRAPS instead of saturating and an overflow becomes a finite value.
    GoldenSpec("unarypower", _unary_power, note="x**2"),
    GoldenSpec(
        "sqrtcustom",
        _sqrt,
        domain=(0.0, float("inf")),
        note="torch.sqrt; DOMAIN [0, inf) -- x<0 is out-of-domain (nan)",
    ),
    GoldenSpec(
        "rsqrtcompat",
        _rsqrt,
        domain=(0.0, float("inf")),
        note="torch.rsqrt; DOMAIN (0, inf) -- 0 is the pole, x<0 out-of-domain",
    ),
    GoldenSpec("fmod", _fmod, note="fmod(x,2.0)"),
    GoldenSpec("remainder", _remainder, note="remainder(x,2.0)"),
    # -- LUT / piecewise.
    GoldenSpec("clamp", _clamp, note="clamp(x,-1,1); EXACT piecewise-linear"),
    GoldenSpec("clamp-fresh", _clamp, note="clamp(x,-1,1); fresh_cpp arm"),
    GoldenSpec("hardtanh", _hardtanh, note="clamp(x,-1,1); EXACT piecewise-linear"),
    GoldenSpec("hardmish", _hardmish, note="x*clamp(0.5x+1,0,1)"),
    GoldenSpec("hardshrink", _hardshrink, note="hardshrink lambda=0.5"),
    GoldenSpec("softshrink", _softshrink, note="softshrink lambda=0.5"),
    GoldenSpec("prelu", _prelu, note=f"x if x>=0 else {PRELU_SLOPE}*x; EXACT"),
    # -- exact.
    GoldenSpec("identity", _identity, note="x; EXACT (expect 0 ULP)"),
]

# ─────────────────────────────────────────────────────────────────────────────
# laneMU corpus extension (2026-09-29): the 62 Float16_b unary rows of
# sweep_2x2_ops.tsv. Unreachable until the streamer became width-aware -- the
# 4-bytes-per-element payload was refused against a 2048-byte bf16 tile. They are
# the STRONGEST rows in the corpus, not the weakest: a bf16 row has only 65536
# distinct inputs, so one 65536-pattern band is EXHAUSTIVE and these ops are
# PROVEN over their whole input space rather than sampled at nine strata.
#
# The 19 `-fitted` rows are here too. They are the block a static audit called
# the highest-suspicion class (12 of 21 *_fitted.h files predicted defective) and
# the block no previous leg could reach at all.
#
# `domain` is the MATHEMATICAL domain as everywhere above, never the fit range.
# ─────────────────────────────────────────────────────────────────────────────
_CORPUS_BF16 = [
    # -- exact / piecewise-linear: expect 0 ULP over the whole space.
    GoldenSpec("abs", _abs, note="|x|; EXACT"),
    GoldenSpec("negative", _neg, note="-x; EXACT"),
    GoldenSpec("ceil-fresh", _ceil, note="ceil(x); EXACT (inf/nan pass through)"),
    GoldenSpec("relu", _relu_max, note=f"relu(min(x,{RELU_MAX_THRESHOLD})); EXACT piecewise-linear"),
    GoldenSpec("unarymaxmin-max", _unary_max, note=f"max(x,{UNARY_MAX_MIN_VALUE}); EXACT"),
    GoldenSpec("unarymaxmin-min", _unary_min, note=f"min(x,{UNARY_MAX_MIN_VALUE}); EXACT"),
    GoldenSpec("threshold", _threshold, note=f"x if x>{THRESHOLD_T} else {THRESHOLD_V}; EXACT"),
    GoldenSpec("threshold-fresh", _threshold, note="threshold; fresh_cpp arm"),
    GoldenSpec("threshold-fitted", _threshold, note="threshold; fitted_cpp arm"),
    GoldenSpec("fill", _fill, note=f"constant {FILL_CONST_VALUE} for every input; EXACT"),
    GoldenSpec("fill-fresh", _fill, note="fill; fresh_cpp arm"),
    GoldenSpec("activations", _hardsigmoid, note="hardsigmoid = clamp(x/6+1/2,0,1); the corpus `activations` row IS Hardsigmoid"),
    GoldenSpec("hardsigmoid-fresh", _hardsigmoid, note="hardsigmoid; fresh_cpp arm"),
    GoldenSpec("square", _square, note="x*x"),
    GoldenSpec("square-fresh", _square, note="x*x; fresh_cpp arm"),
    # -- bounded / squashing. The softsign and erf defects were both this class:
    #    a bounded function returning a value outside or at the wrong end of its
    #    own range. Exhaustive coverage is worth most here.
    GoldenSpec("tanh", _tanh, note="tanh(x); bounded on (-1,1)"),
    GoldenSpec("tanh-fresh", _tanh, note="tanh; fresh_cpp arm"),
    GoldenSpec("tanh-fitted", _tanh, note="tanh; fitted_cpp arm"),
    GoldenSpec(
        "tanhlut-fresh",
        _tanh,
        atol=0.16,
        rtol=0.05,
        note="tanh; LICENSED 3-region SFPLUT (row's own custom_atol=0.16) -- the "
        "sister of the already-fixed sigmoid_lut_licensed",
    ),
    GoldenSpec("tanhshrink", _tanhshrink, note="x - tanh(x)"),
    GoldenSpec("tanhshrink-fresh", _tanhshrink, note="x - tanh(x); fresh_cpp arm"),
    GoldenSpec("tanhderivative-fitted", _tanh_derivative_true, note="TRUE sech^2; the fitted tanh_bw row"),
    GoldenSpec("sigmoid-fitted", _sigmoid, note="torch.sigmoid; fitted_cpp arm"),
    GoldenSpec(
        "sigmoidappx",
        _sigmoid,
        atol=0.13,
        rtol=0.05,
        note="torch.sigmoid; SigmoidAppx is a LICENSED coarse LUT (CUSTOM_TOLERANCES 0.13/0.05)",
    ),
    GoldenSpec(
        "sigmoidappx-tree",
        _sigmoid,
        atol=0.13,
        rtol=0.05,
        note="torch.sigmoid; the PWL-dataflow arm of the same licensed LUT contract",
    ),
    GoldenSpec("silu", _silu, note="x*sigmoid(x)"),
    GoldenSpec("silu-fresh", _silu, note="x*sigmoid(x); fresh_cpp arm"),
    GoldenSpec("gelu", _gelu_exact, note="exact (erf) gelu"),
    GoldenSpec("gelu-fresh", _gelu_exact, note="exact (erf) gelu; fresh_cpp arm"),
    GoldenSpec("gelu-fitted", _gelu_exact, note="exact (erf) gelu; fitted_cpp arm"),
    GoldenSpec("gelu-licensed", _gelu_exact, note="exact (erf) gelu; licensed_cpp arm"),
    # -- exponential family: the exp.h 255.0-clamp and the expm1 sign mechanisms.
    GoldenSpec("exp", _exp, note="torch.exp"),
    GoldenSpec("exp-fitted", _exp, note="torch.exp; fitted_cpp arm"),
    GoldenSpec("exp2", _exp2, note="torch.exp2"),
    GoldenSpec("exp2-fresh", _exp2, note="torch.exp2; fresh_cpp arm"),
    GoldenSpec("expm1-fitted", _expm1, note="torch.expm1; fitted_cpp arm"),
    GoldenSpec("celu", _celu, note=f"celu alpha={CELU_ALPHA}"),
    GoldenSpec("celu-fitted", _celu, note="celu; fitted_cpp arm"),
    GoldenSpec("elu", _elu, note=f"elu alpha={ELU_ALPHA}"),
    GoldenSpec("elu-fresh", _elu, note="elu; fresh_cpp arm"),
    GoldenSpec("elu-fitted", _elu, note="elu; fitted_cpp arm"),
    GoldenSpec("selu-fitted", _selu, note="selu with the default scale/alpha; fitted_cpp arm"),
    GoldenSpec("mish-fitted", _mish, note="x*tanh(softplus(x)); fitted_cpp arm"),
    # -- series / fitted polynomial: the i0/i1 unreduced-Maclaurin and the
    #    digamma/lgamma negative-branch mechanisms.
    GoldenSpec("i0-fitted", _i0, note="torch.special.i0; fitted_cpp arm"),
    GoldenSpec("i1-fitted", _i1, note="torch.special.i1; fitted_cpp arm"),
    GoldenSpec(
        "digamma-fitted",
        _digamma,
        note="torch.digamma; fitted_cpp arm; poles at the non-positive integers",
    ),
    GoldenSpec(
        "polygamma-fitted",
        _polygamma,
        note=f"torch.polygamma(n={POLYGAMMA_ORDER}, x) trigamma; fitted_cpp arm; same pole set",
    ),
    # -- DOMAINED. The mathematical domain, never the fit range. log(0) = -inf is
    #    a LIMIT and therefore the correct answer, so 0 is the domain's lower
    #    boundary and not a licensed pole; only x < 0 is out of domain.
    GoldenSpec("log", _log, domain=(0.0, float("inf")), note="torch.log; DOMAIN [0,inf) -- log(0)=-inf is correct, x<0 undefined"),
    GoldenSpec("log-fresh", _log, domain=(0.0, float("inf")), note="torch.log; fresh_cpp arm"),
    GoldenSpec("log-fitted", _log, domain=(0.0, float("inf")), note="torch.log; fitted_cpp arm"),
    GoldenSpec("log1p", _log1p, domain=(-1.0, float("inf")), note="torch.log1p; DOMAIN [-1,inf) -- log1p(-1)=-inf is correct"),
    GoldenSpec("log1p-fresh", _log1p, domain=(-1.0, float("inf")), note="torch.log1p; fresh_cpp arm"),
    GoldenSpec("log1p-fitted", _log1p, domain=(-1.0, float("inf")), note="torch.log1p; fitted_cpp arm"),
    GoldenSpec("sqrt", _sqrt, domain=(0.0, float("inf")), note="torch.sqrt; DOMAIN [0,inf)"),
    GoldenSpec("sqrt-fresh", _sqrt, domain=(0.0, float("inf")), note="torch.sqrt; fresh_cpp arm"),
    GoldenSpec("rsqrt-fresh", _rsqrt, domain=(0.0, float("inf")), note="torch.rsqrt; fresh_cpp arm; 0 is the pole"),
    GoldenSpec("rsqrt-fitted", _rsqrt, domain=(0.0, float("inf")), note="torch.rsqrt; fitted_cpp arm; 0 is the pole"),
    GoldenSpec(
        "acosh-fitted",
        _acosh,
        domain=(1.0, float("inf")),
        note="torch.acosh; DOMAIN [1,inf) -- acosh(1)=0 exactly, x<1 undefined",
    ),
    GoldenSpec(
        "trigonometry",
        _acosh,
        domain=(1.0, float("inf")),
        note="the corpus `trigonometry` row's mathop IS Acosh (trigonometry.h's "
        "representative); DOMAIN [1,inf)",
    ),
    GoldenSpec(
        "trigonometry-fresh",
        _acosh,
        domain=(1.0, float("inf")),
        note="Acosh; fresh_cpp arm; DOMAIN [1,inf)",
    ),
    # -- reciprocal: defined on every real but 0, so NO domain -- the pole is
    #    excluded pointwise by RECIPROCAL_POLE_OPS, the same way the gamma poles
    #    are. A domain of (0,inf) here would license the entire negative axis,
    #    where 1/x is an ordinary defined value.
    GoldenSpec("recip", _reciprocal, note="1/x; pole at 0, no domain restriction"),
    GoldenSpec("recip-ilv2", _reciprocal, note="1/x; the ilv2-scheduled arm; pole at 0"),
]

# ─────────────────────────────────────────────────────────────────────────────
# laneMU: the blaze and coverage vehicles' POINTWISE rows. Their test files had
# no SFPU_STREAM hook at all -- the hook now lives in `_run_blaze` /
# `_run_coverage`, the single funnel every one of their correctness rows passes
# through, so the whole family gained it in one place.
#
# `-t8` / `-t32` are the same op at tile_count 8 / 32. Identical math, so
# identical golden; they are separate corpus rows and get separate specs because
# the board is keyed on the row.
#
# Tolerances are each row's OWN (the custom_atol/custom_rtol its test passes),
# not a blanket default: `clampedsilu-clamped` is an exact clamp and gets 0/0,
# while `logitsoftcap` legitimately carries 0.25 absolute.
# ─────────────────────────────────────────────────────────────────────────────
def _blaze_rows(base, math, atol=0.05, rtol=0.05, note=""):
    """One spec per corpus row of a blaze op: the 1-tile row and its t8/t32 twins."""
    return [
        GoldenSpec(f"blaze-{base}{suffix}", math, atol=atol, rtol=rtol,
                   note=note + (f"; tile_count={tc}" if tc != 1 else ""))
        for suffix, tc in (("", 1), ("-t8", 8), ("-t32", 32))
    ]


_CORPUS_BLAZE = (
    _blaze_rows("clampedsilu-gate", _blaze_csilu_gate, 0.02, 0.05,
                f"clamp(x,max={CSILU_LIMIT})*sigmoid({CSILU_ALPHA}*that) -- GPT-OSS SwiGLU gate")
    + _blaze_rows("clampedsilu-up", _blaze_csilu_up, 0.05, 0.05,
                  f"clamp(x,+-{CSILU_LIMIT})+1")
    + _blaze_rows("clampedsilu-clamped", _blaze_csilu_clamped, 0.0, 0.0,
                  f"clamp(x,+-{CSILU_LIMIT}); EXACT (the row's own gate is atol=rtol=0)")
    + _blaze_rows("situ-gate", _blaze_situ_gate, 0.03, 0.05,
                  f"{SITU_BETA}*tanh(x/{SITU_BETA})*sigmoid(x)")
    + _blaze_rows("scaledtanh", _blaze_scaledtanh, 0.03, 0.05,
                  f"{SITU_BETA}*tanh(x/{SITU_BETA})")
    + _blaze_rows("logitsoftcap", _blaze_logitsoftcap, 0.25, 0.05,
                  f"{SOFTCAP_CAP}*tanh(x); Gemma-style logit cap, bounded on (-{SOFTCAP_CAP},{SOFTCAP_CAP})")
    + _blaze_rows("siluscaled", _blaze_siluscaled, 0.02, 0.05,
                  f"s*(sx*sigmoid(sx)) with s={SILU_SCALE}")
    + _blaze_rows("addrsqrt", _blaze_addrsqrt, 0.05, 0.05,
                  f"rsqrt(x+{ADD_RSQRT_EPS}); RMSNorm idiom")
    + _blaze_rows("sdpaexp", _blaze_sdpaexp, 0.05, 0.05, "exp(x); the SDPA exp row")
)

# ─────────────────────────────────────────────────────────────────────────────
# The dest_acc=Yes rows the predecessor lane asked for, and the answer they give.
#
# That lane found sigmoid(-89) CLEAN on the Float32->Float32 dest_acc=No row and
# said the audit's "the true value 2.22736e-39 is representable" claim is about an
# fp32 DEST, so testing it needs a dest_acc=Yes ROW. These are that row -- with a
# bf16 INPUT, so one band is also exhaustive.
#
# SIGMOID_SUBNORMAL_NOTE. The row exists and is run, and the answer is that no
# widening of Dest can make that value observable, for a reason that is arithmetic
# rather than a limit of the harness: bf16 and fp32 have the SAME 8-bit exponent
# field, so both have 2^-126 as their smallest NORMAL. 2.22736e-39 is 2^-128.55 --
# a subnormal in fp32 exactly as much as in bf16 -- and SFPU arithmetic flushes
# subnormals at 2^-126 whatever the Dest width. The audit's premise conflates
# "representable in fp32" (true, fp32 has subnormals down to 2^-149) with
# "survives SFPU arithmetic" (false). Measured, not argued: these rows run.
# ─────────────────────────────────────────────────────────────────────────────
_CORPUS_DESTACC = [
    GoldenSpec(
        "sigmoid-destacc",
        _sigmoid,
        dst_acc=True,
        note="torch.sigmoid on Float16_b->Float32 dest_acc=Yes: a 32-bit DEST and an "
        "fp32 output, EXHAUSTIVE over the bf16 input space. The row the predecessor "
        "lane named for the audit's rank 1; see SIGMOID_SUBNORMAL_NOTE",
    ),
    GoldenSpec(
        "softplus-destacc",
        _softplus,
        dst_acc=True,
        note="softplus on Float16_b->Float32 dest_acc=Yes -- the audit's rank 6b names "
        "the bf16 arm specifically, and this is that arm at full Dest width",
    ),
    # The fp32-DEST rows for the 2026-09-30 i0/i1/expm1cw overflow fixes. Those fixes
    # were measured only at Float32->Float32 dest_acc=No, i.e. a 16-bit Dest and the
    # bf16 arm; these rows grade the same bodies at a 32-bit Dest with an fp32 output,
    # exhaustive over the bf16 input space (one band).
    GoldenSpec(
        "i0-destacc",
        _i0,
        dst_acc=True,
        note="torch.special.i0 on Float16_b->Float32 dest_acc=Yes: fp32 DEST/output",
    ),
    GoldenSpec(
        "i1-destacc",
        _i1,
        dst_acc=True,
        note="torch.special.i1 on Float16_b->Float32 dest_acc=Yes: fp32 DEST/output",
    ),
    GoldenSpec(
        "expm1cw-destacc",
        _expm1,
        dst_acc=True,
        note="torch.expm1 on Float16_b->Float32 dest_acc=Yes: fp32 DEST/output",
    ),
]

# Two further single-row vehicles whose test file had no hook and whose row IS
# pointwise. Their siblings in the same families are NOT (see _NOT_POINTWISE):
# sdpametal / sdpafw transform only SdpaSfpuGolden.TRANSFORMED_COLS and pass the
# rest of the tile through, which is a column-index dependence.
_CORPUS_VEHICLES = [
    GoldenSpec(
        "sdpa",
        _sdpa_exp_unclamped,
        note=f"exp(x*{SDPA_EXP_SCALE}); the upper-unclamped 21f exp row, whole tile",
    ),
    GoldenSpec(
        "binopscalar",
        _binop_scalar_add,
        note=f"x + {BINOP_SCALAR_ADD} (MathOperation.ScalarAdd at the row's scalar)",
    ),
]

_CORPUS_COVERAGE = [
    GoldenSpec("addrsqrt-fresh", _blaze_addrsqrt, note=f"rsqrt(x+{ADD_RSQRT_EPS}); coverage vehicle"),
    GoldenSpec("smoothstep-fresh", _cov_smoothstep,
               note=f"t^2(3-2t) on t=clamp((x-{SMOOTHSTEP_EDGE0})*{SMOOTHSTEP_INV_DELTA},0,1)"),
    GoldenSpec("copydest-fresh", _identity, atol=0.0, rtol=0.0,
               note="identity move Dst tile 0 -> tile 1; EXACT (the row's own gate is atol=rtol=0)"),
]

# Rows whose test file now HAS the hook but which still carry no single-operand
# POINTWISE float contract. Listed by their exact corpus row name and with the
# mechanism, not as a category: "the file has no hook" has stopped being the
# reason, so the real reason has to be stated per row.
#
# Two distinct reasons, kept apart because they are not the same answer:
#   not-pointwise  out[i] is not a function of in[i] alone (neighbour element,
#                  row reduction, or the element's index), so no value-indexed
#                  band can express it.
#   exact-int      out[i] IS a function of in[i], but the contract is
#                  exact-integer. A bf16 ULP distance is the wrong metric for it
#                  (the same reason absint32/bitwisenot are already refused
#                  above); it wants an exact-int accumulator, not a float one.
_NOT_POINTWISE = {}
for _suffix in ("", "-t8", "-t32"):
    _NOT_POINTWISE["blaze-zeropad" + _suffix] = (
        "not-pointwise: the output depends on the element's ROW INDEX, not its "
        "value -- rows [24,32) are scrubbed to +0.0 -- so out[i] is not a function "
        "of in[i] and a value-indexed band cannot express it"
    )
    _NOT_POINTWISE["blaze-rope" + _suffix] = (
        "not-pointwise: intra-face pair rotation. out[2k] and out[2k+1] are a "
        "function of BOTH in[2k] and in[2k+1] and of the cos/sin tile in buffer_B "
        "-- two input elements and a second operand per output element"
    )
for _pool in ("max", "sum"):
    for _suffix in ("", "-t8", "-t32", "-cl", "-cl-t32", "-walk", "-walk-t32"):
        _NOT_POINTWISE[f"blaze-sdpareducerow-{_pool}{_suffix}"] = (
            f"not-pointwise: row reduction ({_pool.upper()} over 32 lanes) -- one "
            "output element is a function of 32 input elements"
        )
_NOT_POINTWISE.update(
    {
        "rotate90-fresh": "not-pointwise: out[i] = -in[i^1] (even i) / in[i^1] (odd i) "
        "-- the output reads its NEIGHBOUR element and flips sign on index parity",
        "tiledprod-fresh": "not-pointwise: running elementwise product down 9 vector "
        "rows -- out[r] is a function of in[0..r]",
        "zeropad-fresh": "not-pointwise: rows [24,32) are scrubbed to +0.0 by INDEX, "
        "so out[i] is not a function of in[i]",
        "customadd-fresh": "not-pointwise: genuinely two-operand (a+b with b a second "
        "full tile whose value varies per element index), so fixing b does not reduce "
        "it to one operand",
        "intsum-fresh": "not-pointwise: strided in-tile int32 row reductions -- out[i] "
        "is a sum of 8 (COL) or 4 (ROW) input rows",
        "blaze-sparsekfilter": "exact-int: pointwise, but the contract is an exact "
        "int32 bank-address filter (y = bank-hit ? (slot+1)<<shift : 0) graded at "
        "atol=rtol=0. A bf16 ULP distance is not its metric",
        "blaze-sparsekfilter-t8": "exact-int: see blaze-sparsekfilter",
        "blaze-sparsekfilter-t32": "exact-int: see blaze-sparsekfilter",
        "sparsekfilter-fresh": "exact-int: same bank-address filter on the coverage "
        "vehicle, graded at atol=rtol=0; exact-integer contract, not a float ULP one",
        "unarybitwise-fresh": "exact-int: y = x XOR 0x5A5A0FF0 on the int32 view. "
        "Exact by construction and pointwise, but a bit-pattern identity is not a "
        "float ULP surface (the absint32/bitwisenot precedent)",
    }
)

# The kernels' CLAIMED accuracy ranges, lifted from helpers/sfpu_domains.py
# _OP_DOMAIN_REGISTRY (the stimulus interval the harness itself grades each op
# over) and the kernel headers. REPORTING ONLY -- nothing in the ULP path reads
# it, and it is deliberately NOT GoldenSpec.domain.
#
# It exists so a stratum can be attributed honestly. Two of the five strata
# (+11.0 and +3.1965e38) sit outside almost every one of these intervals, and
# sfpu_domains.py says plainly of the gamma family that a probe at their
# boundary "produces a failure that is neither a bug nor fixable". A fit
# degrading where it never claimed anything is OUT-OF-CLAIM, not a defect; a
# wrong sign, a wrong constant, or a wrapped exponent is a defect wherever it
# lands -- which is why the erf finding at x = 11 was real.
CLAIMED_ACCURACY_DOMAIN: dict[str, tuple] = {
    "erf": (-3.0, 3.0),
    "erfc": (-3.0, 3.0),
    "erfinv": (-0.99, 0.99),
    "sigmoid": (-8.0, 8.0),
    "sigmoid-fresh": (-8.0, 8.0),
    "sigmoidlut-fresh": (-8.0, 8.0),
    "softsign": (-5.0, 5.0),
    "tanhderivative": (-5.0, 5.0),
    "tanhderivative-lut": (-3.0, 3.0),
    "tanhderivlut-fresh": (-3.0, 3.0),
    "heaviside": (-5.0, 5.0),
    "digamma": (0.1, 50.0),
    "digamma-fresh": (0.1, 50.0),
    "lgamma": (1.0, 15.0),
    "polygamma": (0.5, 10.0),
    "i0": (-3.75, 3.75),
    "i1": (-3.75, 3.75),
    "i1-fresh": (-3.75, 3.75),
    "expm1": (-5.0, 5.0),
    "expm1-fresh": (-5.0, 5.0),
    "expm1cw": (-5.0, 5.0),
    "expm1cw-fresh": (-5.0, 5.0),
    "cbrt": (-27.0, 27.0),
    "cbrt-fresh": (-27.0, 27.0),
    "mish": (-5.0, 5.0),
    "selu": (-5.0, 5.0),
    "softplus": (-5.0, 30.0),
    "softplus-fresh": (-5.0, 30.0),
    "xielu": (-5.0, 5.0),
    "xielu-fresh": (-5.0, 5.0),
    "unarypower": (-4.0, 4.0),
    "unarypower-fresh": (-4.0, 4.0),
    "sqrtcustom": (0.0, 100.0),
    "rsqrtcompat": (1e-2, 100.0),
    "fmod": (-5.0, 5.0),
    "fmod-fresh": (-5.0, 5.0),
    "remainder": (-5.0, 5.0),
    "remainder-fresh": (-5.0, 5.0),
    "clamp": (-2.0, 2.0),
    "clamp-fresh": (-2.0, 2.0),
    "hardtanh": (-2.0, 2.0),
    "hardtanh-fresh": (-2.0, 2.0),
    "hardmish": (-4.0, 4.0),
    "hardmish-fresh": (-4.0, 4.0),
    "hardshrink": (-4.0, 4.0),
    "hardshrink-fresh": (-4.0, 4.0),
    "softshrink": (-5.0, 5.0),
    "softshrink-fresh": (-5.0, 5.0),
    "softsign-fresh": (-5.0, 5.0),
    "prelu": (-5.0, 5.0),
    "identity": (-10.0, 10.0),
    "geluappx-fresh": (-5.0, 5.0),
    "rpow": (-10.0, 10.0),
    "rdiv": (-10.0, 10.0),
    "erf-fresh": (-3.0, 3.0),
    "erfc-fresh": (-3.0, 3.0),
    "erfinv-fresh": (-0.99, 0.99),
    # laneMU additions. Read out of the LIVE helpers.sfpu_domains
    # _OP_DOMAIN_REGISTRY (and for_op_pipeline for the four format-dependent ops
    # Exp/Exp2/Reciprocal/Square), not inferred from kernel headers -- this table
    # decides whether an out-of-contract stratum is written up as a defect or as
    # a fit degrading where it never promised anything, so it has to be the
    # interval the harness itself grades over.
    "abs": (-10.0, 10.0),                    # Abs
    "negative": (-10.0, 10.0),               # Neg
    "ceil-fresh": (-10.0, 10.0),             # Ceil
    "relu": (-5.0, 5.0),                     # ReluMax
    "unarymaxmin-max": (-5.0, 5.0),          # UnaryMax
    "unarymaxmin-min": (-5.0, 5.0),          # UnaryMin
    "threshold": (-5.0, 5.0),                # Threshold
    "threshold-fresh": (-5.0, 5.0),
    "threshold-fitted": (-5.0, 5.0),
    "fill": (0.0, 1.0),                      # Fill
    "fill-fresh": (0.0, 1.0),
    "activations": (-4.0, 4.0),              # Hardsigmoid
    "hardsigmoid-fresh": (-4.0, 4.0),
    "square": (-1000.0, 1000.0),             # Square (format-dependent -> same both)
    "square-fresh": (-1000.0, 1000.0),
    "tanh": (-5.0, 5.0),                     # Tanh
    "tanh-fresh": (-5.0, 5.0),
    "tanh-fitted": (-5.0, 5.0),
    "tanhlut-fresh": (-5.0, 5.0),            # Tanh (the LUT row's own mathop)
    "tanhshrink": (-5.0, 5.0),               # Tanhshrink
    "tanhshrink-fresh": (-5.0, 5.0),
    "tanhderivative-fitted": (-5.0, 5.0),    # TanhDerivative
    "sigmoid-fitted": (-8.0, 8.0),           # Sigmoid
    "sigmoidappx": (-5.0, 5.0),              # SigmoidAppx
    "sigmoidappx-tree": (-5.0, 5.0),
    "silu": (-5.0, 5.0),                     # Silu
    "silu-fresh": (-5.0, 5.0),
    "gelu": (-5.0, 5.0),                     # Gelu
    "gelu-fresh": (-5.0, 5.0),
    "gelu-fitted": (-5.0, 5.0),
    "gelu-licensed": (-5.0, 5.0),
    "exp": (-100.0, 16.0),                   # Exp (for_op_pipeline)
    "exp-fitted": (-100.0, 16.0),
    "exp2": (-100.0, 23.0),                  # Exp2 (for_op_pipeline)
    "exp2-fresh": (-100.0, 23.0),
    "expm1-fitted": (-5.0, 5.0),             # Expm1
    "celu": (-5.0, 5.0),                     # Celu
    "celu-fitted": (-5.0, 5.0),
    "elu": (-5.0, 5.0),                      # Elu
    "elu-fresh": (-5.0, 5.0),
    "elu-fitted": (-5.0, 5.0),
    "selu-fitted": (-5.0, 5.0),              # Selu
    "mish-fitted": (-5.0, 5.0),              # Mish
    "i0-fitted": (-3.75, 3.75),              # I0
    "i1-fitted": (-3.75, 3.75),              # I1
    "digamma-fitted": (0.1, 50.0),           # Digamma
    "polygamma-fitted": (0.5, 10.0),         # Polygamma
    "log": (1e-4, 1000.0),                   # Log
    "log-fresh": (1e-4, 1000.0),
    "log-fitted": (1e-4, 1000.0),
    "log1p": (-0.99, 10.0),                  # Log1p
    "log1p-fresh": (-0.99, 10.0),
    "log1p-fitted": (-0.99, 10.0),
    "sqrt": (0.0, 100.0),                    # Sqrt
    "sqrt-fresh": (0.0, 100.0),
    "rsqrt-fresh": (1e-4, 100.0),            # Rsqrt
    "rsqrt-fitted": (1e-4, 100.0),
    "acosh-fitted": (1.0, 10.0),             # Acosh
    "trigonometry": (1.0, 10.0),
    "trigonometry-fresh": (1.0, 10.0),
    "recip": (0.0, 1.0),                     # Reciprocal (for_op_pipeline)
    "recip-ilv2": (0.0, 1.0),
    "sigmoid-destacc": (-8.0, 8.0),   # Sigmoid
    "softplus-destacc": (-5.0, 30.0),  # Softplus
    "i0-destacc": (-3.75, 3.75),       # I0
    "i1-destacc": (-3.75, 3.75),       # I1
    "expm1cw-destacc": (-5.0, 5.0),    # Expm1Cw
    "sdpa": (-20.0, 0.0),          # the row's own swept input_range
    "binopscalar": (-1.0, 1.0),    # Elwadd's _OP_DOMAIN_REGISTRY interval
    # blaze / coverage rows: the interval each row's own StimuliSpec sweeps.
    "blaze-clampedsilu-gate": (-4.0, 4.0),
    "blaze-clampedsilu-gate-t8": (-4.0, 4.0),
    "blaze-clampedsilu-gate-t32": (-4.0, 4.0),
    "blaze-clampedsilu-up": (-4.0, 4.0),
    "blaze-clampedsilu-up-t8": (-4.0, 4.0),
    "blaze-clampedsilu-up-t32": (-4.0, 4.0),
    "blaze-clampedsilu-clamped": (-4.0, 4.0),
    "blaze-clampedsilu-clamped-t8": (-4.0, 4.0),
    "blaze-clampedsilu-clamped-t32": (-4.0, 4.0),
    "blaze-situ-gate": (-4.0, 4.0),
    "blaze-situ-gate-t8": (-4.0, 4.0),
    "blaze-situ-gate-t32": (-4.0, 4.0),
    "blaze-scaledtanh": (-4.0, 4.0),
    "blaze-scaledtanh-t8": (-4.0, 4.0),
    "blaze-scaledtanh-t32": (-4.0, 4.0),
    "blaze-logitsoftcap": (-4.0, 4.0),
    "blaze-logitsoftcap-t8": (-4.0, 4.0),
    "blaze-logitsoftcap-t32": (-4.0, 4.0),
    "blaze-siluscaled": (-4.0, 4.0),
    "blaze-siluscaled-t8": (-4.0, 4.0),
    "blaze-siluscaled-t32": (-4.0, 4.0),
    "blaze-addrsqrt": (0.05, 6.0),
    "blaze-addrsqrt-t8": (0.05, 6.0),
    "blaze-addrsqrt-t32": (0.05, 6.0),
    "blaze-sdpaexp": (-8.0, 0.0),
    "blaze-sdpaexp-t8": (-8.0, 0.0),
    "blaze-sdpaexp-t32": (-8.0, 0.0),
    "addrsqrt-fresh": (0.05, 6.0),
    "smoothstep-fresh": (-1.0, 1.0),
    "copydest-fresh": (-1.0, 1.0),
}


# The gamma family's poles are the non-positive INTEGERS. They need naming
# separately from a domain because torch does not signal them: trigamma(-1) comes
# back as 1.29e15 in fp32 and 6.58e32 in fp64 -- finite, precision-dependent
# noise, where the true value is +inf. So "the fp64 golden is non-finite" does NOT
# find these, and neither oracle is computing the function there.
#
# Excluded POINTWISE, deliberately. Excluding the negative half-line instead would
# license digamma(-1.5) = +0.703157, an ordinary defined value where the kernel
# returns -10.2929 -- a sign flip, and a real defect.
# `lgamma` is deliberately NOT here. The LLK node is the Stirling STAGE, whose
# argument is z = (x < 0.5) ? 1-x : x and is therefore always >= 0.5: the stage
# function is finite on the whole real line and has no pole to exclude. Excluding the
# non-positive integers for it would license the kernel at x = -1, -2, -3, ... where
# it has a perfectly ordinary answer. See _lgamma.
GAMMA_POLE_OPS = frozenset(
    {
        "digamma",
        "digamma-fresh",
        "digamma-fitted",
        "polygamma",
        "polygamma-fitted",
    }
)


def at_gamma_pole(op: str, x: float) -> bool:
    """True iff `op` is a gamma-family op and `x` is one of its poles."""
    if op not in GAMMA_POLE_OPS:
        return False
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(xf):
        return False
    return xf <= 0.0 and float(xf).is_integer()


# The rest of the registry's singular inputs, per op. Needed for the same reason
# GAMMA_POLE_OPS is: "the fp64 golden is non-finite" does NOT mean "undefined".
# I1(3.3e38) and expm1(3.3e38) are +inf in fp64 because they OVERFLOWED it, and
# +inf is then the correct fp32 answer -- a kernel returning a finite value or a
# -inf there is wrong, which is exactly the rpow/expm1cw signature. Only a genuine
# singular input is undefined, and those are enumerable.
RECIPROCAL_POLE_OPS = frozenset(
    {"rsqrtcompat", "rdiv", "recip", "recip-ilv2", "rsqrt-fresh", "rsqrt-fitted"}
)


def at_pole(op: str, x: float) -> bool:
    """True iff `x` is a singular input of `op` -- undefined, not merely huge."""
    if at_gamma_pole(op, x):
        return True
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return op in RECIPROCAL_POLE_OPS and xf == 0.0


def _pole_mask(op: str, values: np.ndarray) -> np.ndarray:
    """Vectorized at_pole over an array of fp32 inputs."""
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if op in GAMMA_POLE_OPS:
        gamma = finite & (x <= 0.0) & (x == np.floor(x))
    else:
        gamma = np.zeros(x.shape, dtype=bool)
    recip = (x == 0.0) if op in RECIPROCAL_POLE_OPS else np.zeros(x.shape, dtype=bool)
    return gamma | recip


def claim_status(op: str, x: float) -> str:
    """Is `x` inside the kernel's DOCUMENTED accuracy range? Reporting only.

    'no-claim' means no interval is recorded for the op -- which is a statement
    about this table, not a licence.
    """
    claim = CLAIMED_ACCURACY_DOMAIN.get(op)
    if claim is None:
        return "no-claim"
    lo, hi = claim
    return "in-claim" if lo <= x <= hi else "out-of-claim"


# Ops with no honest single torch reference on this streamer config (stated, not faked).
_UNSUPPORTED = {
    "absint32": "integer abs on the int32 view — golden is exact-int, not a float ULP; "
    "int32 leg checkable via kind=int32 (abs(int)) but not modeled here yet",
    "bitwisenot": "integer ~ on the int32 view — exact-int, not float ULP",
    "unaryshift-fresh": "integer shift — exact-int, not float ULP",
    "comp": "NotEqualZero predicate (0/1) — exact boolean, trivially correct; no ULP surface",
    "eqz-fresh": "EqualZero predicate (0/1) — exact boolean; no ULP surface",
    "unarycomp-fresh": "unary compare predicate (0/1) — exact boolean; no ULP surface",
    "castfp32tofp16a": "Float32/dest_acc=Yes cast (fp32 dst, distinct format path) — "
    "bit-lattice cast proven exact by laneCT 2^32; not a torch-ULP surface",
    # laneMT corpus extension: same reasons as their -fresh siblings above.
    "unarycomp": "UnaryGe predicate (0/1) — exact boolean; no ULP surface",
    "logicalnot": "LogicalNotUnary predicate (0/1) — exact boolean; no ULP surface",
    "isinfisnan": "Isinf predicate (0/1) — exact boolean; no ULP surface",
    "unaryshift": "integer LeftShift on the int32 view — exact-int, not float ULP",
}

REGISTRY: dict[str, GoldenSpec] = {}
for _s in (
    _DIVERGENT + _BITEXACT + _CORPUS_UNARY + _CORPUS_BF16 + _CORPUS_BLAZE
    + _CORPUS_COVERAGE + _CORPUS_VEHICLES + _CORPUS_DESTACC
):
    REGISTRY[_s.op] = _s
for _op, _why in _UNSUPPORTED.items():
    REGISTRY[_op] = GoldenSpec(
        _op, None, kind="unsupported", note=_why, checkable=False
    )
# A row with no pointwise contract is registered as such, with its mechanism, so
# `--golden <row>` answers "this op has no single-operand pointwise contract
# because <reason>" instead of "no golden registered", which reads as an omission.
for _op, _why in _NOT_POINTWISE.items():
    if _op in REGISTRY:
        raise RuntimeError(f"{_op} is both graded and refused")
    REGISTRY[_op] = GoldenSpec(
        _op,
        None,
        kind=("exact-int" if _why.startswith("exact-int") else "not-pointwise"),
        note=_why,
        checkable=False,
    )
if {op for op, spec in REGISTRY.items() if spec.domain is not None} != set(
    UNARY_DOMAIN_PARTITION_OPS
):
    raise RuntimeError("golden registry/domain class vocabulary drift")


def get_spec(op: str) -> Optional[GoldenSpec]:
    return REGISTRY.get(op)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming correctness accumulator — one per leg, folded chunk by chunk, no retention.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorrectnessAccumulator:
    spec: GoldenSpec
    # Element width of the raw band the device consumed / produced. 4 = the
    # original fp32/int32 leg, where the raw u32 is bf16-TRUNCATED on the way
    # into a 16-bit DEST. 2 = the 16-bit band mode, where the raw u16 IS the
    # bf16 code the SFPU receives -- so a 65536-pattern band is the row's whole
    # input space, not one value of it. The golden output pipeline is the same
    # either way: UnarySFPUGolden.__call__ reaches bf16-round -> NaN->+inf ->
    # _apply_ftz(2^-126) for (Float16_b, Float16_b, dest_acc=No) exactly as it
    # does for (Float32, Float32, dest_acc=No), because dst_format is Float16_b
    # in both and _FTZ_THRESHOLD is keyed on the OUTPUT format.
    in_bytes: int = 4
    out_bytes: int = 4
    patterns: int = 0
    max_ulp: float = 0.0
    max_ulp_input_u32: int = -1
    n_out_of_tol: int = 0
    first_witness_u32: int = -1
    first_witness_class: str = ""
    first_witness_dev: float = 0.0
    first_witness_golden: float = 0.0
    # The same three numbers restricted to GRADED inputs: inside spec.domain and
    # not at a pole. On a 1-value stratum this is the same as the global count; on
    # an exhaustive 65536-pattern band it is the difference between "15982 out of
    # tolerance" and "15982 out of tolerance, every one of them at an input where
    # acosh is undefined". A licensed miss must not read as a defect, and a real
    # defect must not hide behind thousands of licensed ones.
    n_out_graded: int = 0
    max_ulp_graded: float = 0.0
    max_ulp_graded_input: int = -1
    graded_witness_u32: int = -1
    graded_witness_class: str = ""
    graded_witness_dev: float = 0.0
    graded_witness_golden: float = 0.0
    n_graded: int = 0
    # IN-CLAIM: graded AND inside the kernel's DOCUMENTED accuracy range
    # (CLAIMED_ACCURACY_DOMAIN). On an exhaustive band this is the number that
    # decides the headline: a fitted body degrading at x = 1e30 never promised
    # anything there, but the same body wrong at x = 3.2 inside its declared
    # [1, 10] is a defect. Reporting only in the sense that no gate reads it --
    # but it is the difference between "out-of-claim" and "defect" in the writeup,
    # and one global counter cannot express it.
    # Out-of-tolerance inputs whose device answer IS the golden evaluated at the
    # FTZ-flushed operand -- i.e. explained by the oracles not modelling input FTZ
    # rather than by the kernel being wrong.
    n_out_ftz_explained: int = 0
    # Out-of-tolerance inputs explained by the packer's NaN->inf being
    # SIGN-PRESERVING where the golden's convert_nan_to_inf hardcodes +inf.
    n_out_nan_sign_explained: int = 0
    # ...and by the kernel not PROPAGATING NaN at all (it returns an ordinary
    # value where both oracles say NaN). Distinct from the sign question.
    n_out_nan_nonprop: int = 0
    # Patterns where the NAN-OPERAND INFINITY SIGN POLICY was the reason a pair of
    # opposite-signed infinities was accepted. Reported so the policy is never silent.
    n_nan_inf_sign_policy: int = 0
    n_in_claim: int = 0
    n_out_in_claim: int = 0
    max_ulp_in_claim: float = 0.0
    claim_witness_u32: int = -1
    claim_witness_dev: float = 0.0
    claim_witness_golden: float = 0.0
    # tanhderiv extra: distance to the TRUE math (sech^2), reported alongside the LUT contract.
    max_ulp_true: float = 0.0
    class_ulp: dict[str, tuple[int, float]] = field(default_factory=dict)

    def _classify(self, u32: int, xf: float) -> str:
        del u32  # class semantics are explicitly post-bf16-truncation
        one = np.array([xf], dtype=np.float32)
        return next(
            name
            for name, mask in unary_input_classes(one, self.spec.domain).items()
            if bool(mask[0])
        )

    def update(self, chunk_start: int, valid_count: int, dev_bytes: bytes) -> None:
        """Fold one chunk: dev_bytes = valid_count fp32 little-endian device outputs."""
        with np.errstate(all="ignore"):
            self._update(chunk_start, valid_count, dev_bytes)

    def _update(self, chunk_start: int, valid_count: int, dev_bytes: bytes) -> None:
        if valid_count <= 0:
            return
        u32 = np.arange(chunk_start, chunk_start + valid_count, dtype=np.uint32)
        if self.in_bytes == 2:
            # The raw pattern IS the bf16 code; no truncation to model.
            xin = _bf16_bits_to_f32(u32 & np.uint32(0xFFFF))
        else:
            xin = bf16_truncate(u32)  # what the SFPU actually sees
        # INPUT FTZ. The SFPU never receives a subnormal operand, so the golden must
        # not evaluate at one. `xin` stays the delivered bf16 pattern (the input
        # CLASSIFICATION is about what was delivered); `xop` is what the datapath
        # computes on. See input_ftz for the per-op silicon evidence.
        xop = xin if self.spec.op in INPUT_FTZ_EXEMPT else input_ftz(xin)
        hp = self.spec.evaluate(xin)  # fp64 true math at the operand the SFPU sees
        fmt = format_golden_f32_acc if self.spec.dst_acc else format_golden_f32_noacc
        golden = fmt(hp)  # fp32-container reference
        if self.out_bytes == 2:
            dev = _bf16_bits_to_f32(
                np.frombuffer(dev_bytes[: valid_count * 2], dtype="<u2").astype(
                    np.uint32
                )
            ).astype(np.float32)
        else:
            dev = np.frombuffer(dev_bytes[: valid_count * 4], dtype="<f4").astype(
                np.float32
            )

        # The golden infinities that are CONVERTED NaNs, not computed overflows.
        nan_src = np.isnan(np.asarray(hp, dtype=np.float64))
        ulp, within = numeric_comparison(
            golden, dev, self.spec.atol, self.spec.rtol, nan_source=nan_src
        )
        gd = golden.astype(np.float64)
        dd = dev.astype(np.float64)
        self.n_nan_inf_sign_policy += int(
            np.count_nonzero(
                nan_src
                & np.isinf(gd)
                & np.isinf(dd)
                & (np.signbit(gd) != np.signbit(dd))
            )
        )
        classes = unary_input_classes(xin, self.spec.domain)
        _fold_class_ulps(self.class_ulp, ulp, classes)
        out = ~within

        # GRADED mask: where the function is defined. Out-of-domain normals are
        # excluded by the class partition; poles are excluded POINTWISE (the
        # non-positive integers for the gamma family, 0 for the reciprocals) --
        # never by excluding a half-line, which would license digamma(-1.5).
        # A non-finite fp64 golden is NOT a pole and is NOT excluded: I1(3.3e38)
        # is +inf because it overflowed fp64, so +inf is the correct answer there
        # and a finite device result is a defect.
        graded = np.ones(xin.shape, dtype=bool)
        if self.spec.domain is not None:
            graded &= ~classes["out_of_domain_finite_normal"]
        if self.spec.op in GAMMA_POLE_OPS or self.spec.op in RECIPROCAL_POLE_OPS:
            graded &= ~_pole_mask(self.spec.op, xin)
        self.n_graded += int(np.count_nonzero(graded))
        g_ulp = np.where(graded, ulp, -1.0)
        if g_ulp.size:
            gi = int(np.argmax(np.where(np.isfinite(g_ulp), g_ulp, -1.0)))
            if graded[gi] and ulp[gi] > self.max_ulp_graded:
                self.max_ulp_graded = float(ulp[gi])
                self.max_ulp_graded_input = int(u32[gi])
        # FALSIFICATION probe for the input-FTZ model, not an excuse for a miss. The
        # golden above already evaluates at the flushed operand; this counts the
        # RESIDUAL out-of-tolerance patterns at a subnormal operand that the UNFLUSHED
        # golden would have explained instead. Nonzero means this op does NOT flush its
        # input and the model is wrong for it. Only subnormal operands can be affected,
        # so this is cheap.
        sub_in = (np.abs(xin.astype(np.float64)) < BF16_TINY) & (xin != 0)
        if np.any(out & sub_in):
            idx = np.flatnonzero(out & sub_in)
            raw_golden = fmt(self.spec.math(xin[idx]))  # deliberately UNflushed
            _, raw_within = numeric_comparison(
                raw_golden, dev[idx], self.spec.atol, self.spec.rtol
            )
            self.n_out_ftz_explained += int(np.count_nonzero(raw_within))

        # RESIDUAL NaN accounting. format_golden_f32_noacc now converts a NaN to a
        # SIGN-PRESERVED infinity, and the min/max bodies now model the SFPU's
        # sign-magnitude total order, so both of these count what is LEFT.
        #
        # (a) OPERAND-SIGN residual. The golden's NaN carries the sign its own math
        #     produced (identity keeps it, negation flips it, np.abs CLEARS it). This
        #     counts the misses a golden that instead forced the OPERAND's sign would
        #     have explained. It is nonzero exactly where the kernel fails to
        #     canonicalize a NaN that the mathematical function does canonicalize --
        #     `abs`, where the device returns -inf for a negative NaN although |x| can
        #     never be negative. That is a kernel deviation, recorded here, NOT
        #     licensed: the golden stays strict.
        # (b) NON-PROPAGATION. tanh(nan) -> 1.0, threshold(nan) -> 10.0: the kernel
        #     returns an ordinary value where the golden says NaN. A kernel semantics
        #     question this records rather than settles. min/max are no longer counted
        #     here, because their NaN behaviour is now MODELLED rather than attributed.
        nan_in = np.isnan(xin.astype(np.float64))
        if np.any(out & nan_in):
            idx = np.flatnonzero(out & nan_in)
            hp_n = np.asarray(self.spec.evaluate(xin[idx]), dtype=np.float64)
            signed = np.where(
                np.isnan(hp_n), np.copysign(np.inf, xin[idx].astype(np.float64)), hp_n
            )
            g_signed = np.asarray(signed, dtype=np.float32)
            _, ok_sign = numeric_comparison(
                g_signed, dev[idx], self.spec.atol, self.spec.rtol
            )
            self.n_out_nan_sign_explained += int(np.count_nonzero(ok_sign))
            self.n_out_nan_nonprop += int(
                np.count_nonzero(~ok_sign & np.isnan(hp_n) & np.isfinite(dev[idx]))
            )

        claim = CLAIMED_ACCURACY_DOMAIN.get(self.spec.op)
        if claim is not None:
            lo, hi = claim
            in_claim = graded & (xin >= lo) & (xin <= hi)
        else:
            in_claim = np.zeros(xin.shape, dtype=bool)
        self.n_in_claim += int(np.count_nonzero(in_claim))
        c_out = out & in_claim
        n_c = int(np.count_nonzero(c_out))
        if in_claim.any():
            ci = int(np.argmax(np.where(in_claim & np.isfinite(ulp), ulp, -1.0)))
            if in_claim[ci] and ulp[ci] > self.max_ulp_in_claim:
                self.max_ulp_in_claim = float(ulp[ci])
        if n_c:
            self.n_out_in_claim += n_c
            if self.claim_witness_u32 < 0:
                c = int(np.argmax(c_out))
                self.claim_witness_u32 = int(u32[c])
                self.claim_witness_dev = float(dev[c])
                self.claim_witness_golden = float(golden[c])

        g_out = out & graded
        n_g = int(np.count_nonzero(g_out))
        if n_g:
            self.n_out_graded += n_g
            if self.graded_witness_u32 < 0:
                k = int(np.argmax(g_out))
                self.graded_witness_u32 = int(u32[k])
                self.graded_witness_class = self._classify(int(u32[k]), float(xin[k]))
                self.graded_witness_dev = float(dev[k])
                self.graded_witness_golden = float(golden[k])

        # running max ULP + its input
        if ulp.size:
            i = int(np.nanargmax(np.where(np.isfinite(ulp), ulp, -1.0)))
            if ulp[i] > self.max_ulp:
                self.max_ulp = float(ulp[i])
                self.max_ulp_input_u32 = int(u32[i])

        # out-of-tolerance count + first witness
        n_out = int(np.count_nonzero(out))
        if n_out:
            self.n_out_of_tol += n_out
            if self.first_witness_u32 < 0:
                j = int(np.argmax(out))
                self.first_witness_u32 = int(u32[j])
                self.first_witness_class = self._classify(int(u32[j]), float(xin[j]))
                self.first_witness_dev = float(dev[j])
                self.first_witness_golden = float(golden[j])

        # tanhderiv: also track distance to the TRUE sech^2 (reporting the licensed gap)
        if self.spec.op == "tanhderivlut-fresh":
            true_hp = _tanh_derivative_true(xop)
            true_g = format_golden_f32_noacc(true_hp)
            ulp_true = bf16_bitdistance(true_g, dev)
            if ulp_true.size:
                m = float(np.nanmax(np.where(np.isfinite(ulp_true), ulp_true, -1.0)))
                if m > self.max_ulp_true:
                    self.max_ulp_true = m

        self.patterns += valid_count

    def result_line(self, leg: str) -> str:
        w = self.first_witness_u32
        extra = (
            f",max_ulp_true_sech2={self.max_ulp_true:.0f}"
            if self.spec.op == "tanhderivlut-fresh"
            else ""
        )
        return (
            f"SFPU_CORRECTNESS,leg={leg},op={self.spec.op},patterns={self.patterns},"
            f"max_bf16_ulp={self.max_ulp:.0f},max_ulp_input=0x{max(self.max_ulp_input_u32,0):08x},"
            f"in_bytes={self.in_bytes},out_bytes={self.out_bytes},"
            f"n_out_of_tol={self.n_out_of_tol},"
            f"within_contract={self.n_out_of_tol == 0},"
            f"first_witness=0x{max(w,0):08x},first_witness_class={self.first_witness_class or '-'},"
            f"witness_dev={self.first_witness_dev!r},witness_golden={self.first_witness_golden!r},"
            f"n_graded={self.n_graded},n_out_graded={self.n_out_graded},"
            f"n_out_ftz_explained={self.n_out_ftz_explained},"
            f"n_out_nan_sign_explained={self.n_out_nan_sign_explained},"
            f"n_out_nan_nonprop={self.n_out_nan_nonprop},"
            f"n_nan_inf_sign_policy={self.n_nan_inf_sign_policy},"
            f"max_ulp_graded={self.max_ulp_graded:.0f},"
            f"max_ulp_graded_input=0x{max(self.max_ulp_graded_input,0):08x},"
            f"graded_witness=0x{max(self.graded_witness_u32,0):08x},"
            f"graded_witness_class={self.graded_witness_class or '-'},"
            f"graded_witness_dev={self.graded_witness_dev!r},"
            f"graded_witness_golden={self.graded_witness_golden!r},"
            f"n_in_claim={self.n_in_claim},n_out_in_claim={self.n_out_in_claim},"
            f"max_ulp_in_claim={self.max_ulp_in_claim:.0f},"
            f"claim_witness=0x{max(self.claim_witness_u32,0):08x},"
            f"claim_witness_dev={self.claim_witness_dev!r},"
            f"claim_witness_golden={self.claim_witness_golden!r},"
            f"atol={self.spec.atol},rtol={self.spec.rtol},"
            "zero_sign_policy=tolerance_equal_not_bitexact,"
            "nan_inf_sign_policy=not_a_numeric_quantity_for_a_converted_nan,"
            f"class_ulp={format_class_ulp(self.class_ulp)}{extra}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# binarypow (laneMQ two-operand streamer) — the highest-interest divergent op.
# Joint J in [0,2^32): base16 = J>>16, exp16 = J&0xFFFF (raw bf16 patterns). The
# device writes pow(base, exp) to the EVEN tile of each tile pair (sfpu_binary_test.cpp
# `call(tile, tile+1, tile)`); odd tiles stay the 0xA5 clear sentinel. Output Float16_b
# (2 bytes), dest_acc=No, contract atol=rtol=0.05 (passed_test Float16_b default).
# Golden mirrors BinarySFPUGolden._pow: (base_fp32 ** exp_fp32) -> bf16.
# ─────────────────────────────────────────────────────────────────────────────
BINARY_POW_ATOL = 0.05
BINARY_POW_RTOL = 0.05
_ELEMS_PER_TILE = 1024

# Dispatch constants for the binary rows, from the ISCLOSE dispatch bit patterns
# and torch's own defaults, which the kernel hard-codes.
ISCLOSE_RTOL = 1e-5
ISCLOSE_ATOL = 1e-8


@dataclass
class BinaryGoldenSpec:
    """A two-operand true-math golden over the joint bf16 x bf16 space.

    `math(a, b)` takes two fp64 arrays of the EXACT bf16 values the kernel
    receives (raw patterns, so subnormals/NaN payloads are delivered) and returns
    fp64 true math. Same contract as the unary GoldenSpec; the difference is only
    the arity, so the ULP/class/tolerance machinery below is shared verbatim.
    """

    op: str
    math: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]
    atol: float = 0.05
    rtol: float = 0.05
    note: str = ""
    checkable: bool = True
    kind: str = "bf16_joint"


def _b_pow(a, b):
    with np.errstate(all="ignore"):
        return np.power(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _b_sub(a, b):
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)


def _b_eq(a, b):
    return (np.asarray(a, dtype=np.float64) == np.asarray(b, dtype=np.float64)).astype(
        np.float64
    )


def _b_fmod(a, b):
    with np.errstate(all="ignore"):
        return _np(torch.fmod(_t(a), _t(b)))


def _b_remainder(a, b):
    with np.errstate(all="ignore"):
        return _np(torch.remainder(_t(a), _t(b)))


def _b_atan2(a, b):
    # calculate_sfpu_atan2 computes atan2(in0, in1) = atan2(y, x), y = operand A.
    return _np(torch.atan2(_t(a), _t(b)))


def _b_max(a, b):
    # SFPU compare order, not np.maximum -- see sfpu_max.
    return sfpu_max(a, b)


def _b_min(a, b):
    return sfpu_min(a, b)


def _b_isclose(a, b):
    af = np.asarray(a, dtype=np.float64)
    bf = np.asarray(b, dtype=np.float64)
    # torch.isclose semantics exactly (equal_nan=False): the tolerance test applies
    # ONLY to two finite values. Same-signed infinities are close; every other pair
    # involving a non-finite value is not. Writing it as a bare
    # `|a-b| <= atol + rtol*|b|` gets inf-vs-0 wrong, because rtol*inf is inf and
    # `inf <= inf` is True -- which would report 0.0 as close to +inf.
    both_finite = np.isfinite(af) & np.isfinite(bf)
    with np.errstate(all="ignore"):
        close = both_finite & (
            np.abs(af - bf) <= (ISCLOSE_ATOL + ISCLOSE_RTOL * np.abs(bf))
        )
    both_inf = np.isinf(af) & np.isinf(bf) & (np.signbit(af) == np.signbit(bf))
    return (close | both_inf).astype(np.float64)


def _b_mask(a, b):
    # calculate_mask: data (A) passes through where the mask (B) is nonzero, else 0.
    af = np.asarray(a, dtype=np.float64)
    return np.where(np.asarray(b, dtype=np.float64) != 0.0, af, af * 0.0)


# The 12 joint-pointwise Float16_b binary corpus rows. `logsigmoid` is NOT here
# and must not be: its two operands are PAIRED -- the kernel takes exp(-x) as its
# second operand and the test bakes that into the stimuli -- so a band that sweeps
# B independently of A violates the kernel's own precondition and would grade it
# against inputs it is not defined for.
BINARY_REGISTRY: dict[str, BinaryGoldenSpec] = {
    spec.op: spec
    for spec in [
        BinaryGoldenSpec("binarypow", _b_pow, note="a**b (BinarySFPUGolden._pow)"),
        BinaryGoldenSpec("binarypow-fresh", _b_pow, note="a**b; fresh_cpp arm"),
        BinaryGoldenSpec("binary-float", _b_sub, note="a-b (SfpuElwsub)"),
        BinaryGoldenSpec("binarycomp", _b_eq, note="1.0 if a==b else 0.0 (SfpuElwEq)"),
        BinaryGoldenSpec("binaryfmod", _b_fmod, note="fmod(a,b) -- sign of a"),
        BinaryGoldenSpec(
            "binaryremainder", _b_remainder, note="remainder(a,b) -- sign of b"
        ),
        BinaryGoldenSpec("atan2", _b_atan2, note="atan2(a,b) = atan2(y,x)"),
        BinaryGoldenSpec("atan2-fitted", _b_atan2, note="atan2(a,b); fitted_cpp arm"),
        BinaryGoldenSpec("minmax-max", _b_max, note="max(a,b)"),
        BinaryGoldenSpec("minmax-min", _b_min, note="min(a,b)"),
        BinaryGoldenSpec(
            "isclose",
            _b_isclose,
            note=f"|a-b| <= {ISCLOSE_ATOL}+{ISCLOSE_RTOL}|b| -> 1.0/0.0",
        ),
        BinaryGoldenSpec("isclose-fresh", _b_isclose, note="isclose; fresh_cpp arm"),
        BinaryGoldenSpec(
            "mask", _b_mask, note="data A passes through where mask B != 0, else 0"
        ),
    ]
}


# The binary corpus rows that a joint bf16 band cannot grade, each with its own
# reason. Registered rather than absent, so `--golden <row>` says why.
#
# Fourteen of the fifteen 32-bit binary rows are EXACT-INTEGER. That is not a
# harness gap that a wider band would close, and it is worth being precise about
# why, because "run it anyway" is the wrong instinct here:
#   * the joint input space of two Int32 operands is 2^64, not 2^32. The joint
#     index the binary streamer enumerates is base16<<16 | exp16 -- sixteen bits
#     per operand -- and there is no 32-bit index that addresses a 2^64 space, so
#     an exhaustive sweep does not merely cost more, it is not expressible.
#   * and there is nothing for a ULP band to find. Every one of them (gcd, lcm,
#     mulint32, the shifts, the int adds/subs, divint32floor, rsubint32) is an
#     EXACT integer function: no polynomial, no LUT, no bounded range, no setexp
#     exponent-field write. A bf16 ULP distance is not their metric, and their
#     correctness question -- "is the integer arithmetic right" -- is an
#     exact-equality question that the 2x2 correctness node already answers and
#     that formal_equiv is the right tool for. Manufacturing an ULP test for
#     `a & b` would be inventing coverage, not adding it.
BINARY_UNREACHABLE = {
    "logsigmoid": "paired operands: the kernel takes exp(-x) as its SECOND operand "
    "and the test bakes that into the stimuli, so a joint band that varies B "
    "independently of A feeds the kernel a pair it is not defined for. Its golden "
    "reads only A -- there is no two-operand contract to grade",
    "binary-bcast": "broadcast row (bcast_dim=ROW): the second operand is a "
    "replicated row, so out[i] = f(a[i], b[row(i)]) depends on the element's "
    "position and the joint (a,b) index does not describe it",
    "addint": "exact-integer (SfpuElwadd on Int32): joint space 2^64, no "
    "approximation surface, exact-equality contract",
    "subint": "exact-integer (SfpuElwsub on Int32): as addint",
    "binarybitwise": "exact-integer (SfpuBitwiseAnd on Int32): a bit-pattern "
    "identity, not a float ULP surface",
    "mulint32": "exact-integer (SfpuMulInt32): joint space 2^64, exact-equality "
    "contract",
    "mulint32-fresh": "exact-integer (SfpuMulInt32, fresh arm): as mulint32",
    "divint32floor": "exact-integer (SfpuDivInt32Floor): exact floor division, no "
    "approximation surface",
    "divint32floor-fresh": "exact-integer (SfpuDivInt32Floor, fresh arm)",
    "gcd": "exact-integer (SfpuGcd): a number-theoretic function with no "
    "polynomial/LUT/bounded surface; exact-equality contract",
    "gcd-fresh": "exact-integer (SfpuGcd, fresh arm)",
    "lcm": "exact-integer (SfpuLcm): as gcd",
    "lcm-fresh": "exact-integer (SfpuLcm, fresh arm)",
    "leftshift-fresh": "exact-integer (SfpuElwLeftShift): a bit-pattern identity",
    "shift": "exact-integer (SfpuElwRightShift): a bit-pattern identity",
    "rsubint32": "exact-integer (SfpuRsubInt32): exact integer subtraction",
}
for _op, _why in BINARY_UNREACHABLE.items():
    if _op in BINARY_REGISTRY:
        raise RuntimeError(f"{_op} is both graded and refused")
    BINARY_REGISTRY[_op] = BinaryGoldenSpec(
        _op,
        None,
        note=_why,
        checkable=False,
        kind=("exact-int" if _why.startswith("exact-integer") else "not-pointwise"),
    )


def get_binary_spec(op: str) -> Optional[BinaryGoldenSpec]:
    return BINARY_REGISTRY.get(op)


def binary_pow_golden_bf16(base16: np.ndarray, exp16: np.ndarray) -> np.ndarray:
    """pow(base, exp) per BinarySFPUGolden._pow: (a_fp32 ** b_fp32) rounded to bf16, as fp32."""
    a = _bf16_bits_to_f32(base16).astype(np.float64)
    b = _bf16_bits_to_f32(exp16).astype(np.float64)
    with np.errstate(all="ignore"):
        hp = np.power(
            a, b
        )  # fp64 pow (a,b are exact bf16 values); matches (fp32**fp32) target
    return _round_bf16_as_f32(hp).astype(np.float32)


@dataclass
class BinaryPowAccumulator:
    """Streaming correctness for a binary row: device even-tile outputs vs true math.

    Generalized from the binarypow-only form: `spec` carries the two-operand
    golden, so every joint-pointwise binary row rides the same accumulator instead
    of one class per op. Left unset it keeps the original pow behaviour, so the
    existing binarypow callers and the selftest are unchanged.

    Output always lives in the EVEN tile of each pair -- every binary op in
    sources/sfpu_binary_test.cpp is dispatched as `call(tile, tile + 1, tile)`.
    """

    spec: Optional[BinaryGoldenSpec] = None
    atol: float = BINARY_POW_ATOL
    rtol: float = BINARY_POW_RTOL
    joints: int = 0
    max_ulp: float = 0.0
    max_ulp_joint: int = -1
    n_out_of_tol: int = 0
    first_witness_joint: int = -1
    first_witness_class: str = ""
    first_witness_dev: float = 0.0
    first_witness_golden: float = 0.0
    class_ulp: dict[str, tuple[int, float]] = field(default_factory=dict)

    def update(
        self, dispatch_start: int, pairs: int, result_region_bytes: bytes
    ) -> None:
        with np.errstate(all="ignore"):
            self._update(dispatch_start, pairs, result_region_bytes)

    def __post_init__(self) -> None:
        if self.spec is not None:
            self.atol = self.spec.atol
            self.rtol = self.spec.rtol

    @property
    def op(self) -> str:
        return self.spec.op if self.spec is not None else "binarypow"

    def _golden_hp(self, base16: np.ndarray, exp16: np.ndarray) -> np.ndarray:
        """The PRE-conversion high-precision golden, so a caller can tell a converted
        NaN from a computed overflow (see numeric_comparison)."""
        a = _bf16_bits_to_f32(base16).astype(np.float64)
        b = _bf16_bits_to_f32(exp16).astype(np.float64)
        with np.errstate(all="ignore"):
            if self.spec is None:
                return np.asarray(a**b, dtype=np.float64)
            return np.asarray(self.spec.math(a, b), dtype=np.float64)

    def _golden(self, base16: np.ndarray, exp16: np.ndarray) -> np.ndarray:
        if self.spec is None:
            return binary_pow_golden_bf16(base16, exp16)
        with np.errstate(all="ignore"):
            hp = self._golden_hp(base16, exp16)
        return format_golden_f32_noacc(hp).astype(np.float32)

    def _update(self, dispatch_start: int, pairs: int, res: bytes) -> None:
        tile_bytes = _ELEMS_PER_TILE * 2  # bf16 tile = 1024 * 2 bytes
        for p in range(pairs):
            joint0 = dispatch_start + p * _ELEMS_PER_TILE
            base16 = (joint0 >> 16) & 0xFFFF
            lo = joint0 & 0xFFFF
            exp16 = (np.arange(_ELEMS_PER_TILE, dtype=np.uint32) + lo).astype(np.uint16)
            base_arr = np.full(_ELEMS_PER_TILE, base16, dtype=np.uint16)
            golden = self._golden(base_arr, exp16)  # fp32 (bf16-valued)

            even_off = (2 * p) * tile_bytes  # output lives in the EVEN tile of the pair
            dev16 = np.frombuffer(res[even_off : even_off + tile_bytes], dtype="<u2")
            if dev16.size < _ELEMS_PER_TILE:
                raise ValueError(
                    f"binarypow dispatch@{dispatch_start} pair {p}: short result region"
                )
            dev = _bf16_bits_to_f32(dev16.astype(np.uint32)).astype(np.float32)

            base_values = _bf16_bits_to_f32(base_arr)
            exp_values = _bf16_bits_to_f32(exp16.astype(np.uint32))
            # A converted NaN's infinity has a non-numeric sign; see numeric_comparison.
            nan_src = np.isnan(
                np.asarray(self._golden_hp(base_arr, exp16), dtype=np.float64)
            )
            ulp, within = numeric_comparison(
                golden, dev, self.atol, self.rtol, nan_source=nan_src
            )
            _fold_class_ulps(
                self.class_ulp,
                ulp,
                binary_input_classes(base_values, exp_values),
            )
            out = ~within

            i = int(np.nanargmax(np.where(np.isfinite(ulp), ulp, -1.0)))
            if ulp[i] > self.max_ulp:
                self.max_ulp = float(ulp[i])
                self.max_ulp_joint = joint0 + i
            n_out = int(np.count_nonzero(out))
            if n_out:
                self.n_out_of_tol += n_out
                if self.first_witness_joint < 0:
                    j = int(np.argmax(out))
                    self.first_witness_joint = joint0 + j
                    one_base = base_values[j : j + 1]
                    one_exp = exp_values[j : j + 1]
                    self.first_witness_class = next(
                        name
                        for name, mask in binary_input_classes(one_base, one_exp).items()
                        if bool(mask[0])
                    )
                    self.first_witness_dev = float(dev[j])
                    self.first_witness_golden = float(golden[j])
            self.joints += _ELEMS_PER_TILE

    def result_line(self, leg: str) -> str:
        w = max(self.first_witness_joint, 0)
        return (
            f"SFPU_CORRECTNESS,leg={leg},op={self.op},joints={self.joints},"
            f"max_bf16_ulp={self.max_ulp:.0f},max_ulp_joint=0x{max(self.max_ulp_joint,0):08x},"
            f"n_out_of_tol={self.n_out_of_tol},within_contract={self.n_out_of_tol == 0},"
            f"first_witness=0x{w:08x},first_witness_class={self.first_witness_class or '-'},"
            f"witness_dev={self.first_witness_dev!r},witness_golden={self.first_witness_golden!r},"
            f"atol={self.atol},rtol={self.rtol},"
            "zero_sign_policy=tolerance_equal_not_bitexact,"
            "nan_inf_sign_policy=not_a_numeric_quantity_for_a_converted_nan,"
            f"class_ulp={format_class_ulp(self.class_ulp)}"
        )

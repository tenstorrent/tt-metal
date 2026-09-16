#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ttpoly.spec.units — accuracy metrics, rounding-mode/precision enums, FTZ policy.

THE TWO STANDARD HEADLINE METRICS (per the literature + TT's own
ttnn-eltwise-op-tester + numpy/pytorch)
================================================================================
1. **PURE ULP** — :func:`ulp_error_pure` / :func:`max_ulp_pure`. The absolute
   error from the FTZ'd high-precision golden to the target-precision result,
   scaled by the spacing at the FTZ'd, target-precision golden:
   ``|FTZ(true) - RN_p(comp)| / ulp_p(FTZ(RN_p(true)))``. This matches TT's
   ttnn-eltwise-op-tester, including its FP32 arithmetic for the final error and
   quotient, so a correctly rounded result can measure up to 0.5 ULP rather
   than being forced to zero. KNOWN CAVEAT: it inflates negligible-magnitude
   errors at zeros/roots (a tiny absolute error divided by the tiny spacing near
   zero reads as a large ULP). This is a shared property of native TTNN's metric,
   not an artifact unique to us — it is documented, not hidden.
2. **ML-tolerance pass-rate** — :func:`ml_pass_rate`. Percentage of points with
   ``|err| <= atol + rtol*|true|`` (``atol = rtol = 1e-3``, the bf16
   pytorch/numpy default). The honest ML-accuracy metric; robust to the near-zero
   pure-ULP inflation above.

These two are the HEADLINE numbers. The beta-floored ``ulp_error`` below is a
LABELED, NON-HEADLINE helper kept for continuity — it is NOT "the ULP". Do not
report it as the canonical ULP; prefer PURE ULP + ML-pass for any headline.

----------------------------------------------------------------------------
THE BETA-FLOORED ULP HELPER (non-headline; kept, not deleted)
----------------------------------------------------------------------------
``ulp_error`` is the *signed total-order bit-distance* between the
correctly-rounded reference and the computed value, measured in units of the
function's own output resolution. One principled definition, correct for BOTH
positive-steep functions (acosh/lgamma, where naive *spacing* explodes near a
root) AND sign-crossing functions (swish/mish/gelu, where naive *bit-distance*
explodes at a zero crossing). No per-function switching, no ``min(spacing,
bitdistance)``, no masking that hides a real error.

Definition (per point):

    ref   = FTZ( RN_bf16(y_true) )            # correctly-rounded golden, FTZ'd
    comp  = FTZ( bf16(y_comp)   )             # silicon output, FTZ'd
    scale = max(|ref|, |comp|)                # "ULP of the result" convention
    beta  = spacing_bf16( max|ref| over domain )   # one bf16 step at output scale
    ULP   = |ref - comp| / max( spacing_bf16(scale), beta )

Why this is well-defined, sign-correct, and root-stable
-------------------------------------------------------
* **Sign-correct.** ``|ref - comp|`` with ``scale = max(|ref|,|comp|)`` is the
  textbook Goldberg "ULP of the result". A genuine wrong-sign error on a value
  of real magnitude m yields ``~2m / spacing(m)`` ≈ 2·2^mantissa_bits — correctly
  *huge*. It is reported, not hidden.
* **Root-stable.** The only thing that "explodes" the naive bit-distance metric
  is the dense bf16 exponent ladder hugging zero: two opposite-sign *tiny*
  values are thousands of representable codes apart yet differ by a physically
  negligible magnitude. The ``beta`` floor (one bf16 step at the function's own
  output scale) caps the denominator there: below ``beta`` the two values are
  within a single bf16 ULP of the *result's* magnitude — bf16 cannot resolve a
  difference finer than ``beta`` at that scale — so a sign flip across zero costs
  ≤ 1, never 24000. ``beta`` is derived from the data (``max|ref|``), so there
  is no per-function constant.
* **Reduces to ~0.5 for a correctly-rounded output everywhere**, including at and
  near zero crossings and roots: if ``comp == RN_bf16(y_true)`` then
  ``|ref - comp| == 0`` → ULP 0; a half-ULP rounding gives 0.5; one bf16 step
  gives exactly 1.
* **No spacing-explosion (acosh/lgamma).** The old metric divided by
  ``spacing(RN_bf16(y_true))``; at a root ``y_true→0`` that spacing → ~1e-41 and
  *any* nonzero output blew up (lgamma '221'). Here the denominator is floored at
  ``beta``, the *coarsest* meaningful step, so a tiny near-root error stays ~1.
* **No bit-distance-explosion (swish/mish).** The old raw bit-distance counted
  the full signed-magnitude code span from the smallest negative bf16 to the
  smallest positive bf16 across a sign crossing (24590). The ``beta`` floor
  collapses that sub-resolution ladder; only a genuinely large-magnitude error
  survives.

Why the OLD numbers exploded (root cause)
-----------------------------------------
* swish raw bit-distance 24590: it compared signed-magnitude bf16 *codes* whose
  span across zero counts every exponent from 2^-1 down to 2^-126, even though
  both values were ~5e-10 and the absolute error was ~1e-9 (silicon is correct
  near zero). Artifact of counting representable codes, not magnitude.
* acosh spacing 127.9: ``error / spacing(y_true)`` sampled near the steep x→1
  region where ``spacing(y_true)`` is tiny; a legacy CSV-sampling artifact that
  does not reproduce against the real kernel (real worst is 1 bf16 ULP at
  x=81.5). The sound metric's ``beta`` floor prevents the small-spacing blow-up.

FTZ is applied to BOTH operands BEFORE the distance (silicon flushes bf16
subnormals to signed zero), and is an explicit first-class step.

``ulp_spacing`` (Goldberg nextafter spacing) is retained as a public helper and
as the building block of ``beta``.

Reference: Goldberg, "What every computer scientist should know about
floating-point arithmetic"; tt-metal models/common/utility_functions.py:576.
"""

import enum
import math
from fractions import Fraction

import os as _os

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from ttpoly.precision import bf16 as _bf16


class RoundingMode(enum.Enum):
    """Rounding modes used across the precision model."""

    RNE = "rne"  # round to nearest even (IEEE default; reference downcast)
    RTZ = "rtz"  # round toward zero (legacy SFPU-CAST label; the modeled bf16
    #             # output cast is RNE — see eval.py:_cast_output and bf16.py)


class Precision(enum.Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"

    @classmethod
    def coerce(cls, value):
        if isinstance(value, cls):
            return value
        v = str(value).lower()
        aliases = {
            "float32": cls.FP32,
            "fp32": cls.FP32,
            "bfloat16": cls.BF16,
            "bf16": cls.BF16,
            "float16": cls.FP16,
            "half": cls.FP16,
            "fp16": cls.FP16,
        }
        if v not in aliases:
            raise ValueError(f"Unknown precision: {value!r}")
        return aliases[v]


# Smallest normal magnitude shared by FP32 and BF16 exponent range (2**-126).
MIN_NORMAL = _bf16.MIN_NORMAL


def _ulp_torch(x):
    """Goldberg ULP via torch.nextafter (port of extract_accuracy._ulp_torch)."""
    abs_x = torch.abs(x)
    nxt = torch.nextafter(abs_x, torch.tensor(math.inf, dtype=x.dtype))
    ulp_value = nxt - abs_x
    dtype_max = torch.finfo(x.dtype).max
    max_epsilon = dtype_max - torch.nextafter(
        torch.tensor(dtype_max, dtype=x.dtype),
        torch.tensor(-math.inf, dtype=x.dtype),
    )
    return torch.where(abs_x == dtype_max, max_epsilon, ulp_value)


def _float_ulp_spacing_numpy(x, dtype):
    """numpy nextafter spacing for FP32/FP16 (port of extract_accuracy)."""
    with np.errstate(over="ignore", invalid="ignore"):
        x_typed = np.asarray(x, dtype=dtype)
        abs_x = np.abs(x_typed)
        inf = np.array(np.inf, dtype=dtype)
        neg_inf = np.array(-np.inf, dtype=dtype)
        next_up = np.nextafter(abs_x, inf)
        ulp_value = next_up.astype(np.float32) - abs_x.astype(np.float32)
        dtype_info = np.finfo(dtype)
        dtype_max = np.array(dtype_info.max, dtype=dtype)
        previous_max = np.nextafter(dtype_max, neg_inf)
        max_epsilon = np.float32(dtype_max) - np.float32(previous_max)
        ulp_value = np.where(abs_x == dtype_max, max_epsilon, ulp_value)
    return ulp_value.astype(np.float32, copy=False)


def _bf16_bits_to_float32(bits):
    bits32 = np.asarray(bits, dtype=np.uint32)
    return (bits32 << np.uint32(16)).view(np.float32)


def _bf16_ulp_spacing_numpy(x):
    """numpy nextafter spacing for BF16 (port of extract_accuracy)."""
    x32 = np.abs(np.asarray(x, dtype=np.float32))
    bits = x32.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    bf16_bits = ((bits + bias) >> np.uint32(16)).astype(np.uint32)
    current = _bf16_bits_to_float32(bf16_bits)
    max_bf16 = np.uint32(0x7F7F)
    next_bits = np.where(bf16_bits < max_bf16, bf16_bits + np.uint32(1), bf16_bits)
    next_up = _bf16_bits_to_float32(next_bits)
    ulp_value = next_up - current
    max_epsilon = _bf16_bits_to_float32(max_bf16) - _bf16_bits_to_float32(max_bf16 - np.uint32(1))
    ulp_value = np.where(bf16_bits == max_bf16, max_epsilon, ulp_value)
    ulp_value = np.where(np.isfinite(x32), ulp_value, np.nan)
    return ulp_value.astype(np.float32, copy=False)


def ulp_spacing_numpy(x, precision="fp32"):
    """Goldberg ULP spacing, numpy-only (the cross-check / fallback path)."""
    p = Precision.coerce(precision)
    if p is Precision.BF16:
        return _bf16_ulp_spacing_numpy(x)
    if p is Precision.FP16:
        return _float_ulp_spacing_numpy(x, np.float16)
    return _float_ulp_spacing_numpy(x, np.float32)


def ulp_spacing(x, precision="fp32"):
    """Goldberg ULP spacing for each value (torch fast path, numpy fallback)."""
    p = Precision.coerce(precision)
    if torch is None:
        return ulp_spacing_numpy(x, precision)
    x_torch = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    if p is Precision.BF16:
        x_torch = x_torch.to(torch.bfloat16)
    elif p is Precision.FP16:
        x_torch = x_torch.to(torch.float16)
    ulp_torch = _ulp_torch(x_torch)
    if ulp_torch.dtype in (torch.bfloat16, torch.float16):
        ulp_torch = ulp_torch.to(torch.float32)
    return ulp_torch.numpy()


# DEFAULT FLIPPED 2026-08-24 to post_round. Blackhole flushes denormal outputs
# AFTER rounding (BlackholeA0/TensixTile/TensixCoprocessor/SFPMAD.md), so keying
# the numerator's flush on the ROUNDED golden is the only order that scores this
# silicon correctly. pre_round -- the upstream tester's order, and Wormhole's --
# charges a correctly-rounded MIN_NORMAL 2^mantissa_bits AND scores a wrongly
# flushed 0 as perfect, i.e. it ranks the two engines backwards. Set
# TTPOLY_ULP_FLUSH_ORDER=pre_round to reproduce any number published before this
# date, or to score Wormhole silicon. Upstream: nmauriceTT/ttnn-eltwise-op-tester#2
_DEFAULT_FLUSH_ORDER = _os.environ.get("TTPOLY_ULP_FLUSH_ORDER", "post_round")
if _DEFAULT_FLUSH_ORDER not in ("pre_round", "post_round"):
    raise ValueError(
        "TTPOLY_ULP_FLUSH_ORDER must be 'pre_round' (upstream/WH-compatible, the "
        "default) or 'post_round' (matches Blackhole SFPMAD)"
    )


def round_to_target(values, precision, flush_to_zero=True):
    """Round mathematical values to a target format using canonical RNE.

    The returned FP32 containers hold the target-format values.  In particular,
    this function performs IEEE overflow rounding: a finite mathematical value
    at or beyond the target's overflow midpoint becomes signed infinity.  Code
    deciding whether a golden belongs to a finite-numeric or expected-nonfinite
    population must classify this result, not the wider source value.

    ``flush_to_zero`` is applied after rounding, matching the default Blackhole
    output policy.  Callers retain the original high-precision values for error
    measurement; this helper owns only the target-format disposition.
    """
    p = Precision.coerce(precision)
    with np.errstate(over="ignore", invalid="ignore"):
        v = np.asarray(values, dtype=np.float64).astype(np.float32)
        if p is Precision.BF16:
            out = _bf16.to_bf16(v, flush_to_zero=flush_to_zero, rounding_mode="rne")
        elif p is Precision.FP16:
            out = v.astype(np.float16).astype(np.float32)
            if flush_to_zero:
                out = _bf16.apply_ftz(out)
        else:
            out = v
            if flush_to_zero:
                out = _bf16.apply_ftz(out)
    return np.atleast_1d(np.asarray(out, dtype=np.float32))


def _downcast_ftz(values, precision, flush_to_zero):
    """Compatibility alias for :func:`round_to_target`."""
    return round_to_target(values, precision, flush_to_zero)


def ulp_error(y_true, y_pred, precision="bf16", inputs=None, flush_to_zero=True):
    """RETIRED floored-ULP metric (do NOT use as a headline accuracy number).

    This computes a floored ULP whose denominator floor ``beta`` is a single
    GLOBAL value = spacing(max|ref| over the domain). That global floor makes the
    metric DYNAMIC-RANGE BLIND: on wide-range functions (e.g. i0, multigammaln)
    a gross error in the small-output region is divided by the large domain-peak
    floor and reads as near-zero, so the metric is anti-correlated with real
    accuracy there. The headline accuracy metrics are now the silicon PURE ULP
    (:func:`ulp_error_pure`) and the ML-tolerance pass rate; this function is kept
    only to populate the inert ``bf16_maxulp`` schema column and the legacy s40
    verify gate. It must not be reported or gated on in the paper.

    Per point::

        ref   = FTZ( RN_<prec>(y_true) )
        comp  = FTZ( <prec>(y_pred)   )
        scale = max(|ref|, |comp|)
        beta  = spacing( max|ref| over the domain )   # single GLOBAL floor (the bug)
        ULP   = |ref - comp| / max( spacing(scale), beta )

    ``flush_to_zero`` applies FTZ to BOTH operands (silicon behavior).
    Returns the per-point floored-ULP array (NaN only where ``ref`` is non-finite).
    """
    Precision.coerce(precision)  # validate the precision label (raises on unknown)
    y_true = np.atleast_1d(np.asarray(y_true, dtype=np.float64))
    y_pred = np.atleast_1d(np.asarray(y_pred, dtype=np.float64))

    # Correctly-rounded reference and the silicon-domain computed value, both
    # rounded to the target precision with FTZ applied FIRST.
    ref = _downcast_ftz(y_true, precision, flush_to_zero).astype(np.float64)
    comp = _downcast_ftz(y_pred, precision, flush_to_zero).astype(np.float64)

    finite = np.isfinite(ref) & np.isfinite(comp)

    # beta: one ULP-spacing at the function's own output scale (data-derived, no
    # per-function constant). The coarsest meaningful step the outputs occupy;
    # this is the root-stability floor.
    if np.any(finite):
        out_scale = np.max(np.abs(ref[finite]))
    else:
        out_scale = 1.0
    beta = float(np.asarray(ulp_spacing(np.float32(out_scale), precision)).reshape(-1)[0])
    if not np.isfinite(beta) or beta <= 0.0:
        beta = float(MIN_NORMAL)

    # ULP-of-the-result: spacing at the larger of the two operands, floored at
    # beta so the dense near-zero ladder cannot inflate a sub-resolution error.
    scale = np.maximum(np.abs(ref), np.abs(comp)).astype(np.float32)
    local_spacing = np.asarray(ulp_spacing(scale, precision), dtype=np.float64)
    denom = np.maximum(local_spacing, beta)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ulp_errors = np.abs(ref - comp) / denom

    out = np.asarray(ulp_errors, dtype=np.float64).copy()
    out[~finite] = np.nan
    return out


def ulp_error_unfloored(y_true, y_pred, precision="bf16", inputs=None, flush_to_zero=True):
    """ULP-DISTANCE between two rounded floats — :func:`ulp_error` WITHOUT the beta floor.

    NOT Goldberg's error-in-ulps (that canonical metric is :func:`ulp_error_pure`).
    Because it rounds BOTH operands, a correctly-rounded result reads exactly 0 and
    it understates TT's tester / the canonical metric by up to 0.5 ULP. It is a
    ulp-*distance* ("ulps between two floats", Dawson-2012 style), not the distance
    to the high-precision real value; do not label it "Goldberg".

    Identical to :func:`ulp_error` except the denominator is the plain
    ULP-spacing at ``max(|ref|, |comp|)`` with NO global ``beta`` floor::

        ref   = FTZ( RN_<prec>(y_true) )
        comp  = FTZ( <prec>(y_pred)   )
        scale = max(|ref|, |comp|)
        ULP   = |ref - comp| / spacing(scale)

    This is a ulp-DISTANCE between two rounded floats (rounds both operands, so a
    correctly-rounded output reads exactly 0) — NOT Goldberg's error-in-ulps and
    NOT the citeable "ULP of the result" (that is :func:`ulp_error_pure`, which
    keeps the high-precision golden in the numerator and so scores a correctly-
    rounded result up to 0.5). It merely drops the nonstandard dynamic-range floor
    that :func:`ulp_error` documents as "the bug". Removing the floor re-exposes
    near-zero / near-root error on BOTH sides, so it tracks :func:`ulp_error_pure`
    on real deficits while understating it by up to 0.5 on correctly-rounded points.

    Unlike the floored metric there is no global (domain-wide) term, so this is
    chunk-independent and safe for streaming/exhaustive evaluation.

    Returns the per-point ULP array (NaN only where ``ref`` or ``comp`` is
    non-finite).
    """
    Precision.coerce(precision)  # validate the precision label (raises on unknown)
    y_true = np.atleast_1d(np.asarray(y_true, dtype=np.float64))
    y_pred = np.atleast_1d(np.asarray(y_pred, dtype=np.float64))

    ref = _downcast_ftz(y_true, precision, flush_to_zero).astype(np.float64)
    comp = _downcast_ftz(y_pred, precision, flush_to_zero).astype(np.float64)
    finite = np.isfinite(ref) & np.isfinite(comp)

    # ULP-of-the-result: spacing at the larger of the two operands, NO beta floor.
    scale = np.maximum(np.abs(ref), np.abs(comp)).astype(np.float32)
    denom = np.asarray(ulp_spacing(scale, precision), dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ulp_errors = np.abs(ref - comp) / denom

    out = np.asarray(ulp_errors, dtype=np.float64).copy()
    out[~finite] = np.nan
    return out


def ulp_error_pure(y_true, y_pred, precision="bf16", inputs=None, flush_to_zero=True, flush_order=_DEFAULT_FLUSH_ORDER):
    """PURE ULP error — TT's ``ttnn-eltwise-op-tester`` metric, AND the canonical
    Goldberg (1991) "error in ulps": |high-precision golden - comp| / ulp(rounded
    golden). THIS is the real Goldberg ULP-of-the-result (despite their names,
    neither :func:`ulp_error` nor :func:`ulp_error_unfloored` is Goldberg's metric).

    This is high-precision golden error scaled by the spacing at the rounded
    golden.  It is deliberately *not* the distance between two rounded values:
    that older implementation made every correctly rounded result exactly zero
    and therefore understated the tester's result by as much as 0.5 ULP.

    This is intentionally distinct from :func:`ulp_error` (the beta-floored
    "sound"/ML-tolerance lens kept as a labeled secondary metric). The pure metric
    legitimately reports large ULP wherever a fit loses relative precision near a
    root or singularity — that near-singularity loss is real, not a metric
    artifact, and the beta floor was hiding it.

    Per point::

        golden = FTZ(y_true)                       # high-precision golden
        rounded_golden = FTZ(RN_<prec>(y_true))   # denominator location
        comp = RN_<prec>(y_pred)                  # target-precision result
        denom = ulp_<prec>(rounded_golden)
        ULP = fp32(|golden - comp|) / fp32(denom)

    The tester performs its final error arithmetic in FP32; reproducing that
    detail here makes parity exact rather than merely mathematically close. The
    denominator is the spacing at ``rounded_golden`` ALONE (not
    ``max(|golden|,|comp|)``, not a domain-peak beta). ``ulp_spacing`` returns
    the smallest subnormal spacing at zero, so a small error on a true zero
    stays finite (and may be very large).

    ``flush_to_zero`` applies the tester's binary32/BF16 FTZ threshold
    (``2**-126``) to the high-precision and rounded goldens. The tester does not
    post-process the device result: an unexpected device subnormal remains a
    measurable error. Returns the per-point pure-ULP array; undefined points are
    NaN, while a defined infinite error remains ``+inf``.
    """
    Precision.coerce(precision)  # validate the precision label (raises on unknown)
    y_true = np.atleast_1d(np.asarray(y_true, dtype=np.float64))
    y_pred = np.atleast_1d(np.asarray(y_pred, dtype=np.float64))

    # The tester preserves the FP64 golden for the numerator and only downcasts
    # a copy to locate its target-precision ULP.  FTZ is applied to the FP64
    # golden without first casting it to FP32, which would discard the very
    # rounding residual this metric is meant to measure.
    rounded_golden = _downcast_ftz(y_true, precision, flush_to_zero).astype(np.float64)

    golden = y_true.copy()
    if flush_to_zero:
        if flush_order == "post_round":
            # Blackhole flushes AFTER rounding: BlackholeA0/TensixTile/
            # TensixCoprocessor/SFPMAD.md -- "If the output (after rounding) is
            # denormal, it'll be flushed to sign-preserved zero." So a true value
            # in the top half-ULP below MIN_NORMAL rounds UP onto MIN_NORMAL and
            # is NOT flushed, and the correct device answer there is MIN_NORMAL.
            # Keying the numerator's flush on the ROUNDED golden makes both sides
            # of the quotient describe the same value, which fixes the artifact in
            # BOTH directions: a correct MIN_NORMAL output scores ~0.5 instead of
            # 2^mantissa_bits, and a wrong 0 output scores 2^mantissa_bits instead
            # of 0.0. See nmauriceTT/ttnn-eltwise-op-tester#2.
            flush = np.isfinite(rounded_golden) & (rounded_golden == 0.0)
        else:
            # "pre_round" (default): bit-compatible with the upstream tester,
            # which flushes the UNROUNDED fp64 golden at 2**-126. Wormhole
            # documents flushing before rounding, so this matches WH silicon and
            # every number published to date -- but on Blackhole it charges a
            # correctly-rounded result 2^mantissa_bits, and scores a wrongly
            # flushed 0 as perfect.
            flush = np.isfinite(golden) & (np.abs(golden) < MIN_NORMAL)
        golden[flush] = 0.0
    # The TTNN tensor is already in the target dtype and is not explicitly FTZ'd
    # by compare_with_golden. Round array callers to that dtype, but preserve an
    # unexpected subnormal output so the metric exposes it rather than hiding it.
    comp = _downcast_ftz(y_pred, precision, False).astype(np.float64)

    defined_golden = np.isfinite(golden) & np.isfinite(rounded_golden)

    # Match compare_with_golden exactly: both operands of the final quotient are
    # FP32 even though the golden used to form the numerator is FP64.
    denom = np.asarray(
        ulp_spacing(np.abs(rounded_golden).astype(np.float32), precision),
        dtype=np.float32,
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        abs_error = np.abs(golden - comp).astype(np.float32)
        ulp_errors = (abs_error / denom).astype(np.float64)

    out = np.asarray(ulp_errors, dtype=np.float64).copy()
    # A finite golden with an infinite device result (or an overflowing FP32
    # quotient) is +inf, exactly as in the tester. Only an undefined golden,
    # NaN device result, or invalid spacing is NaN.
    out[~defined_golden | ~np.isfinite(denom) | (denom <= 0.0)] = np.nan
    return out


class _Float32ExactAccumulator:
    """Chunk-invariant max/mean reducer for non-negative binary32 metrics.

    ``compare_with_golden`` produces binary32 ULP values. Summing those values
    with floating-point chunk reductions makes the exhaustive mean depend on
    FIFO/chunk boundaries. Every finite binary32 value is instead accumulated
    as an integer multiple of ``2**-149``; the exact rational mean is rounded to
    Python float only once in :attr:`mean`.

    NaNs are undefined samples and are excluded, matching ``np.nanmean``.
    Positive infinity is a defined worst-case error and makes max/mean infinite.
    """

    # With at most 2**28 values per bincount, a bin's sum of 24-bit integer
    # significands is below 2**52 and is therefore exact in float64.
    _EXACT_BINCOUNT_BLOCK = 1 << 28

    def __init__(self):
        self.count = 0
        self.units = 0
        self.has_infinity = False
        self._max_finite = float("nan")

    def update(self, values):
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        for start in range(0, array.size, self._EXACT_BINCOUNT_BLOCK):
            block = array[start : start + self._EXACT_BINCOUNT_BLOCK]
            defined = block[~np.isnan(block)]
            self.count += int(defined.size)
            if not defined.size:
                continue
            if np.any(defined < 0):
                raise ValueError("exact binary32 metric accumulator requires non-negative values")
            if np.any(np.isinf(defined)):
                self.has_infinity = True

            finite = defined[np.isfinite(defined)]
            if not finite.size:
                continue
            chunk_max = float(np.max(finite))
            if not np.isfinite(self._max_finite) or chunk_max > self._max_finite:
                self._max_finite = chunk_max

            bits = finite.view(np.uint32)
            exponents = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.intp)
            fractions = bits & np.uint32(0x7FFFFF)
            significands = np.where(
                exponents == 0,
                fractions,
                fractions | np.uint32(0x800000),
            ).astype(np.float64)
            bins = np.bincount(exponents, weights=significands, minlength=255)
            for exponent in np.flatnonzero(bins):
                significand_sum = int(bins[exponent])
                shift = 0 if exponent == 0 else int(exponent) - 1
                self.units += significand_sum << shift
        return self

    @property
    def maximum(self):
        if self.has_infinity:
            return float("inf")
        return self._max_finite if self.count else float("nan")

    @property
    def mean(self):
        if not self.count:
            return float("nan")
        if self.has_infinity:
            return float("inf")
        return float(Fraction(self.units, self.count * (1 << 149)))

    def additive_state(self):
        """Return the exact, JSON-safe state needed to combine disjoint runs.

        ``units`` is emitted as decimal text because a complete binary32-space
        reduction can exceed JSON's interoperable integer range.  Consumers
        must still authenticate and replay retained raw shards for canonical
        evidence; this state makes each chunk independently auditable and
        catches accidental metric aggregation by rounded chunk means.
        """
        return {
            "schema": "binary32_nonnegative_exact_accumulator_v1",
            "count": self.count,
            "units_2^-149": str(self.units),
            "has_infinity": self.has_infinity,
            "max_finite": (self._max_finite if np.isfinite(self._max_finite) else None),
        }


def _summarize_float32_metric(values):
    """Return chunk-independent ``(max, mean, defined_count)``."""
    accumulator = _Float32ExactAccumulator().update(values)
    return accumulator.maximum, accumulator.mean, accumulator.count


def max_ulp_pure(y_true, y_pred, precision="bf16", inputs=None, flush_to_zero=True, flush_order=_DEFAULT_FLUSH_ORDER):
    e = ulp_error_pure(y_true, y_pred, precision, inputs, flush_to_zero, flush_order=flush_order)
    maximum, _mean, count = _summarize_float32_metric(e)
    return maximum if count else float("nan")


def mean_ulp_pure(y_true, y_pred, precision="bf16", inputs=None, flush_to_zero=True, flush_order=_DEFAULT_FLUSH_ORDER):
    e = ulp_error_pure(y_true, y_pred, precision, inputs, flush_to_zero, flush_order=flush_order)
    _maximum, mean, count = _summarize_float32_metric(e)
    return mean if count else float("nan")


def ml_pass_rate(y_true, y_pred, atol=1e-3, rtol=1e-3, precision="bf16", flush_to_zero=True):
    """STANDARD ML-tolerance pass-rate — the honest ML-accuracy headline metric.

    Percentage of points satisfying the numpy/pytorch ``allclose`` predicate

        |y_pred - y_true| <= atol + rtol * |y_true|

    with ``atol = rtol = 1e-3`` (the bf16 pytorch/numpy default tolerance). This
    is reported ALONGSIDE :func:`ulp_error_pure` (PURE ULP) as one of the two
    standard headline metrics. Unlike a pure-ULP max, a pass-rate is robust to
    the known near-zero inflation (a negligible-magnitude error near a root passes
    ML tolerance even when its relative/pure-ULP value is large) — it answers the
    real question "does this fit behave for ML?".

    Both operands are rounded to ``precision`` (RNE) with FTZ applied first, to
    match silicon behavior, then compared against the (un-rounded) golden in the
    tolerance. The golden ``y_true`` is the high-precision reference.

    Returns the pass-rate as a percentage in ``[0, 100]`` (NaN if no finite
    points). Points where the golden or computed value is non-finite are excluded.
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=np.float64))
    y_pred = np.atleast_1d(np.asarray(y_pred, dtype=np.float64))

    comp = _downcast_ftz(y_pred, precision, flush_to_zero).astype(np.float64)

    abs_err = np.abs(y_true - comp)
    tol = float(atol) + float(rtol) * np.abs(y_true)
    # The denominator is every point with a DEFINED golden. A finite golden with a
    # non-finite computed value (a collapsed / overflowed output) is a FAILURE, not
    # a point to drop — otherwise a partially-NaN prediction reports a clean pass
    # rate on the surviving points (P0-02). Points whose golden is non-finite are a
    # genuinely undefined reference and stay excluded.
    defined = np.isfinite(y_true) & np.isfinite(tol)
    if not np.any(defined):
        return float("nan")
    passed = np.isfinite(abs_err) & (abs_err <= tol)
    return float(100.0 * np.mean(passed[defined]))


def max_ulp(y_true, y_pred, precision="bf16", inputs=None, flush_to_zero=True):
    e = ulp_error(y_true, y_pred, precision, inputs, flush_to_zero)
    return float(np.nanmax(e)) if np.any(np.isfinite(e)) else float("nan")


def mean_ulp(y_true, y_pred, precision="bf16", inputs=None, flush_to_zero=True):
    e = ulp_error(y_true, y_pred, precision, inputs, flush_to_zero)
    return float(np.nanmean(e)) if np.any(np.isfinite(e)) else float("nan")


def max_ulp_unfloored(y_true, y_pred, precision="bf16", inputs=None, flush_to_zero=True):
    e = ulp_error_unfloored(y_true, y_pred, precision, inputs, flush_to_zero)
    maximum, _mean, count = _summarize_float32_metric(e)
    return maximum if count else float("nan")


def mean_ulp_unfloored(y_true, y_pred, precision="bf16", inputs=None, flush_to_zero=True):
    e = ulp_error_unfloored(y_true, y_pred, precision, inputs, flush_to_zero)
    _maximum, mean, count = _summarize_float32_metric(e)
    return mean if count else float("nan")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
SFPU / FPU operation domain registry and helpers.

Maps every MathOperation to safe per-operand input domains (OperandSpecs).
Provides for_op() to look up domains by op + format, and
exclude_undefined()/exclude_intervals()/exclude_values() to subtract known-undefined
regions from a user-supplied StimuliSpec.
"""

from __future__ import annotations

import copy
import math
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple, Union

import torch

from .format_config import MX_FORMAT_MAX_NORMAL, MX_FORMAT_MIN_NORMAL, DataFormat
from .llk_params import MathOperation
from .sfpu_dispatch_constants import (
    CLAMP_MAX,
    CLAMP_MIN,
    HARDSHRINK_LAMBDA,
    INT_MAXMIN_SCALAR,
    RELU_MAX_THRESHOLD,
    RELU_MIN_THRESHOLD,
    SOFTPLUS_THRESHOLD,
    SOFTSHRINK_LAMBDA,
    THRESHOLD_T,
    UNARY_COMP_THRESHOLD,
    UNARY_MAX_MIN_VALUE,
)
from .stimuli_generator import DistributionKind, StimuliSpec

# ─────────────────────────────────────────────────────────────────────────────
# OperandSpecs
# ─────────────────────────────────────────────────────────────────────────────


class Operand(str, Enum):
    """Identifies which operand of an OperandSpecs a value refers to."""

    A = "spec_A"
    B = "spec_B"
    C = "spec_C"


@dataclass
class OperandSpecs:
    """Per-operand input domain specs returned by for_op.

    For binary ops where operands need different domains (e.g. divisor avoids
    zero), spec_A and spec_B differ; unary ops need only spec_A.
    spec_B defaults to a copy of spec_A when "None".

    *spec_C* is the ternary family's third operand -- a divisor for ``addcdiv`` and
    ``snake_beta``, so the one that carries the pole. It defaults to a copy of *spec_B* the
    way spec_B defaults to a copy of spec_A, so an entry naming only spec_A still resolves
    to three identical specs.
    """

    spec_A: StimuliSpec
    spec_B: Optional[StimuliSpec] = None
    spec_C: Optional[StimuliSpec] = None

    def __post_init__(self) -> None:
        if self.spec_B is None:
            self.spec_B = copy.deepcopy(self.spec_A)
        if self.spec_C is None:
            self.spec_C = copy.deepcopy(self.spec_B)

    def spec_for(self, operand: "Operand") -> Optional[StimuliSpec]:
        """The spec for *operand*, so callers can select one without a chain of ifs."""
        return getattr(self, operand.value)


# ─────────────────────────────────────────────────────────────────────────────
# Picking which format bounds the domain
# ─────────────────────────────────────────────────────────────────────────────

# Largest finite magnitude each format can hold. Only formats with a narrower
# exponent field than bfloat16 need an entry; every other format shares
# bfloat16's ceiling and is therefore never the binding constraint.
#
# The MX rows come from MX_FORMAT_MAX_NORMAL rather than being restated here: the two
# fp8 encodings are easy to transpose by hand, and a transposed pair silently *widens*
# a domain instead of failing. MxFp8R is E5M2 (ceiling 57344, the wide one) and MxFp8P
# is E4M3 (ceiling 448, the narrow one) -- which is the polarity the builders below
# already assume, e.g. _square_spec caps MxFp8P at +-20 and groups MxFp8R with Float16.
_FORMAT_MAX_MAGNITUDE: Dict[DataFormat, float] = {
    **MX_FORMAT_MAX_NORMAL,  # MxFp4 (e2m1) 6, MxFp8P (e4m3) 448, MxFp8R (e5m2) 57344
    DataFormat.Float16: 65504.0,  # e5m10
    # Plain E4M3 with no per-block scale to lift it, so the same 448 ceiling as MxFp8P.
    DataFormat.Fp8_e4m3: 448.0,
}

_BF16_MAX_MAGNITUDE = 3.3895314e38

# Smallest positive *normal* of each format: the other end of the exponent range from
# _FORMAT_MAX_MAGNITUDE, and where the subnormal band begins.
#
# Sourced from torch.finfo / the MX tables rather than written as 2**-126 and 2**-14, for one
# specific reason: golden_generators._FTZ_THRESHOLD is built from the *same* torch.finfo call,
# and a probe placed on the wrong side of the flush-to-zero cliff tests nothing while reading
# as though it tests everything. test_sfpu_domains pins the two against each other.
#
# fp32 and bfloat16 share 2^-126 -- bfloat16 is fp32's exponent field with a shorter mantissa,
# which is also why bfloat16 never binds the *magnitude* of a probe and has no
# _FORMAT_MAX_MAGNITUDE row. Float16 is the narrow one at 2^-14.
#
# The block-float formats are deliberately absent. Their smallest representable element is set
# by the exponent shared across the 16-element block, not by the element, so "min normal of
# Bfp8_b" is not a property of the format -- format_extremes() raises for them rather than
# returning a number that would be wrong for every block but one.
_FORMAT_MIN_NORMAL: Dict[DataFormat, float] = {
    **MX_FORMAT_MIN_NORMAL,  # MxFp8R 2^-14, MxFp8P 2^-6, MxFp4 1.0
    DataFormat.Float32: float(torch.finfo(torch.float32).tiny),
    DataFormat.Float16_b: float(torch.finfo(torch.bfloat16).tiny),
    DataFormat.Float16: float(torch.finfo(torch.float16).tiny),
    # Plain E4M3 with no per-block scale, so the same 2^-6 as MxFp8P.
    DataFormat.Fp8_e4m3: MX_FORMAT_MIN_NORMAL[DataFormat.MxFp8P],
}


def narrowest_range_format(*formats: Optional[DataFormat]) -> DataFormat:
    """Return whichever of *formats* has the smallest representable magnitude.

    A safe input domain is bounded by the narrowest float format anywhere in the
    pipeline, not just the input one. exp over (-100, 80) peaks at ~5.5e34: fine
    into a Float32 output, saturates a Float16 one. Passing a single format keeps
    the previous input-only behaviour, and ties resolve to the first argument, so
    callers should pass the input format first.
    """
    candidates = [fmt for fmt in formats if fmt is not None]
    if not candidates:
        raise ValueError("narrowest_range_format() requires at least one format")
    return min(
        candidates,
        key=lambda fmt: _FORMAT_MAX_MAGNITUDE.get(fmt, _BF16_MAX_MAGNITUDE),
    )


def _two_state_flag(value: Union[bool, Enum, None], param: str, enum_name: str) -> bool:
    """Normalise a two-state Enum-or-bool test flag to a plain bool.

    DestAccumulation and ApproximationMode wrap True and False, so ``bool(member)`` is True
    for *both* — ``.No`` included, which silently selects the True branch. Read ``.value``
    for an enum, take a bool as-is, reject anything else.

    *enum_name* is checked too, because both enums' ``.Yes`` wraps True and duck-typing on
    ``.value`` would accept the wrong one. Matched on class name rather than class so this
    module stays free of llk_params imports beyond MathOperation.
    """
    flag = getattr(value, "value", value)
    if not isinstance(flag, bool):
        raise TypeError(
            f"{param} must be a bool or a {enum_name} member, got "
            f"{value!r} ({type(value).__name__})"
        )
    if isinstance(value, Enum) and type(value).__name__ != enum_name:
        raise TypeError(
            f"{param} must be a bool or a {enum_name} member, got the "
            f"{type(value).__name__} member {value!r} -- both wrap a bool, so this would "
            f"otherwise select a branch silently"
        )
    return flag


def _approx_mode_flag(approx_mode: Union[bool, Enum]) -> bool:
    """Normalise an approximation-mode flag to a plain bool. See _two_state_flag."""
    return _two_state_flag(approx_mode, "approx_mode", "ApproximationMode")


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific domain builders
# ─────────────────────────────────────────────────────────────────────────────

# The two e4m3 formats share an encoding and so a ceiling of 448, which is the narrowest
# any builder below narrows for. Listed together everywhere, so a builder cannot honour one
# and hand the other the wide-format branch.
_E4M3_FORMATS = (DataFormat.MxFp8P, DataFormat.Fp8_e4m3)

# e5m2 tops out at 57344 and Float16 at 65504: close enough to share a tier.
_E5M2_AND_FLOAT16 = (DataFormat.Float16, DataFormat.MxFp8R)


# ── Exp family: range on the registry, accuracy behind ApproximationMode.Yes ──
#
# Two ceilings bound the exp family's positive side, and they belong in different places:
#
#   * **Range** — exp overflows an 8-bit exponent near x = 88.7. True in both modes, so it
#     lives in the registry entries below.
#   * **Accuracy** — the *approximation* overshoots the golden by ~5.7% past ~8 (measured on
#     Wormhole; see _APPROX_EXP_ACCURACY_XFAIL in test_eltwise_unary_sfpu). One mode only, so it
#     lives in _APPROX_ACCURACY_MAX, applied by for_op() at ApproximationMode.Yes.
#
# The registry entry serves both modes, so an accuracy bound written there also withholds
# (16, 80] from the *accurate* path — with it the exponent-overflow region and all large-exp
# saturation into Float16_b and Bfp8_b.
#
# MEASURED on a Wormhole n300: the full unary sweep drives Exp, Exp2 and ExpWithBase over these
# widened domains at ApproximationMode.No -- the only mode the accurate path runs in -- and passes
# with no custom tolerance and no xfail. If it ever drifts, the fix is a mode-conditional
# custom_rtol, not a re-narrowed registry entry, since the entry serves both modes.
#
# ExpWithBase's entry below is correct but currently unreachable: the op is in STANDARD_SWEEP_OPS,
# which drives ApproximationMode.No only, so the ceiling never fires and the swept domain is the
# range bound (high=160, argument 80). Kept rather than deleted so that enrolling it in
# BROAD_SWEEP_OPS cannot silently hand the approximation an argument of 80, which is ten times the
# ~8 where the overshoot starts. test_sfpu_domains pins the unreachability so the entry cannot be
# mistaken for active coverage.
_APPROX_ACCURACY_MAX: Dict[MathOperation, float] = {
    MathOperation.Exp: 16.0,
    # exp2(x) = exp(x * ln2), so exp's argument ceiling of 16 lands at x = 16 / ln2 ~ 23.
    MathOperation.Exp2: 23.0,
    # exp_with_base computes exp(0.5*x), so double exp's ceiling puts its argument on it.
    MathOperation.ExpWithBase: 32.0,
}


def _exp_spec(fmt: DataFormat) -> OperandSpecs:
    """Safe input range for exp(x) per format to avoid overflow.

    Range-bound only; the approximation's accuracy ceiling is applied on top by for_op()
    from _APPROX_ACCURACY_MAX. See the section comment above.
    """
    if fmt in _E4M3_FORMATS:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    elif fmt in _E5M2_AND_FLOAT16:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    else:
        # the lower bound is intentionally pushed to -100.0 so we cross the SFPU's negative-side
        # sanitization boundary near x ≈ -88.5 (where InputClamping::ClampToNegative saturates inputs
        # in the fast/approx exp path).
        #
        # The positive side is bounded by range: exp overflows an 8-bit exponent near
        # x = 88.7, and 80 leaves margin below it. Narrower output formats pull this in
        # further through for_op_pipeline.
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-100.0, high=80.0)
    return OperandSpecs(spec_A=spec)


def _exp_with_base_spec(fmt: DataFormat) -> OperandSpecs:
    """Input range for exp_with_base, which computes exp(0.5*x).

    Keep the negative reach of _exp_spec (low=-100 crosses the SFPU's negative-side
    sanitization boundary near x ~ -88.5). The 0.5 scale halves the argument, so the
    positive side is double _exp_spec's to put the argument under the same ceiling -- in
    either mode, since _APPROX_ACCURACY_MAX doubles it the same way.
    """
    if fmt in _E4M3_FORMATS:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    elif fmt in _E5M2_AND_FLOAT16:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    else:
        spec = StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-100.0, high=160.0
        )
    return OperandSpecs(spec_A=spec)


def _exp2_spec(fmt: DataFormat) -> OperandSpecs:
    """Safe input range for exp2(x) = 2^x per format to avoid overflow.

    Range-bound only, as _exp_spec; the approximation ceiling comes from
    _APPROX_ACCURACY_MAX.
    """
    if fmt in _E4M3_FORMATS:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-7.0, high=7.0)
    elif fmt in _E5M2_AND_FLOAT16:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-14.0, high=14.0)
    else:
        # 2^100 still fits an 8-bit exponent, so the positive side is not range-bound the
        # way _exp_spec is; the negative side matches its reach past the clamp.
        spec = StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-100.0, high=100.0
        )
    return OperandSpecs(spec_A=spec)


# Block-float formats: a group of 16 elements shares one exponent, so an element's
# usable precision depends on how far below its block maximum it sits, not just on
# the mantissa width.
_BLOCK_FLOAT_FORMATS = (DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Bfp2_b)


def _reciprocal_spec(fmt: DataFormat) -> OperandSpecs:
    """Safe input range for 1/x, tightened for block-float inputs.

    A 1000:1 ratio inside a 16-element block quantizes the smallest elements to
    zero, sending the golden to inf. How tight the ratio has to be scales with the
    mantissa width: 10:1 suffices for Bfp8_b's 7 bits, but at Bfp4_b's 3 bits ~6% of
    that window still lands below the block's representable step and collapses to
    zero. 4:1 keeps every element representable with margin (the smallest survivor
    at 10:1 is 16.0, so the floor is not placed right on the boundary).
    """
    if fmt == DataFormat.Bfp4_b:
        spec = StimuliSpec.uniform(intervals=[(-100.0, -25.0), (25.0, 100.0)])
    elif fmt in _BLOCK_FLOAT_FORMATS:
        spec = StimuliSpec.uniform(intervals=[(-100.0, -10.0), (10.0, 100.0)])
    else:
        spec = StimuliSpec.uniform(intervals=[(-100.0, -0.1), (0.1, 100.0)])
    return OperandSpecs(spec_A=spec)


def _square_spec(fmt: DataFormat) -> OperandSpecs:
    """Safe input range for square(x) = x^2 per format to avoid overflow."""
    if fmt in _E4M3_FORMATS:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-20.0, high=20.0)
    elif fmt in _E5M2_AND_FLOAT16:
        spec = StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-200.0, high=200.0
        )
    else:
        spec = StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-1000.0, high=1000.0
        )
    return OperandSpecs(spec_A=spec)


# ─────────────────────────────────────────────────────────────────────────────
# SFPU / FPU operation domain registry
# ─────────────────────────────────────────────────────────────────────────────
#
# Maps every MathOperation to either:
#   OperandSpecs          — format-independent safe input domains
#   callable              — (DataFormat) -> OperandSpecs for format-sensitive ops
#
# For unary operations spec_B is omitted (defaults to a copy of spec_A).
# For binary operations where operands require different domains the entry
# uses explicit spec_A and spec_B.
#

_OP_DOMAIN_REGISTRY: Dict[
    MathOperation,
    Union[OperandSpecs, Callable[[DataFormat], OperandSpecs]],
] = {
    # ── SFPU unary ────────────────────────────────────────────────────────────
    # abs: all reals; include negative branch
    MathOperation.Abs: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # acosh: domain x >= 1
    MathOperation.Acosh: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=1.0, high=10.0)
    ),
    # asinh: all reals
    MathOperation.Asinh: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # atanh: domain |x| < 1. The log1p reformulation is stable across the whole
    # interior including the small-x region (catastrophic cancellation in the old
    # form) and close to ±1, so sweep nearer the boundary; stay just inside ±1 to
    # avoid the exact ±inf endpoints (covered separately by special-case tests).
    MathOperation.Atanh: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-0.999, high=0.999
        )
    ),
    # celu: exercises both the exponential branch (x < 0) and linear (x >= 0)
    MathOperation.Celu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # cos: cover the full unit circle
    MathOperation.Cos: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-math.pi, high=math.pi
        )
    ),
    # elu: exercises the exponential branch (x < 0)
    MathOperation.Elu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # erfinv: domain |x| < 1; stay just inside ±1 to avoid the ±inf endpoints.
    MathOperation.Erfinv: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-0.99, high=0.99)
    ),
    # heaviside: cover both the negative (->0) and positive (->1) branches.
    MathOperation.Heaviside: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # exp: format-specific overflow threshold
    MathOperation.Exp: _exp_spec,
    # exp2: format-specific overflow threshold
    MathOperation.Exp2: _exp2_spec,
    # exp_with_base computes exp(0.5*x). It needs its own (wider-on-the-positive-side)
    # domain: the 0.5 scale halves the argument, so reusing plain exp's high would cap the
    # argument at half the reach exp's own domain is allowed. Doubling it puts the argument
    # back on the same ceiling, in both modes. See _exp_with_base_spec.
    MathOperation.ExpWithBase: _exp_with_base_spec,
    # fill: the hardware ignores the input value; any range is fine
    MathOperation.Fill: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=1.0)
    ),
    # gelu: gaussian-sampled (mean=0, std=3) — most inputs near 0, but still some large ones.
    MathOperation.Gelu: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.GAUSSIAN,
            mean=0.0,
            std=3.0,
            low=-5.0,
            high=5.0,
        )
    ),
    # gelu_appx: LUT approximation of gelu — same Gaussian spread as gelu so both
    # the near-0 transition and the saturating tails exercise the piecewise LUT.
    MathOperation.GeluAppx: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.GAUSSIAN,
            mean=0.0,
            std=3.0,
            low=-5.0,
            high=5.0,
        )
    ),
    # gelu_tanh: tanh approximation of gelu — same Gaussian spread exercises both
    # tails (saturation) and values near 0 (the +-0 sign path).
    MathOperation.GeluTanh: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.GAUSSIAN, mean=0.0, std=3.0)
    ),
    # gelu_derivative: d/dx gelu; Gaussian spread hits both saturating tails
    # (->0 and ->1) and the transition region around 0.
    MathOperation.GeluDerivative: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.GAUSSIAN,
            mean=0.0,
            std=3.0,
            low=-5.0,
            high=5.0,
        )
    ),
    # hardsigmoid: linear region between -3 and 3, clipped outside
    MathOperation.Hardsigmoid: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-4.0, high=4.0)
    ),
    # log: domain x > 0; log-uniform spans several decades
    MathOperation.Log: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.LOG_UNIFORM, low=1e-4, high=1e3
        )
    ),
    # log_with_base (log2): same positive domain as natural log.
    MathOperation.LogWithBase: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.LOG_UNIFORM, low=1e-4, high=1e3
        )
    ),
    # log1p: domain x > -1; log1p(x) = log(1 + x)
    MathOperation.Log1p: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-0.99, high=10.0)
    ),
    # neg: all reals
    MathOperation.Neg: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # reciprocal: domain x != 0; avoid a small band around 0 and cover both signs.
    # Format-sensitive: block-float inputs need a tighter ratio, see _reciprocal_spec.
    MathOperation.Reciprocal: _reciprocal_spec,
    # relu / relu_max / relu_min / threshold: include negatives (zero branch)
    MathOperation.Relu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    MathOperation.ReluMax: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    MathOperation.ReluMin: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # lrelu: leaky ReLU with slope 0.1; span both signs so the negative
    # (scaled) branch and the positive (pass-through) branch are exercised.
    MathOperation.Lrelu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    MathOperation.Threshold: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # rsqrt: domain x > 0; log-uniform covers a wide positive range
    MathOperation.Rsqrt: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.LOG_UNIFORM, low=1e-4, high=100.0
        )
    ),
    # rsqrt_compat (legacy reciprocal-root): domain x > 0. Keep the range a bit
    # tighter than accurate rsqrt — the compat approximation loses accuracy at the
    # extreme small-input end (rsqrt -> very large).
    MathOperation.RsqrtCompat: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.LOG_UNIFORM, low=1e-2, high=100.0
        )
    ),
    # reciprocal_compat (legacy exponent-difference reciprocal): same domain as the
    # accurate Reciprocal -- everything except the pole, both signs.
    MathOperation.ReciprocalCompat: _reciprocal_spec,
    # expm1_cw (component-wise expm1): same safe range as the standalone expm1.
    MathOperation.Expm1Cw: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # sigmoid: cover both saturation regions
    MathOperation.Sigmoid: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-8.0, high=8.0)
    ),
    # silu: silu(x) = x * sigmoid(x); cover saturation + linear regions
    MathOperation.Silu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # softshrink: piecewise around ±lambda (0.5); span both shrink branches and the zero band
    MathOperation.Softshrink: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # softsign: softsign(x) = x / (1 + |x|); defined for all reals
    MathOperation.Softsign: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # mish: mish(x) = x * tanh(softplus(x)); defined for all reals, cover saturation
    MathOperation.Mish: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # selu: piecewise at x==0; span both the linear (x>=0) and exp (x<0) branches
    MathOperation.Selu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # i0: modified Bessel I0; kernel poly approx is only valid on |x| <= 3.75
    MathOperation.I0: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-3.75, high=3.75)
    ),
    # i1: modified Bessel I1; poly path valid on |x| <= ~3.75 (asymptotic beyond)
    MathOperation.I1: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-3.75, high=3.75)
    ),
    # erf / erfc: span both tails and the transition through 0
    MathOperation.Erf: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-3.0, high=3.0)
    ),
    MathOperation.Erfc: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-3.0, high=3.0)
    ),
    # expm1: exp(x)-1; keep within a range that avoids fp overflow
    MathOperation.Expm1: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # cbrt: defined for all reals; span both signs to exercise the sign path
    MathOperation.Cbrt: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-27.0, high=27.0)
    ),
    # sign / signbit: span both signs and near-zero
    MathOperation.Sign: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.Signbit: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    # tanh_derivative = sech^2(x); cover the saturating tails
    MathOperation.TanhDerivative: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # legacy LUT tanh_derivative: same math but tanh comes from a coarse piecewise
    # LUT that suffers catastrophic cancellation in 1 - tanh^2 for |x| > ~3.4, so
    # keep the domain inside the LUT's accurate region.
    MathOperation.TanhDerivativeLut: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-3.0, high=3.0)
    ),
    # hardmish: piecewise on [-2, 0]; span past both clamp knees
    MathOperation.Hardmish: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-4.0, high=4.0)
    ),
    # lgamma: single-tile Stirling kernel is accurate for x >= ~0.5; avoid the poles at x<=0
    MathOperation.Lgamma: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=1.0, high=15.0)
    ),
    # digamma: LUT fit on [0.01, 102]; keep positive to avoid the poles at x<=0
    MathOperation.Digamma: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.1, high=50.0)
    ),
    # identity: pass-through; any range is valid
    MathOperation.Identity: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # prelu: leaky slope on the negative side; span both signs
    MathOperation.Prelu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # rpow: 2**x; keep exponent bounded to avoid fp overflow
    MathOperation.Rpow: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-4.0, high=4.0)
    ),
    # power: x**2 (fixed integer exponent); span both signs
    MathOperation.UnaryPower: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-4.0, high=4.0)
    ),
    # fmod / remainder: divisor fixed to 2.0; span both signs
    MathOperation.Fmod: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    MathOperation.Remainder: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # unary comparisons against threshold 0.5; span it for a mix of 0/1 outputs
    MathOperation.UnaryGt: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.UnaryLt: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.UnaryGe: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.UnaryLe: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    # unary max/min against value 0.0; span both signs
    MathOperation.UnaryMax: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    MathOperation.UnaryMin: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # polygamma (order 1, trigamma): poles at x<=0, so keep positive
    MathOperation.Polygamma: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.5, high=10.0)
    ),
    # xielu: piecewise activation; span both signs across the knee at 0
    MathOperation.Xielu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # hardshrink: piecewise around +/-lambda (0.5); span past both knees
    MathOperation.Hardshrink: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-4.0, high=4.0)
    ),
    # softplus: smooth; span both signs and past the linear threshold (20) so the
    # kernel's linear-passthrough branch (input > threshold -> softplus(x) ~= x) is covered.
    MathOperation.Softplus: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=30.0)
    ),
    # sigmoid_appx: LUT approximation of sigmoid; span both signs across the knee at 0
    MathOperation.SigmoidAppx: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # sqrt_custom: domain x >= 0
    MathOperation.SqrtCustom: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=100.0)
    ),
    # add1: x + 1; defined for all reals
    MathOperation.Add1: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    MathOperation.CastFp32ToFp16a: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-100000.0, high=100000.0
        )
    ),
    # comparison-to-zero: span both signs so the </<=/>/>= branches are exercised
    MathOperation.EqualZero: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.NotEqualZero: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.LessThanZero: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.GreaterThanZero: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.LessThanEqualZero: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.GreaterThanEqualZero: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    # rdiv: value / x; keep x away from 0 to avoid the reciprocal blow-up
    MathOperation.Rdiv: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=1.0, high=8.0)
    ),
    # clamp/hardtanh: bounds fixed to [-1, 1]; span past both bounds to exercise clamping
    MathOperation.Clamp: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    MathOperation.Hardtanh: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0)
    ),
    # sin: cover the full unit circle
    MathOperation.Sin: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-math.pi, high=math.pi
        )
    ),
    # tan: stay inside the poles at +-pi/2 (~1.5708); tan grows rapidly near them.
    MathOperation.Tan: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.3, high=1.3)
    ),
    # atan: defined for all reals; span both signs and the saturating tails.
    MathOperation.Atan: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # asin/acos: domain [-1, 1]; stay just inside to avoid the NaN region for |x|>1.
    MathOperation.Asin: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-0.99, high=0.99)
    ),
    MathOperation.Acos: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-0.99, high=0.99)
    ),
    # sinh/cosh: keep the range moderate so exp(|x|) stays well within fp range.
    MathOperation.Sinh: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    MathOperation.Cosh: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # round: round-half-to-even to integer; span both signs across integer knees.
    MathOperation.Round: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # floor/ceil/trunc/frac: defined for all reals, but each of floor and ceil differs
    # from trunc on one side only -- floor on the negative side (floor(-1.5) = -2 vs
    # trunc's -1), ceil on the positive side (ceil(1.5) = 2 vs trunc's 1) -- so the
    # domain has to span both signs to tell the three apart at all. Same range as round
    # for the same reason: enough integer knees inside the interval that the random
    # sweep lands near several of them.
    MathOperation.Floor: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    MathOperation.Ceil: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    MathOperation.Trunc: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # frac keeps the sign of x (frac(x) = x - trunc(x)), so the negative half is a
    # distinct branch rather than a mirror of the positive one.
    MathOperation.Frac: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # sqrt: domain x >= 0
    MathOperation.Sqrt: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=100.0)
    ),
    # square: format-specific overflow threshold
    MathOperation.Square: _square_spec,
    # tanh: cover saturation regions (saturates near ±1 for |x| > ~3)
    MathOperation.Tanh: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # tanhshrink(x) = x - tanh(x): odd function, so the negative half exercises a
    # distinct sign path. Same range as tanh — past |x| ~ 3 tanh saturates and the
    # result degenerates to x, while near 0 the subtraction cancels down to ~x^3/3
    # (small absolute values, covered by atol rather than rtol).
    MathOperation.Tanhshrink: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # topk family: operation sorts/merges; any values are valid
    MathOperation.TopKLocalSort: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    MathOperation.TopKMerge: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    MathOperation.TopKRebuild: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # ── FPU binary ────────────────────────────────────────────────────────────
    MathOperation.Elwadd: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    MathOperation.Elwmul: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    MathOperation.Elwsub: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    # ── SFPU binary ───────────────────────────────────────────────────────────
    MathOperation.SfpuElwadd: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    MathOperation.SfpuElwmul: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    MathOperation.SfpuElwsub: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    # div: srcA is the dividend (any value); srcB is the divisor.
    # Use uniform over two bands to exercise both negative and positive divisors
    # while avoiding a small region around 0.
    MathOperation.SfpuElwdiv: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-2.0, high=2.0),
        spec_B=StimuliSpec.uniform(
            intervals=[(-10.0, -0.1), (0.1, 10.0)],
        ),
    ),
    MathOperation.SfpuElwrsub: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    # pow: srcA is the base (non-negative for non-integer exponents); srcB is the
    # exponent (non-negative to keep output finite).
    #
    # Bounded by accuracy, not representable range: a**b evaluates as exp(b * ln a), and
    # measured on Blackhole the *relative* error is ~flat in the operands (10-13% across
    # b * ln a = 3.30 to 11.09) rather than growing with them -- so the old high=3 was the
    # fixed 5% rtol talking, not the op. BINARY_CUSTOM_TOLERANCES gives it rtol=0.15, and
    # A <= 8 / B <= 4 puts the worst-case product at 4 * ln 8 = 8.32. A <= 16 is left out
    # because it drives |a**b| to within 1.06x of Float16's ceiling -- an overflow test
    # dressed as an accuracy one. See BINARY_CUSTOM_TOLERANCES.
    MathOperation.SfpuElwpow: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=8.0),
        spec_B=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=4.0),
    ),
    # xlogy: x * log(y) element-wise. srcA (x): x >= 0 so xlogy(0, y) = 0 is well-defined.
    # srcB (y): y > 0 so log(y) is finite; log-uniform spans several decades.
    #
    # x's ceiling is an absolute-accuracy bound: error is dominated by x * abs_err(ln y), so
    # it grows with x while a fixed atol does not. Measured on Blackhole, max absolute error
    # is linear in x -- 0.25 / 0.50 / 1.00 / 2.00 at x <= 4 / 8 / 16 / 32 in Float16_b, most
    # of it output quantization (a bfloat16 ULP is already 0.5 at |golden| ~ 72) rather than
    # the kernel. BINARY_CUSTOM_TOLERANCES gives it atol=0.6, so x doubles to 8. y keeps its
    # full log-uniform span -- narrowing it made the failures worse, not better.
    MathOperation.SfpuXlogy: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=8.0),
        spec_B=StimuliSpec(
            distribution=DistributionKind.LOG_UNIFORM, low=1e-4, high=10.0
        ),
    ),
    MathOperation.SfpuAddTopRow: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    # shift ops: operate on integer bit patterns; both operands in [0, 255]
    MathOperation.SfpuElwLeftShift: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=255.0)
    ),
    MathOperation.SfpuElwLogicalRightShift: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=255.0)
    ),
    MathOperation.SfpuElwRightShift: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=255.0)
    ),
    # ── Reduce ────────────────────────────────────────────────────────────────
    MathOperation.ReduceColumn: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    MathOperation.ReduceRow: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
    MathOperation.ReduceScalar: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-1.0, high=1.0)
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# for_op — registry lookup
# ─────────────────────────────────────────────────────────────────────────────


def _clip_high(spec: StimuliSpec, ceiling: float, op: MathOperation) -> None:
    """Lower *spec*'s upper bound to *ceiling* in place, if it reaches past it.

    Only low/high specs are supported, which is what every _APPROX_ACCURACY_MAX op uses.
    An interval spec raises rather than being silently left unclipped — that would hand the
    approximation the very region the ceiling exists to withhold.
    """
    if spec.intervals:
        raise ValueError(
            f"MathOperation.{op.name} has an approximation-mode ceiling in "
            f"_APPROX_ACCURACY_MAX but an interval-based domain, which _clip_high cannot "
            f"narrow. Clip the intervals explicitly, or drop the table entry."
        )
    if spec.high is not None and spec.high > ceiling:
        spec.high = ceiling


def _domain_of(spec: StimuliSpec) -> Tuple:
    """The part of *spec* that says which values it draws, and nothing else.

    `StimuliSpec` is a plain dataclass, so `==` would also compare `distribution`, `seed` and
    the per-face fields -- how the values are drawn, not which -- and report "distinct
    per-operand domains" for two operands over the same range sampled differently.
    """
    return (spec.low, spec.high, tuple(spec.intervals or ()))


def _apply_approx_ceiling(
    result: OperandSpecs,
    op: MathOperation,
    approx_mode: Union[bool, Enum, None],
) -> None:
    """Narrow *result* to *op*'s approximation-mode accuracy ceiling, in place.

    A no-op unless *approx_mode* normalises to True and *op* has an entry. The ceiling only
    ever narrows, so a format branch already tighter than it (the e4m3 exp domain stops at
    5.0) is left alone.
    """
    if approx_mode is None:
        return
    if not _approx_mode_flag(approx_mode):
        return
    ceiling = _APPROX_ACCURACY_MAX.get(op)
    if ceiling is None:
        return
    # Every operand, not just A and B. __post_init__ deep-copies C from B before this runs, so
    # clipping only A and B leaves an OperandSpecs whose spec_for(Operand.C) disagrees with the
    # other two -- inert while nothing reads C off an approx-clipped result, and wrong the moment
    # something does.
    others = [s for s in (result.spec_B, result.spec_C) if s is not None]
    if any(_domain_of(s) != _domain_of(result.spec_A) for s in others):
        raise ValueError(
            f"MathOperation.{op.name} has an approximation-mode ceiling but distinct "
            f"per-operand domains. _APPROX_ACCURACY_MAX is written for the unary exp "
            f"family; decide per operand before adding a multi-operand op to it."
        )
    _clip_high(result.spec_A, ceiling, op)
    for spec in others:
        _clip_high(spec, ceiling, op)


def for_op(
    op: MathOperation,
    data_format: DataFormat = DataFormat.Float16_b,
    distribution_a: Optional[Union[DistributionKind, Callable]] = None,
    distribution_b: Optional[Union[DistributionKind, Callable]] = None,
    approx_mode: Union[bool, Enum, None] = None,
) -> OperandSpecs:
    """Return OperandSpecs with safe input domains for *op* and *data_format*.

    Args:
        op: Target math operation.
        data_format: Input data format; controls the numeric range and
            precision used to choose safe per-op input domains (e.g. tighter
            ranges for narrower MX/BFP formats).
        distribution_a: Optional override for spec_A. When None (default),
            spec_A uses the per-op default from the registry — typically
            UNIFORM, but some ops use LOG_UNIFORM, GAUSSIAN, or interval
            uniforms. When set, only the distribution is overridden; all
            other fields on the returned spec stay unchanged, so the safe
            per-op domain is preserved. Some fields may become unused for
            the new distribution, but they are kept as-is. The caller may
            pass either a DistributionKind or a callable accepted by
            StimuliSpec.distribution.
        distribution_b: Same as distribution_a, applied to spec_B. To
            apply the same override to both operands, pass it explicitly
            on both arguments.
        approx_mode: The mode the variant will run in, as an ApproximationMode
            member or a bool. In the approximating mode, ops in
            _APPROX_ACCURACY_MAX have their positive side narrowed to their
            accuracy ceiling; the registry entry carries only the range bound.
            None applies no narrowing, for callers that check no tolerance.

    Returns:
        OperandSpecs with per-operand domain specs.

    Raises:
        KeyError: If *op* is not in the registry.
        TypeError: If any distribution argument is neither a DistributionKind
            member nor a callable.
        ValueError: If overriding to LOG_UNIFORM or LOG_UNIFORM_LINSPACE
            while the spec's domain includes non-positive values.
    """
    entry = _OP_DOMAIN_REGISTRY.get(op)
    if entry is None:
        registered = sorted(o.name for o in _OP_DOMAIN_REGISTRY)
        raise KeyError(
            f"MathOperation.{op.name} has no entry in the stimuli domain "
            f"registry. Add an OperandSpecs entry to _OP_DOMAIN_REGISTRY.\n"
            f"Currently registered ({len(registered)}): {registered}"
        )
    if callable(entry):
        result = copy.deepcopy(entry(data_format))
    else:
        result = copy.deepcopy(entry)

    if distribution_a is not None:
        _validate_distribution_override(distribution_a, result.spec_A)
        result.spec_A.distribution = distribution_a
    if distribution_b is not None:
        if result.spec_B is None:
            raise ValueError(
                f"distribution_b={distribution_b!r} was given but "
                f"MathOperation.{op.name} has no spec_B (single-operand op). "
                f"Drop distribution_b, or override distribution_a instead."
            )
        _validate_distribution_override(distribution_b, result.spec_B)
        result.spec_B.distribution = distribution_b

    _apply_approx_ceiling(result, op, approx_mode)

    return result


def _spec_span(spec: StimuliSpec) -> float:
    """Total measure of the values *spec* is allowed to draw."""
    if spec.intervals:
        return sum(high - low for low, high in spec.intervals)
    return spec.high - spec.low


def _tighter_spec(a: StimuliSpec, b: StimuliSpec) -> StimuliSpec:
    """Whichever of *a* / *b* draws from the smaller domain; ties keep *a*."""
    if a == b:
        return a
    return a if _spec_span(a) <= _spec_span(b) else b


def for_op_pipeline(
    op: MathOperation,
    input_format: DataFormat,
    output_format: Optional[DataFormat] = None,
    **kwargs,
) -> OperandSpecs:
    """Return safe input domains for *op* over a whole input->output pipeline.

    Two different constraints pick two different formats, and resolving against
    either one alone drops the other:

    * **Range** is bounded by the narrowest exponent range anywhere in the
      pipeline. exp over (-100, 80) is fine into a Float32 output and saturates
      a Float16 one, so the *output* format has to be able to narrow the domain.
    * **Precision** is a property of the *input* format alone. A block-float
      input has already spent its relative precision by the time the op runs,
      and a wider output cannot give it back. Resolving Bfp8_b -> Float16
      against Float16 alone restores reciprocal's 1000:1 interval — the exact
      spread _reciprocal_spec exists to avoid, which quantizes small block
      elements to zero and sends the golden to inf.

    So resolve against both formats and keep whichever spec is tighter per
    operand. Both constraints only ever *narrow* a domain, so the tighter of the
    two satisfies both. Ops with a format-independent registry entry resolve
    identically either way and are unaffected.
    """
    by_input = for_op(op, input_format, **kwargs)
    range_format = narrowest_range_format(input_format, output_format)
    if range_format == input_format:
        return by_input

    by_range = for_op(op, range_format, **kwargs)
    # C as well as A and B: rebuilding only A and B would drop a registered third-operand
    # domain, and __post_init__ would then refill spec_C from spec_B -- a wrong answer that
    # looks like a default.
    return OperandSpecs(
        spec_A=_tighter_spec(by_input.spec_A, by_range.spec_A),
        spec_B=_tighter_spec(by_input.spec_B, by_range.spec_B),
        spec_C=_tighter_spec(by_input.spec_C, by_range.spec_C),
    )


def _validate_distribution_override(
    distribution: Union[DistributionKind, Callable],
    spec: StimuliSpec,
) -> None:
    """Catch the obvious incompatibilities between *distribution* and *spec*'s
    existing fields early, instead of letting them fail deep inside
    generate_face / generate_stimuli.

    Currently checked:
      - distribution must be a DistributionKind member or a callable
      - LOG_UNIFORM / LOG_UNIFORM_LINSPACE requires strictly positive bounds
        across spec.low/spec.high or every interval in spec.intervals
      - GAUSSIAN_LINSPACE does not support spec.intervals at all
    """
    if not (callable(distribution) or isinstance(distribution, DistributionKind)):
        raise TypeError(
            f"distribution must be DistributionKind or callable, got "
            f"{type(distribution).__name__!r}: {distribution!r}"
        )

    if distribution == DistributionKind.GAUSSIAN_LINSPACE and spec.intervals:
        raise ValueError(
            f"Cannot override to GAUSSIAN_LINSPACE: spec carries intervals "
            f"{spec.intervals!r}, which gaussian_linspace does not support."
        )

    if distribution in (
        DistributionKind.LOG_UNIFORM,
        DistributionKind.LOG_UNIFORM_LINSPACE,
    ):
        if spec.intervals:
            for lo, hi in spec.intervals:
                if lo <= 0 or hi <= 0:
                    raise ValueError(
                        f"Cannot override to {distribution.name}: "
                        f"spec intervals include non-positive bounds {spec.intervals!r}"
                    )
        elif spec.low <= 0 or spec.high <= 0:
            raise ValueError(
                f"Cannot override to {distribution.name}: spec range "
                f"[{spec.low}, {spec.high}] includes non-positive values"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Undefined-region subtraction
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Which family each registered op belongs to
#
# _OP_DOMAIN_REGISTRY is keyed by MathOperation and mixes the unary SFPU ops with the
# binary ones, the FPU eltwise ops and the reduce family. That is fine for for_op(), but it
# means a suite cannot ask "is my op list complete?" -- the unary sweep can check that no op
# sits in two of its profiles and that every op it drives has a domain, but not that every
# unary op is actually driven, so an op added to the registry and to no test goes untested
# silently.
#
# Recording the family closes that. Each family is named positively, one set per
# arity/engine, and the unary family is whatever is left over -- so an op registered
# without being classified here lands in sfpu_unary_ops() and trips the unary sweep's
# exhaustiveness assert, which is the prompt to put it in the right set below.
#
# They are flat sets rather than a field on OperandSpecs because several entries are shared
# with the perf sweeps and the accuracy harness, and those consumers do not want an arity
# opinion imposed on them.
# ─────────────────────────────────────────────────────────────────────────────

# Eltwise binary run on the FPU rather than the SFPU (test_eltwise_binary.py).
_FPU_ELTWISE_OPS: FrozenSet[MathOperation] = frozenset(
    {
        MathOperation.Elwadd,
        MathOperation.Elwmul,
        MathOperation.Elwsub,
    }
)

# Applied by the packer (STACC_RELU), not the SFPU. Relu is the entry that looks like it
# should be a unary SFPU op: it has a domain, but `relu` is not a member of SfpuType at
# all, so driving it through the unary test fails to compile.
_PACKER_OPS: FrozenSet[MathOperation] = frozenset({MathOperation.Relu})

# Binary SFPU ops (test_eltwise_binary_sfpu.py). Registered ones only -- that suite also drives
# ~30 int, comparison and bitwise ops that have no domain entry and so cannot be keys
# here; they are declared in its own _UNREGISTERED_BINARY_OPS instead.
_SFPU_BINARY_OPS: FrozenSet[MathOperation] = frozenset(
    {
        MathOperation.SfpuAddTopRow,
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwdiv,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuElwrsub,
        MathOperation.SfpuXlogy,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLogicalRightShift,
    }
)

# Ternary SFPU ops (test_sfpu_ternary.py). Membership is what "is this a ternary op"
# keys off -- the cat-B partition below, and _NON_SFPU_UNARY_OPS, which keeps a ternary op
# out of sfpu_unary_ops() should one ever gain a domain entry.
#
# Listed rather than derived from _OP_DOMAIN_REGISTRY the way _SFPU_BINARY_OPS is, because
# no ternary op has a registry entry: OperandSpecs grew spec_C but nothing fills it in yet,
# so a derivation would produce the empty set and every consumer would silently see no
# ternary ops at all. That is the state this set was left in, and it is why the family had
# no cat B: TERNARY_SPECIALS_READY_OPS is partitioned against *this* set.
#
# TTNNWhere is a member despite not going through _run_sfpu_ternary: it is a three-operand
# SFPU kernel driven by the same source and the same suite, and leaving it out would make
# the cat-B totality check below pass by not asking about it.
_SFPU_TERNARY_OPS: FrozenSet[MathOperation] = frozenset(
    {
        MathOperation.SfpuAddcmul,
        MathOperation.SfpuAddcdiv,
        MathOperation.SfpuLerp,
        MathOperation.SfpuSnakeBeta,
        MathOperation.TTNNWhere,
    }
)

# Reduce family (test_sfpu_reduce*.py).
_REDUCE_OPS: FrozenSet[MathOperation] = frozenset(
    {
        MathOperation.ReduceColumn,
        MathOperation.ReduceRow,
        MathOperation.ReduceScalar,
    }
)

# Ops with no unary SFPU kernel, so they must not appear in the unary sweep.
_NON_SFPU_UNARY_OPS: FrozenSet[MathOperation] = (
    _FPU_ELTWISE_OPS | _PACKER_OPS | _SFPU_BINARY_OPS | _SFPU_TERNARY_OPS | _REDUCE_OPS
)

# Unary SFPU ops that are registered but deliberately not in the correctness sweep.
_UNARY_OPS_NOT_SWEPT: Dict[MathOperation, str] = {
    MathOperation.TopKLocalSort: "perf-only; whole-op topk is covered by test_topk.py",
    MathOperation.TopKMerge: "perf-only; whole-op topk is covered by test_topk.py",
    MathOperation.TopKRebuild: "perf-only; whole-op topk is covered by test_topk.py",
}


def sfpu_unary_ops() -> FrozenSet[MathOperation]:
    """Every registered op that has a unary SFPU kernel.

    Everything registered that is not claimed by one of the family sets above. The unary
    sweep drives exactly this set minus _UNARY_OPS_NOT_SWEPT, so a newly registered op is
    swept by default and only escapes by being classified into a family or exempted.
    """
    return frozenset(_OP_DOMAIN_REGISTRY) - _NON_SFPU_UNARY_OPS


_SFPU_UNDEFINED_RANGES: Dict[
    MathOperation,
    Dict[Operand, List[Tuple[float, float]]],
] = {
    # ── Unary: only spec_A has a hole ────────────────────────────────────────
    MathOperation.Reciprocal: {Operand.A: [(-1e-6, 1e-6)]},
    MathOperation.ReciprocalCompat: {Operand.A: [(-1e-6, 1e-6)]},
    MathOperation.Log: {Operand.A: [(-float("inf"), 1e-6)]},
    MathOperation.Sqrt: {Operand.A: [(-float("inf"), 0.0)]},
    MathOperation.Atanh: {
        Operand.A: [(-float("inf"), -1.0 + 1e-6), (1.0 - 1e-6, float("inf"))]
    },
    MathOperation.Log1p: {Operand.A: [(-float("inf"), -1.0 + 1e-6)]},
    MathOperation.Rsqrt: {Operand.A: [(-float("inf"), 1e-6)]},
    MathOperation.Acosh: {Operand.A: [(-float("inf"), 1.0)]},
    # erfinv: defined only on the open interval (-1, 1)
    MathOperation.Erfinv: {
        Operand.A: [(-float("inf"), -1.0 + 1e-6), (1.0 - 1e-6, float("inf"))]
    },
    # ── Binary: per-operand holes ────────────────────────────────────────────
    # div: divisor (srcB) must avoid 0
    MathOperation.SfpuElwdiv: {Operand.B: [(-1e-6, 1e-6)]},
    # xlogy: y (srcB) must be > 0 for log(y) to be finite
    MathOperation.SfpuXlogy: {Operand.B: [(-float("inf"), 1e-6)]},
    # pow: base (srcA) must be > 0 for the exp(b·log(a)) implementation
    MathOperation.SfpuElwpow: {Operand.A: [(-float("inf"), 1e-6)]},
}


def _subtract_intervals(
    base: List[Tuple[float, float]],
    holes: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Take a list of base intervals and cut out all the "hole" intervals.
    Returns the remaining pieces as a sorted list of non-overlapping [lo, hi] ranges.
    """
    result: List[Tuple[float, float]] = []
    for lo, hi in base:
        current = [(lo, hi)]
        for h_lo, h_hi in holes:
            next_segments: List[Tuple[float, float]] = []
            for s_lo, s_hi in current:
                if h_hi <= s_lo or h_lo >= s_hi:
                    next_segments.append((s_lo, s_hi))
                    continue
                if h_lo > s_lo:
                    next_segments.append((s_lo, h_lo))
                if h_hi < s_hi:
                    next_segments.append((h_hi, s_hi))
            current = next_segments
        result.extend(current)
    result.sort()
    return result


def exclude_intervals(
    spec: StimuliSpec,
    holes: List[Tuple[float, float]],
) -> StimuliSpec:
    """Return a copy of *spec* with the given *holes* subtracted from its domain.

    - If spec.intervals is set, those are the base domain.
    - Otherwise [spec.low, spec.high] is used as a single base interval.
    - Raises ValueError if nothing remains after subtraction.
    """
    new_spec = copy.deepcopy(spec)

    if new_spec.intervals:
        base = new_spec.intervals
    else:
        base = [(new_spec.low, new_spec.high)]

    defined = _subtract_intervals(base, holes)
    if not defined:
        raise ValueError(
            f"exclude_intervals produced empty domain from {base} "
            f"minus holes {holes}"
        )

    new_spec.intervals = defined
    return new_spec


def exclude_values(
    spec: StimuliSpec,
    values: List[float],
    epsilon: float = 1e-6,
) -> StimuliSpec:
    """Return a copy of *spec* with tiny intervals around each value excluded.

    For each *v* in *values*, the interval [v - epsilon, v + epsilon] is
    subtracted from the domain.
    """
    holes = [(v - epsilon, v + epsilon) for v in values]
    return exclude_intervals(spec, holes)


def exclude_undefined(
    op: MathOperation,
    spec: StimuliSpec,
    operand: Operand = Operand.A,
) -> StimuliSpec:
    """Return a copy of *spec* with its domain clipped to where *op* is defined
    for the named *operand*.

    Looks up the undefined regions for (*op*, *operand*) in
    _SFPU_UNDEFINED_RANGES and delegates to exclude_intervals.  Returns *spec*
    unchanged if the op (or that operand) has no registered undefined regions.

    Args:
        op: Target math operation.
        spec: Input stimuli spec to clip.
        operand: Which operand the spec corresponds to (Operand.A or Operand.B).
            For unary ops use Operand.A (the default).  For binary ops with
            per-operand restrictions (e.g. div, xlogy, pow), pass the operand
            whose domain you are sanitizing.
    """
    op_ranges = _SFPU_UNDEFINED_RANGES.get(op, {})
    undefined = op_ranges.get(operand)
    if not undefined:
        return spec
    return exclude_intervals(spec, undefined)


def exclude_undefined_pair(
    op: MathOperation,
    specs: "OperandSpecs",
) -> "OperandSpecs":
    """Apply per-operand undefined-region subtraction to every operand of an OperandSpecs.

    Convenience wrapper around exclude_undefined.  Returns a deep copy so the
    caller can mutate further without aliasing the registry.

    Named "_pair" from when OperandSpecs held two operands; it covers C as well now.
    """
    op_ranges = _SFPU_UNDEFINED_RANGES.get(op, {})
    new = copy.deepcopy(specs)
    if Operand.A in op_ranges:
        new.spec_A = exclude_intervals(new.spec_A, op_ranges[Operand.A])
    if Operand.B in op_ranges and new.spec_B is not None:
        new.spec_B = exclude_intervals(new.spec_B, op_ranges[Operand.B])
    if Operand.C in op_ranges and new.spec_C is not None:
        new.spec_C = exclude_intervals(new.spec_C, op_ranges[Operand.C])
    return new


# ─────────────────────────────────────────────────────────────────────────────
# Edge values
#
# Everything above answers "what is safe to draw at random". Everything below answers
# "which single values are worth hitting on purpose", which is a different question with
# a different answer, and the two must not share a table.
#
# The reason they cannot is mechanical: exclude_intervals() *always* rewrites its result
# into the `intervals` form, and the interval sampler consumes two torch.rand draws per
# element where the plain low/high sampler consumes one. So adding an entry to
# _SFPU_UNDEFINED_RANGES re-draws that op's whole stimulus set even when the subtraction
# removes nothing (verified: uniform(1, 8) and intervals=[(1, 8)] are the same
# distribution and different numbers at the same seed). Edge metadata therefore lives in
# its own tables and never touches the random-draw path.
#
# The second reason is semantic. A hole in _SFPU_UNDEFINED_RANGES is a *guard band* —
# Reciprocal's is (-1e-6, 1e-6), so its edges are ±1e-6 and the mathematically
# interesting point, exactly 0, is inside the hole and never produced. Probing wants the
# singularity itself, so _OP_SINGULARITIES records it directly.
# ─────────────────────────────────────────────────────────────────────────────

# Mantissa bits *after* the implicit leading 1, i.e. what sets the step between adjacent
# representable values. Sourced from the same encodings _FORMAT_MAX_MAGNITUDE documents.
# For the block floats this is the per-element magnitude width that utils.py's
# _bfp_block_aware_compare uses (7 / 3 / 1).
_FORMAT_MANTISSA_BITS: Dict[DataFormat, int] = {
    DataFormat.Float32: 23,
    DataFormat.Tf32: 10,
    DataFormat.Float16: 10,  # e5m10
    DataFormat.Float16_b: 7,  # e8m7
    DataFormat.Bfp8: 7,
    DataFormat.Bfp8_b: 7,
    DataFormat.Bfp4_b: 3,
    DataFormat.Bfp2_b: 1,
    DataFormat.MxFp8R: 2,  # e5m2
    DataFormat.MxFp8P: 3,  # e4m3
    DataFormat.Fp8_e4m3: 3,  # e4m3
    DataFormat.MxFp4: 1,  # e2m1
}

# Fall back to bfloat16's precision for anything unlisted: it is the coarsest of the
# common float formats, so a probe spaced for it is spaced widely enough for the rest.
_DEFAULT_MANTISSA_BITS = 7

# Stimuli are built as fp32 host-side, so a narrower format's width is subtracted from
# fp32's to get the number of mantissa bits the datapath drops.
_FLOAT32_MANTISSA_BITS = _FORMAT_MANTISSA_BITS[DataFormat.Float32]


def dest_truncation_mask(dst_format: DataFormat) -> int:
    """The bits a 32-bit datum keeps when it lands in a 16-bit *dst_format* Dest.

    Derived from the mantissa-width table above rather than written as a literal, so a width
    change moves the mask with it. Both goldens truncate the operand on the way in at
    dest_acc=No; an FP16 Dest keeps three more mantissa bits than a BF16 one.
    """
    dropped = _FLOAT32_MANTISSA_BITS - _FORMAT_MANTISSA_BITS[dst_format]
    return (0xFFFFFFFF << dropped) & 0xFFFFFFFF


def format_ulp(fmt: DataFormat, magnitude: float = 1.0) -> float:
    """Distance to the next representable value of *fmt* near |*magnitude*|.

    This is what a boundary probe has to be offset by. A fixed epsilon does not work:
    at a boundary of 1.0 in Float16_b, 1.0 - 1e-6 *is* 1.0, so the "just inside" and
    "at" probes collapse onto one point and the pair silently tests half of what it
    reads as testing.

    For the block-float formats the real step is set by the exponent shared across the
    16-element block, not by the element's own magnitude, so the value returned here is
    a lower bound for a small element sitting inside a block with a large maximum. That
    is the safe direction for a probe (too fine merely wastes a value; too coarse walks
    past the boundary), but it means a block-float probe cannot be assumed distinct.
    """
    bits = _FORMAT_MANTISSA_BITS.get(fmt, _DEFAULT_MANTISSA_BITS)
    magnitude = abs(magnitude)
    if magnitude == 0.0 or not math.isfinite(magnitude):
        # No exponent to work from. Use the step at 1.0, which is representable in every
        # float format and is a visible distance from zero in all of them.
        return 2.0**-bits
    return 2.0 ** (math.floor(math.log2(magnitude)) - bits)


def probe_spacing_format(
    fmt: DataFormat,
    dest_acc: Optional[Union[bool, Enum]] = None,
) -> DataFormat:
    """Which format's ULP a boundary probe has to step by to survive the datapath.

    narrowest_range_format() ranks on exponent range and so bounds a probe's **magnitude**;
    its **spacing** is a mantissa question neither the input nor the output format settles.
    With ``dest_acc=No`` the DEST holds 16 bits whatever the input is, so an fp32 probe
    stepped by an fp32 ULP truncates straight back onto the boundary it was meant to
    straddle. The goldens model that truncation as ``& 0xFFFF0000``.

    Acosh is what this bites today: at singularity (1.0, ABOVE) with ``(Float32, Float16_b)``
    the above-pole probe is ``0x3F800002``, which arrives as 1.0 under ``dest_acc=No`` -- a
    second copy of the pole probe. Nothing false-passes; the probe simply stops probing.

    Only a 32-bit *float* fmt is coarsened, and only to Float16_b. Narrower formats land in a
    DEST no coarser than themselves, and mantissa truncation is not what a 16-bit integer DEST
    does. ``dest_acc=None`` keeps the format-only behaviour.

    This is the *format* half of the rule; probe_beside() applies it per boundary **and per
    side**, because whether the narrowing destroys a probe depends on both.
    """
    if dest_acc is None or _dest_acc_flag(dest_acc):
        return fmt
    if fmt.is_integer() or not fmt.is_32_bit():
        return fmt
    return DataFormat.Float16_b


def _truncate_mantissa(value: float, fmt: DataFormat) -> float:
    """*value* with its mantissa truncated to *fmt*'s width, keeping its exponent.

    Models what a narrower DEST does to a wider datum, which the goldens spell for the
    bfloat16 case as ``& 0xFFFF0000``. Round-toward-zero rather than round-to-nearest: this
    is only ever used to ask "is this probe still distinct from its boundary", and nearest
    can move a probe *away* from the boundary but never onto it, so truncating is the
    conservative direction.
    """
    if not math.isfinite(value):
        return value
    dropped = _FLOAT32_MANTISSA_BITS - _FORMAT_MANTISSA_BITS.get(
        fmt, _DEFAULT_MANTISSA_BITS
    )
    if dropped <= 0:
        return value
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    raw &= 0xFFFFFFFF ^ ((1 << dropped) - 1)
    return struct.unpack("<f", struct.pack("<I", raw))[0]


def probe_beside(
    point: float,
    direction: int,
    range_fmt: DataFormat,
    step_fmt: DataFormat,
    ulps: int = 1,
) -> float:
    """A probe *direction* (+1 / -1) off *point*: the tightest one that still arrives distinct.

    Steps by *ulps* ULPs of *range_fmt* first, which is the closest to the boundary the
    stimulus format can express. Widens to *step_fmt*'s ULP only when that fine probe would
    be quantized back onto *point* by a narrower datapath.

    Deciding per **side** keeps this to the probes that actually collapse. At a pole of 1.0
    under a 16-bit DEST, stepping up gives ``0x3F800002``, which truncates back to 1.0;
    stepping *down* crosses into the next binade (``0x3F7FFFFF`` -> 0.99609375) and stays
    distinct, so asin, acos, atanh, erfinv and log1p keep their tighter below-1.0 probes.
    Zero-poles keep theirs too, since bfloat16 carries fp32's full exponent range.
    """
    fine = point + direction * ulps * format_ulp(range_fmt, point)
    # Both sides truncated: the datapath truncates the boundary as well as the probe, and the
    # question is whether they stay distinct *as the kernel sees them*. Every point registered
    # today is bfloat16-exact, so this is equivalent to comparing against the bare point -- it
    # stops being equivalent as soon as one is not.
    if _truncate_mantissa(fine, step_fmt) != _truncate_mantissa(point, step_fmt):
        return fine
    return point + direction * ulps * format_ulp(step_fmt, point)


# Exact singular points of each op, per operand — the pole, the branch cut, the value
# where the function stops being defined. New per-op data, and deliberately so: the
# undefined-range table records a guard band wide enough to keep a random draw away from
# a singularity, which is not the same thing as the singularity's location.
#
# Held back on purpose: Lgamma, Digamma and Polygamma all have poles at 0 and the
# negative integers, but their kernels are polynomial/LUT fits that only claim accuracy
# well inside a positive domain (see their registry comments). A probe at their boundary
# tests a value the kernel never promised, which produces a failure that is neither a bug
# nor fixable. Add them if and when the kernels claim that range.
# Each singularity carries **which side of it the op is defined on**, because that decides
# whether a probe one ULP away is a legitimate edge case or a value the kernel never
# promised anything about:
#
#   BOTH  - defined on both sides (1/x for x<0 and x>0); probe both, plus the point.
#   ABOVE - defined only above (log, sqrt); probe the point and above it.
#   BELOW - defined only below (the +1 end of asin/atanh).
#
# Measured need for this: without it, probing the undefined side failed for 8 ops, and in
# every case the *golden* was the thing that broke — torch-backed goldens return `inf`
# where the mathematical answer is `nan` (log(-eps) -> golden inf against a finite
# hardware result; sqrt(-eps) -> golden inf against hardware 0). Those are golden defects
# worth fixing, but they are not what a boundary probe is for, and they are Phase 5 work.
# Pass include_undefined=True to probe the far side anyway once the goldens model it.
SingularitySide = Enum("SingularitySide", "BOTH ABOVE BELOW")
_BOTH, _ABOVE, _BELOW = (
    SingularitySide.BOTH,
    SingularitySide.ABOVE,
    SingularitySide.BELOW,
)

_OP_SINGULARITIES: Dict[
    MathOperation, Dict[Operand, Tuple[Tuple[float, SingularitySide], ...]]
] = {
    # 1/x and everything built on a reciprocal: pole at exactly 0, defined either side.
    MathOperation.Reciprocal: {Operand.A: ((0.0, _BOTH),)},
    MathOperation.Rdiv: {Operand.A: ((0.0, _BOTH),)},  # rdiv(x) = 2.0 / x
    # log family: log(0) = -inf and negative arguments are undefined.
    MathOperation.Log: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.LogWithBase: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.Log1p: {Operand.A: ((-1.0, _ABOVE),)},  # log1p(x) = log(1 + x)
    # sqrt / rsqrt: 0 is the edge of the domain, and rsqrt's pole as well.
    MathOperation.Sqrt: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.SqrtCustom: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.Rsqrt: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.RsqrtCompat: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.ReciprocalCompat: {Operand.A: ((0.0, _BOTH),)},
    # Inverse functions defined only on (-1, 1) or [-1, 1]: the interior is the defined
    # side, so -1 is probed upward and +1 downward.
    MathOperation.Atanh: {Operand.A: ((-1.0, _ABOVE), (1.0, _BELOW))},
    MathOperation.Erfinv: {Operand.A: ((-1.0, _ABOVE), (1.0, _BELOW))},
    MathOperation.Asin: {Operand.A: ((-1.0, _ABOVE), (1.0, _BELOW))},
    MathOperation.Acos: {Operand.A: ((-1.0, _ABOVE), (1.0, _BELOW))},
    MathOperation.Acosh: {Operand.A: ((1.0, _ABOVE),)},
    # Binary: the singularity sits on one specific operand.
    MathOperation.SfpuElwdiv: {Operand.B: ((0.0, _BOTH),)},
    MathOperation.SfpuXlogy: {Operand.B: ((0.0, _ABOVE),)},
    MathOperation.SfpuElwpow: {Operand.A: ((0.0, _ABOVE),)},
    # fmod / remainder divide by B, so B = 0 is their pole. Neither has an entry in
    # _SFPU_UNDEFINED_RANGES — they are on the positive-only format default — and adding
    # one there would re-roll their stimuli (see the section header). Recording the
    # singularity here instead is free, because this table never touches the draw path.
    MathOperation.SfpuBinaryFmod: {Operand.B: ((0.0, _BOTH),)},
    MathOperation.SfpuBinaryRemainder: {Operand.B: ((0.0, _BOTH),)},
    # atan2(y, x) has no pole, but x = 0 is a branch point: atan2(y, +0) = +/-pi/2 by the
    # sign of y, and crossing to x < 0 with y held at 0 jumps the result to +/-pi. Registered
    # on operand B (x) and probed from both sides. Nothing else reaches it -- atan2 keeps the
    # format default, whose draw is positive-only.
    MathOperation.SfpuAtan2: {Operand.B: ((0.0, _BOTH),)},
    # Ternary: the pole is on the *third* operand. addcdiv is a + value * b / c and
    # snake_beta is a + sin(b*a)^2 / c, so c = 0 is a pole for both, and
    # _ternary_default_specs holds c in uniform(1, 2) -- so only the edge sweep reaches it.
    MathOperation.SfpuAddcdiv: {Operand.C: ((0.0, _BOTH),)},
    MathOperation.SfpuSnakeBeta: {Operand.C: ((0.0, _BOTH),)},
}


def ops_with_singularity(
    operand: Optional[Operand] = None,
) -> FrozenSet[MathOperation]:
    """Every op with a recorded singularity, optionally only those on *operand*.

    This is how an edge sweep enrols its ops: intersect it with the ops the suite can
    drive rather than listing them, so adding an entry to _OP_SINGULARITIES is enough to
    get the op probed. A hand-written second list is the failure mode -- the table grows
    and the sweep silently does not.
    """
    if operand is None:
        return frozenset(_OP_SINGULARITIES)
    return frozenset(
        op for op, per_operand in _OP_SINGULARITIES.items() if operand in per_operand
    )


def _dedup_representable(values: List[float], fmt: DataFormat) -> List[float]:
    """Sort *values* and drop those *fmt* cannot tell apart from the previous one.

    Two probes closer together than half a ULP quantize to the same value on the way to
    the device, so keeping both spends a stimulus slot on a duplicate. Non-finite values
    are kept verbatim and appended after the sorted finite values, in their original
    relative order.

    **-0.0 is exempt.** It compares equal to +0.0 and is zero ULPs away from it, so a
    plain numeric dedup discards it — and for signbit, sign, heaviside and reciprocal the
    difference between the two zeros is the entire point of the probe (1/+0 = +inf against
    1/-0 = -inf, and the SFPU returns +inf for both, which is what forced Bfp4_b's tighter
    reciprocal domain). Signed zeros are therefore keyed by sign, not by value.
    """
    finite = sorted(v for v in values if math.isfinite(v))
    non_finite = [v for v in values if not math.isfinite(v)]
    kept: List[float] = []
    seen_zeros = set()
    for v in finite:
        if v == 0.0:
            # math.copysign distinguishes the two zeros where == cannot.
            sign = math.copysign(1.0, v)
            if sign in seen_zeros:
                continue
            seen_zeros.add(sign)
            kept.append(v)
            continue
        if kept and abs(v - kept[-1]) < 0.5 * format_ulp(
            fmt, max(abs(v), abs(kept[-1]))
        ):
            continue
        kept.append(v)
    return kept + non_finite


def boundary_probes(
    op: MathOperation,
    operand: Operand = Operand.A,
    fmt: DataFormat = DataFormat.Float16_b,
    ulps: int = 2,
    include_undefined: bool = False,
    step_fmt: Optional[DataFormat] = None,
) -> List[float]:
    """Values straddling every boundary of *op*'s defined region for *operand*.

    Two sources, in order of preference for the operand asked about:

    * _OP_SINGULARITIES, when it has an entry: the exact singular point *p*, plus one
      probe on each side the op is actually *defined* on (see SingularitySide).
      ``include_undefined=True`` adds the far side too.
    * _SFPU_UNDEFINED_RANGES otherwise: each finite guard-band edge gives the edge plus
      one probe on its defined side. This is the fallback that makes a *newly* declared
      hole yield probes with no second table entry, which is the "derive it from data
      that already exists" property worth keeping.

    Deliberately not both. A guard band sits a fixed 1e-6 off the true boundary, so the
    two sources disagree about which binade the boundary is in — at a boundary of 1.0 the
    band edge 0.999999 has half the ULP that 1.0 has, and the pair emits a third probe
    that is neither the boundary nor a clean step from it. The singularity is strictly
    better information, so where it exists it wins outright.

    ``eps`` is *ulps* steps of *fmt* at the boundary's own magnitude (see format_ulp),
    not a fixed constant, so the probes stay distinct in low-precision formats.

    *step_fmt* widens that step where the datapath is coarser than *fmt* — pass
    probe_spacing_format(fmt, dest_acc) to account for a 16-bit DEST holding a 32-bit
    format. It defaults to *fmt*, which is the format-only behaviour. See probe_beside().

    Returns a sorted list with format-indistinguishable duplicates removed. Values are
    *not* clipped to what *fmt* can represent or to the op's registered domain — that is
    the caller's job, and for a sweep pairing input with output formats it has to be done
    against the narrowest format in the pipeline, not against one end of it.
    """
    probes: List[float] = []
    step_fmt = fmt if step_fmt is None else step_fmt

    singularities = _OP_SINGULARITIES.get(op, {}).get(operand, ())
    if singularities:
        for point, side in singularities:
            probes.append(point)
            if point == 0.0:
                # A zero pole is two probes, not one: 1/+0 is +inf and 1/-0 is -inf, and the
                # same split runs through div, fmod, remainder, xlogy, atan2, addcdiv and
                # snake_beta. _dedup_representable() keys zeros by sign rather than by value,
                # so both survive it.
                #
                # copysign rather than a -0.0 literal because the registered point is only
                # *conventionally* +0.0 -- deriving the opposite sign keeps this correct if an
                # entry is ever written as -0.0.
                #
                # Emitted unconditionally here. edge_values() drops it again on the pipelines
                # that flatten a -0.0 to +0.0, which is a delivery question this function has
                # no dest_acc to answer.
                probes.append(math.copysign(0.0, -math.copysign(1.0, point)))
            if include_undefined or side in (
                SingularitySide.BOTH,
                SingularitySide.BELOW,
            ):
                probes.append(probe_beside(point, -1, fmt, step_fmt, ulps))
            if include_undefined or side in (
                SingularitySide.BOTH,
                SingularitySide.ABOVE,
            ):
                probes.append(probe_beside(point, +1, fmt, step_fmt, ulps))
    else:
        for lo, hi in _SFPU_UNDEFINED_RANGES.get(op, {}).get(operand, ()):
            if math.isfinite(lo):
                probes += [probe_beside(lo, -1, fmt, step_fmt, ulps), lo]
            if math.isfinite(hi):
                probes += [hi, probe_beside(hi, +1, fmt, step_fmt, ulps)]

    # Dedup against *fmt*, not *step_fmt*: probe_beside() has already guaranteed every probe
    # survives the datapath, and a step_fmt dedup would then discard the tight below-boundary
    # probes it deliberately kept (1.0 and 0.99999976 are under half a bfloat16 ULP apart).
    return _dedup_representable(probes, fmt)


# ─────────────────────────────────────────────────────────────────────────────
# Shared special-value lists, by format class
# ─────────────────────────────────────────────────────────────────────────────

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
UINT32_MAX = 2**32 - 1


# +0.0 and -0.0 are listed separately and both matter: signbit, sign, heaviside,
# reciprocal and the comparison-to-zero ops all distinguish them, and reciprocal is the
# op where they disagree most visibly (1/+0 = +inf against 1/-0 = -inf, and the SFPU
# returns +inf for both — that sign disagreement is what forced Bfp4_b's tighter
# reciprocal domain).
FLOAT_SPECIALS: Tuple[float, ...] = (
    float("inf"),
    float("-inf"),
    float("nan"),
    0.0,
    -0.0,
)


def _is_negative_zero(value: float) -> bool:
    """True only for -0.0. `value == -0.0` is also true for +0.0, so read the sign bit."""
    return value == 0.0 and math.copysign(1.0, value) < 0.0


# Shift amounts worth driving on purpose, shared by the unary and binary shift sweeps: the
# in-range ends (0, 31), the first out-of-range value (32), larger ones, and negatives.
#
# The *amounts* only, deliberately not the rule for what they produce -- three consumers,
# two behaviours:
#
#   binary shifts        every out-of-range amount produces 0, both signs
#   unary left shift     the same: calculate_left_shift zeroes the result
#   unary right shift    calculate_right_shift clamps the amount to 31 and shifts anyway, so
#                        a negative operand yields -1 rather than 0
#
# Each golden states its own kernel's rule. Shared here because both suites drive it: binary
# shift ops take the amount as an operand, unary ones as a compile-time immediate
# (SFPU_SHIFT_AMOUNT).
# fmt: off
SHIFT_EDGE_AMOUNTS: Tuple[int, ...] = (
    0, 1, 2, 7, 15, 16, 30, 31,      # in range
    32, 33, 40, 63, 100, 1000,       # >= 32
    -1, -5, -32, -1000,              # < 0
)
# fmt: on


def integer_specials(fmt: DataFormat) -> Tuple[int, ...]:
    """Extreme and near-extreme values for an integer *fmt*.

    Derived from the format's width rather than hard-coded to 32 bits, so Int16/Int8 get
    their own extremes instead of int32's clamped down to something meaningless.

    The signed minimum is included and is the one value that cannot be delivered through
    StimuliSpec: CustomStrategy clamps integers through _get_integer_bounds, which
    returns ``info.min + 1`` because Dst stores integers as sign-magnitude and the
    INT_MIN bit pattern is "negative zero" there. Deliver it as a raw override tensor
    (see _build_shift_edge_case_src) and expect it to fail on hardware — that is a
    documented HW limitation, not a test gap.
    """
    if not fmt.is_integer():
        raise ValueError(f"{fmt.name} is not an integer format")
    bits = int(fmt.size) * 8
    if fmt.name.startswith("UInt"):
        return (0, 1, 2**bits - 1)
    signed_min = -(2 ** (bits - 1))
    return (signed_min, signed_min + 1, -1, 0, 1, 2 ** (bits - 1) - 1)


def format_specials(fmt: DataFormat) -> Tuple[float, ...]:
    """IEEE specials for a float *fmt*, integer extremes for an integer one."""
    if fmt.is_integer():
        return integer_specials(fmt)
    return FLOAT_SPECIALS


# ─────────────────────────────────────────────────────────────────────────────
# Cat F — finite magnitude extremes
#
# The counterpart to format_specials() at the other end: everything above is about values
# outside the reals, this is about the two ends of the format's own exponent range. Dumping
# _OP_DOMAIN_REGISTRY shows why it is needed -- the widest registered domain is Square at
# +/-1000 and CastFp32ToFp16a at +/-1e5, everything else is inside +/-100, so the input space
# jumps from ~10 straight to infinity and nothing occupies the 30-odd decades in between or
# the band immediately above zero. The goldens model flush-to-zero carefully
# (golden_generators._FTZ_THRESHOLD) and, until this, no test drove an input that triggered it.
#
# Kept as its own axis rather than folded into `specials`, because the delivery rules differ
# and the failure classes are unrelated: a subnormal survives an unpack that destroys a NaN,
# and an overflowing *result* is a pack question rather than an unpack one. Folding them would
# give one gate answering two questions and one xfail covering two causes.
# ─────────────────────────────────────────────────────────────────────────────


def format_extremes(fmt: DataFormat) -> Tuple[float, ...]:
    """Finite magnitude extremes of *fmt*, both signs.

    Four magnitudes:

      * the ceiling, where an op that grows must saturate rather than wrap;
      * the largest step below it, so a kernel that special-cases the ceiling alone still has
        a neighbour to get right;
      * the smallest normal, the last magnitude the format holds at full precision;
      * half of that, which is subnormal by construction -- a real subnormal in a format that
        keeps them (Float16) and a value on the flushed side of the cliff in one that does not
        (Float32, Float16_b). One probe, and which of the two it is depends on the format
        rather than on this function guessing.

    The ceiling comes from _FORMAT_MAX_MAGNITUDE with the same bfloat16 fallback
    clip_to_format() uses -- for Float32 that is bfloat16's 3.39e38 rather than fp32's 3.40e38,
    the codebase's standing rule that a pipeline is bounded by its narrowest float -- and is
    then rounded down onto that format's grid, so a probe emitted here is exactly representable
    and is never one clip_to_format() goes on to drop.

    Raises for an integer format -- integer extremes are cat C and come from integer_specials(),
    which knows they cannot be delivered through a StimuliSpec at all -- and for a block-float
    one, whose smallest element is a property of the block's shared exponent rather than of the
    format.
    """
    if fmt.is_integer():
        raise ValueError(
            f"{fmt.name} is an integer format: its extremes are cat C, from "
            "integer_specials(), not cat F"
        )
    if fmt not in _FORMAT_MIN_NORMAL:
        raise ValueError(
            f"{fmt.name} has no per-element smallest normal (see _FORMAT_MIN_NORMAL): its "
            "small end is set by the exponent shared across a block, so a cat-F probe would "
            "be wrong for every block but one"
        )
    # Whose ceiling this actually is. A format with no _FORMAT_MAX_MAGNITUDE row is one at
    # least as wide as bfloat16, and the fallback it takes is *bfloat16's* ceiling -- so the
    # value must be rounded to bfloat16's grid, not to fp32's. Unrounded, the decimal literal
    # 3.3895314e38 sits a hair above the true bfloat16 maximum and is therefore not a value any
    # 16-bit Dest in the pipeline can hold: the probe would quantize on the way in and stop
    # being the ceiling, which is the "a literal like 88.7 becomes 88.5 in bf16" failure in its
    # least visible form.
    ceiling_fmt = fmt if fmt in _FORMAT_MAX_MAGNITUDE else DataFormat.Float16_b
    ceiling = _truncate_mantissa(
        _FORMAT_MAX_MAGNITUDE.get(fmt, _BF16_MAX_MAGNITUDE), ceiling_fmt
    )
    # One ULP of the *ceiling's* format, for the same reason: an fp32-sized step below a
    # bfloat16 ceiling truncates straight back onto it under a 16-bit Dest, which is the
    # collapse probe_beside() exists to avoid at a pole.
    below_ceiling = _truncate_mantissa(
        ceiling - format_ulp(ceiling_fmt, ceiling), ceiling_fmt
    )
    min_normal = _FORMAT_MIN_NORMAL[fmt]
    subnormal = min_normal / 2.0
    magnitudes = (ceiling, below_ceiling, min_normal, subnormal)
    return tuple(-m for m in magnitudes) + magnitudes


def extremes_safe(
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: Union[bool, Enum],
) -> bool:
    """May format_extremes() be injected on this (input, output, dest_acc) triple?

    Deliberately **not** specials_safe(), and the difference is the point of having two gates.
    specials_safe() encodes two measured breakers about non-finites -- a Float16 anywhere in
    the pipeline, and a 16-bit input into a 32-bit Dest -- both of which are about +/-inf and
    NaN surviving an unpack. A finite magnitude has no such problem: it is an ordinary datum
    with an extreme exponent, and every pipeline that carries 1.0 carries 3.39e38 and 2^-126
    as well.

    What it does exclude is block-float and MX on either leg, and for a reason specific to cat
    F rather than borrowed from cat B: those formats share one exponent across a 16-element
    block, and this probe list spans thirty-odd decades *inside a single block*. The shared
    exponent follows the ceiling and the small end of the list quantizes to zero -- so the
    probe would be destroyed by the container rather than by anything the kernel did. Integer
    legs are excluded because cat F is a float question; integer extremes are cat C.

    ``dest_acc`` is accepted for symmetry with specials_safe() and normalised the same way, so
    a caller passing the enum member cannot silently select the wrong branch. It does not
    currently change the answer: the 16-bit Dest a dest_acc=No pipeline uses is bfloat16 (or
    Float16, where the pipeline already has one), and both carry the exponent range of the
    range format that clip_to_format() bounds the probe by. Kept in the signature because a
    Dest narrower in *exponent* than the input would change that, and a caller would then have
    no way to say so.
    """
    _dest_acc_flag(dest_acc)

    if input_format.is_integer() or output_format.is_integer():
        return False
    if input_format.is_block_float() or input_format.is_mx_format():
        return False
    if output_format.is_block_float() or output_format.is_mx_format():
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Op-specific discrete edges
#
# Only points that are not already a domain boundary: piecewise knees, comparison
# thresholds, exact rounding ties. The registry's random domains are chosen to land
# *near* several of these; this table is what lands *on* them.
#
# Every dispatch constant this table probes at is imported from
# sfpu_dispatch_constants, which UnarySFPUGolden reads too — so there is one number, not
# a copy per consumer. Restating them here was the drift bug the module exists to
# prevent: change the golden's threshold and a table with its own copy keeps probing a
# point that is no longer a threshold, which reads as full coverage while testing
# nothing. Values still written literally below (hardsigmoid's [-3, 3], the rounding
# ties, the integer knees) are properties of the mathematics rather than of a kernel's
# dispatch, so there is nothing on the golden side for them to drift from.
# ─────────────────────────────────────────────────────────────────────────────

# Ops whose only interesting point is exactly zero, and where +0.0/-0.0 may differ.
_ZERO_EDGE_OPS = (
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
    MathOperation.Sign,  # -1 / 0 / +1, so 0 is the whole middle branch
    MathOperation.Signbit,  # true for -0.0, false for +0.0
    MathOperation.Heaviside,  # returns the dispatch value 0.5 at exactly 0
    # Relu is deliberately absent: its knee is at 0 like the rest, but relu is applied by
    # the packer (STACC_RELU) and is not a member of SfpuType, so no SFPU probe can reach
    # it. See _NON_SFPU_UNARY_OPS.
    MathOperation.Lrelu,  # LRELU_NEGATIVE_SLOPE applies below 0
    MathOperation.Prelu,  # PRELU_SLOPE applies below 0
    MathOperation.Elu,
    MathOperation.Celu,
    MathOperation.Selu,
    MathOperation.Xielu,  # _xielu switches alpha_p/alpha_n at 0
)

# Ops whose value at exactly zero is known and exactly representable, so a fit that is off by a
# constant at the origin is visible there and nowhere else.
#
# These are smooth at zero rather than piecewise, so this is not a knee in the sense the
# comment above describes -- it is an *exact-value* probe, and it earns its place for the same
# reason a knee does: the registered domains are symmetric uniforms of order 1 to 8, and a
# random draw over one lands on exactly 0.0 essentially never. What it pins:
#
#   erf(+/-0) = +/-0     tanh(+/-0) = +/-0     gelu(+/-0) = +/-0     expm1(+/-0) = +/-0
#   I1(+/-0) = +/-0      power(+/-0) = 0       cast(+/-0) = +/-0
#   erfc(+/-0) = 1       sigmoid(+/-0) = 0.5   gelu_derivative(+/-0) = 0.5
#   tanh_derivative(+/-0) = 1                  rpow(+/-0) = 1
#
# Measured on a Blackhole p150 before enrolling; all fourteen agree within tolerance on every
# cell, and two of them do so with a constant error the probe now holds in place:
# erfc(+0) = 0.99612 and erfc(-0) = 1.00388 against 1.0 -- sign-asymmetric at the origin, which
# nothing else in the suite would show -- and tanh_derivative(+/-0) = 0.99920 against 1.0 on a
# 32-bit dest. Both are inside atol today; if either fit drifts, this is what reports it.
#
# The gamma family is absent and excludes *itself*: Digamma, Lgamma and Polygamma have poles at
# zero and registered domains that start above it (0.1, 1.0, 0.5), so the membership test below
# leaves them out without a second list to maintain. That is the same reasoning recorded above
# _OP_SINGULARITIES for their poles -- a probe there tests a value the kernel never promised.
_EXACT_AT_ZERO_OPS = (
    MathOperation.Erf,
    MathOperation.Erfc,
    MathOperation.Gelu,
    MathOperation.GeluDerivative,
    MathOperation.Sigmoid,
    MathOperation.SigmoidAppx,
    MathOperation.Tanh,
    MathOperation.TanhDerivative,
    MathOperation.TanhDerivativeLut,
    MathOperation.Expm1Cw,
    MathOperation.I1,
    MathOperation.Rpow,
    MathOperation.UnaryPower,
    MathOperation.CastFp32ToFp16a,
)

_COMPARISON_EDGE_OPS = (
    MathOperation.UnaryGt,
    MathOperation.UnaryLt,
    MathOperation.UnaryGe,
    MathOperation.UnaryLe,
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
)

# The integer binary ops whose answer at zero is a knee rather than an ordinary value:
# gcd(0, x) = |x| and lcm(0, x) = 0 are identities, and 0 is the annihilator for the multiply
# and the neutral element for max/min. 1 comes with it because it is the multiplicative
# identity gcd and lcm are next most likely to disagree on.
#
# Registered here rather than listed in the test that drives them, which is convention 1: an op
# joins a sweep by gaining a table entry. It also makes the coverage ledger tell the truth --
# with the values in a test-file list, cat D read "no knee registered" for ops that were being
# driven at one, which is the ledger claiming *less* than the suite delivers.
#
# The divisor ops are absent: a zero divisor is undefined for them (see
# _INT_ZERO_UNDEFINED_DIVISOR in the binary suite), and a zero *dividend* is an ordinary value
# rather than a knee, so there is nothing to register.
_INT_ZERO_KNEE_OPS = (
    MathOperation.SfpuGcd,
    MathOperation.SfpuLcm,
    MathOperation.SfpuMulInt32,
    MathOperation.SfpuMaxInt32,
    MathOperation.SfpuMinInt32,
    MathOperation.SfpuMaxUint32,
    MathOperation.SfpuMinUint32,
)

_OP_EDGE_POINTS: Dict[MathOperation, Tuple[float, ...]] = {
    **{op: (0.0, -0.0) for op in _ZERO_EDGE_OPS},
    **{op: (0.0, -0.0) for op in _EXACT_AT_ZERO_OPS},
    **{op: (0.0, 1.0) for op in _INT_ZERO_KNEE_OPS},
    # UnaryGt/Lt/Ge/Le reach the edge sweep through edge_spec(). UnaryEq and UnaryNe do not
    # -- they are outside _OP_DOMAIN_REGISTRY, so their consumer is
    # test_eltwise_unary_sfpu._threshold_op_stimuli_spec, which reads op_edge_points() directly to
    # place the exact threshold in its stimuli, as the int32 comparison ops below do.
    **{op: (UNARY_COMP_THRESHOLD,) for op in _COMPARISON_EDGE_OPS},
    # logical_not(x) = (x == 0). Same shape as _ZERO_EDGE_OPS but it is a threshold op
    # rather than a sign op, so keep it named. (LogicalNotUnary is an alias of this
    # member — see the note in llk_params.py — so listing both would be one key.) Also
    # outside the registry, and read by _threshold_op_stimuli_spec as above.
    MathOperation.LogicalNot: (0.0, -0.0),
    # unary max/min compare x against UNARY_MAX_MIN_VALUE. Keyed on the constant rather
    # than folded into _ZERO_EDGE_OPS: it happens to be 0.0 today, and if it moves the
    # probe has to move with it.
    MathOperation.UnaryMax: (UNARY_MAX_MIN_VALUE, -UNARY_MAX_MIN_VALUE),
    MathOperation.UnaryMin: (UNARY_MAX_MIN_VALUE, -UNARY_MAX_MIN_VALUE),
    # Clamp / hardtanh bounds.
    MathOperation.Clamp: (CLAMP_MIN, CLAMP_MAX),
    MathOperation.Hardtanh: (CLAMP_MIN, CLAMP_MAX),
    # Shrinkage lambdas: below |lambda| the op returns 0.
    MathOperation.Softshrink: (-SOFTSHRINK_LAMBDA, SOFTSHRINK_LAMBDA),
    MathOperation.Hardshrink: (-HARDSHRINK_LAMBDA, HARDSHRINK_LAMBDA),
    # torch hardsigmoid is piecewise on [-3, 3].
    MathOperation.Hardsigmoid: (-3.0, 3.0),
    # hardmish(x) = x * clamp(0.5x + 1, 0, 1): the clamp saturates at x = -2 and x = 0.
    MathOperation.Hardmish: (-2.0, 0.0),
    # Below THRESHOLD_T the output jumps to THRESHOLD_V.
    MathOperation.Threshold: (THRESHOLD_T,),
    # relu_max clamps above at its threshold, and keeps relu's own knee at 0.
    MathOperation.ReluMax: (0.0, RELU_MAX_THRESHOLD),
    MathOperation.ReluMin: (RELU_MIN_THRESHOLD,),
    # softplus goes linear at its threshold.
    MathOperation.Softplus: (SOFTPLUS_THRESHOLD,),
    # Round-half-to-even ties, where the kernel's _round_even_ and a naive round differ.
    MathOperation.Round: (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5),
    # Integer knees. floor/ceil differ from trunc only on the negative side, and frac
    # keeps the sign of x, so each list has to span both.
    MathOperation.Floor: (-2.0, -1.0, 0.0, 1.0, 2.0),
    MathOperation.Ceil: (-2.0, -1.0, 0.0, 1.0, 2.0),
    MathOperation.Trunc: (-1.0, 0.0, 1.0),
    MathOperation.Frac: (-1.5, -1.0, 1.0, 1.5),
    # Integer scalar comparisons against UnarySFPUGolden._int_maxmin_scalar. These four
    # are not in _OP_DOMAIN_REGISTRY, so sfpu_unary_ops() never puts them in an edge
    # sweep and edge_spec() never sees them: their consumer is
    # test_eltwise_unary_sfpu._int_unary_stimuli_spec, which reads op_edge_points() directly to
    # place the exact comparison tie in its stimuli. Keep that call in mind before
    # editing — dropping these entries makes the tie untestable rather than merely
    # unlisted.
    MathOperation.UnaryMaxInt32: (INT_MAXMIN_SCALAR,),
    MathOperation.UnaryMinInt32: (INT_MAXMIN_SCALAR,),
    MathOperation.UnaryMaxUint32: (INT_MAXMIN_SCALAR,),
    MathOperation.UnaryMinUint32: (INT_MAXMIN_SCALAR,),
    # IEEE pow(x, 0) == 1 for every x, including a negative base. (-2)**0 must stay +1
    # rather than picking up the odd-integer sign flip; 2**0 is the matching positive
    # control. Cartesian-producted with Operand.B's zero encodings in
    # _OP_OPERAND_EDGE_POINTS, these are the committed (base, -0.0) pairs.
    MathOperation.SfpuElwpow: (-2.0, 2.0),
}


# Cat D for an operand other than A. _OP_EDGE_POINTS describes the op's own input, which for
# a unary or binary op is operand A. A ternary op breaks that: lerp is a + c * (b - a), so
# its interesting values are properties of the *weight*, operand C. pow's interesting
# exponent encodings live here for the same reason: the singularity is on the base.
#
# A second per-operand table rather than a nested _OP_EDGE_POINTS: one of the 44 entries has
# per-operand structure, and a dict layer on all of them would read worse.
_OP_OPERAND_EDGE_POINTS: Dict[MathOperation, Dict[Operand, Tuple[float, ...]]] = {
    # lerp interpolates at c = 0 (result is exactly a), reaches b at c = 1, and
    # *extrapolates* beyond it for c > 1 -- three different behaviours of one kernel, none
    # of which the default uniform(-1, 1) weight lands on. 2.0 is the extrapolating probe;
    # -1.0 extrapolates the other way, which is the same branch and a different sign.
    MathOperation.SfpuLerp: {Operand.C: (-1.0, 0.0, 1.0, 2.0)},
    # IEEE pow(x, 0) == 1, and SFPSETCC's contract excludes negative zero, so the kernel
    # compares on setsgn(pow, 0). Without -0.0 here, B falls through to edge_counterparts()
    # which only contributes +0.0, and removing setsgn would leave 0**0 green while
    # 0**-0.0 went back to inf. 0.0/1.0/2.0 are the counterparts this entry replaces, so
    # 0**1 and 0**2 stay in the sweep as the over-firing-guard controls. -0.0 is dropped
    # by edge_values() where negative_zero_delivered() is false, so the datacopy path
    # does not claim coverage for a sign it flattened.
    MathOperation.SfpuElwpow: {Operand.B: (-0.0, 0.0, 1.0, 2.0)},
}


def op_edge_points(
    op: MathOperation, operand: Operand = Operand.A
) -> Tuple[float, ...]:
    """Discrete edges of *op* for *operand* that are not already a domain boundary.

    Operand A reads _OP_EDGE_POINTS, the op's own knees. Any other operand reads
    _OP_OPERAND_EDGE_POINTS (lerp's weight boundaries, pow's exponent encodings).
    Returns () when there is nothing to probe.
    """
    if operand == Operand.A:
        return _OP_EDGE_POINTS.get(op, ())
    return _OP_OPERAND_EDGE_POINTS.get(op, {}).get(operand, ())


# ─────────────────────────────────────────────────────────────────────────────
# Where IEEE specials can actually be injected
#
# A cat-B sweep must not be a plain product over formats x dest_acc. Measured on
# Wormhole (n150) by driving the isinf/isposinf/isneginf/isnan/isfinite predicates over the
# full 5x5 format matrix x both dest_acc values with no skips — 250 variants, 85 failing.
# The predicates are the right instrument because their output is 0.0/1.0, representable in
# every format including the block floats, so a failure isolates "the input's specialness did
# not survive unpack" from "the output cannot express a non-finite result".
#
# Two independent breakers came out of it:
#
#   1. A Float16 (e5m10) anywhere in the pipeline. As an *input* it never preserves specials
#      — all 5 predicates fail on all 5 output formats at both dest_acc, 10/10 cells. As an
#      *output* it fails too, unless a 32-bit input is paired with dest_acc=Yes:
#      Float32->Float16 at dest_acc=No fails all five, the exact pair Blackhole already
#      guards in _skip_bh_unsupported_float_combo.
#
#   2. A 16-bit input with dest_acc=Yes. Float16_b there fails isinf, isneginf and isnan
#      while isposinf and isfinite pass — +inf survives, -inf and NaN do not. Precisely the
#      "bf16->fp32 dest unpack does not preserve -inf/nan, mangling is_neg/is_nan" already
#      recorded on test_eltwise_unary_sfpu_isinf_isnan, now with per-predicate detail.
#
# A third constraint is not measurable this way and is applied statically: block-float and MX
# *inputs* cannot carry specials at all. Verified host-side — quantize_input_to_unpack_format()
# destroys NaN for Bfp8_b and Bfp4_b (±inf survives). A predicate passing on a block-float
# input is therefore vacuous, golden and hardware agreeing there is no NaN because neither
# saw one. Those rows are excluded rather than trusted.
# ─────────────────────────────────────────────────────────────────────────────

# Float formats whose unpack quantization leaves +inf, -inf and NaN intact, so the golden
# still evaluates the op at the special the test meant to inject.
_SPECIALS_CARRYING_INPUTS: FrozenSet[DataFormat] = frozenset(
    {DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b}
)


# Ops whose golden defines a result for non-finite *inputs*, and may therefore have cat-B
# specials injected. This is the golden-side gate; specials_safe() above is the
# pipeline-side one, and both have to pass. Neither implies the other: a pipeline can
# deliver NaN perfectly to a golden that has no answer for it.
#
# Empty, and that is a measurement rather than caution. Injecting specials on every triple
# specials_safe() allows gives **272 failures out of 564 variants** -- 48% -- and the
# failures are not the (format, dest_acc) matrix, which is gated correctly, but goldens
# that return inf where IEEE says nan and so on. The expansion plan's rule of thumb was
# "default to injecting the edge; xfail the handful the golden cannot yet express"; the
# measurement says the handful is half the op list, which makes cat B golden work rather
# than a stimulus change.
#
# Per op rather than one global bool, because the global only had two states: no specials
# anywhere, or ~270 xfails -- and 270 xfails is not coverage, it is a monument. An op joins
# here once its golden defines a result at +inf, -inf, NaN, +0.0 and -0.0, carrying the
# reason it is ready; the sweeps then inject specials for that op alone. That turns the
# remaining cat-B work into a series of one-op commits rather than one that cannot land.
#
# The first tranche. Eight of the ten needed no golden change -- their goldens route through
# torch, which is IEEE-correct at every special, verified host-side before enrolling. Sin and
# Cos needed one: they called math.sin / math.cos, which *raise* on a non-finite input rather
# than returning NaN, and now route through torch.
#
# What this does not claim: the *sign of a zero result* is a separate, arch-dependent question
# -- SFPMAD flushes negative zero to positive on Wormhole and preserves it on Blackhole -- so
# Neg(+0) -> -0 and Reciprocal(-inf) -> -0 are covered by the measurement that arch-gated the
# binary suite's negative-zero class.
SPECIALS_READY_OPS: Dict[MathOperation, str] = {
    MathOperation.Identity: "Pass-through: every special maps to itself. Green on "
    "Blackhole across all safe triples.",
    MathOperation.Abs: "Magnitude: |+/-inf| = +inf, |NaN| = NaN, |+/-0| = 0. Green on "
    "Blackhole.",
    MathOperation.Exp: "IEEE: exp(+inf) = +inf, exp(-inf) = 0, exp(+/-0) = 1. Green on "
    "Blackhole.",
    MathOperation.Sin: "sin(+/-inf) = NaN, sin(NaN) = NaN, sin(+/-0) = +/-0. Golden moved "
    "off math.sin, which *raised* on a non-finite input rather than returning one. Green "
    "on Blackhole.",
    MathOperation.Cos: "cos(+/-inf) = NaN, cos(NaN) = NaN, cos(+/-0) = 1. Golden moved off "
    "math.cos for the same reason as Sin. Green on Blackhole.",
    MathOperation.Neg: "neg(+/-inf) = -/+inf, neg(NaN) = NaN, neg(+/-0) = -/+0. Enrolled "
    "once the golden stopped mangling a NaN's sign through a 16-bit Dest (see "
    "cast_to_dest_dtype): Neg is the one op here that produces a *negative* NaN, so it was "
    "the op that measured the defect. Green on Blackhole.",
    MathOperation.Reciprocal: "IEEE: 1/+/-inf = +/-0, 1/+/-0 = +/-inf, 1/NaN = NaN. The "
    "kernel does not propagate NaN (returns +0), which is a genuine kernel divergence and "
    "is xfailed per combination rather than papered over in the golden.",
    MathOperation.Sqrt: "IEEE: sqrt(+inf) = +inf, sqrt(-inf) = NaN, sqrt(NaN) = NaN, "
    "sqrt(+/-0) = +/-0. The kernel returns NaN for sqrt(-0) where IEEE gives -0, but only "
    "where a -0 is actually delivered (unpack-to-dest); xfailed there.",
    MathOperation.Rsqrt: "IEEE: rsqrt(+inf) = +0, rsqrt(-inf) = NaN, rsqrt(NaN) = NaN, "
    "rsqrt(+/-0) = +/-inf. Same -0 divergence as Sqrt and the same unpack-to-dest scoping.",
}

# The third tranche, enrolled in bulk: all 84 unenrolled ops with a golden, driven over the full
# specials set on every Blackhole-reachable triple. 48 agreed with their goldens everywhere and
# are enrolled here; the other 34 diverge and stay out until each is understood.
#
# "Agreed" means the golden and the kernel give the same answer at every special the pipeline
# delivers, which is what this suite can establish -- not that the golden is independently
# correct. The risk is small because these goldens route through torch or are plain arithmetic.
_SPECIALS_READY_UNCHANGED: Tuple[MathOperation, ...] = (
    MathOperation.Acosh,
    MathOperation.Add1,
    MathOperation.Asinh,
    MathOperation.Atan,
    MathOperation.Atanh,
    MathOperation.Cbrt,
    MathOperation.Ceil,
    MathOperation.Celu,
    MathOperation.Cosh,
    MathOperation.Elu,
    MathOperation.EqualZero,
    MathOperation.Exp2,
    MathOperation.ExpWithBase,
    MathOperation.Expm1,
    MathOperation.Floor,
    MathOperation.Fmod,
    MathOperation.GeluAppx,
    MathOperation.GeluTanh,
    MathOperation.GreaterThanEqualZero,
    MathOperation.GreaterThanZero,
    MathOperation.Hardmish,
    MathOperation.LessThanEqualZero,
    MathOperation.LessThanZero,
    MathOperation.Log1p,
    MathOperation.Lrelu,
    MathOperation.Mish,
    MathOperation.NotEqualZero,
    MathOperation.Prelu,
    MathOperation.Remainder,
    MathOperation.Round,
    MathOperation.Selu,
    MathOperation.Signbit,
    MathOperation.Silu,
    MathOperation.Sinh,
    MathOperation.Softplus,
    MathOperation.Softshrink,
    MathOperation.Softsign,
    MathOperation.Tanhshrink,
    MathOperation.Threshold,
    MathOperation.Trunc,
    MathOperation.UnaryLe,
    MathOperation.UnaryLt,
    MathOperation.UnaryMax,
    MathOperation.Xielu,
)

# The three that needed the same one-line fix as _sin and _cos: math.acos / math.asin / math.tan
# *raise* ValueError("math domain error") on a non-finite input rather than returning NaN, so a
# cat-B probe reached them as a test error. Every remaining `math.*` call in a unary golden is a
# latent repeat.
_SPECIALS_READY_TORCH_ROUTED: Tuple[MathOperation, ...] = (
    MathOperation.Acos,
    MathOperation.Asin,
    MathOperation.Tan,
)

SPECIALS_READY_OPS.update(
    {
        op: "Enrolled in the third tranche with no golden change: driven over the full "
        "specials set on every Blackhole-reachable triple and agreed with its golden at "
        "each one. See _SPECIALS_READY_UNCHANGED for what that does and does not establish."
        for op in _SPECIALS_READY_UNCHANGED
    }
)
SPECIALS_READY_OPS.update(
    {
        op: "Enrolled in the third tranche once its golden moved off math.acos / math.asin / "
        "math.tan, which raise on a non-finite input instead of returning NaN -- the same "
        "defect _sin and _cos carried. Green on Blackhole afterwards."
        for op in _SPECIALS_READY_TORCH_ROUTED
    }
)

# Three of the 34 the sweep left out were the *golden's* fault: a guard written for finite inputs
# ("did this overflow?", "is this inside the shrink band?") answers false for NaN and routes it
# somewhere wrong. Listed separately because each needed a fix.
SPECIALS_READY_OPS.update(
    {
        MathOperation.Square: "square(+/-inf) = +inf, square(NaN) = NaN. The golden tested "
        "isfinite(x * x) to detect overflow, which is also false for NaN, so it reported inf "
        "where the kernel correctly returns NaN. Green on Blackhole after the fix.",
        MathOperation.I0: "I0(+/-inf) = +inf (I0 is even and unbounded), I0(NaN) = NaN. "
        "torch.special.i0 returns NaN at +/-inf, which is a torch limitation rather than the "
        "mathematics -- the kernel was the correct party here and the golden was not.",
        MathOperation.Hardshrink: "hardshrink(NaN) = NaN. The golden's |x| > lambda test is "
        "false for NaN, which sent it to the shrink-to-zero branch; torch and the kernel both "
        "propagate. Green on Blackhole after the fix.",
    }
)

# The comparison family, enrolled once the goldens learned the SFPU's total order.
#
# These seven look like kernel divergences -- each returns its own upper-bound dispatch
# constant where IEEE says NaN -- but SFPGT, SFPLE and SFPSWAP all route through
# SignMagIsSmaller(), which documents
#
#     -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN
#
# so +NaN outranks every finite value and clamping one lands on the upper bound. The goldens
# were the wrong party; sfpu_total_order_key models the documented behaviour and these enrol
# as ordinary passes. Verified on Wormhole too, which has no SFPGT/SFPLE but documents the
# same order on SFPSWAP.
#
# UnaryLt, UnaryLe and UnaryMax are absent because they were already enrolled: under the total
# order their answers coincide with IEEE's at a +NaN.
SPECIALS_READY_OPS.update(
    {
        MathOperation.UnaryGt: "x > 0.5 under the SFPU's total order, in which +NaN is the "
        "largest FP32 value, so NaN > 0.5 is true. Not IEEE, which makes it false.",
        MathOperation.UnaryGe: "As UnaryGt.",
        MathOperation.UnaryMin: "min(x, 0.0) under the total order. +NaN is the maximum, so "
        "the minimum is the other operand -- which is why this diverged where UnaryMax did not.",
        MathOperation.Clamp: "clamp(x, -1, 1) applied as metal `calculate_clamp` applies it: "
        "max(x, min) then min(x, max), both SFPSWAP total-order folds, so a +NaN outranks "
        "everything, survives the max, and lands on max via the min.",
        MathOperation.Hardtanh: "clamp(x, -1, 1) via metal `calculate_hardtanh`, i.e. "
        "`sfpi::clamp` -- the same SFPSWAP max-then-min composition as Clamp's kernel, so the "
        "two share one golden by construction. The identity is pinned host-side.",
        MathOperation.ReluMax: "_relu_max_body_: a total-order `> threshold` replaces a NaN "
        "with the threshold, and the relu clamp then sees a finite value.",
        MathOperation.Hardsigmoid: "x * (1/6) + 0.5 through the same _relu_max_body_ the "
        "kernel shares with ReluMax, so a NaN clamps to 1.0.",
    }
)

# The five scalar binops. They have their own suite and golden (ScalarBinopGolden), so no unary
# tranche reached them, and cat B is their *entire* edge story: each is `x (+|-|*|/) c` for a
# compile-time c, smooth in x with no pole and no knee, so edge_spec() returns None unless
# specials are on. Their goldens are plain fp32 torch arithmetic and needed no fix.
SPECIALS_READY_OPS.update(
    {
        MathOperation.ScalarAdd: "x + c: +/-inf and NaN pass through the add, +/-0 + c = c. "
        "Golden is plain fp32 arithmetic.",
        MathOperation.ScalarSub: "x - c: as ScalarAdd with the sign of c.",
        MathOperation.ScalarMul: "x * c: +/-inf * c keeps the sign for a finite non-zero c, "
        "NaN propagates, and +/-0 * c keeps the zero's sign.",
        MathOperation.ScalarDiv: "x / d, which the host turns into x * (1/d) at compile time, "
        "so it is ScalarMul at the kernel and d never reaches the device. A divide-by-zero is "
        "therefore unreachable through this op.",
        MathOperation.ScalarRsub: "c - x: +inf and -inf swap, NaN propagates, c - (+/-0) = c.",
    }
)

# Fill is enrolled, but read its entry narrowly: the kernel writes a compile-time constant and
# its golden ignores the input, so the probe asserts only that a non-finite input does not
# corrupt the fill -- nothing about NaN semantics.
SPECIALS_READY_OPS[MathOperation.Fill] = (
    "Input-independent: fill writes a constant, so every special maps to that constant. The "
    "probe asserts the fill survives a non-finite input, not any NaN semantics. Green on "
    "Blackhole."
)

# Three facts from the Neg/Reciprocal/Sqrt/Rsqrt tranche that the entries above rest on.
#
# **The Dest-write cast.** torch's fp32 -> bfloat16 cast canonicalises every NaN to 0xFFFF,
# sign bit set, so a NaN crossing a 16-bit Dest came back negative whatever its true sign was
# -- which is why the defect showed only at dest_acc=No and read as "Neg(NaN) mangled": Neg is
# the one op here whose NaN is genuinely negative. cast_to_dest_dtype models the Dest write as
# the truncation it is, and Signbit's enrolment via _SPECIALS_READY_UNCHANGED is only sound
# because of it.
#
# **A zero's sign is invisible to this comparator.** passed_test() judges by torch.isclose, a
# both-NaN clause and PCC, and -0.0 == +0.0 under all three, so Neg(+0) -> -0 and
# Reciprocal(-inf) -> -0 can neither fail nor XPASS. Asserting one would need a bitwise
# comparator -- a suite-wide change.
#
# **-0.0 delivery**, measured: at dest_acc=No, Reciprocal, Rsqrt and Sqrt treat -0 exactly as
# +0 (1/-0 -> +inf, rsqrt(-0) -> +inf, sqrt(-0) -> +0); at dest_acc=Yes with a 32-bit input
# they do not (sqrt(-0), rsqrt(-0) -> NaN). That is the unpack_to_dest split, and it is what
# scopes Sqrt's and Rsqrt's xfails to unpack-to-dest.
#
# The fourth tranche. The three families each carry a *_SPECIALS_READY_OPS and a
# _*_SPECIALS_NOT_READY dict, and the totality test asserts the partition -- except that the
# unary family only ever had the first half. 28 ops sat outside cat B with nothing recording
# whether that was a decision or an omission.
#
# Driven on a Blackhole p150 over the full specials set, on every specials-safe cell, after a
# host-side pass confirming all 28 goldens *answer* at every special (none raises, unlike the
# third tranche's math.acos / math.asin / math.tan). Three agreed and are enrolled below; the
# 25 that did not are in _UNARY_SPECIALS_NOT_READY, grouped by measured cause.
#
# ONE RECORD HERE WAS STALE, which is the reason to re-measure rather than transcribe. This
# comment used to read:
#
#     Log  +inf -> golden +inf, hw 88.5   (~ln(FLT_MAX): the kernel clamps a non-finite
#          -inf -> golden NaN,  hw 84.3    input to the format maximum and takes the log
#          NaN  -> golden NaN,  hw 89.1    of that, so no non-finite input survives)
#
# It is no longer true. Measured now, Log(+inf) = +inf, Log(-inf) = NaN and Log(NaN) = NaN, all
# agreeing with the golden on every safe cell -- the kernel does not clamp. The note was
# accurate when written and went quietly false, which is the failure mode convention 3 warns
# about from the other direction: a reason string is a claim about hardware with a shelf life.
SPECIALS_READY_OPS.update(
    {
        MathOperation.Log: "IEEE: log(+inf) = +inf, log(-inf) = NaN, log(NaN) = NaN, "
        "log(+/-0) = -inf. The kernel no longer clamps a non-finite input to the format "
        "maximum -- the behaviour this file recorded for years -- so it now agrees with the "
        "golden on every safe cell. Green on Blackhole.",
        MathOperation.LogWithBase: "As Log: the base is a compile-time multiply on the "
        "result, so every non-finite answer is Log's scaled, and the scale cannot rescue or "
        "break one. Green on Blackhole.",
        MathOperation.ReluMin: "min(x, threshold) through the same _relu_max_body_ the "
        "comparison family enrolled on: a total-order compare, in which +NaN is the largest "
        "value, so relu_min(NaN) is the threshold. Measured green on Blackhole by driving the "
        "spec directly -- read this entry as a verdict with no consumer yet, because "
        "test_eltwise_unary_sfpu_edges skips ReluMin outright for tt-llk issue #1120. The "
        "enrolment takes effect the day that skip lifts; it is here so the skip is the only "
        "thing holding the op back.",
    }
)

# The 25 that diverge. Five causes, not twenty-five -- which is what the grouping records, and
# it is also the honest limit of what one tranche establishes. Each entry says what was
# measured; none claims the op has been individually investigated.

# (1) A reciprocal composition returning +0 for 1/NaN. Identical to the divergence already
# recorded against unary Reciprocal, reached through the same SFPARECIP, and section 5.6 Q1's
# composition question rather than a new one per op.
_UNARY_NOT_READY_RECIPROCAL: Tuple[MathOperation, ...] = (
    MathOperation.Rdiv,
    MathOperation.ReciprocalCompat,
    MathOperation.RsqrtCompat,
    MathOperation.Sigmoid,
    MathOperation.TanhDerivative,
)

# (2) A LUT or polynomial fit evaluated at a non-finite. SFPLUTFP32 documents no NaN/inf
# handling and a polynomial has no notion of one, so each returns whatever its fit produces
# from a saturated or wrapped argument -- Erf(NaN) = 1.0, Gelu(-inf) = 0.0,
# TanhDerivativeLut(+/-inf) = +inf, Digamma(NaN) = 89. What the composition should do there is
# an LLK decision, not an ISA one.
_UNARY_NOT_READY_LUT: Tuple[MathOperation, ...] = (
    MathOperation.Digamma,
    MathOperation.Erf,
    MathOperation.Erfc,
    MathOperation.Gelu,
    MathOperation.GeluDerivative,
    MathOperation.Lgamma,
    MathOperation.Polygamma,
    MathOperation.SigmoidAppx,
    MathOperation.Tanh,
    MathOperation.TanhDerivativeLut,
)

# (3) The kernel saturates a non-finite input to +/-1.1547668e37, which _i1_bessel's own
# comment already records. Its golden was fixed for correctness rather than to enrol it.
_UNARY_NOT_READY_SATURATES: Tuple[MathOperation, ...] = (MathOperation.I1,)

# (4) The *golden* is the wrong party, and in a way already fixed once elsewhere. Both of these
# route a NaN through a finite-input comparison that is false for it, landing in the zero
# branch: Sign(NaN) reports 0 and Heaviside(NaN) reports 0.5. The kernel answers 1.0 for both,
# which is what the SFPU's total order gives -- +NaN is the largest FP32 value -- and is the
# same reading that let UnaryGt, UnaryGe, UnaryMin, Clamp, Hardtanh, ReluMax and Hardsigmoid
# enrol. So these are the two most likely next enrolments, and they need a golden change first,
# not a measurement. Kept out until then rather than enrolled against a golden known to be
# wrong.
#
# Both also carry the separate, genuine SFPSETCC negative-zero divergence recorded in
# test_eltwise_unary_sfpu -- that one is the kernel's and would survive the golden fix.
_UNARY_NOT_READY_GOLDEN_TOTAL_ORDER: Tuple[MathOperation, ...] = (
    MathOperation.Sign,
    MathOperation.Heaviside,
)

# (5) Each its own question, and none of them a composition the groups above cover. Grouped
# only by "not yet understood", which is what the reason string says.
_UNARY_NOT_READY_UNGROUPED: Dict[MathOperation, str] = {
    MathOperation.CastFp32ToFp16a: "cast(NaN) reaches L1 as +inf on Float32->Float32 at "
    "dest_acc=Yes, where a NaN should survive. A cast is the one op whose whole job is the "
    "format conversion the pack path also performs, so which of the two substituted the "
    "infinity needs a read-back to say.",
    MathOperation.Expm1Cw: "expm1_cw(+inf) returns NaN where the golden and IEEE give +inf. "
    "Only on the 32-bit-dest cell; the narrower ones agree, which points at the subtraction "
    "of 1 rather than at the exponential.",
    MathOperation.Erfinv: "erfinv(-inf) returns -inf where the golden gives +inf, on top of "
    "the erfinv(+/-1) saturation already recorded as a divergence in the edge sweep. The two "
    "may be one cause; nothing establishes that yet.",
    MathOperation.Frac: "frac(-inf) returns +inf. frac(x) = x - trunc(x) is NaN at either "
    "infinity mathematically, and the golden says -inf, so all three disagree -- the golden "
    "needs settling before the kernel can be judged.",
    MathOperation.Rpow: "rpow(-inf) returns +inf where the golden gives 0. base**(-inf) is 0 "
    "for a base above 1, so the kernel's exp(x * ln base) composition is not carrying the "
    "sign of the exponent through.",
    MathOperation.SqrtCustom: "sqrt_custom(+inf) returns NaN and sqrt_custom(-inf) returns "
    "+inf -- both signs wrong in opposite directions, on the 32-bit-dest cell only. Sqrt "
    "itself is enrolled and correct there, so this is the custom variant's own path.",
    MathOperation.UnaryPower: "power(NaN) returns +inf on the 32-bit-dest cell. Its exponent "
    "is a compile-time immediate, so this is the same exp(b * ln a) composition SfpuElwpow is "
    "held out of cat B for.",
}

_UNARY_SPECIALS_NOT_READY: Dict[MathOperation, str] = {}
_UNARY_SPECIALS_NOT_READY.update(
    {
        op: "composition through a reciprocal, which returns +0 for 1/NaN -- the divergence "
        "unary Reciprocal already carries, through the same SFPARECIP. Section 5.6 Q1."
        for op in _UNARY_NOT_READY_RECIPROCAL
    }
)
_UNARY_SPECIALS_NOT_READY.update(
    {
        op: "a LUT or polynomial fit evaluated at a non-finite, which SFPLUTFP32 documents no "
        "handling for, so the result is whatever the fit produces from a saturated argument. "
        "Section 5.6 Q1."
        for op in _UNARY_NOT_READY_LUT
    }
)
_UNARY_SPECIALS_NOT_READY.update(
    {
        op: "the kernel saturates a non-finite input to +/-1.1547668e37 rather than "
        "propagating it; see this op's golden comment, which records the same."
        for op in _UNARY_NOT_READY_SATURATES
    }
)
_UNARY_SPECIALS_NOT_READY.update(
    {
        op: "the *golden* misroutes a NaN through a finite-input comparison into its zero "
        "branch; the kernel's answer is the SFPU total order's, in which +NaN is the largest "
        "value. Needs a golden change, not a measurement -- see the group comment."
        for op in _UNARY_NOT_READY_GOLDEN_TOTAL_ORDER
    }
)
_UNARY_SPECIALS_NOT_READY.update(_UNARY_NOT_READY_UNGROUPED)

assert not (
    set(SPECIALS_READY_OPS) & set(_UNARY_SPECIALS_NOT_READY)
), "an op cannot be both enrolled in cat B and recorded as not ready for it"


def _dest_acc_flag(dest_acc: Union[bool, Enum]) -> bool:
    """Normalise a 32-bit-destination flag to a plain bool.

    DestAccumulation is an Enum whose two members wrap True and False, so ``bool(member)``
    is True for *both* of them -- ``DestAccumulation.No`` included. A caller that passes
    the member directly instead of ``dest_acc == DestAccumulation.Yes`` therefore does not
    get an error, it gets every triple evaluated as the 32-bit-dest case, which silently
    flips specials on and off for whole rows of the matrix.

    Read ``.value`` when handed the enum, take a bool as-is, and reject anything else
    rather than guessing. The enum itself is not imported here: this module deliberately
    carries no llk_params test-side types beyond MathOperation, and duck-typing on
    ``.value`` keeps it that way.
    """
    return _two_state_flag(dest_acc, "dest_acc", "DestAccumulation")


def specials_safe(
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: Union[bool, Enum],
) -> bool:
    """May FLOAT_SPECIALS be injected on this (input, output, dest_acc) triple?

    ``dest_acc`` is a 32-bit-destination flag: either a plain bool or a
    ``DestAccumulation`` member. Both are accepted because the member is the natural
    thing for a caller to have, and its truthiness is a trap -- see _dest_acc_flag.

    Returns False for anything not positively established, so a new format defaults to
    "do not inject" rather than to a wall of failures with one root cause. Every rule
    below reproduces a measured verdict; see the section comment for the measurement.
    """
    dest_acc = _dest_acc_flag(dest_acc)

    if input_format not in _SPECIALS_CARRYING_INPUTS:
        return False  # block-float / MX / integer input cannot carry them at all

    if input_format == DataFormat.Float16:
        return False  # breaker 1: never preserves specials, any output, any dest_acc

    if output_format == DataFormat.Float16:
        # breaker 1 on the output side: only a 32-bit input into a 32-bit dest survives.
        if not (input_format.is_32_bit() and dest_acc):
            return False

    if not input_format.is_32_bit() and dest_acc:
        return False  # breaker 2: 16-bit -> fp32 dest unpack loses -inf and NaN

    if output_format.is_block_float() or output_format.is_mx_format():
        # Not a measured failure — the predicates pass here because their result is 0/1.
        # Excluded on the golden's behalf: an inf/NaN result inside a block whose shared
        # exponent is finite is not a value the format can express, so neither the
        # lattice nor the tolerance criterion in passed_test means anything for it.
        return False

    return True


def negative_zero_delivered(
    input_format: DataFormat, dest_acc: Optional[Union[bool, Enum]]
) -> bool:
    """Does a -0.0 written to L1 still have its sign when the SFPU reads it?

    Only on the unpack-to-dest path -- a 32-bit input at dest_acc=Yes. Everywhere else the
    datum goes through SrcA and the datacopy, and the LREG holds +0.0. Measured two ways: the
    signbit/sign/heaviside divergence partition, and Reciprocal/Rsqrt/Sqrt over the same probe
    (1/-0 -> +inf, rsqrt(-0) -> +inf, sqrt(-0) -> +0 at dest_acc=No; NaN for the latter two at
    dest_acc=Yes, a distinct answer that says a real -0 arrived).

    Strictly narrower than specials_safe(), which asks whether a pipeline preserves
    non-finites at all: several triples it accepts carry +/-inf and NaN intact while flattening
    -0.0. Sending the probe there costs an xfail per variant that blames the kernel for a datum
    it never received.

    dest_acc=None means the caller does not know the pipeline, so keep the probe.
    """
    if dest_acc is None:
        return True
    return input_format.is_32_bit() and _dest_acc_flag(dest_acc)


def _is_subnormal(value: float, fmt: DataFormat) -> bool:
    """Is *value* a non-zero magnitude below *fmt*'s smallest normal?

    Formats with no per-element smallest normal (the block floats) answer False: their small
    end is the block's shared exponent, which is not a property of a single value.
    """
    min_normal = _FORMAT_MIN_NORMAL.get(fmt)
    if min_normal is None:
        return False
    return value != 0.0 and abs(value) < min_normal


def subnormal_delivered(
    input_format: DataFormat, dest_acc: Optional[Union[bool, Enum]]
) -> bool:
    """Does a subnormal written to L1 still have a non-zero magnitude when the SFPU reads it?

    Only on the unpack-to-dest path -- a 32-bit input at dest_acc=Yes. Everywhere else the datum
    goes through SrcA and the datacopy, which normalises, and the LREG holds +0.0.

    Measured on a Blackhole p150 by driving +/-2^-127 through Ceil, Floor, Sign and Signbit --
    the four cat-F tranche-1 ops whose answer at a subnormal differs from their answer at zero,
    which is what makes them the instrument. On Float16_b->Float16_b at dest_acc=Yes,
    Float16_b->Float32 at dest_acc=Yes and Float32->Float32 at dest_acc=No, all four answer as
    though the input were +0.0: ceil(+2^-127) -> 0 rather than 1, floor(-2^-127) -> 0 rather
    than -1, sign(+/-2^-127) -> 0, signbit(-2^-127) -> 0. On Float32->Float32 at dest_acc=Yes
    all four agree with the golden. Round, Trunc and Abs agree everywhere, which is the control:
    their answer at a subnormal is the same as their answer at zero, so the flush is invisible
    to them.

    Same partition as negative_zero_delivered(), and the same cause -- SrcA cannot hold either
    datum -- but kept as its own predicate because it is its own measured fact about its own
    class of value, and because the two would have to be un-merged the moment a pipeline handled
    one and not the other.

    Where this is False the probe is dropped rather than xfailed, for the reason Signbit's six
    former entries record: an xfail there would blame the kernel for a datum it never received,
    and no kernel change could ever clear it. The assertion it forgoes -- that the pipeline
    *does* flush, and that the op then behaves as it does at zero -- would need the goldens to
    model the flush on the input side, which is a suite-wide change rather than a probe.

    dest_acc=None means the caller does not know the pipeline, so keep the probe.
    """
    if dest_acc is None:
        return True
    return input_format.is_32_bit() and _dest_acc_flag(dest_acc)


def nan_survives_to_l1(
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: Optional[Union[bool, Enum]],
) -> bool:
    """Does a NaN the kernel produces reach L1 still a NaN, or as a signed infinity?

    Keyed on (dst_format, output) so it mirrors UnarySFPUGolden's own preservation rule rather
    than restating its result: the golden keeps a NaN for {(Float16, Float16),
    (Float32, Float16), (Float32, Float32)} and routes everything else through
    convert_nan_to_inf, which rewrites exponent and mantissa and leaves the sign bit alone. So
    wherever this is False, a NaN arrives at the comparator as +/-inf and its sign is suddenly
    load-bearing. SFPSTORE documents the hardware half on both arches ("NaN is also converted
    to infinity, so software is advised to avoid NaN inputs for this conversion").

    This asks about the *output* leg where negative_zero_delivered() asks about the input leg:
    of the six triples specials_safe() accepts on the {Float16_b, Float32} matrix, five narrow
    somewhere and only Float32->Float32 at dest_acc=Yes carries a NaN the whole way.

    dest_acc=None means the caller does not know the pipeline, so keep the assertion.
    """
    if dest_acc is None:
        return True

    # The Dest format the golden derives from the same two inputs. Block-float and MX are
    # not reachable here -- specials_safe() rejects them on both legs before this is asked.
    if _dest_acc_flag(dest_acc):
        dst_format = DataFormat.Float32
    elif DataFormat.Float16 in (input_format, output_format):
        dst_format = DataFormat.Float16
    else:
        dst_format = DataFormat.Float16_b

    return (dst_format, output_format) in {
        (DataFormat.Float16, DataFormat.Float16),
        (DataFormat.Float32, DataFormat.Float16),
        (DataFormat.Float32, DataFormat.Float32),
    }


# The ops whose NaN result is one the kernel *invents*, not one it forwards.
#
# IEEE 754 leaves the sign of an invalid-operation default unspecified, and the two arches
# promise different things about what the SFPU emits (SFPMAD.md):
#
#   Blackhole  "it is always the canonical NaN with bit pattern 0x7fc00000"
#   Wormhole   "the least significant bit of the mantissa is guaranteed to be set; other
#               bits of the mantissa might or might not be set, and the sign bit might or
#               might not be set"
#
# So a golden may assert this sign on Blackhole and may not on Wormhole. It stays invisible
# while the NaN remains a NaN -- passed_test's both-NaN clause accepts either sign -- and
# becomes a +inf/-inf disagreement the moment nan_survives_to_l1() is False.
#
# Measured on a Wormhole n300 by driving the specials set through every enrolled op: these ten,
# and no others, emit a NaN whose sign disagrees with UnarySFPUGolden's canonicalisation.
# ScalarRsub is the same cause in the scalar suite -- `c - x` builds its NaN through SFPMAD
# rather than forwarding the operand's, diverging where ScalarAdd/Sub/Mul/Div do not.
#
# Measured rather than derived, because "does this kernel generate a NaN or forward one" is a
# fact about the kernel that no property of the format axis predicts. This is *not* a table of
# the observed signs -- recording those would assert exactly what the ISA declines to promise.
# See UnarySFPUGolden._NAN_SIGN_TRANSPARENT_OPS for the other side of the partition: Neg, Abs
# and Identity move the sign bit, so for them it means something.
#
# Membership is by observed disagreement, not by kernel shape: GeluTanh and Xielu build a NaN
# through SFPMAD in the same `inf + (-inf)` shape as the gated GeluAppx, but read back raw on
# Float32->Float32 dest_acc=Yes they come out `0x7FC00001`, sign clear, where Cos, Mish and
# Silu come out `0xFFC00001`. They agree with the golden and are correctly out. That is
# evidence rather than a guarantee -- as it is for every op absent from this list -- so if one
# of them ever fails these cells, add it here.
GENERATED_NAN_SIGN_OPS: FrozenSet[MathOperation] = frozenset(
    {
        MathOperation.Cos,
        MathOperation.Fmod,
        MathOperation.GeluAppx,
        MathOperation.Hardmish,
        MathOperation.Mish,
        MathOperation.Rsqrt,
        MathOperation.ScalarRsub,
        MathOperation.Silu,
        MathOperation.Sin,
        MathOperation.Softsign,
        MathOperation.Tan,
    }
)


def nan_sign_is_unspecified(
    mathop: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: Optional[Union[bool, Enum]],
) -> bool:
    """Would this variant assert the sign of a NaN that the ISA leaves unspecified?

    Both halves have to hold: the op invents a NaN (GENERATED_NAN_SIGN_OPS), and the pipeline
    turns that NaN's sign into an observable +/-inf (not nan_survives_to_l1). The caller
    supplies the third -- the architecture, since Blackhole's SFPMAD does promise a canonical
    NaN and the assertion is sound there.
    """
    return mathop in GENERATED_NAN_SIGN_OPS and not nan_survives_to_l1(
        input_format, output_format, dest_acc
    )


def specials_after_nan_sign_gate(
    mathop: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: Optional[Union[bool, Enum]],
    specials: bool,
    on_wormhole: bool,
) -> bool:
    """*specials*, with cat B switched off where the NaN sign would be unspecified.

    The probe is switched off rather than the variant skipped, because edge_values() puts the
    cat-A, cat-D and cat-B probes in one list that edge_spec() wraps as a single
    StimuliSpec.custom -- dropping the variant would take the pole and knee assertions sharing
    that tensor with it (Hardmish's (-2.0, 0.0) knee, Rsqrt's 0.0 pole). Callers still skip
    when the narrowed spec comes back None, i.e. ops with nothing but cat B to drive.

    *on_wormhole* is passed rather than read here so this module stays free of the device
    imports ChipArchitecture pulls in; both edge sweeps share this one rule.

    Not an xfail: SFPMAD.md says the sign may be either, so the same hardware could satisfy or
    break that claim run to run. There is nothing to assert until the golden accepts both
    infinities on a substituted NaN, which is a change to convert_nan_to_inf's contract rather
    than to this gate. See tt-metal#52938.
    """
    if (
        specials
        and on_wormhole
        and nan_sign_is_unspecified(mathop, input_format, output_format, dest_acc)
    ):
        return False
    return specials


# Cat B for the *binary* SFPU family -- the golden-side gate, as SPECIALS_READY_OPS is for the
# unary and scalar families. A separate dict rather than an extension of it because membership is
# per family: SfpuElwadd and Add1 share neither an implementation nor a Dest path, so "the unary
# namesake is ready" says nothing about this one.
#
# Both gates still have to pass -- this one says the *golden* defines an answer for a non-finite
# operand, specials_safe() says the *pipeline* delivers one intact.
#
# Measured before enrolling, host-side and then on a Wormhole n150, over all 21 candidates and
# every specials-safe cell. All 21 goldens answer at all 25 (special, special) pairs -- none
# raises, unlike the unary tranche. The six comparisons and min/max answered wrongly, modelling
# IEEE's unordered comparison rather than the SFPU's total order; fixed in BinarySFPUGolden
# before any was enrolled, since enrolling first would have recorded seven kernel divergences
# the ISA specifies as correct. 12 agreed everywhere and are enrolled here; 9 diverge and stay
# out, in _BINARY_SPECIALS_NOT_READY.
BINARY_SPECIALS_READY_OPS: Dict[MathOperation, str] = {
    # Plain SFPMAD arithmetic: "if any input is NaN or +/-Infinity, then the result will be NaN or
    # +/-Infinity, following the usual IEEE754 rules". Green on Wormhole on all safe cells.
    MathOperation.SfpuElwadd: "IEEE: inf+x = inf, inf+(-inf) = NaN, NaN+x = NaN. Plain SFPMAD, "
    "which the ISA specifies as IEEE for a non-finite input. Green on Wormhole.",
    MathOperation.SfpuElwsub: "As SfpuElwadd; inf-inf = NaN is the case worth having.",
    MathOperation.SfpuElwmul: "IEEE: inf*x = inf, inf*0 = NaN, +/-0 signs multiply. Green on "
    "Wormhole.",
    MathOperation.SfpuElwrsub: "As SfpuElwsub with the operands reversed.",
    # Total order -- and the reason max/min enrol on a model the six comparisons could not: their
    # kernel is a bare SFPSWAP(VEC_MIN_MAX) with no NaN guard.
    MathOperation.SfpuBinaryMax: "binary_max_min is a bare SFPSWAP(VEC_MIN_MAX) with no NaN "
    "guard, so the documented total order reaches the result: +NaN is the maximum, -NaN the "
    "minimum. Golden models sfpu_max, not torch.maximum -- those agree on +NaN by coincidence "
    "and differ on -NaN. Green on Wormhole.",
    MathOperation.SfpuBinaryMin: "As SfpuBinaryMax, and the op that made the difference visible: "
    "torch.minimum propagates a NaN where the total order returns the other operand.",
    # IEEE unordered, because these kernels reject a NaN operand before comparing. The guard is
    # quoted per sequence in BinarySFPUGolden, above _lt.
    MathOperation.SfpuElwEq: "calculate_binary_comp_fp32_equal rejects a NaN operand "
    "(SFPIADD(inf, |a|+|b|, CC_GTE0)) so its pre-stored default stands and eq(NaN, x) = 0 -- "
    "IEEE's unordered answer, deliberately, not the SFPSWAP total order. Green on Wormhole.",
    MathOperation.SfpuElwNe: "As SfpuElwEq; its default result is 1, so ne(NaN, x) = 1.",
    MathOperation.SfpuElwLt: "calculate_binary_comp_fp32_strict_ordered pre-stores 0 and guards "
    "the store with the same 'rejects NaN' predicate, so lt(NaN, x) = 0. Green on Wormhole.",
    MathOperation.SfpuElwGt: "As SfpuElwLt, operands swapped.",
    MathOperation.SfpuElwLe: "calculate_binary_comp_fp32_weak_ordered pre-stores 1, rejects if "
    "false, then stores 0 under 'abs(a) + abs(b) > inf; a or b is NaN'. So le(NaN, x) = 0.",
    MathOperation.SfpuElwGe: "As SfpuElwLe, operands swapped.",
}

# The 9 that diverge, and what each waits on -- five causes rather than nine investigations.
# Measured on a Wormhole n150. None is enrolled on a guess: a reason string written to make a
# variant green becomes a permanent, plausible-looking claim about the hardware.
_BINARY_SPECIALS_NOT_READY: Dict[MathOperation, str] = {
    # (1) Composition through a reciprocal / log / exp -- the binary half of the audit's section
    # 5.9, where 23 unary ops sit behind the same question. Each builds its result from a primitive
    # the ISA specifies only inside a stated range (SFPARECIP gives accuracy bounds for 0 <= x < 2;
    # SFPLUTFP32 documents no NaN/inf handling), so what the composition does with a non-finite
    # input is an LLK decision rather than an ISA one. One answer decides all six.
    MathOperation.SfpuElwdiv: "composition: reciprocal + Newton-Raphson. Section 5.6 Q1.",
    MathOperation.SfpuXlogy: "composition: x * log(y). Section 5.6 Q1.",
    MathOperation.SfpuElwpow: "composition: exp(b * ln a). Section 5.6 Q1.",
    MathOperation.SfpuBinaryFmod: "composition: quotient via reciprocal. Section 5.6 Q1.",
    MathOperation.SfpuBinaryRemainder: "composition: as fmod. Section 5.6 Q1.",
    MathOperation.SfpuAtan2: "composition: ratio plus a format-specific polynomial. Diverges on "
    "2 cells rather than 4, so its non-finite handling is partial. Section 5.6 Q1.",
    # (2) Compare-against-zero on an operand that may be a NaN. calculate_mask is
    # `v_if(mask == 0)`, which lowers to SFPSETCC -- whose contract is conditioned "provided that
    # VC is neither negative zero nor any kind of NaN". Identical to what holds Sign and Heaviside
    # out of the unary cat B, so it is section 5.6's third question rather than a new one.
    MathOperation.SfpuMask: "the mask operand reaches SFPSETCC, whose contract excludes a NaN "
    "operand. Section 5.6 Q3, with Sign and Heaviside.",
    # (3) Its own question, and a narrow one. ckernel_sfpu_isclose documents torch.isclose
    # semantics including EQUAL_NAN=false ("any NaN input => result = 0") and bit-inspects for
    # +/-Inf against NaN -- so both sides claim to agree and do not. Needs one per-cell read-back
    # to say whether the golden or the kernel's inf path is wrong, before either is touched.
    MathOperation.SfpuIsclose: "golden and kernel both claim torch.isclose semantics and "
    "disagree at a non-finite operand; needs a per-cell read-back to say which is wrong.",
    # (4) Operand B is *derived*, not free. This entry used to read "effectively unary -- operand
    # B is read only on the x > 4 branch and the golden ignores it", which was true of the test
    # rather than of the kernel: the sweep stopped at x = 3.9 so the branch never ran. It runs
    # now, the golden models it, and the kernel was measured returning exactly -in1 there (max
    # |hw - (-exp(-x))| = 0 over 256 lanes on Blackhole, and doubling in1 doubles the result).
    #
    # That does not make it a cat-B candidate, and the reason is structural. The kernel's
    # contract is in1 == exp(-in0), so the two operands are not independent -- but cat B here is
    # driven by edge_pair_values(), a cartesian *product* of two independently-chosen lists. A
    # NaN placed in B against a finite A is not exp(-A) and so is not a stimulus the kernel has
    # any contract about; the pair asserts nothing whichever way it comes out. Enrolling
    # logsigmoid needs a probe that derives B from A, which is a different mechanism from the
    # one this sweep has, not a different entry in this table.
    MathOperation.SfpuLogsigmoid: "operand B is derived (in1 == exp(-in0)), and cat B *here* "
    "is a product of two independently-chosen lists, so a special in B is not a stimulus the "
    "kernel has a contract about. It stays out of this sweep for that reason and is driven "
    "instead by test_eltwise_binary_sfpu_logsigmoid_specials, which derives B from A -- so "
    "read this as 'not ready for the product mechanism', not as 'has no cat B'.",
    # (5) Not element-wise, so the sweep's whole shape does not fit it. add_top_row reduces one
    # tile's first row across the other operand, and BinarySFPUGolden returns for it *before*
    # the Dest and pack modelling -- it raises rather than reporting a generated-NaN mask. A
    # cat-B probe would be compared against a golden that models neither step, which is the
    # defect the binary suite's own "0/0 returns inf" retraction was about.
    MathOperation.SfpuAddTopRow: "not element-wise: the golden returns before the Dest and "
    "pack modelling and cannot report a generated-NaN mask, so a non-finite would be judged "
    "against a reference that models neither step.",
}

assert not (
    set(BINARY_SPECIALS_READY_OPS) & set(_BINARY_SPECIALS_NOT_READY)
), "an op cannot be both enrolled in cat B and recorded as not ready for it"


# ─────────────────────────────────────────────────────────────────────────────
# Cat B, ternary family
#
# Same two-dict shape as the binary family above, and for the same reason: a *_READY_OPS entry
# says the op's *golden* defines an answer at a non-finite operand, specials_safe() says the
# *pipeline* delivers one intact, and neither implies the other.
#
# Measured before enrolling, in the two passes convention 3 asks for.
#
# Host-side first, over all 5x5x5 (special, special, special) triples on every op: all four
# TernarySFPUGolden operations and WhereGolden *answer* at every triple, none raises. That is
# the check the unary tranche failed -- math.acos / math.asin / math.tan raise
# ValueError("math domain error") on a non-finite rather than returning NaN -- and these goldens
# passing it is not luck: they are plain torch arithmetic with no math.* call and no
# finite-input guard, so they carry neither of the two defects that tranche found.
#
# Then on a Blackhole p150, through test_sfpu_ternary_operand_edges' specials_in class and
# test_ttnn_where_specials: each op's specials driven into A, into B and into C separately, over
# both specials_safe cells the ternary format axis reaches (Float16_b->Float16_b at dest_acc=No
# and Float32->Float32 at dest_acc=Yes; the other two are excluded by specials_safe's breaker 2
# and by the suite's own fp32-at-dest_acc=No skip).
#
# THE GOLDENS HAD TO BE FIXED FIRST, and the fix is the same one the binary suite's retraction
# describes. Neither TernarySFPUGolden nor WhereGolden modelled the store into Dest or the pack
# out of it, so on a pipeline too narrow to hold a NaN the packer's substituted infinity
# (SFPSTORE: "NaN is also converted to infinity") read as the kernel having computed one. That
# alone accounted for 10 of the 16 failing cells in the first pass -- every op, every operand,
# on Float16_b->Float16_b. With both goldens modelling it, addcmul and lerp are green
# everywhere, and what is left is four genuine per-op divergences recorded as xfails in
# test_sfpu_ternary rather than papered over here.
#
# All five ops are enrolled. _TERNARY_SPECIALS_NOT_READY is empty and stays present: it is
# where an op whose golden does *not* define an answer belongs, and an empty dict is a
# different statement from a missing one -- the totality check in test_sfpu_domains partitions
# _SFPU_TERNARY_OPS across both, so an op added to the family with no verdict fails collection.
# ─────────────────────────────────────────────────────────────────────────────

TERNARY_SPECIALS_READY_OPS: Dict[MathOperation, str] = {
    MathOperation.SfpuAddcmul: "a + value*b*c is plain SFPMAD arithmetic, which the ISA "
    "specifies as IEEE for a non-finite input: inf*0 = NaN, inf+(-inf) = NaN, NaN propagates. "
    "Green on Blackhole on both safe cells for all three operands.",
    MathOperation.SfpuLerp: "a + c*(b - a), also plain SFPMAD, and the op where inf - inf "
    "arises from an ordinary interpolation rather than from a contrived operand. Green on "
    "Blackhole on both safe cells for all three operands.",
    MathOperation.SfpuAddcdiv: "a + value*b/c. Enrolled on the strength of operands A and B, "
    "which agree everywhere, and of c = +/-inf, where value*b/c = +/-0 and the result is a. "
    "c = NaN is a genuine kernel divergence -- the reciprocal returns +0 instead of "
    "propagating, exactly as unary Reciprocal does -- and is xfailed per cell rather than "
    "papered over in the golden.",
    MathOperation.SfpuSnakeBeta: "a + sin(b*a)^2/c. Enrolled on the same footing as addcdiv: "
    "the pipeline delivers the probe and the golden defines an answer at every triple. Two "
    "divergences are xfailed -- c = NaN through the same reciprocal, and a non-finite reaching "
    "the sin, which SFPLUTFP32 documents no handling for.",
    MathOperation.TTNNWhere: "An element-wise select, so every special is returned verbatim "
    "from whichever operand was chosen and there is no arithmetic to specify. Green on "
    "Blackhole for all three operands once WhereGolden modelled the pack's NaN substitution, "
    "except a -0.0 *condition* on the unpack-to-dest path, which is xfailed with the same "
    "SFPSETCC caveat that scopes Sign and Heaviside.",
}

# Empty by measurement, not by omission -- see the section comment. Kept so the totality check
# has both halves of the partition to test against.
_TERNARY_SPECIALS_NOT_READY: Dict[MathOperation, str] = {}

assert not (
    set(TERNARY_SPECIALS_READY_OPS) & set(_TERNARY_SPECIALS_NOT_READY)
), "an op cannot be both enrolled in cat B and recorded as not ready for it"


# ─────────────────────────────────────────────────────────────────────────────
# Cat F enrolment
#
# Opt-in per op, for the same reason SPECIALS_READY_OPS is: driving exp(3.39e38) on an op whose
# golden does not model saturation produces a wall of failures with one root cause, and a wall
# of failures is indistinguishable from no measurement at all.
#
# THE FIRST TRANCHE, and what "first" means here. These are the ops whose behaviour at a
# magnitude extreme is uncontroversial and whose golden is plain arithmetic: magnitude, sign,
# the rounding family, the pass-throughs and the four sweep-reachable comparisons. None of them
# can overflow -- every one either returns its input, a bounded constant, or a magnitude no
# larger than the input -- so this tranche says nothing about saturation. That is deliberate:
# saturation is the ops whose *result* leaves the format, and each of those wants its own probe
# list and its own assertion rather than a shared enrolment.
#
# Measured on a Blackhole p150 over the sweep's whole format axis, in two passes.
#
# The first pass failed 12 of 70 cells across exactly four ops -- Ceil, Floor, Sign and Signbit
# -- and every failing lane was the +/-2^-127 subnormal, on every cell except Float32->Float32
# at dest_acc=Yes. That is not four findings, it is one: the datacopy path hands the kernel
# +0.0, so ceil(+2^-127) -> 0, floor(-2^-127) -> 0, sign(+/-2^-127) -> 0 and signbit(-2^-127)
# -> 0 are all "the op behaved correctly at zero". Round, Trunc and Abs passed throughout and
# are the control: their answer at a subnormal *is* their answer at zero, so the flush is
# invisible to them and their agreement establishes nothing about it either way. The finding is
# recorded as subnormal_delivered(), which keeps the probe off the pipelines that flatten it,
# and the second pass was green on all 70.
#
# So each entry below is a claim about the ceiling, its neighbour and the smallest normal on
# every cell, and about the subnormal on the one cell that delivers one.
# ─────────────────────────────────────────────────────────────────────────────

# The rounding family, whose enrolment settles a question nothing else in the suite asks: above
# 2**mantissa every float is already an integer, so at the ceiling and its neighbour all four
# must be the identity. They are, on every cell.
_EXTREMES_READY_ROUNDING: Tuple[MathOperation, ...] = (
    MathOperation.Floor,
    MathOperation.Ceil,
    MathOperation.Trunc,
    MathOperation.Round,
)

# The four comparisons the unary edge sweep can drive. UnaryEq and UnaryNe are outside
# _OP_DOMAIN_REGISTRY and so never reach it -- see the note above _OP_EDGE_POINTS.
_EXTREMES_READY_COMPARISONS: Tuple[MathOperation, ...] = (
    MathOperation.UnaryGt,
    MathOperation.UnaryLt,
    MathOperation.UnaryGe,
    MathOperation.UnaryLe,
)

EXTREMES_READY_OPS: Dict[MathOperation, str] = {
    MathOperation.Abs: "Magnitude: |x| = |x| at every representable magnitude, and the "
    "result can never leave the format the input came from. Green on Blackhole.",
    MathOperation.Neg: "Sign flip, likewise closed over the format. Green on Blackhole.",
    MathOperation.Identity: "Pass-through: every extreme maps to itself, which makes it the "
    "op that measures the *pipeline* rather than any arithmetic. Green on Blackhole.",
    MathOperation.Fill: "Input-independent -- the kernel writes a compile-time constant -- so "
    "read this narrowly: the probe asserts that an extreme input does not corrupt the fill, "
    "and nothing about the extreme itself. Green on Blackhole.",
    MathOperation.Sign: "-1 / 0 / +1, so the answer is decided by the sign bit and by whether "
    "the datum is zero -- which is what makes it one of the four instruments that measured "
    "subnormal_delivered(). Green on Blackhole once the probe stopped being sent where the "
    "datacopy path flattens it.",
    MathOperation.Signbit: "Reads the sign bit directly. Same role in the "
    "subnormal_delivered() measurement as Sign, and green on the same terms.",
}
EXTREMES_READY_OPS.update(
    {
        op: "Above 2**mantissa every float is already an integer, so at the ceiling and its "
        "neighbour this op must be the identity -- an assertion nothing else in the suite "
        "makes. Ceil and Floor also measured subnormal_delivered(), being the two whose answer "
        "just above and just below zero differs from their answer at it. Green on Blackhole."
        for op in _EXTREMES_READY_ROUNDING
    }
)
EXTREMES_READY_OPS.update(
    {
        op: "x compared against UNARY_COMP_THRESHOLD, so an extreme operand is an ordinary "
        "one on the far side of the threshold and the answer is 0.0 or 1.0 -- representable "
        "everywhere, which is why the comparisons are cheap to enrol. Green on Blackhole."
        for op in _EXTREMES_READY_COMPARISONS
    }
)


def _deliverable_extremes(
    range_fmt: DataFormat,
    input_format: DataFormat,
    dest_acc: Optional[Union[bool, Enum]],
) -> List[float]:
    """format_extremes(*range_fmt*) minus the probes this pipeline cannot deliver.

    One filter for both cat-F entry points, so edge_values(extremes=True) and extreme_values()
    cannot come to disagree about which probes are sent. Today that is the subnormal alone: the
    ceiling, its neighbour and the smallest normal all survive every pipeline cat F runs on,
    because a 16-bit Dest is bfloat16 or Float16 and both carry the exponent range that
    clip_to_format() has already bounded the probe by.

    The gate keys off *input_format* rather than *range_fmt*: delivery is a property of how the
    datum reaches the LREG, not of the magnitudes the pipeline can represent. Same distinction
    edge_values() draws for the -0.0 probe.
    """
    return [
        v
        for v in format_extremes(range_fmt)
        if not _is_subnormal(v, range_fmt)
        or subnormal_delivered(input_format, dest_acc)
    ]


def extreme_values(
    input_format: DataFormat,
    output_format: Optional[DataFormat] = None,
    dest_acc: Optional[Union[bool, Enum]] = None,
) -> List[float]:
    """The cat-F probe for this pipeline, and nothing else.

    edge_values(extremes=True) returns cat F *alongside* the op's poles and knees, which is
    right for a caller building one tensor per op; a sweep that wants one failure class per
    variant needs cat F on its own, and deriving it by set-differencing two edge_values() calls
    is the kind of cleverness that stops being right the moment either list gains a duplicate.

    No *op* argument, deliberately: the format's ceiling and its subnormal band are properties
    of the pipeline, not of the function evaluated on them. Which ops may be *driven* at them
    is EXTREMES_READY_OPS' question and is asked separately.

    Pass *dest_acc* to get the subnormal probe right; see subnormal_delivered(). Omitting it
    keeps every probe, which is the right default for a caller that does not know the pipeline
    and the wrong one for a test that does.
    """
    range_fmt = narrowest_range_format(input_format, output_format)
    return _dedup_representable(
        clip_to_format(
            _deliverable_extremes(range_fmt, input_format, dest_acc), range_fmt
        ),
        range_fmt,
    )


def generated_nan_sign_is_asserted(
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: Optional[Union[bool, Enum]],
    on_wormhole: bool,
) -> bool:
    """Would this pipeline make a *generated* NaN's sign load-bearing on Wormhole?

    The binary-family twin of nan_sign_is_unspecified(), taking no op argument -- which is the
    whole difference. There, membership is a measured per-op fact (GENERATED_NAN_SIGN_OPS). Here
    the caller already knows, because the binary edge sweep partitions its probe *by what the
    golden answers*: the `both_zero` and `nan_golden` classes are exactly the pairs where finite
    operands produce a NaN, so every element of them is generated by construction.

    Two conditions, same as the unary gate:
      * the pipeline narrows, so the NaN leaves as a signed infinity (not nan_survives_to_l1);
      * the arch leaves that sign unspecified, i.e. Wormhole -- `SFPMAD.md` says the sign "might
        or might not be set" there and promises canonical 0x7fc00000 on Blackhole.

    Where this is True there is nothing sound to assert *yet*. The assertion to restore is "an
    infinity of either sign", which is a change to the comparator rather than to this gate --
    see tt-metal#52938.
    """
    return on_wormhole and not nan_survives_to_l1(input_format, output_format, dest_acc)


def specials_safe_formats(
    formats: List["InputOutputFormat"],  # noqa: F821 - test-side type, duck-typed
    dest_acc: Union[bool, Enum],
) -> List["InputOutputFormat"]:  # noqa: F821
    """Filter an input_output_formats() list down to the triples that carry specials.

    Normalises *dest_acc* once here rather than leaving it to the per-format calls, so a
    bad argument raises even when *formats* is empty.
    """
    dest_acc = _dest_acc_flag(dest_acc)
    return [
        f for f in formats if specials_safe(f.input_format, f.output_format, dest_acc)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# edge_spec — the one builder the per-family edge tests call
# ─────────────────────────────────────────────────────────────────────────────


def clip_to_format(values: List[float], fmt: DataFormat) -> List[float]:
    """Drop finite values *fmt* cannot represent; keep non-finite ones verbatim.

    Non-finite values are the *point* of a cat-B probe, so they are never clipped — the
    decision about whether they belong at all is specials_safe()'s, made before this.
    """
    limit = _FORMAT_MAX_MAGNITUDE.get(fmt, _BF16_MAX_MAGNITUDE)
    return [v for v in values if not math.isfinite(v) or abs(v) <= limit]


def edge_values(
    op: MathOperation,
    input_format: DataFormat,
    output_format: Optional[DataFormat] = None,
    operand: Operand = Operand.A,
    specials: bool = False,
    include_undefined: bool = False,
    dest_acc: Optional[Union[bool, Enum]] = None,
    extremes: bool = False,
) -> List[float]:
    """Every value worth hitting on purpose for (*op*, *operand*) in this pipeline.

    Four sources, matching the audit's edge categories:
      * cat A — boundary_probes(): the op's singularities, straddled.
      * cat D — op_edge_points(): knees, thresholds, exact rounding ties.
      * cat B — format_specials(), only when *specials* is True. The caller decides via
        specials_safe(input_format, output_format, dest_acc); it is off by default
        because injecting them on the wrong triple is a wall of failures with one root
        cause (see the section above).
      * cat F — format_extremes(), only when *extremes* is True, decided by the caller via
        EXTREMES_READY_OPS and extremes_safe(). A separate flag from *specials* and not a
        synonym for it: the delivery rules differ and so do the failure classes, so folding
        the two would give one xfail covering an unpack question and a saturation one.

    Clipped against the *narrowest* format in the pipeline, not the input format. This is
    the part the plan's original one-format signature got wrong: a caller that passes a
    spec to a driver bypasses the driver's own for_op_pipeline() resolution entirely
    (eltwise_unary_sfpu only resolves when spec_A is None), so a probe near a format
    ceiling would otherwise reach a Float16 or MxFp4 output unclipped and overflow.

    Range and spacing resolve separately — *range_fmt* clips magnitudes, *step_fmt* sizes the
    ULP steps and the dedup. Pass *dest_acc* to get the second one right; see
    probe_spacing_format().
    """
    range_fmt = narrowest_range_format(input_format, output_format)
    step_fmt = probe_spacing_format(range_fmt, dest_acc)
    vals = list(
        boundary_probes(
            op,
            operand,
            range_fmt,
            include_undefined=include_undefined,
            step_fmt=step_fmt,
        )
    )
    # Cat D, per operand. For A this is _OP_EDGE_POINTS, the op's own knees; a binary op's
    # B-side knees are domain boundaries and come from cat A instead. A *ternary* op's third
    # operand can have knees of its own (lerp's weight), so every operand is asked.
    vals += list(op_edge_points(op, operand))
    if specials:
        # Specials are an exponent-range property, so they key off range_fmt: it is what
        # decides integer extremes vs IEEE non-finites, and what clip_to_format honours.
        vals += list(format_specials(range_fmt))
    if extremes:
        # Cat F off range_fmt for the same reason cat B is: the ceiling and the subnormal band
        # are properties of the narrowest exponent range in the pipeline, and format_extremes()
        # is built against the very table clip_to_format() clips with, so nothing it emits is
        # then dropped. The caller has already asked extremes_safe(), which rejects the integer
        # and block-float legs format_extremes() would raise on.
        vals += _deliverable_extremes(range_fmt, input_format, dest_acc)

    if not range_fmt.is_integer() and not negative_zero_delivered(
        input_format, dest_acc
    ):
        # One pass over every source, not one per source. -0.0 now arrives from three of them
        # -- a registered zero pole (cat A), the comparison-to-zero ops' knees (cat D) and
        # FLOAT_SPECIALS (cat B) -- and a filter applied per source is a filter a fourth source
        # can be added without. On the datacopy path the LREG holds +0.0, so sending the probe
        # there blames the kernel for a datum it never received; six standing Signbit xfails
        # were recording exactly that before the gate existed.
        #
        # The gate keys off input_format, not range_fmt: delivery is a property of how the datum
        # reaches the LREG, not of the magnitudes the pipeline can represent.
        vals = [v for v in vals if not _is_negative_zero(v)]
    return _dedup_representable(clip_to_format(vals, range_fmt), range_fmt)


def edge_spec(
    op: MathOperation,
    input_format: DataFormat,
    output_format: Optional[DataFormat] = None,
    operand: Operand = Operand.A,
    specials: bool = False,
    include_undefined: bool = False,
    dest_acc: Optional[Union[bool, Enum]] = None,
    extremes: bool = False,
    **kwargs,
) -> Optional[StimuliSpec]:
    """edge_values() as a StimuliSpec, or None if *op* has no edge worth probing.

    Returns None rather than an empty spec so a caller can fall back to the op's random
    domain: 47 of the 97 unary SFPU ops are smooth everywhere with no knee and no pole,
    and for those an edge sweep has nothing to add beyond cat B.

    ``custom`` with ``cycle=True``: the values are tiled across every face rather than
    written at its head with a zero tail. The zero tail was not free. The median edge list
    across the swept unary ops is four values against a 256-element face, so ~98% of the
    tensor was 0.0 — which put the probes in lanes 0-3 of the first vector op only, hiding
    any lane-position-dependent defect, and left PCC and every other aggregate a statement
    about the filler rather than about the probe. It also drove an out-of-domain 0.0 into
    Acosh, Log and Rsqrt on every edge variant without recording that it was doing so.
    ``0.0`` is a registered pole or knee wherever it is meaningful, so cycling loses nothing
    and an op that wants it gets it from _OP_SINGULARITIES or _OP_EDGE_POINTS explicitly.

    A caller can pass ``cycle=False`` to get the old behaviour back for a probe that
    genuinely depends on the tail — the int comparison sweep is the one place that does.

    Note ``custom`` is per-face only (generate_full_tensor raises), so the pattern repeats
    in every face; ``custom_faces`` is available when faces must differ.

    Integer formats: format_specials() returns the integer extremes, but INT_MIN cannot
    be delivered through any spec — CustomStrategy clamps through _get_integer_bounds,
    which returns info.min + 1. Deliver integer extremes as a raw override tensor
    instead (see _build_shift_edge_case_src); this raises rather than silently clamping.
    """
    vals = edge_values(
        op,
        input_format,
        output_format,
        operand,
        specials,
        include_undefined,
        dest_acc,
        extremes,
    )
    if not vals:
        return None
    if input_format.is_integer() and specials:
        raise ValueError(
            f"edge_spec(specials=True) cannot deliver integer extremes for "
            f"{input_format.name}: StimuliSpec.custom clamps INT_MIN to INT_MIN + 1. "
            f"Use a raw src_A_override tensor instead."
        )
    kwargs.setdefault("cycle", True)
    return StimuliSpec.custom(values=vals, seed=0, **kwargs)


# Representative counterpart values for the operand that has *no* edge of its own. A
# divisor-zero probe only means something when paired against a positive, a negative and a
# zero numerator — three distinct cases — so the plain operand contributes a small spread
# rather than one arbitrary value.
_EDGE_COUNTERPARTS: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _in_spec_domain(spec: Optional[StimuliSpec], value: float) -> bool:
    """Is *value* inside what *spec* is allowed to draw? True when *spec* is None."""
    if spec is None:
        return True
    if spec.intervals:
        return any(lo <= value <= hi for lo, hi in spec.intervals)
    if spec.low is None or spec.high is None:
        return True
    return spec.low <= value <= spec.high


def edge_counterparts(
    op: MathOperation,
    fmt: DataFormat,
    operand: Operand = Operand.A,
) -> List[float]:
    """In-domain representative values for an operand with no edge of its own.

    Clipped to the op's registered domain for that operand where one exists, so pairing
    pow's base-zero probe against an exponent of -2 (outside its registered [0, 3]) cannot
    happen. Ops with no registry entry at all keep the full counterpart spread.
    """
    try:
        specs = for_op(op, fmt)
    except KeyError:
        return list(_EDGE_COUNTERPARTS)
    return [
        v for v in _EDGE_COUNTERPARTS if _in_spec_domain(specs.spec_for(operand), v)
    ]


def edge_pair_values(
    op: MathOperation,
    input_format: DataFormat,
    output_format: Optional[DataFormat] = None,
    specials: bool = False,
    include_undefined: bool = False,
    dest_acc: Optional[Union[bool, Enum]] = None,
) -> List[Tuple[float, float]]:
    """Cartesian product of both operands' edge values, for a binary op.

    The product matters more than element-wise pairing here: a divisor of 0 against a
    positive, a negative and a zero numerator are three different cases, and element-wise
    pairing would test one of them. Whichever operand has no edge of its own contributes
    edge_counterparts() instead, so the other operand's edge is still crossed with a
    spread.

    Returns [] when neither operand has anything to probe, which is the caller's cue to
    skip rather than drive a meaningless variant.
    """
    a = edge_values(
        op,
        input_format,
        output_format,
        Operand.A,
        specials,
        include_undefined,
        dest_acc,
    )
    b = edge_values(
        op,
        input_format,
        output_format,
        Operand.B,
        specials,
        include_undefined,
        dest_acc,
    )
    if not a and not b:
        return []
    if not a:
        a = edge_counterparts(op, input_format, Operand.A)
    if not b:
        b = edge_counterparts(op, input_format, Operand.B)
    if not a or not b:
        return []
    return [(x, y) for x in a for y in b]


# ─────────────────────────────────────────────────────────────────────────────
# The coverage ledger
#
# Every gap this suite has closed was found by reading code and running the registries by
# hand. Nothing stated, per op, *which classes of input value it has actually seen* -- so an
# op could sit for a release with a positive-only uniform and look fully covered.
#
# The ingredients were already here: SPECIALS_READY_OPS and _BINARY_SPECIALS_NOT_READY are
# exactly this kind of ledger for one class, and the *_READY / *_NOT_READY partition tests are
# exactly this kind of totality check for another. What follows generalises them across
# classes.
#
# TWO RULES IT IS BUILT ON.
#
# **Derived, never declared.** A cell reads COVERED only if the machinery that builds stimuli
# actually emits a value of that class for that op -- the question is put to
# ops_with_singularity(), op_edge_points(), the *_READY_OPS dicts and EXTREMES_READY_OPS, not
# to a hand-maintained matrix. A matrix can claim coverage the sweep does not deliver; this
# cannot.
#
# **Unrecorded is its own state.** The honest answer for most (op, class) pairs today is
# neither "covered" nor "does not apply" but "nobody has decided" -- 28 unary ops sit outside
# cat B with no per-op reason, for instance. Collapsing that into "not covered" would make the
# ledger agree with a future where someone *had* decided and the answer was no. The three
# states stay distinct, and the ratchet below is what turns UNRECORDED into a reason over time.
# ─────────────────────────────────────────────────────────────────────────────


class EdgeClass(str, Enum):
    """The classes of input value a sweep can drive at, and where each one comes from."""

    A = "singularities"  # _OP_SINGULARITIES, straddled by boundary_probes()
    B = "ieee_specials"  # format_specials() behind the *_READY_OPS gates
    C = "integer_extremes"  # integer_specials(), through a raw override
    D = "knees"  # _OP_EDGE_POINTS / _OP_OPERAND_EDGE_POINTS
    E = "operand_parameters"  # shift amounts, compile-time scalars
    F = "magnitude_extremes"  # format_extremes() behind EXTREMES_READY_OPS
    G = "signed_zero_at_a_pole"  # a -0.0 delivered *to* a registered pole


#: A cell's value when the machinery does emit a value of that class for that op.
COVERED = "covered"

#: A cell's value when nothing records whether the class applies. Distinct from a reason:
#: this is the state the ratchet exists to shrink.
UNRECORDED = "unrecorded"

# The two classes an integer-only op cannot have, written once because the ledger reaches them
# by subtraction rather than by a table -- see SuiteCoverage.float_driven.
_CAT_B_INTEGER_ONLY = (
    "integer-only op: no IEEE specials to inject, so format_specials() returns the integer "
    "extremes instead and the question is cat C's"
)
_CAT_C_OUT_OF_RANGE = (
    "the kernel is documented invalid at the format extremes and measured wrong there; see "
    "_INT_EXTREMES_OUT_OF_RANGE in the binary suite for the bound and the values"
)
_CAT_F_INTEGER_ONLY = (
    "integer-only op: no subnormal band and no float ceiling, so format_extremes() raises "
    "for its formats by design"
)


@dataclass(frozen=True)
class SuiteCoverage:
    """What the suites drive that this module cannot derive from its own tables.

    Two of the seven classes are delivered by machinery that lives in the test modules --
    integer extremes as raw override tensors, operand-as-parameter as compile-time template
    arguments -- and a third fact, which ops are driven on an integer format at all, decides
    whether cat C even applies. None of them has a table here to ask.

    Passed in rather than restated. A second copy of a list inside this module is exactly how a
    ledger comes to claim coverage no sweep delivers, which is the failure mode the whole thing
    exists to prevent; test_sfpu_domains builds this from the suites themselves and pins it
    against them. Every field defaults to empty, so a caller without the facts gets UNRECORDED
    rather than a confident wrong answer.
    """

    #: Ops driven at the integer extremes (cat C).
    integer_extremes: FrozenSet[MathOperation] = frozenset()
    #: Ops driven on an integer format at all. Outside this, cat C does not apply.
    integer_driven: FrozenSet[MathOperation] = frozenset()
    #: Ops driven on an integer format whose kernel is documented invalid at the extremes, so
    #: cat C is excluded by decision rather than unrecorded. The per-op reason lives in the
    #: suite's own table; only membership is needed here.
    integer_extremes_excluded: FrozenSet[MathOperation] = frozenset()
    #: Ops driven on a float format at all. The complement of this within *integer_driven* is
    #: the integer-only set, for which cat B and cat F do not apply -- an integer op has no
    #: IEEE specials to be given and no subnormal band to probe. Derived by subtraction rather
    #: than declared, so an op that gains a float sweep stops being integer-only on its own.
    #:
    #: Supplying *integer_driven* without this one makes every integer-driven op look
    #: integer-only. That is right today -- the two sets are disjoint, and test_sfpu_domains
    #: asserts it -- and would stop being right for an op driven on both, which is why the
    #: assertion is there rather than a comment.
    float_driven: FrozenSet[MathOperation] = frozenset()
    #: Ops taking an operand as a parameter -- a shift amount, a compile-time scalar (cat E).
    operand_parameters: FrozenSet[MathOperation] = frozenset()
    #: Ops whose cat B is driven by a purpose-built variant rather than by the shared
    #: product-of-two-lists sweep -- logsigmoid, whose operand B has to be derived from A. The
    #: *_SPECIALS_READY_OPS gate governs the shared sweep only, so without this the ledger
    #: reports "explained" for an op that is in fact covered.
    specials_derived: FrozenSet[MathOperation] = frozenset()
    #: Ops driven at a magnitude extreme by a purpose-built saturation sweep rather than
    #: through EXTREMES_READY_OPS (cat F). The two halves of cat F have different gates --
    #: enrolment for the ops that cannot overflow, a hand-chosen probe list for the ones whose
    #: *result* leaves the format -- and the ledger has to count both or it reports less
    #: coverage than the suite delivers.
    saturation: FrozenSet[MathOperation] = frozenset()
    #: Ops a suite drives that have no _OP_DOMAIN_REGISTRY entry, so _ledger_ops() cannot
    #: find them. The ~30 integer and bitwise binary ops are the population this exists for.
    extra_ops: FrozenSet[MathOperation] = frozenset()


def _ledger_ops(suite: SuiteCoverage) -> FrozenSet[MathOperation]:
    """Every op the ledger has an opinion about.

    The three families this module knows how to classify, minus the ops deliberately kept out
    of the correctness sweep, plus whatever the suites declare they drive without a registry
    entry. An op that gains a registry entry lands here automatically, which is what makes the
    totality check a real gate rather than a list to remember to edit.
    """
    return (
        (sfpu_unary_ops() - set(_UNARY_OPS_NOT_SWEPT))
        | _SFPU_BINARY_OPS
        | _SFPU_TERNARY_OPS
        | suite.extra_ops
    )


_CAT_B_READY: Dict[MathOperation, str] = {
    **SPECIALS_READY_OPS,
    **BINARY_SPECIALS_READY_OPS,
    **TERNARY_SPECIALS_READY_OPS,
}

_CAT_B_NOT_READY: Dict[MathOperation, str] = {
    **_UNARY_SPECIALS_NOT_READY,
    **_BINARY_SPECIALS_NOT_READY,
    **_TERNARY_SPECIALS_NOT_READY,
}


def _has_knee(op: MathOperation) -> bool:
    return bool(op_edge_points(op)) or any(
        op_edge_points(op, operand) for operand in Operand
    )


def _emits_negative_zero_at_a_pole(op: MathOperation) -> bool:
    """Does boundary_probes() actually emit a -0.0 for *op*, on any operand?

    Cat G is "a -0.0 delivered to a registered pole", so an op with no zero pole has nowhere to
    deliver one and the class does not apply to it -- reciprocal, div, fmod, remainder, xlogy,
    atan2, addcdiv, snake_beta and the compat variants are the ones that do.

    Asked of the probe builder rather than of _OP_SINGULARITIES directly, which is the ledger's
    standing rule: a cell reads COVERED only if the machinery that builds stimuli emits the
    value. Reading the table instead would report coverage from the moment a pole was
    *registered*, which was true for years before anything drove the second zero at it.
    """
    return any(
        _is_negative_zero(value)
        for operand in Operand
        for value in boundary_probes(op, operand, DataFormat.Float32)
    )


def coverage_ledger(
    suite: Optional[SuiteCoverage] = None,
) -> Dict[MathOperation, Dict[EdgeClass, str]]:
    """Per op, per class: COVERED, UNRECORDED, or the reason the class is not driven.

    See SuiteCoverage for the three facts this module cannot derive and why they are arguments
    rather than a second table.
    """
    suite = suite or SuiteCoverage()
    integer_only = suite.integer_driven - suite.float_driven
    ledger: Dict[MathOperation, Dict[EdgeClass, str]] = {}
    for op in _ledger_ops(suite):
        cells: Dict[EdgeClass, str] = {}

        cells[EdgeClass.A] = (
            COVERED
            if op in ops_with_singularity()
            else "smooth everywhere: no entry in _OP_SINGULARITIES"
        )

        if op in _CAT_B_READY or op in suite.specials_derived:
            cells[EdgeClass.B] = COVERED
        elif op in _CAT_B_NOT_READY:
            cells[EdgeClass.B] = _CAT_B_NOT_READY[op]
        elif op in integer_only:
            cells[EdgeClass.B] = _CAT_B_INTEGER_ONLY
        else:
            cells[EdgeClass.B] = UNRECORDED

        if op in suite.integer_extremes:
            cells[EdgeClass.C] = COVERED
        elif op in suite.integer_extremes_excluded:
            cells[EdgeClass.C] = _CAT_C_OUT_OF_RANGE
        elif op in suite.integer_driven:
            cells[EdgeClass.C] = UNRECORDED
        else:
            cells[EdgeClass.C] = "no integer form driven by any suite"

        cells[EdgeClass.D] = (
            COVERED
            if _has_knee(op)
            else "no knee, threshold or rounding tie registered"
        )

        cells[EdgeClass.E] = (
            COVERED
            if op in suite.operand_parameters
            else "takes no operand as a parameter (no shift amount, no compile-time scalar)"
        )

        if op in EXTREMES_READY_OPS or op in suite.saturation:
            cells[EdgeClass.F] = COVERED
        elif op in integer_only:
            cells[EdgeClass.F] = _CAT_F_INTEGER_ONLY
        else:
            cells[EdgeClass.F] = UNRECORDED

        cells[EdgeClass.G] = (
            COVERED
            if _emits_negative_zero_at_a_pole(op)
            else "no singularity at zero, so there is no pole to deliver a -0.0 to"
        )

        ledger[op] = cells
    return ledger


def coverage_counts(
    ledger: Dict[MathOperation, Dict[EdgeClass, str]],
) -> Dict[EdgeClass, Dict[str, int]]:
    """How many ops are covered / unrecorded / explained, per class."""
    counts: Dict[EdgeClass, Dict[str, int]] = {}
    for edge_class in EdgeClass:
        states = [cells[edge_class] for cells in ledger.values()]
        counts[edge_class] = {
            COVERED: sum(1 for s in states if s == COVERED),
            UNRECORDED: sum(1 for s in states if s == UNRECORDED),
            "explained": sum(1 for s in states if s not in (COVERED, UNRECORDED)),
        }
    return counts


def format_coverage_report(ledger: Dict[MathOperation, Dict[EdgeClass, str]]) -> str:
    """The ledger as a text matrix, so the remaining gaps are a one-line query.

    `.` is covered, `-` is explained-as-not-applicable, `?` is unrecorded. Ops are listed
    alphabetically and the per-class totals come last.
    """
    glyph = {COVERED: ".", UNRECORDED: "?"}
    header = "  ".join(c.name for c in EdgeClass)
    width = max(len(op.name) for op in ledger) if ledger else 0
    lines = [f"{'op'.ljust(width)}  {header}", f"{'-' * width}  {'-' * len(header)}"]
    for op in sorted(ledger, key=lambda o: o.name):
        row = "  ".join(
            glyph.get(ledger[op][c], "-").center(len(c.name)) for c in EdgeClass
        )
        lines.append(f"{op.name.ljust(width)}  {row}")

    counts = coverage_counts(ledger)
    lines.append("")
    lines.append(f"{len(ledger)} ops.  . covered   - not applicable   ? unrecorded")
    for edge_class in EdgeClass:
        c = counts[edge_class]
        lines.append(
            f"  {edge_class.name} {edge_class.value:<22} "
            f"covered {c[COVERED]:>3}   n/a {c['explained']:>3}   "
            f"unrecorded {c[UNRECORDED]:>3}"
        )
    return "\n".join(lines)


def suite_coverage_from_tests() -> SuiteCoverage:
    """Build a SuiteCoverage by asking the suites what they drive.

    The import is a layering inversion -- this module sits below the tests -- and it is
    confined to this function, which only the CLI and test_sfpu_domains call, so it never
    happens at import time. The alternative was a second copy of four lists inside this module,
    which is the drift the ledger exists to prevent.
    """
    import test_eltwise_binary_sfpu as binary
    import test_eltwise_unary_sfpu as unary
    import test_sfpu_ternary as ternary

    return SuiteCoverage(
        # The shift ops belong here: _SHIFT_EDGE_VALUES contains INT32_MAX, and INT32_MIN with
        # its own filter and xfail, so the shift edge sweeps do drive the integer extremes.
        # _UINT32_BINARY_OPS: the uint32 high-range sweep drives 0, 1 and 2**32 - 1, which is
        # exactly integer_specials(UInt32). It reaches the extremes by a different route than
        # _INT_EXTREME_OPS and the ledger has to count both or it under-reports the class --
        # the same omission W13 fixed for the shift ops.
        integer_extremes=frozenset(binary._INT_EXTREME_OPS)
        | frozenset(binary._SHIFT_EDGE_OPS)
        | frozenset(binary._UINT32_BINARY_OPS)
        | frozenset(unary._INT_UNARY_OPS),
        integer_extremes_excluded=frozenset(binary._INT_EXTREMES_OUT_OF_RANGE),
        integer_driven=frozenset(binary._INT_DRIVEN_BINARY_OPS)
        | frozenset(binary._INT_BINARY_STIMULI)
        | frozenset(unary._INT_UNARY_OPS),
        operand_parameters=frozenset(unary._UNARY_SHIFT_OPS)
        | frozenset(binary._SHIFT_EDGE_OPS)
        | frozenset(ternary._SCALAR_OPS),
        specials_derived=frozenset({MathOperation.SfpuLogsigmoid}),
        saturation=frozenset(unary._SATURATION_PROBES)
        | frozenset(binary._BINARY_SATURATION_PAIRS),
        float_driven=frozenset(binary._CLASSIFIED_STIMULI_OPS)
        - frozenset(binary._INT_DRIVEN_BINARY_OPS)
        | frozenset(unary._EDGE_SWEEP_OPS),
        # _CLASSIFIED_STIMULI_OPS is the binary suite's own declaration of what reaches
        # sfpu_binary(). Without it, an op with no _OP_DOMAIN_REGISTRY entry is absent from the
        # ledger entirely rather than under-reported -- SfpuLogsigmoid was, which is worse than
        # a wrong cell because nothing in the totality checks can see an op that is not there.
        extra_ops=frozenset(binary._INT_DRIVEN_BINARY_OPS)
        | frozenset(binary._INT_BINARY_STIMULI)
        | frozenset(binary._CLASSIFIED_STIMULI_OPS),
    )


def _report_main() -> None:
    """`python -m helpers.sfpu_domains --report`."""
    print(format_coverage_report(coverage_ledger(suite_coverage_from_tests())))


if __name__ == "__main__":  # pragma: no cover - developer entry point
    import sys

    if "--report" in sys.argv:
        _report_main()
    else:
        print("usage: python -m helpers.sfpu_domains --report")

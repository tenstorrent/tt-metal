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

from .format_config import MX_FORMAT_MAX_NORMAL, DataFormat
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

    *spec_C* exists for the ternary family and defaults to a copy of *spec_B*, mirroring
    how spec_B defaults to a copy of spec_A. It is the third operand of ``addcdiv``,
    ``addcmul``, ``lerp`` and ``snake_beta`` -- and for two of those it is a **divisor**, so
    it is the operand that carries the pole. Before it existed, `_ternary_default_specs`
    reused B for C with a comment saying so, which meant a ternary singularity had nowhere
    to be registered and the ``c -> 0`` pole was unreachable by construction.

    Defaulting rather than requiring it keeps every existing consumer working unchanged:
    a unary or binary entry that names only spec_A still resolves to three identical specs.
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

    Both DestAccumulation and ApproximationMode are Enums whose two members wrap True and
    False, so ``bool(member)`` is True for *both* of them — the ``.No`` member included. A
    caller that passes the member directly rather than comparing it therefore gets no error,
    it gets every case evaluated as the True branch, which silently flips whole rows of
    behaviour. Read ``.value`` when handed an enum, take a bool as-is, and reject anything
    else rather than guessing.

    The enums themselves are not imported here: this module deliberately carries no
    llk_params test-side types beyond MathOperation, and duck-typing on ``.value`` keeps it
    that way.
    """
    flag = getattr(value, "value", value)
    if not isinstance(flag, bool):
        raise TypeError(
            f"{param} must be a bool or a {enum_name} member, got "
            f"{value!r} ({type(value).__name__})"
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
# Two different ceilings bound the exp family's positive side, and they belong in two
# different places:
#
#   * **Range** — exp overflows an 8-bit exponent near x = 88.7. That is a property of the
#     op and the format, true in both approximation modes, so it lives in the registry
#     entries below.
#   * **Accuracy** — the *approximation* overshoots the golden by ~5.7% once its argument
#     passes ~8 (measured on Wormhole; see _APPROX_EXP_ACCURACY_XFAIL in test_sfpu_unary).
#     That is a property of one mode only, so it lives in _APPROX_ACCURACY_MAX and is
#     applied by for_op() when the caller says ApproximationMode.Yes.
#
# Keeping the accuracy bound on the shared registry entry was the bug: one entry is consumed
# by both modes, so narrowing it to 16 also took (16, 80] away from the *accurate* path —
# with it the whole exponent-overflow region and all large-exp saturation into Float16_b and
# Bfp8_b. ExpWithBase was the sharper case still: it sits in STANDARD_SWEEP_OPS, which runs
# ApproximationMode.No only, so its bound was narrowed from 160 to 32 for a limit the op
# never executes.
#
# NOT YET MEASURED: the accurate path over (16, 80] has never been isolated on hardware. The
# evidence on record is approximation-specific — the xfail is gated on
# ApproximationMode.Yes and names "Approximate exp" — and phase 1 (#52172) shipped high=80
# for both modes, so this restores a previously-shipped domain rather than inventing one. It
# still wants a run of the Exp/Exp2 broad sweep at ApproximationMode.No before being called
# green; if the accurate path does drift here, the fix is a mode-conditional custom_rtol,
# not a re-narrowed registry entry.
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
    # Bounded by accuracy rather than by representable range, but **no longer by the
    # default tolerance**: a**b is evaluated as exp(b * ln a), and measuring the error
    # across four candidate domains on Blackhole showed the *relative* error is ~flat in
    # the operands (10.0% at b*ln a = 3.30, 13.4% at 8.32, 10.3% at 11.09) rather than
    # growing with them. So what capped this domain at 3 was the fixed 5% rtol, not the op.
    # BINARY_CUSTOM_TOLERANCES gives it rtol=0.15 and the domain widens to the range bound:
    # A <= 8, B <= 4 puts the worst-case product at 4 * ln 8 = 8.32, 2.5x the old reach.
    #
    # A <= 16 is deliberately not taken. It is accurate (10.3%), but drives |a**b| to
    # 6.2e4, within a factor of 1.06 of Float16's 65504 ceiling -- an overflow test dressed
    # up as an accuracy one. See BINARY_CUSTOM_TOLERANCES for the full measurement.
    MathOperation.SfpuElwpow: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=8.0),
        spec_B=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=4.0),
    ),
    # xlogy: computes x * log(y) element-wise
    # srcA (x): x >= 0 so xlogy(0, y) = 0 is well-defined
    # srcB (y): y > 0 so log(y) is finite; log-uniform spans several decades
    #
    # x's ceiling is an absolute-accuracy bound: the error is dominated by
    # x * abs_err(ln y), so it grows with x while a fixed atol does not. Measured on
    # Blackhole, max absolute error is exactly linear in x -- 0.25 / 0.50 / 1.00 / 2.00 at
    # x <= 4 / 8 / 16 / 32 in Float16_b -- which confirms that model and is why no fixed
    # atol can hold. BINARY_CUSTOM_TOLERANCES gives it atol=0.6, so x doubles to 8.
    #
    # Most of the Float16_b number is *output quantization*, not the kernel: at |golden| ~ 72
    # a bfloat16 ULP is already 0.5. The Float32 column is the kernel's own error and is 4x
    # smaller (0.058 / 0.116 / 0.232 / 0.464). y keeps its full log-uniform span -- narrowing
    # it made the failures worse, not better.
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
    if result.spec_B is not None and result.spec_B != result.spec_A:
        raise ValueError(
            f"MathOperation.{op.name} has an approximation-mode ceiling but distinct "
            f"per-operand domains. _APPROX_ACCURACY_MAX is written for the unary exp "
            f"family; decide per operand before adding a binary op to it."
        )
    _clip_high(result.spec_A, ceiling, op)
    if result.spec_B is not None:
        _clip_high(result.spec_B, ceiling, op)


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
        approx_mode: The approximation mode the variant will run in, as an
            ApproximationMode member or a bool. When it is the approximating
            mode, ops in _APPROX_ACCURACY_MAX have their positive side
            narrowed to the ceiling their approximation stays accurate to;
            the registry entry itself carries only the range bound. Leaving
            it None applies no narrowing, which is right for callers that
            check no tolerance (the accuracy harness) and for the pre-existing
            format-only behaviour.

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
    return OperandSpecs(
        spec_A=_tighter_spec(by_input.spec_A, by_range.spec_A),
        spec_B=_tighter_spec(by_input.spec_B, by_range.spec_B),
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

# Binary SFPU ops (test_sfpu_binary.py). Registered ones only -- that suite also drives
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

# Ternary SFPU ops (test_sfpu_ternary.py). Empty because no ternary op has a domain entry
# yet: OperandSpecs carries A and B only, so that suite builds its own per-operand specs
# and reuses B for C. The set exists so the first registered ternary op lands here rather
# than silently in sfpu_unary_ops().
_SFPU_TERNARY_OPS: FrozenSet[MathOperation] = frozenset()

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
    """Apply per-operand undefined-region subtraction to both operands of an
    OperandSpecs in one call.

    Convenience wrapper around exclude_undefined.  Returns a deep copy so the
    caller can mutate further without aliasing the registry.
    """
    op_ranges = _SFPU_UNDEFINED_RANGES.get(op, {})
    new = copy.deepcopy(specs)
    if Operand.A in op_ranges:
        new.spec_A = exclude_intervals(new.spec_A, op_ranges[Operand.A])
    if Operand.B in op_ranges and new.spec_B is not None:
        new.spec_B = exclude_intervals(new.spec_B, op_ranges[Operand.B])
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

# Stimuli are built as fp32 host-side, so it is fp32's mantissa a narrower format's width is
# subtracted from to get the number of bits the datapath drops. Read from the table rather
# than written as 23 twice.
_FLOAT32_MANTISSA_BITS = _FORMAT_MANTISSA_BITS[DataFormat.Float32]


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

    Two formats bound a probe, and the one the caller resolves first is only half of it.
    narrowest_range_format() ranks on _FORMAT_MAX_MAGNITUDE, i.e. exponent range, and so
    bounds a probe's **magnitude**. Its **spacing** is a mantissa question, and neither the
    input nor the output format settles that one: with ``dest_acc=No`` the DEST holds 16 bits
    whatever the input format is, so an fp32 probe stepped by an fp32 ULP is truncated
    straight back onto the boundary it was meant to straddle. The goldens model the same
    truncation as ``& 0xFFFF0000``.

    Acosh is the op this bites today. Its singularity is (1.0, ABOVE), and
    ``(Float32, Float16_b)`` ties on the bfloat16 ceiling and resolves to Float32, so
    ``eps = 2 * 2**-23`` and the above-pole probe is ``0x3F800002``. Under ``dest_acc=No``
    that arrives as 1.0 — a second copy of the pole probe, which is exactly the collapse
    format_ulp() exists to prevent. Both ``(Float32 -> Float16_b, No)`` and
    ``(Float32 -> Float32, No)`` are collected by test_eltwise_unary_sfpu_edges, so a
    mantissa-keyed choice of format alone would not fix it. Nothing false-passes either way:
    the probe simply stops probing.

    Only a 32-bit *float* fmt is coarsened, and only to Float16_b. A format that is already
    16-bit or narrower keeps its own spacing, because the DEST it lands in is no coarser than
    it is (Tf32 reports ``is_32_bit() == False``, so it is not caught here either). An integer
    format keeps its own too: mantissa truncation is not what a 16-bit integer DEST does, and
    substituting a float format's ULP there would be meaningless rather than conservative.

    ``dest_acc=None`` keeps the format-only behaviour, for callers that have no dest_acc to
    give.

    This is the *format* half of the rule. probe_beside() applies it per boundary **and per
    side**, because whether the narrowing actually destroys a probe depends on both.
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

    In practice only the bfloat16 width is exercised — probe_spacing_format() either returns
    Float16_b or the format it was given, and in the latter case probe_beside() never asks.
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

    Deciding per **side** is what keeps this to the one probe that was actually collapsing.
    At a pole of 1.0 the two sides behave differently under a 16-bit DEST: stepping up gives
    ``0x3F800002``, same binade, and truncation returns it to 1.0 — a second copy of the pole
    probe, which is Acosh's ``(1.0, ABOVE)`` and the whole of the collapse. Stepping *down*
    crosses into the next binade (``0x3F7FFFFF`` -> 0.99609375) and stays distinct, so asin,
    acos, atanh, erfinv and log1p keep their tighter below-1.0 probes rather than being
    loosened along with it. Zero-poles keep theirs for the same reason: bfloat16 carries
    fp32's full exponent range, so 2**-23 survives as a visible distance from zero.
    """
    fine = point + direction * ulps * format_ulp(range_fmt, point)
    if _truncate_mantissa(fine, step_fmt) != point:
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
    # Ternary: the pole is on the *third* operand, which is why OperandSpecs grew a spec_C.
    # addcdiv is a + value * b / c and snake_beta is a + sin(b*a)^2 / c, so c = 0 is a
    # genuine pole for both. _ternary_default_specs holds c in uniform(1, 2) for exactly
    # this reason, which means the random sweep can never reach it -- the edge sweep is the
    # only thing that does.
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
    format. It defaults to *fmt*, which is the format-only behaviour. See probe_step().

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

_COMPARISON_EDGE_OPS = (
    MathOperation.UnaryGt,
    MathOperation.UnaryLt,
    MathOperation.UnaryGe,
    MathOperation.UnaryLe,
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
)

_OP_EDGE_POINTS: Dict[MathOperation, Tuple[float, ...]] = {
    **{op: (0.0, -0.0) for op in _ZERO_EDGE_OPS},
    # UnaryGt/Lt/Ge/Le reach the edge sweep through edge_spec() like everything else.
    # UnaryEq and UnaryNe do not: they are outside _OP_DOMAIN_REGISTRY, so sfpu_unary_ops()
    # never puts them in _EDGE_SWEEP_OPS. Their consumer is
    # test_sfpu_unary._threshold_op_stimuli_spec, which reads op_edge_points() directly to
    # place the exact threshold in its stimuli — the same arrangement as the int32
    # comparison ops below. Dropping these entries makes that test raise rather than
    # silently drift off the threshold.
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
    # test_sfpu_unary._int_unary_stimuli_spec, which reads op_edge_points() directly to
    # place the exact comparison tie in its stimuli. Keep that call in mind before
    # editing — dropping these entries makes the tie untestable rather than merely
    # unlisted.
    MathOperation.UnaryMaxInt32: (INT_MAXMIN_SCALAR,),
    MathOperation.UnaryMinInt32: (INT_MAXMIN_SCALAR,),
    MathOperation.UnaryMaxUint32: (INT_MAXMIN_SCALAR,),
    MathOperation.UnaryMinUint32: (INT_MAXMIN_SCALAR,),
}


# Cat D for an operand other than A. _OP_EDGE_POINTS describes the op's own input, which
# for a unary op is the only operand and for a binary op is the one whose knees are the
# op's knees. A ternary op breaks that: lerp is a + c * (b - a), so its interesting values
# are properties of the *weight*, operand C, and nothing about A or B.
#
# Kept as a second, per-operand table rather than by generalising _OP_EDGE_POINTS to a
# nested dict: 43 of the 44 entries have no per-operand structure, and paying a dict layer
# on all of them to express one op's would make every existing entry noisier to read.
_OP_OPERAND_EDGE_POINTS: Dict[MathOperation, Dict[Operand, Tuple[float, ...]]] = {
    # lerp interpolates at c = 0 (result is exactly a), reaches b at c = 1, and
    # *extrapolates* beyond it for c > 1 -- three different behaviours of one kernel, none
    # of which the default uniform(-1, 1) weight lands on. 2.0 is the extrapolating probe;
    # -1.0 extrapolates the other way, which is the same branch and a different sign.
    MathOperation.SfpuLerp: {Operand.C: (-1.0, 0.0, 1.0, 2.0)},
}


def op_edge_points(
    op: MathOperation, operand: Operand = Operand.A
) -> Tuple[float, ...]:
    """Discrete edges of *op* for *operand* that are not already a domain boundary.

    Operand A reads _OP_EDGE_POINTS, the op's own knees. Any other operand reads
    _OP_OPERAND_EDGE_POINTS, which today holds only lerp's weight boundaries. Returns ()
    when there is nothing to probe.
    """
    if operand == Operand.A:
        return _OP_EDGE_POINTS.get(op, ())
    return _OP_OPERAND_EDGE_POINTS.get(op, {}).get(operand, ())


# ─────────────────────────────────────────────────────────────────────────────
# Where IEEE specials can actually be injected
#
# A cat-B sweep must not be a plain product over formats x dest_acc. Measured on
# Wormhole (n150) by driving the isinf/isposinf/isneginf/isnan/isfinite predicates over
# the full 5x5 format matrix x both dest_acc values with no skips — 250 variants, of
# which 85 fail. The predicates are the right instrument because their output is 0.0/1.0,
# representable in every format including the block floats, so a failure isolates "the
# input's specialness did not survive unpack" from "the output cannot express a
# non-finite result".
#
# Two independent breakers came out of it:
#
#   1. A Float16 (e5m10) anywhere in the pipeline. As an *input* it never preserves
#      specials — all 5 predicates fail on all 5 output formats at both dest_acc values,
#      10/10 cells. As an *output* it fails too, unless a 32-bit input is paired with
#      dest_acc=Yes: Float32->Float16 at dest_acc=No fails all five, which is the exact
#      pair Blackhole already guards in _skip_bh_unsupported_float_combo.
#
#   2. A 16-bit input with dest_acc=Yes. Float16_b input at dest_acc=Yes fails isinf,
#      isneginf and isnan while isposinf and isfinite pass — i.e. +inf survives and -inf
#      and NaN do not. That is precisely the "bf16->fp32 dest unpack does not preserve
#      -inf/nan, mangling is_neg/is_nan" already recorded on
#      test_eltwise_unary_sfpu_isinf_isnan, now with the per-predicate detail.
#
# A third constraint is not measurable this way and is applied statically: block-float
# and MX *inputs* cannot carry specials in the first place. Verified host-side —
# quantize_input_to_unpack_format() destroys NaN for Bfp8_b and Bfp4_b (±inf survives).
# So a predicate passing on a block-float input is vacuous: golden and hardware agree
# that there is no NaN, because neither ever saw one. Those rows are excluded rather
# than trusted.
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
# The first tranche, ordered by how unambiguous the answer is rather than by how interesting
# the op is: the trivially-defined ops come first so that a failure there is the *harness*
# and not the semantics, and only then the ops whose specials compose with a pole.
#
# Eight of the ten needed no golden change at all -- their goldens already route through
# torch and torch is IEEE-correct at every special. Verified host-side before enrolling any
# of them, which is the check to repeat for the next tranche:
#
#   Identity   +inf -> +inf   -inf -> -inf   NaN -> NaN
#   Neg        +inf -> -inf   -inf -> +inf   NaN -> NaN
#   Abs        +inf -> +inf   -inf -> +inf   NaN -> NaN
#   Sqrt       +inf -> +inf   -inf -> NaN    +/-0 -> 0
#   Rsqrt      +inf -> 0      -inf -> NaN    +0 -> +inf   -0 -> -inf
#   Reciprocal +inf -> 0      -inf -> 0      +0 -> +inf   -0 -> -inf
#   Exp        +inf -> +inf   -inf -> 0      +/-0 -> 1
#   Log        +inf -> +inf   -inf -> NaN    +/-0 -> -inf
#
# Sin and Cos needed one: their goldens called math.sin / math.cos, which *raise*
# ValueError("math domain error") on a non-finite input rather than returning anything, on
# the recorded assumption that the input is "never not finite". Cat B is exactly what breaks
# that assumption. Both now route through torch, which gives NaN.
#
# Note what this does NOT claim. The *sign of a zero result* is a separate question, and one
# the ISA already answers: SFPMAD flushes negative zero to positive zero on Wormhole and
# preserves it on Blackhole. Neg(+0) -> -0 and Reciprocal(-inf) -> -0 therefore depend on the
# arch, not on the golden, and are covered by the same measurement that arch-gated the binary
# suite's negative-zero class.
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
}

# The next tranche, measured on Blackhole and deliberately NOT enrolled yet.
#
# Driving specials through these five found real divergences -- but a majority of them are
# *golden* defects, and enrolling an op whose golden is wrong would record a kernel xfail for
# a test-side bug. Each needs its golden settled first, which is exactly the per-op work cat
# B was always going to be. Measured per probe (Float32 input, the only specials-carrying
# input format on Blackhole):
#
#   Neg        NaN -> golden +inf, hw -inf   (dest_acc=No: the *golden* mangles NaN)
#              +0  -> golden +0,   hw -0     (dest_acc=Yes: the HARDWARE is IEEE-correct
#                                             here and the golden is not)
#   Reciprocal NaN -> golden +inf/NaN, hw +0 (NaN is not propagated by the kernel)
#              -inf -> golden +0,  hw -0     (hardware IEEE-correct, golden is not)
#   Sqrt       -0  -> golden +0,   hw NaN    (dest_acc=Yes)
#   Rsqrt      -0  -> golden -inf, hw NaN    (dest_acc=Yes)
#   Log        +inf -> golden +inf, hw 88.5  (~ln(FLT_MAX): the kernel clamps a non-finite
#              -inf -> golden NaN,  hw 84.3   input to the format maximum and takes the log
#              NaN  -> golden NaN,  hw 89.1   of that, so no non-finite input survives)
#
# Two conclusions worth keeping separate from the op list:
#
#  1. **Log saturates its input.** Every non-finite input comes back as a finite number near
#     ln(FLT_MAX). That is a kernel behaviour, not a golden one, and it is the single largest
#     cat-B finding so far -- worth raising with kernel owners alongside RsqrtCompat(0).
#  2. **-0.0 delivery is now answered.** At dest_acc=No, Reciprocal, Rsqrt and Sqrt all treat
#     -0 *exactly* as +0 (1/-0 -> +inf, rsqrt(-0) -> +inf, sqrt(-0) -> +0). At dest_acc=Yes
#     with a 32-bit input they do not (sqrt(-0) -> NaN, rsqrt(-0) -> NaN). That is the
#     unpack_to_dest split, measured independently of the signbit/sign/heaviside partition it
#     was inferred from -- see the coverage audit. The probe genuinely is not delivered on the
#     datacopy path.
_SPECIALS_NEXT_TRANCHE: FrozenSet[MathOperation] = frozenset(
    {
        MathOperation.Neg,
        MathOperation.Reciprocal,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
        MathOperation.Log,
    }
)


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
) -> List[float]:
    """Every value worth hitting on purpose for (*op*, *operand*) in this pipeline.

    Three sources, matching the audit's edge categories:
      * cat A — boundary_probes(): the op's singularities, straddled.
      * cat D — op_edge_points(): knees, thresholds, exact rounding ties.
      * cat B — format_specials(), only when *specials* is True. The caller decides via
        specials_safe(input_format, output_format, dest_acc); it is off by default
        because injecting them on the wrong triple is a wall of failures with one root
        cause (see the section above).

    Clipped against the *narrowest* format in the pipeline, not the input format. This is
    the part the plan's original one-format signature got wrong: a caller that passes a
    spec to a driver bypasses the driver's own for_op_pipeline() resolution entirely
    (eltwise_unary_sfpu only resolves when spec_A is None), so a probe near a format
    ceiling would otherwise reach a Float16 or MxFp4 output unclipped and overflow.

    Range and spacing are then resolved separately — *range_fmt* clips magnitudes,
    *step_fmt* sizes the ULP steps and the dedup. Pass *dest_acc* to get the second one
    right; see probe_spacing_format() for the collapse that follows from leaving it out.
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
    # Cat D, per operand. For operand A this is _OP_EDGE_POINTS, the op's own knees. A
    # binary op's B-side knees, where they exist, are domain boundaries and come from cat A
    # instead -- but a *ternary* op's third operand can have real knees of its own (lerp's
    # weight), so op_edge_points() is asked about every operand rather than only A.
    vals += list(op_edge_points(op, operand))
    if specials:
        # Specials are an exponent-range property, so they key off range_fmt: it is what
        # decides integer extremes vs IEEE non-finites, and what clip_to_format honours.
        vals += list(format_specials(range_fmt))
    return _dedup_representable(clip_to_format(vals, range_fmt), range_fmt)


def edge_spec(
    op: MathOperation,
    input_format: DataFormat,
    output_format: Optional[DataFormat] = None,
    operand: Operand = Operand.A,
    specials: bool = False,
    include_undefined: bool = False,
    dest_acc: Optional[Union[bool, Enum]] = None,
    **kwargs,
) -> Optional[StimuliSpec]:
    """edge_values() as a StimuliSpec, or None if *op* has no edge worth probing.

    Returns None rather than an empty spec so a caller can fall back to the op's random
    domain: 47 of the 97 unary SFPU ops are smooth everywhere with no knee and no pole,
    and for those an edge sweep has nothing to add beyond cat B.

    ``custom`` places the values at the head of every face and zero-fills the remainder,
    which is what we want — a face is far larger than these lists, and 0.0 is itself a
    useful probe. Note it is per-face only (generate_full_tensor raises), so the values
    repeat in every face; ``custom_faces`` is available when faces must differ.

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
    )
    if not vals:
        return None
    if input_format.is_integer() and specials:
        raise ValueError(
            f"edge_spec(specials=True) cannot deliver integer extremes for "
            f"{input_format.name}: StimuliSpec.custom clamps INT_MIN to INT_MIN + 1. "
            f"Use a raw src_A_override tensor instead."
        )
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

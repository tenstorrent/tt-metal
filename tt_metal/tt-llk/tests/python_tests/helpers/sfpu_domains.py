# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
SFPU / FPU operation domain registry and helpers.

Maps every MathOperation to safe per-operand input domains (OperandSpecs).
Provides for_op() to look up domains by op + format, and
exclude_undefined()/exclude_intervals() to subtract known-undefined regions from a
user-supplied StimuliSpec.
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

# Largest finite magnitude each format can *bound a domain* with. Only formats with a
# narrower exponent field than bfloat16 need an entry; every other format shares bfloat16's
# ceiling and is therefore never the binding constraint.
#
# Float32 is deliberately absent even though its ceiling is higher: the fallback makes it tie
# with Float16_b, and narrowest_range_format() resolves that tie to its first argument, which
# is the *stimulus* format. That is what keeps a probe on a (Float32 -> Float16_b) pipeline
# spaced by an fp32 ULP rather than a bfloat16 one -- the output's narrower mantissa bounds
# the result, not how finely the input can be dialled. format_max_magnitude() below is the
# other question, "what can this format hold", and there fp32's own ceiling is the answer.
#
# The MX rows come from MX_FORMAT_MAX_NORMAL rather than being restated here: MxFp8R is
# E5M2 (ceiling 57344) and MxFp8P is E4M3 (ceiling 448).
_FORMAT_MAX_MAGNITUDE: Dict[DataFormat, float] = {
    **MX_FORMAT_MAX_NORMAL,  # MxFp4 (e2m1) 6, MxFp8P (e4m3) 448, MxFp8R (e5m2) 57344
    DataFormat.Float16: 65504.0,  # e5m10
    # Plain E4M3 with no per-block scale to lift it, so the same 448 ceiling as MxFp8P.
    DataFormat.Fp8_e4m3: 448.0,
}

_BF16_MAX_MAGNITUDE = 3.3895314e38
_FP32_MAX_MAGNITUDE = float(torch.finfo(torch.float32).max)


def format_max_magnitude(fmt: DataFormat) -> float:
    """The largest finite magnitude *fmt* itself can represent.

    _FORMAT_MAX_MAGNITUDE answers the neighbouring question -- which format bounds a domain --
    and leaves Float32 out on purpose (see its comment). Callers asking what a format *holds*
    need fp32's real ceiling: cat F probes it, and clip_to_format() must not drop the probe it
    emits there.
    """
    if fmt == DataFormat.Float32:
        return _FP32_MAX_MAGNITUDE
    return _FORMAT_MAX_MAGNITUDE.get(fmt, _BF16_MAX_MAGNITUDE)


def _has_own_ceiling(fmt: DataFormat) -> bool:
    """Does *fmt* carry its own ceiling, or borrow bfloat16's from the fallback?

    The domain-bounding table's rows do, and so does Float32 -- the one format
    format_max_magnitude() adds on top of it.
    """
    return fmt in _FORMAT_MAX_MAGNITUDE or fmt == DataFormat.Float32


# Smallest positive *normal* of each format: the other end of the exponent range from
# _FORMAT_MAX_MAGNITUDE. From torch.finfo / the MX tables, because _FTZ_THRESHOLD comes from
# the same call and a probe on the wrong side of the flush-to-zero cliff tests nothing.
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
    into a Float32 output, saturates a Float16 one. Ties resolve to the first
    argument, so callers should pass the input format first.
    """
    candidates = [fmt for fmt in formats if fmt is not None]
    if not candidates:
        raise ValueError("narrowest_range_format() requires at least one format")
    return min(
        candidates,
        key=lambda fmt: _FORMAT_MAX_MAGNITUDE.get(fmt, _BF16_MAX_MAGNITUDE),
    )


def narrowest_ceiling_format(*formats: Optional[DataFormat]) -> DataFormat:
    """Return whichever of *formats* has the lowest ceiling, fp32's own ceiling included.

    narrowest_range_format()'s sibling, and they differ on exactly one pair. There the
    (Float32, Float16_b) tie resolves to the stimulus format, which is what spaces a probe by
    an fp32 ULP on that pipeline. Here it resolves to Float16_b, because a magnitude probe has
    to survive *both* legs: 3.40e38 into a bfloat16 output is a value the pack cannot hold,
    which is the saturation sweep's subject rather than cat F's.
    """
    candidates = [fmt for fmt in formats if fmt is not None]
    if not candidates:
        raise ValueError("narrowest_ceiling_format() requires at least one format")
    return min(candidates, key=format_max_magnitude)


def _two_state_flag(value: Union[bool, Enum, None], param: str, enum_name: str) -> bool:
    """Normalise a two-state Enum-or-bool test flag to a plain bool.

    DestAccumulation and ApproximationMode wrap True and False, so ``bool(member)`` is True
    for *both* — ``.No`` included, which silently selects the True branch. Read ``.value``
    for an enum, take a bool as-is, reject anything else.

    *enum_name* is checked too, because both enums' ``.Yes`` wraps True and duck-typing on
    ``.value`` would accept the wrong one. Matched on class name so this module needs no
    llk_params import beyond MathOperation.
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
# Two separate ceilings bound the positive side. The range bound (exp overflows an 8-bit
# exponent near x = 88.7) holds in both modes and lives in the registry entries below; the
# approximation's accuracy ceiling holds in one mode only and lives in _APPROX_ACCURACY_MAX,
# applied on top by for_op() at ApproximationMode.Yes.
#
# ExpWithBase's entry is unreachable while the op runs at ApproximationMode.No only, and is
# kept so enrolling it in BROAD_SWEEP_OPS cannot hand the approximation an unbounded
# argument.
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
    mantissa width: 10:1 suffices for Bfp8_b's 7 bits, while Bfp4_b's 3 bits need
    4:1 to keep every element representable.
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
    # atanh: domain |x| < 1. The log1p form is stable across the whole interior, small x
    # included, and close to ±1, so the sweep can run near the boundary; stay just inside
    # ±1 to avoid the exact ±inf endpoints (covered separately by special-case tests).
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
    # exp_with_base computes exp(0.5*x), so its positive side is double exp's to put the
    # argument on the same ceiling. See _exp_with_base_spec.
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
    # relu_max(x) = clamp(x, 0, RELU_MAX_THRESHOLD) -- two cutoffs, and the upper one is a
    # strict `result > threshold`, so the bound has to clear the threshold for a *finite*
    # input to reach it. See _OP_EDGE_POINTS for the straddled cutoffs.
    MathOperation.ReluMax: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.UNIFORM,
            low=-5.0,
            high=2.0 * RELU_MAX_THRESHOLD,
        )
    ),
    # relu_min(x) = max(x, RELU_MIN_THRESHOLD), so the upper bound has to clear the
    # threshold or every input clamps and the golden collapses to that constant.
    MathOperation.ReluMin: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.UNIFORM,
            low=-5.0,
            high=2.0 * RELU_MIN_THRESHOLD,
        )
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
    MathOperation.TopKDefuse: OperandSpecs(
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
    # Bounded by accuracy rather than representable range: a**b evaluates as
    # exp(b * ln a), and the relative error is roughly flat in the operands, so the bounds
    # pair with the rtol in BINARY_CUSTOM_TOLERANCES. A <= 16 is left out because it drives
    # |a**b| to Float16's ceiling, which would make this an overflow test.
    MathOperation.SfpuElwpow: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=8.0),
        spec_B=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=4.0),
    ),
    # xlogy: x * log(y) element-wise. srcA (x): x >= 0 so xlogy(0, y) = 0 is well-defined.
    # srcB (y): y > 0 so log(y) is finite; log-uniform spans several decades.
    #
    # x's ceiling is an absolute-accuracy bound: the error is dominated by
    # x * abs_err(ln y) and so grows with x while a fixed atol does not, which is what pairs
    # it with the atol in BINARY_CUSTOM_TOLERANCES. Most of that error is output
    # quantization rather than the kernel. y keeps its full log-uniform span.
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
    # Every operand, not just A and B: __post_init__ deep-copies C from B before this runs, so
    # clipping only A and B leaves spec_for(Operand.C) disagreeing with the other two.
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
      and a wider output cannot give it back, so resolving Bfp8_b -> Float16
      against the output alone would restore the wide interval _reciprocal_spec
      exists to avoid.

    So resolve against both formats and keep whichever spec is tighter per
    operand. Both constraints only ever *narrow* a domain, so the tighter of the
    two satisfies both.
    """
    by_input = for_op(op, input_format, **kwargs)
    range_format = narrowest_range_format(input_format, output_format)
    if range_format == input_format:
        return by_input

    by_range = for_op(op, range_format, **kwargs)
    # C as well as A and B: rebuilding only A and B would drop a registered third-operand
    # domain, and __post_init__ would then refill spec_C from spec_B.
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
# Which ops the unary sweep must not drive
#
# _OP_DOMAIN_REGISTRY mixes the unary SFPU ops with everything else, so sfpu_unary_ops()
# subtracts the ops that have no unary SFPU kernel and treats the remainder as unary. An op
# registered without being listed here therefore lands in the unary sweep, which is the
# safe direction: it trips that sweep's exhaustiveness assert rather than going untested.
# ─────────────────────────────────────────────────────────────────────────────

# Binary SFPU ops (test_eltwise_binary_sfpu.py). Registered ones only -- that suite also
# drives int, comparison and bitwise ops that have no domain entry and so cannot be keys
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

# No ternary op is registered yet, so there is nothing to subtract for that family: the
# suite builds its own per-operand specs and reuses B for C.
_NON_SFPU_UNARY_OPS: FrozenSet[MathOperation] = _SFPU_BINARY_OPS | frozenset(
    {
        # Eltwise binary on the FPU (test_eltwise_binary.py).
        MathOperation.Elwadd,
        MathOperation.Elwmul,
        MathOperation.Elwsub,
        # Reduce family (test_sfpu_reduce*.py).
        MathOperation.ReduceColumn,
        MathOperation.ReduceRow,
        MathOperation.ReduceScalar,
        # Applied by the packer (STACC_RELU): Relu has a domain and looks unary, but is not
        # a member of SfpuType, so driving it through the unary test fails to compile.
        MathOperation.Relu,
    }
)

# Unary SFPU ops that are registered but deliberately not in the correctness sweep.
_UNARY_OPS_NOT_SWEPT: Dict[MathOperation, str] = {
    MathOperation.TopKLocalSort: "perf-only; whole-op topk is covered by test_topk.py",
    MathOperation.TopKMerge: "perf-only; whole-op topk is covered by test_topk.py",
    MathOperation.TopKRebuild: "perf-only; whole-op topk is covered by test_topk.py",
    MathOperation.TopKDefuse: "perf-only; whole-op topk is covered by test_topk.py",
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
    caller can mutate further without aliasing the registry. Covers operand C too.
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
# "which single values are worth hitting on purpose", and the two must not share a table.
#
# Mechanically they cannot: exclude_intervals() *always* rewrites its result into the
# `intervals` form, and the interval sampler consumes two torch.rand draws per element
# where the plain low/high sampler consumes one. So adding an entry to
# _SFPU_UNDEFINED_RANGES re-draws that op's whole stimulus set even when the subtraction
# removes nothing. Edge metadata therefore lives in its own tables and never touches the
# random-draw path.
#
# Semantically they cannot either. A hole in _SFPU_UNDEFINED_RANGES is a *guard band* —
# Reciprocal's is (-1e-6, 1e-6), so its edges are ±1e-6 and the mathematically
# interesting point, exactly 0, is inside the hole and never produced. Probing wants the
# singularity itself, so _OP_SINGULARITIES records it directly.
# ─────────────────────────────────────────────────────────────────────────────

# Mantissa bits *after* the implicit leading 1, i.e. what sets the step between adjacent
# representable values. For the block floats this is the per-element magnitude width
# (7 / 3 / 1).
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

    Derived from the mantissa-width table above rather than written as a literal. An FP16
    Dest keeps three more mantissa bits than a BF16 one.
    """
    dropped = _FLOAT32_MANTISSA_BITS - _FORMAT_MANTISSA_BITS[dst_format]
    return (0xFFFFFFFF << dropped) & 0xFFFFFFFF


def format_ulp(fmt: DataFormat, magnitude: float = 1.0) -> float:
    """Distance to the next representable value of *fmt* near |*magnitude*|.

    This is what a boundary probe has to be offset by. A fixed epsilon does not work:
    at a boundary of 1.0 in Float16_b, 1.0 - 1e-6 *is* 1.0, so the "just inside" and
    "at" probes collapse onto one point.

    For the block-float formats the real step is set by the exponent shared across the
    16-element block, so the value returned here is a lower bound for a small element
    sitting inside a block with a large maximum. That is the safe direction for a probe,
    but it means a block-float probe cannot be assumed distinct.
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
    straddle. The goldens model that truncation as ``& 0xFFFF0000``. Where it bites, the
    probe does not false-pass -- it collapses onto the boundary and stops probing.

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

    Deciding per **side** keeps this to the probes that actually collapse: at a pole of 1.0
    under a 16-bit DEST, stepping up truncates back onto 1.0, while stepping down crosses
    into the next binade and stays distinct. Zero-poles are unaffected either way, since
    bfloat16 carries fp32's full exponent range.
    """
    fine = point + direction * ulps * format_ulp(range_fmt, point)
    # Both sides truncated: the datapath truncates the boundary as well as the probe, and the
    # question is whether they stay distinct *as the kernel sees them*. Equivalent to
    # comparing against the bare point for a boundary the format holds exactly, and not
    # otherwise.
    if _truncate_mantissa(fine, step_fmt) != _truncate_mantissa(point, step_fmt):
        return fine
    return point + direction * ulps * format_ulp(step_fmt, point)


# Exact singular points of each op, per operand — the pole, the branch cut, the value
# where the function stops being defined. Separate from _SFPU_UNDEFINED_RANGES, which
# records a guard band wide enough to keep a random draw away from a singularity rather
# than the singularity's own location.
#
# Held back on purpose: Lgamma, Digamma and Polygamma all have poles at 0 and the
# negative integers, but their kernels are polynomial/LUT fits that only claim accuracy
# well inside a positive domain (see their registry comments). A probe at their boundary
# tests a value the kernel never promised. Add them if and when the kernels claim it.
#
# Each singularity carries **which side of it the op is defined on**, because that decides
# whether a probe one ULP away is a legitimate edge case or a value the kernel never
# promised anything about:
#
#   BOTH  - defined on both sides (1/x for x<0 and x>0); probe both, plus the point.
#   ABOVE - defined only above (log, sqrt); probe the point and above it.
#   BELOW - defined only below (the +1 end of asin/atanh).
#
# Probing the undefined side fails for several ops today, and on the *golden* rather than
# the kernel: torch-backed goldens return `inf` where the mathematical answer is `nan`
# (log(-eps), sqrt(-eps)). Pass include_undefined=True to probe the far side anyway once
# the goldens model it.
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
    get the op probed, with no second hand-written list to keep in step.
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
    difference between the two zeros is the entire point of the probe. Signed zeros are
    therefore keyed by sign, not by value.
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
      one probe on its defined side, so a newly declared hole yields probes without a
      second table entry.

    Not both. A guard band sits a fixed 1e-6 off the true boundary, so the two sources
    disagree about which binade the boundary is in — at a boundary of 1.0 the band edge
    0.999999 has half the ULP that 1.0 has, and the pair emits a third probe that is
    neither the boundary nor a clean step from it. The singularity wins where it exists.

    Probes are offset by *ulps* steps of *fmt* at the boundary's own magnitude (see
    format_ulp) rather than by a fixed constant, so they stay distinct in low-precision
    formats. *step_fmt* widens that step where the datapath is coarser than *fmt* — pass
    probe_spacing_format(fmt, dest_acc) to account for a 16-bit DEST holding a 32-bit
    format. It defaults to *fmt*. See probe_beside().

    Returns a sorted list with format-indistinguishable duplicates removed. Values are
    *not* clipped to what *fmt* can represent or to the op's registered domain — that is
    the caller's job, and has to be done against the narrowest format in the pipeline.
    """
    probes: List[float] = []
    step_fmt = fmt if step_fmt is None else step_fmt

    singularities = _OP_SINGULARITIES.get(op, {}).get(operand, ())
    if singularities:
        for point, side in singularities:
            probes.append(point)
            if point == 0.0:
                # Cat G. A zero pole is two probes: 1/+0 is +inf and 1/-0 is -inf, and the
                # same split runs through div, fmod, remainder, xlogy, atan2, addcdiv and
                # snake_beta. Emitted unconditionally -- edge_values() drops it on the
                # pipelines that flatten a -0.0, which this function has no dest_acc to judge.
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


# +0.0 and -0.0 are listed separately and both matter: signbit, sign, heaviside,
# reciprocal and the comparison-to-zero ops all distinguish them.
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
# The *amounts* only, deliberately not the rule for what they produce -- three consumers
# with two behaviours between them:
#
#   binary shifts        every out-of-range amount produces 0, both signs
#   unary left shift     the same: calculate_left_shift zeroes the result
#   unary right shift    calculate_right_shift clamps the amount to 31 and shifts anyway, so
#                        a negative operand yields -1 rather than 0
#
# Each golden states its own kernel's rule. The amounts are shared because both suites drive
# them: binary shift ops take the amount as an operand, unary ones as a compile-time immediate
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
    INT_MIN bit pattern is "negative zero" there. Deliver it as a raw override tensor and
    expect it to fail on hardware — that is a documented HW limitation, not a test gap.
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
# format_specials()' counterpart at the other end. The widest registered domain is +/-1000, so
# the input space jumps from ~10 straight to infinity, with nothing in the thirty-odd decades
# between or in the band just above zero the goldens model flush-to-zero for. Its own axis, not
# a flag on `specials`: a subnormal survives an unpack that destroys a NaN.
# ─────────────────────────────────────────────────────────────────────────────


def format_extremes(fmt: DataFormat) -> Tuple[float, ...]:
    """Finite magnitude extremes of *fmt*, both signs: the ceiling, the largest step below it,
    the smallest normal, and half of that.

    Both top magnitudes are rounded onto the *ceiling's* format grid: the bfloat16 fallback
    literal sits a hair above the true maximum, and an fp32-sized step below a bfloat16 ceiling
    truncates back onto it. Raises for integer and block-float formats.
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
    ceiling_fmt = fmt if _has_own_ceiling(fmt) else DataFormat.Float16_b
    ceiling = _truncate_mantissa(format_max_magnitude(fmt), ceiling_fmt)
    below_ceiling = _truncate_mantissa(
        ceiling - format_ulp(ceiling_fmt, ceiling), ceiling_fmt
    )
    min_normal = _FORMAT_MIN_NORMAL[fmt]
    magnitudes = (ceiling, below_ceiling, min_normal, min_normal / 2.0)
    return tuple(-m for m in magnitudes) + magnitudes


def extremes_safe(
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: Union[bool, Enum],
) -> bool:
    """May format_extremes() be injected on this (input, output, dest_acc) triple?

    Deliberately **not** specials_safe(): its breakers are about +/-inf and NaN surviving an
    unpack, and every pipeline that carries 1.0 carries 3.39e38 and 2^-126. What this excludes
    is block-float and MX on either leg, one exponent across 16 elements against a probe list
    spanning thirty-odd decades. ``dest_acc`` does not change the answer today.
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
# Every dispatch constant this table probes at is imported from sfpu_dispatch_constants,
# which UnarySFPUGolden reads too — so there is one number, not a copy per consumer, and a
# threshold cannot move on the golden side without the probe moving with it. Values
# written literally below (hardsigmoid's [-3, 3], the rounding ties, the integer knees) are
# properties of the mathematics rather than of a kernel's dispatch, so there is nothing on
# the golden side for them to drift from.
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

# Ops smooth at zero whose value there is known and exactly representable, so a fit off by a
# constant at the origin is visible there and nowhere else -- the registered domains are
# symmetric uniforms where a random draw never lands on exactly 0.0. Measured on a Blackhole
# p150: all fourteen agree, two with a constant error this probe now holds in place
# (erfc(+0) = 0.99612 against erfc(-0) = 1.00388, tanh_derivative(+/-0) = 0.99920 on a 32-bit
# dest). The gamma family excludes itself: poles at zero and domains starting above it.
# fmt: off
_EXACT_AT_ZERO_OPS = (
    MathOperation.Erf, MathOperation.Erfc, MathOperation.Gelu,
    MathOperation.GeluDerivative, MathOperation.Sigmoid, MathOperation.SigmoidAppx,
    MathOperation.Tanh, MathOperation.TanhDerivative, MathOperation.TanhDerivativeLut,
    MathOperation.Expm1Cw, MathOperation.I1, MathOperation.Rpow,
    MathOperation.UnaryPower, MathOperation.CastFp32ToFp16a,
)
# fmt: on

# The integer binary ops whose answer at zero is a knee: gcd and lcm have identities there, 0
# annihilates the multiply and is neutral for max/min, and 1 comes with it as the
# multiplicative identity. Here rather than in the test, so an op joins by gaining a table
# entry and cat D derives from the same place. The divisor ops are absent -- a zero divisor is
# undefined for them (_INT_ZERO_UNDEFINED_DIVISOR) and a zero dividend is ordinary.
# fmt: off
_INT_ZERO_KNEE_OPS = (
    MathOperation.SfpuGcd, MathOperation.SfpuLcm, MathOperation.SfpuMulInt32,
    MathOperation.SfpuMaxInt32, MathOperation.SfpuMinInt32,
    MathOperation.SfpuMaxUint32, MathOperation.SfpuMinUint32,
)
# fmt: on

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
    **{op: (0.0, -0.0) for op in _EXACT_AT_ZERO_OPS},
    **{op: (0.0, 1.0) for op in _INT_ZERO_KNEE_OPS},
    # UnaryGt/Lt/Ge/Le reach the edge sweep through edge_spec(), which is what consumes these.
    #
    # UnaryEq, UnaryNe and LogicalNot below have no consumer today, and are listed anyway.
    # They are outside _OP_DOMAIN_REGISTRY, so sfpu_unary_ops() never puts them in an edge
    # sweep and edge_spec() never sees them. Kept as the recorded cutoff so registering any
    # of the three later gets a correct edge sweep for free -- but treat them as
    # documentation rather than as something a test reads. The threshold the sweeps actually
    # use lives in _OP_COMPARISON_THRESHOLD, so moving one here alone moves nothing.
    **{op: (UNARY_COMP_THRESHOLD,) for op in _COMPARISON_EDGE_OPS},
    # logical_not(x) = (x == 0). Same shape as _ZERO_EDGE_OPS but it is a threshold op
    # rather than a sign op, so keep it named. (LogicalNotUnary is an alias of this
    # member — see the note in llk_params.py — so listing both would be one key.)
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
    # relu_max clamps above at its threshold, and keeps relu's own knee at 0. Both cutoffs
    # are strict compares (`> threshold`, `< 0`), so each needs a value on either side and
    # not just the cutoff itself: at exactly the threshold the clamp does not fire, and at
    # exactly 0 the relu does not. -0.0 is deliberately absent -- the relu branch compares
    # on SFPSETCC, which is unspecified for a negative zero or a NaN.
    MathOperation.ReluMax: (
        -1.0,
        0.0,
        1.0,
        RELU_MAX_THRESHOLD - 1.0,
        RELU_MAX_THRESHOLD,
        RELU_MAX_THRESHOLD + 1.0,
    ),
    # The threshold alone only proves the clamp branch. Straddle it so the edge face also
    # carries a value that passes through unchanged; 4.0/5.0/6.0 are exact in every format
    # this sweep runs, so the pair does not blur together in bf16.
    MathOperation.ReluMin: (
        RELU_MIN_THRESHOLD - 1.0,
        RELU_MIN_THRESHOLD,
        RELU_MIN_THRESHOLD + 1.0,
    ),
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
    # test_eltwise_unary_sfpu._int_unary_stimuli_spec, which reads op_edge_points()
    # directly to place the exact comparison tie in its stimuli. Dropping an entry makes
    # that tie untestable rather than merely unlisted.
    MathOperation.UnaryMaxInt32: (INT_MAXMIN_SCALAR,),
    MathOperation.UnaryMinInt32: (INT_MAXMIN_SCALAR,),
    MathOperation.UnaryMaxUint32: (INT_MAXMIN_SCALAR,),
    MathOperation.UnaryMinUint32: (INT_MAXMIN_SCALAR,),
    # IEEE pow(x, 0) == 1 for every x, including a negative base. (-2)**0 must stay +1
    # rather than picking up the odd-integer sign flip; 2**0 is the matching positive
    # control. Crossed with Operand.B's zero encodings in _OP_OPERAND_EDGE_POINTS.
    MathOperation.SfpuElwpow: (-2.0, 2.0),
}


# The scalar each threshold-driven op compares against, mirrored from the dispatch constants
# the kernels read.
#
# Explicit rather than positional: several _OP_EDGE_POINTS entries *straddle* their cutoff --
# ReluMin's leads below it and ReluMax's leads with the relu knee -- so index 0 there is a probe
# beside the threshold rather than the threshold itself, and a positional read would place the
# wrong value. Consumers: test_eltwise_unary_sfpu._threshold_op_stimuli_spec.
_OP_COMPARISON_THRESHOLD: Dict[MathOperation, float] = {
    # logical_not(x) = (x == 0) ? 1 : 0.
    MathOperation.LogicalNotUnary: 0.0,
    **{op: UNARY_COMP_THRESHOLD for op in _COMPARISON_EDGE_OPS},
    # The two clamps. relu_min clamps below at its threshold; relu_max clamps *above* at
    # its own, and its relu knee at 0 is a second cutoff the random domain already covers.
    MathOperation.ReluMin: RELU_MIN_THRESHOLD,
    MathOperation.ReluMax: RELU_MAX_THRESHOLD,
    # threshold(x) jumps to THRESHOLD_V below THRESHOLD_T.
    MathOperation.Threshold: THRESHOLD_T,
}


def op_threshold(op: MathOperation) -> Optional[float]:
    """The scalar *op* compares its input against, or None if it has no such scalar.

    Use this rather than reading op_edge_points(op)[0]: the edge-point entries straddle
    their cutoffs, so index 0 is not the threshold for every op that has one.
    """
    return _OP_COMPARISON_THRESHOLD.get(op)


# Cat D for an operand other than A. _OP_EDGE_POINTS describes the op's own input, which for
# a unary or binary op is operand A. A ternary op breaks that: lerp is a + c * (b - a), so
# its interesting values are properties of the *weight*, operand C. pow's interesting
# exponent encodings live here for the same reason: the singularity is on the base.
#
# A second per-operand table rather than a nested _OP_EDGE_POINTS: only a couple of entries
# have per-operand structure, and a dict layer on all of them would read worse.
_OP_OPERAND_EDGE_POINTS: Dict[MathOperation, Dict[Operand, Tuple[float, ...]]] = {
    # lerp: c=0 gives a, c=1 gives b, and c outside [0,1] extrapolates (2.0 and -1.0).
    MathOperation.SfpuLerp: {Operand.C: (-1.0, 0.0, 1.0, 2.0)},
    # IEEE pow(x, 0) == 1; the kernel compares on setsgn, so -0.0 needs its own probe.
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
# A cat-B sweep must not be a plain product over formats x dest_acc. Which cells work is
# established with the isinf/isposinf/isneginf/isnan/isfinite predicates, whose 0.0/1.0 output
# every format can express, so a failure means the input lost its specialness rather than the
# output being unable to hold the result. Three constraints:
#
#   1. Float16 inputs are excluded: although Float16 can encode specials, unpack does not preserve them (any output, any dest_acc).
#   2. A Float16 output loses them too, unless the input is 32-bit at dest_acc=Yes.
#   3. A 16-bit input at dest_acc=Yes keeps +inf but loses -inf and NaN.
#
# Block-float and MX inputs are excluded statically rather than probed: unpack quantization
# destroys NaN (±inf survives), so a predicate passing there is vacuous -- golden and hardware
# agree there is no NaN because neither saw one.
# ─────────────────────────────────────────────────────────────────────────────

# Float formats whose unpack quantization leaves +inf, -inf and NaN intact, so the golden
# still evaluates the op at the special the test meant to inject.
_SPECIALS_CARRYING_INPUTS: FrozenSet[DataFormat] = frozenset(
    {DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b}
)


# Ops whose golden defines a result for non-finite *inputs*, and may therefore have cat-B
# specials injected. The golden-side gate; specials_safe() above is the pipeline-side one and
# both have to pass. Per op rather than one global bool, because about half the goldens still
# have no answer at a special. Says nothing about the sign of a zero result, which is
# arch-dependent -- SFPMAD flushes a negative zero on Wormhole and preserves it on Blackhole.
SPECIALS_READY_OPS: FrozenSet[MathOperation] = frozenset(
    {
        # Golden and kernel agree at every special the pipeline delivers, which is all this
        # suite establishes -- not that the golden is independently right.
        MathOperation.Abs,  # |+/-inf| = +inf, |NaN| = NaN, |+/-0| = 0
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
        MathOperation.Exp,  # exp(+inf) = +inf, exp(-inf) = 0, exp(+/-0) = 1
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
        MathOperation.Identity,  # every special maps to itself
        MathOperation.LessThanEqualZero,
        MathOperation.LessThanZero,
        MathOperation.Log1p,
        MathOperation.Lrelu,
        MathOperation.Mish,
        MathOperation.Neg,  # -/+inf, -/+0; the only negative NaN here (cast_to_dest_dtype)
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
        # Divergences worth reading before trusting one of these: each is xfailed per combination
        # in the sweep rather than smoothed over in the golden.
        MathOperation.Reciprocal,  # 1/+/-0 = +/-inf, 1/+/-inf = +/-0; kernel gives +0 for NaN
        MathOperation.Sqrt,  # sqrt(-inf) = NaN; kernel gives NaN for sqrt(-0), IEEE gives -0
        MathOperation.Rsqrt,  # rsqrt(+/-0) = +/-inf; same -0 divergence as Sqrt
        MathOperation.SqrtCustom,  # sqrt(-inf) gives -inf where IEEE gives NaN (issue #52930)
        # These goldens have to route through torch: math.sin / cos / acos / asin / tan *raise*
        # on a non-finite input instead of returning NaN, so a `math.*` call in a unary golden
        # is the same trap.
        MathOperation.Sin,  # sin(+/-inf) = NaN, sin(+/-0) = +/-0
        MathOperation.Cos,  # cos(+/-inf) = NaN, cos(+/-0) = 1
        MathOperation.Acos,
        MathOperation.Asin,
        MathOperation.Tan,
        # Goldens whose guard has to test for a NaN before anything else: a finite-input
        # question ("did this overflow?", "is this inside the shrink band?") answers false for
        # a NaN too, which would route it down the wrong branch.
        MathOperation.Square,  # +/-inf -> +inf, NaN -> NaN
        MathOperation.I0,  # +/-inf -> +inf (even, unbounded); torch.special.i0 gives NaN there
        MathOperation.Hardshrink,  # NaN -> NaN, not shrunk to zero
        # The comparison family answers on the SFPU's total order, not IEEE's unordered compare:
        # +NaN ranks above every finite value, so a clamp lands on its upper bound rather than
        # propagating. sfpu_total_order_key models that. UnaryLt/Le/Max sit in the block above,
        # since there the two orders agree at a +NaN.
        MathOperation.UnaryGt,  # NaN > 0.5 is true under the total order, false under IEEE
        MathOperation.UnaryGe,  # as UnaryGt
        MathOperation.UnaryMin,  # min(x, 0.0): +NaN is the maximum, so the other operand wins
        MathOperation.Clamp,  # calculate_clamp: max-then-min folds, so a +NaN lands on max
        MathOperation.Hardtanh,  # sfpi::clamp, the same composition as Clamp, so one golden
        MathOperation.ReluMax,  # _relu_max_body_ swaps a NaN for the threshold before the clamp
        # ReluMin rests on the +NaN result alone: FLOAT_SPECIALS carries no -NaN, and one
        # would fold to the threshold, -NaN ranking below -inf.
        MathOperation.ReluMin,  # max(x, threshold) as one SFPSWAP fold, so a +NaN survives
        MathOperation.Hardsigmoid,  # x*(1/6) + 0.5 through _relu_max_body_, so a NaN clamps to 1.0
        # The scalar binops (ScalarBinopGolden). Each is `x (+|-|*|/) c` for a compile-time c, so
        # cat B is their entire edge story -- edge_spec() returns None for them otherwise.
        MathOperation.ScalarAdd,  # +/-inf and NaN pass through; +/-0 + c = c
        MathOperation.ScalarSub,  # as ScalarAdd with the sign of c
        MathOperation.ScalarMul,  # +/-inf * c keeps the sign; +/-0 * c keeps the zero's sign
        MathOperation.ScalarDiv,  # host folds to x * (1/d), so divide-by-zero is unreachable
        MathOperation.ScalarRsub,  # c - x: +inf and -inf swap, NaN propagates
        # Read narrowly: fill writes a compile-time constant and its golden ignores the input, so
        # the probe says only that a non-finite input does not corrupt the fill.
        MathOperation.Fill,
    }
)

# Three facts the entries above rest on:
#
# * torch's fp32 -> bfloat16 cast canonicalises every NaN to a negative one, so
#   cast_to_dest_dtype has to model a 16-bit Dest write as the truncation it is. Signbit's
#   enrolment depends on that.
# * A zero's sign is invisible to passed_test(), which judges by torch.isclose, a both-NaN
#   clause and PCC -- so a wrong zero sign can neither fail nor XPASS.
# * Only a 32-bit input at dest_acc=Yes delivers a -0.0, which is what scopes Sqrt's and
#   Rsqrt's xfails to unpack-to-dest.
#
# One record here went stale: this comment used to say Log clamps a non-finite input to the
# format maximum and logs that. It does not any more (Blackhole p150), so Log and LogWithBase
# are enrolled below.
SPECIALS_READY_OPS = SPECIALS_READY_OPS | frozenset(
    {
        MathOperation.Log,  # log(+inf) = +inf, log(-inf) = NaN, log(+/-0) = -inf
        MathOperation.LogWithBase,  # Log scaled by a compile-time multiply
    }
)

# The other half of the unary partition: the ops measured to diverge, so nothing records cat B
# as off for them by omission. Driven on a Blackhole p150 over the full specials set on every
# specials-safe cell; five causes, not twenty-five, which is what the grouping records.
_UNARY_SPECIALS_NOT_READY: FrozenSet[MathOperation] = frozenset(
    {
        # (1) A reciprocal composition returning +0 for 1/NaN. Identical to the divergence
        # already recorded against unary Reciprocal, through the same SFPARECIP.
        MathOperation.Rdiv,
        MathOperation.ReciprocalCompat,
        MathOperation.RsqrtCompat,
        MathOperation.Sigmoid,
        MathOperation.TanhDerivative,
        # (2) A LUT or polynomial fit evaluated at a non-finite. SFPLUTFP32 documents no
        # NaN/inf handling, so each returns whatever its fit produces from a saturated
        # argument -- Erf(NaN) = 1.0, Gelu(-inf) = 0, TanhDerivativeLut(+/-inf) = +inf,
        # Digamma(NaN) = 89. An LLK decision, not an ISA one.
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
        # (3) Saturates a non-finite input to +/-1.1547668e37, which _i1_bessel's own comment
        # already records.
        MathOperation.I1,
        # (4) The *golden* is the wrong party: both route a NaN through a finite-input
        # comparison that is false for it, landing in the zero branch (Sign -> 0,
        # Heaviside -> 0.5). The kernel answers 1.0, the SFPU total order's reading that let
        # UnaryGt, Clamp, ReluMax and Hardsigmoid enrol. Needs a golden change, not a
        # measurement; their separate SFPSETCC -0.0 divergence survives the fix.
        MathOperation.Sign,
        MathOperation.Heaviside,
        # (5) Each its own question, and none a composition the groups above cover.
        MathOperation.CastFp32ToFp16a,  # cast(NaN) reaches L1 as +inf on fp32 end-to-end
        MathOperation.Expm1Cw,  # expm1_cw(+inf) = NaN; the subtraction of 1, not the exp
        MathOperation.Erfinv,  # erfinv(-inf) = -inf against the golden's +inf
        MathOperation.Frac,  # frac(-inf) = +inf; kernel, golden and IEEE all disagree
        MathOperation.Rpow,  # rpow(-inf) = +inf; the exp(x ln b) sign is not carried
        MathOperation.UnaryPower,  # power(NaN) = +inf, as SfpuElwpow's exp(b ln a) does
    }
)

assert not (
    SPECIALS_READY_OPS & _UNARY_SPECIALS_NOT_READY
), "an op cannot be both enrolled in cat B and recorded as not ready for it"


def _dest_acc_flag(dest_acc: Union[bool, Enum]) -> bool:
    """Normalise a 32-bit-destination flag to a plain bool.

    DestAccumulation is an Enum whose two members wrap True and False, so ``bool(member)``
    is True for *both* of them -- ``DestAccumulation.No`` included, which would evaluate
    every triple as the 32-bit-dest case and silently flip specials on and off for whole
    rows of the matrix.

    Read ``.value`` when handed the enum, take a bool as-is, and reject anything else
    rather than guessing. The enum itself is not imported here: this module carries no
    llk_params test-side types beyond MathOperation, and duck-typing on ``.value`` keeps it
    that way.
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
    "do not inject" rather than to a wall of failures with one root cause. Each rule below
    is one of the breakers described in the section comment.
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
        # Not something the predicates can show — their result is 0/1, so they pass here.
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
    datum goes through SrcA and the datacopy, and the LREG holds +0.0.

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

    Block floats answer False: their small end is the shared exponent, not a per-value fact.
    """
    min_normal = _FORMAT_MIN_NORMAL.get(fmt)
    if min_normal is None:
        return False
    return value != 0.0 and abs(value) < min_normal


def subnormal_delivered(
    input_format: DataFormat, dest_acc: Optional[Union[bool, Enum]]
) -> bool:
    """Does a subnormal written to L1 still have a non-zero magnitude when the SFPU reads it?

    Only on the unpack-to-dest path -- a 32-bit input at dest_acc=Yes; everywhere else it goes
    through SrcA and the datacopy, which normalises. Measured on a Blackhole p150 with
    +/-2^-127 through Ceil, Floor, Sign and Signbit, the four cat-F ops whose answer at a
    subnormal differs from their answer at zero: everywhere but Float32->Float32 at
    dest_acc=Yes all four answer as though the input were +0.0. False drops the probe.
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
    load-bearing. The hardware does the same on both arches: a narrowing store converts a NaN
    to an infinity.

    This asks about the *output* leg where negative_zero_delivered() asks about the input leg:
    of the triples specials_safe() accepts on the {Float16_b, Float32} matrix, only
    Float32->Float32 at dest_acc=Yes carries a NaN the whole way.

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
# IEEE 754 leaves the sign of an invalid-operation default unspecified: Blackhole emits the
# canonical NaN, Wormhole one whose sign may or may not be set. That stays invisible while the
# NaN remains a NaN -- passed_test's both-NaN clause accepts either sign -- and becomes a
# +inf/-inf disagreement the moment nan_survives_to_l1() is False.
#
# Membership is by *observed* disagreement on Wormhole, not by kernel shape: two ops building
# a NaN the same way can land on opposite sides, so absence here is evidence rather than a
# guarantee -- add an op if it ever fails these cells. ScalarRsub is here because `c - x`
# builds its NaN rather than forwarding one. Not a table of the signs seen, which would assert
# what the ISA declines to promise; see UnarySFPUGolden._NAN_SIGN_TRANSPARENT_OPS for the ops
# whose sign does mean something.
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

    Not an xfail: the sign may be either, so the same hardware could satisfy or break that
    claim run to run. There is nothing to assert until the golden accepts both
    infinities on a substituted NaN, which is a change to convert_nan_to_inf's contract rather
    than to this gate.
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
# An op belongs here once it agrees with its golden at every (special, special) pair on every
# specials-safe cell; the ones that diverge are in _BINARY_SPECIALS_NOT_READY. Note that the
# comparisons and min/max answer on the SFPU's total order rather than IEEE's unordered
# comparison, which BinarySFPUGolden models -- read as IEEE they look like kernel divergences
# the ISA specifies as correct.
BINARY_SPECIALS_READY_OPS: FrozenSet[MathOperation] = frozenset(
    {
        # Plain SFPMAD arithmetic, which the ISA specifies as IEEE754 for a non-finite input.
        MathOperation.SfpuElwadd,  # inf+x = inf, inf+(-inf) = NaN, NaN+x = NaN
        MathOperation.SfpuElwsub,  # as SfpuElwadd; inf-inf = NaN is the case worth having
        MathOperation.SfpuElwmul,  # inf*x = inf, inf*0 = NaN, +/-0 signs multiply
        MathOperation.SfpuElwrsub,  # as SfpuElwsub, operands reversed
        # Total order: the kernel is a bare SFPSWAP(VEC_MIN_MAX) with no NaN guard, so +NaN is
        # the maximum and -NaN the minimum -- unlike the comparisons below, which reject a NaN
        # first. The golden models sfpu_max/min, not torch's, which propagate.
        MathOperation.SfpuBinaryMax,
        MathOperation.SfpuBinaryMin,  # torch.minimum propagates a NaN where the order returns b
        # IEEE unordered, because these kernels reject a NaN operand before comparing, leaving
        # their pre-stored default: eq/lt/gt/le/ge(NaN, x) = 0 and ne(NaN, x) = 1.
        MathOperation.SfpuElwEq,  # eq(NaN, x) = 0: the kernel rejects a NaN before comparing
        MathOperation.SfpuElwNe,  # its default result is 1, so ne(NaN, x) = 1
        MathOperation.SfpuElwLt,  # strict_ordered pre-stores 0, so lt(NaN, x) = 0
        MathOperation.SfpuElwGt,  # as SfpuElwLt, operands swapped
        MathOperation.SfpuElwLe,  # weak_ordered pre-stores 1, then 0 if either operand is NaN
        MathOperation.SfpuElwGe,  # as SfpuElwLe, operands swapped
    }
)

# The ops that diverge, grouped by what each waits on. None is enrolled on a guess: an
# enrolment written to make a variant green becomes a permanent claim about the hardware.
_BINARY_SPECIALS_NOT_READY: FrozenSet[MathOperation] = frozenset(
    {
        # Composition through a reciprocal / log / exp. Each builds its result from a primitive
        # the ISA specifies only inside a stated finite range, so what the composition does with
        # a non-finite input is an LLK decision rather than an ISA one, and one answer decides
        # all six.
        MathOperation.SfpuElwdiv,  # reciprocal + Newton-Raphson
        MathOperation.SfpuXlogy,  # x * log(y)
        MathOperation.SfpuElwpow,  # exp(b * ln a)
        MathOperation.SfpuBinaryFmod,  # quotient via reciprocal
        MathOperation.SfpuBinaryRemainder,  # as fmod
        MathOperation.SfpuAtan2,  # ratio plus a format-specific polynomial; 2 cells, not 4
        # Compare-against-zero on an operand that may be a NaN: calculate_mask lowers to
        # SFPSETCC, which is unspecified for a negative zero or a NaN. The same thing that
        # holds Sign and Heaviside out of the unary gate.
        MathOperation.SfpuMask,
        # Kernel and golden both claim torch.isclose semantics and disagree at a non-finite
        # operand; needs a per-cell read-back to say which is wrong before either is touched.
        MathOperation.SfpuIsclose,
        # Operand B is *derived*, not free. This entry used to read "effectively unary -- the
        # kernel reads B only on its x > 4 branch", which was true of the *test*: the sweep
        # stopped at x = 3.9. It runs now, and this is still not a candidate *here* -- cat B in
        # this sweep is a product of two independent lists while the contract is
        # in1 == exp(-in0). Driven by test_eltwise_binary_sfpu_logsigmoid_specials instead.
        MathOperation.SfpuLogsigmoid,
        # Not element-wise: add_top_row reduces one tile's first row across the other operand,
        # and BinarySFPUGolden returns for it *before* the Dest and pack modelling, so a
        # non-finite would be judged against a reference that models neither step.
        MathOperation.SfpuAddTopRow,
    }
)

assert not (
    set(BINARY_SPECIALS_READY_OPS) & set(_BINARY_SPECIALS_NOT_READY)
), "an op cannot be both enrolled in cat B and recorded as not ready for it"


# Cat B, ternary family. Two gates, as everywhere: a *_READY_OPS entry says the op's
# *golden* defines an answer at a non-finite operand, specials_safe() says the *pipeline*
# delivers one intact. Measured host-side over all 5x5x5 special triples, then on a
# Blackhole p150 driving each op's specials into A, B and C separately. The goldens had to
# be fixed first -- neither modelled the store into Dest or the pack out of it, so the
# packer's substituted infinity read as the kernel having computed one, 10 of the 16
# failing cells.

# The ternary family, which the cat-B partition below is asked about. Listed rather than
# derived from _OP_DOMAIN_REGISTRY: no ternary op has a registry entry, so a derivation would
# give the empty set and the totality check would pass by asking about nothing.
_SFPU_TERNARY_OPS: FrozenSet[MathOperation] = frozenset(
    {
        MathOperation.SfpuAddcmul,
        MathOperation.SfpuAddcdiv,
        MathOperation.SfpuLerp,
        MathOperation.SfpuSnakeBeta,
        MathOperation.TTNNWhere,
    }
)

TERNARY_SPECIALS_READY_OPS: FrozenSet[MathOperation] = frozenset(
    {
        # Plain SFPMAD arithmetic, which the ISA specifies as IEEE for a non-finite input:
        # inf*0 = NaN, inf+(-inf) = NaN, NaN propagates. Green on both safe cells for all
        # three operands.
        MathOperation.SfpuAddcmul,  # a + value*b*c
        MathOperation.SfpuLerp,  # a + c*(b - a); inf - inf from an ordinary interpolation
        # Enrolled on the strength of operands A and B, which agree everywhere, and of
        # c = +/-inf, where the quotient is +/-0 and the result is a. c = NaN is a genuine
        # kernel divergence -- the reciprocal returns +0 instead of propagating, exactly as
        # unary Reciprocal does -- and is xfailed per cell rather than papered over.
        MathOperation.SfpuAddcdiv,  # a + value*b/c
        MathOperation.SfpuSnakeBeta,  # a + sin(b*a)^2/c; also diverges at an infinite sin arg
        # An element-wise select, so every special is returned verbatim from whichever operand
        # was chosen and there is no arithmetic to specify. Green once WhereGolden modelled the
        # pack's NaN substitution, except a -0.0 *condition* on the unpack-to-dest path, which
        # carries the same SFPSETCC caveat that scopes Sign and Heaviside.
        MathOperation.TTNNWhere,
    }
)

# Empty by measurement, not by omission. Kept so the totality check has both halves.
_TERNARY_SPECIALS_NOT_READY: FrozenSet[MathOperation] = frozenset()

assert not (
    TERNARY_SPECIALS_READY_OPS & _TERNARY_SPECIALS_NOT_READY
), "an op cannot be both enrolled in cat B and recorded as not ready for it"
assert (
    TERNARY_SPECIALS_READY_OPS | _TERNARY_SPECIALS_NOT_READY
) == _SFPU_TERNARY_OPS, (
    "every ternary op needs a cat-B verdict, and only ternary ops have one"
)


# ─────────────────────────────────────────────────────────────────────────────
# Cat F enrolment
#
# Opt-in per op: driving exp(3.39e38) on a golden that does not model saturation gives a wall
# of failures with one root cause. An entry claims the ceiling, its neighbour and the smallest
# normal on every cell, and the subnormal on the one cell that delivers one.
#
# Tranche 1 is the ops that cannot overflow, so it says nothing about saturation. Measured on a
# Blackhole p150 in two passes: the first failed 12 of 70 cells across Ceil, Floor, Sign and
# Signbit, every failing lane the +/-2^-127 subnormal on every cell but Float32->Float32 at
# dest_acc=Yes -- one finding, not four, recorded as subnormal_delivered(). Tranche 2 made the
# class total: the 74 remaining sweep-reachable float ops, 55 agreeing and 19 in
# _EXTREMES_NOT_READY. What the 55 assert is narrower than it looks -- for an op whose domain
# is a *definedness* bound the answer at 3.39e38 is NaN on both sides.
# ─────────────────────────────────────────────────────────────────────────────

EXTREMES_READY_OPS: FrozenSet[MathOperation] = frozenset(
    {
        # Tranche 1: closed over the format, so the probe asks only whether the pipeline
        # delivered the datum.
        MathOperation.Abs,
        MathOperation.Neg,
        MathOperation.Identity,  # pass-through: measures the pipeline and no arithmetic
        MathOperation.Fill,  # read narrowly: an extreme input must not corrupt the fill
        # Sign and Signbit read the sign bit, so they are two of the four instruments that
        # measured subnormal_delivered(); green once the probe stopped being sent where the
        # datacopy flattens it.
        MathOperation.Sign,
        MathOperation.Signbit,
        # Above 2**mantissa every float is already an integer, so at the ceiling and its
        # neighbour all four must be the identity -- an assertion nothing else here makes.
        # Ceil and Floor were the other two subnormal_delivered() instruments.
        MathOperation.Floor,
        MathOperation.Ceil,
        MathOperation.Trunc,
        MathOperation.Round,
        # x against UNARY_COMP_THRESHOLD: an extreme operand is an ordinary one on the far
        # side, and 0.0/1.0 is representable everywhere. UnaryEq and UnaryNe are outside
        # _OP_DOMAIN_REGISTRY and never reach the sweep -- see the note above _OP_EDGE_POINTS.
        MathOperation.UnaryGt,
        MathOperation.UnaryLt,
        MathOperation.UnaryGe,
        MathOperation.UnaryLe,
        # Tranche 2: measured at format_extremes() on both specials-safe cells and agreed with
        # its golden at every probe.
        MathOperation.Acos,
        MathOperation.Acosh,
        MathOperation.Add1,
        MathOperation.Asin,
        MathOperation.Asinh,
        MathOperation.Atan,
        MathOperation.Atanh,
        MathOperation.CastFp32ToFp16a,
        MathOperation.Cbrt,
        MathOperation.Celu,
        MathOperation.Clamp,
        MathOperation.Elu,
        MathOperation.EqualZero,
        MathOperation.Erf,
        MathOperation.Erfc,
        MathOperation.Fmod,
        MathOperation.Frac,
        MathOperation.Gelu,
        MathOperation.GeluAppx,
        MathOperation.GeluDerivative,
        MathOperation.GeluTanh,
        MathOperation.GreaterThanEqualZero,
        MathOperation.GreaterThanZero,
        MathOperation.Hardmish,
        MathOperation.Hardshrink,
        MathOperation.Hardsigmoid,
        MathOperation.Hardtanh,
        MathOperation.Heaviside,
        MathOperation.I0,
        MathOperation.LessThanEqualZero,
        MathOperation.LessThanZero,
        MathOperation.Log1p,
        MathOperation.Lrelu,
        MathOperation.Mish,
        MathOperation.NotEqualZero,
        MathOperation.Prelu,
        MathOperation.Rdiv,
        MathOperation.ReluMax,
        MathOperation.ReluMin,
        MathOperation.Selu,
        MathOperation.Sigmoid,
        MathOperation.SigmoidAppx,
        MathOperation.Silu,
        MathOperation.Softplus,
        MathOperation.Softshrink,
        MathOperation.Sqrt,
        MathOperation.Tanh,
        MathOperation.TanhDerivative,
        MathOperation.TanhDerivativeLut,
        MathOperation.Tanhshrink,
        MathOperation.Threshold,
        MathOperation.UnaryMax,
        MathOperation.UnaryMin,
        MathOperation.UnaryPower,
        MathOperation.Xielu,
    }
)

# The 19 that diverge, in five groups. Each group says what was measured; none claims the op
# has been investigated on its own.
_EXTREMES_NOT_READY: FrozenSet[MathOperation] = frozenset(
    {
        # (1) Trig at a magnitude no range reduction reaches: sin/cos/tan return +/-inf or NaN
        # at 3.39e38 against a bounded golden. Whether the kernel should reduce that far is an
        # open contract question.
        MathOperation.Sin,
        MathOperation.Cos,
        MathOperation.Tan,
        # (2) The op's arithmetic flushes a subnormal *input* where the golden does not.
        # Measured at 2**-127: log -inf against -88.03, reciprocal +inf against 1.70e38,
        # remainder 0 against 2. The goldens model flush-to-zero on the *result* only, so
        # closing this is a change to every golden, not a probe.
        MathOperation.Log,
        MathOperation.LogWithBase,
        MathOperation.Reciprocal,
        MathOperation.ReciprocalCompat,
        MathOperation.Rsqrt,
        MathOperation.Remainder,
        # (3) The composition overflows or loses the sign at the ceiling: softsign's
        # denominator overflows so it returns 0 against +/-1; expm1_cw and erfinv return a
        # wrongly-signed infinity; rpow and sqrt_custom return +inf against 0 and NaN;
        # rsqrt_compat returns 0 and +inf against NaN.
        MathOperation.Softsign,
        MathOperation.Expm1Cw,
        MathOperation.Erfinv,
        MathOperation.Rpow,
        MathOperation.SqrtCustom,
        MathOperation.RsqrtCompat,
        # (4) Saturates a non-finite *result* to +/-1.1547668e37, the same constant I1's golden
        # comment records for a non-finite input.
        MathOperation.I1,
        # (5) The gamma family, for the reason recorded above _OP_SINGULARITIES for their
        # poles: fits valid only well inside a positive domain.
        MathOperation.Digamma,
        MathOperation.Lgamma,
        MathOperation.Polygamma,
    }
)

assert not (
    EXTREMES_READY_OPS & _EXTREMES_NOT_READY
), "an op cannot be both enrolled in cat F and recorded as not ready for it"


def _deliverable_extremes(
    range_fmt: DataFormat,
    input_format: DataFormat,
    dest_acc: Optional[Union[bool, Enum]],
) -> List[float]:
    """format_extremes(*range_fmt*) minus the probes this pipeline cannot deliver: today the
    subnormal alone. Keyed off *input_format*, not *range_fmt*, delivery being about how the
    datum reaches the LREG."""
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

    Separate from edge_values() because a sweep wants one failure class per variant. No *op*
    argument: the ceiling and the subnormal band are properties of the pipeline, and which ops
    may be *driven* at them is EXTREMES_READY_OPS' question.

    Bounded by narrowest_ceiling_format() rather than narrowest_range_format(): every probe
    here is a magnitude that has to survive both legs of the pipeline, and the two disagree
    on (Float32 -> Float16_b) -- see that function."""
    range_fmt = narrowest_ceiling_format(input_format, output_format)
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
    whole difference. There, membership is a per-op fact (GENERATED_NAN_SIGN_OPS). Here the
    caller already knows, because the binary edge sweep partitions its probe *by what the
    golden answers*: the `both_zero` and `nan_golden` classes are exactly the pairs where finite
    operands produce a NaN, so every element of them is generated by construction.

    Two conditions, same as the unary gate:
      * the pipeline narrows, so the NaN leaves as a signed infinity (not nan_survives_to_l1);
      * the arch leaves that sign unspecified, i.e. Wormhole; Blackhole promises the canonical
        NaN instead.

    Where this is True there is nothing sound to assert *yet*. The assertion to restore is "an
    infinity of either sign", which is a change to the comparator rather than to this gate.
    """
    return on_wormhole and not nan_survives_to_l1(input_format, output_format, dest_acc)


def signed_zero_pole_cells(
    formats: List["InputOutputFormat"],  # noqa: F821 - test-side type, duck-typed
    dest_accs,
) -> Tuple[Tuple[DataFormat, DataFormat, object], ...]:
    """The cells of *formats* x *dest_accs* where a -0.0 driven into a pole reaches the LREG.

    The cat-A twin of the cat-B gate above and, unlike it, takes no specials_safe(): the -0.0
    at a registered pole comes from boundary_probes(), so delivery alone decides it."""
    return tuple(
        (fmt.input_format, fmt.output_format, dest_acc)
        for fmt in formats
        for dest_acc in dest_accs
        if negative_zero_delivered(fmt.input_format, dest_acc)
    )


# Mixed-magnitude block-float blocks. Bfp8_b, Bfp4_b and Bfp2_b share one exponent per
# BLOCK_SPREAD_ELEMENTS-element block, so the stimulus that exercises the format is a
# block of one large element and fifteen small ones -- every other spec here is a
# narrow-range uniform where the shared exponent never bites.

BLOCK_SPREAD_ELEMENTS = 16

# Binary decades below the block's largest element: 4 keeps every element inside Bfp8_b's 7
# magnitude bits and is the control, 12 flushes the tail, 24 flushes most of it.
BLOCK_SPREAD_DECADES = (4, 12, 24)

# The largest element of every block. 1.0 rather than an op's domain ceiling: exact in every
# format, and it keeps the whole spread inside the domain of the ops the sweeps select.
BLOCK_SPREAD_HIGH = 1.0


def block_spread_spec(decades: int, seed: int = 0) -> StimuliSpec:
    """One element at BLOCK_SPREAD_HIGH per block, the rest log-spaced *decades* below it.

    Log-spaced, not linear: what the shared exponent does to an element depends on its distance
    from the block maximum in *binades*. *seed* only makes a multi-operand caller's specs
    distinguishable objects; the pattern is identical across them."""

    def face(size, dtype, generator):
        steps = torch.tensor(
            [0.0]
            + [
                -(decades * i) / (BLOCK_SPREAD_ELEMENTS - 1)
                for i in range(1, BLOCK_SPREAD_ELEMENTS)
            ],
            dtype=torch.float32,
        )
        block = BLOCK_SPREAD_HIGH * torch.pow(2.0, steps)
        return block.repeat(-(-size // BLOCK_SPREAD_ELEMENTS))[:size].to(dtype)

    return StimuliSpec(distribution=face, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# edge_spec — the one builder the per-family edge tests call
# ─────────────────────────────────────────────────────────────────────────────


def clip_to_format(values: List[float], fmt: DataFormat) -> List[float]:
    """Drop finite values *fmt* cannot represent; keep non-finite ones verbatim.

    Non-finite values are the *point* of a cat-B probe, so they are never clipped — the
    decision about whether they belong at all is specials_safe()'s, made before this.
    """
    limit = format_max_magnitude(fmt)
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

    Three sources, one per edge category:
      * cat A — boundary_probes(): the op's singularities, straddled.
      * cat D — op_edge_points(): knees, thresholds, exact rounding ties.
      * cat B — format_specials(), only when *specials* is True. The caller decides via
        specials_safe(input_format, output_format, dest_acc); it is off by default
        because injecting them on the wrong triple is a wall of failures with one root
        cause (see the section above).

    Cat F is not here: a format extreme is a different failure class from a pole or a special,
    and one tensor carrying both would let a saturation xfail hide an unpack one. It has its
    own entry point, extreme_values(), and its own sweeps.

    Clipped against the *narrowest* format in the pipeline, not the input format: a caller
    that passes a spec to a driver bypasses the driver's own for_op_pipeline() resolution
    entirely (eltwise_unary_sfpu only resolves when spec_A is None), so a probe near a
    format ceiling would otherwise reach a Float16 or MxFp4 output unclipped and overflow.

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

    if not range_fmt.is_integer() and not negative_zero_delivered(
        input_format, dest_acc
    ):
        # One pass over every source, not one per source: -0.0 now arrives from a zero pole
        # (cat A), the comparison-to-zero knees (cat D) and FLOAT_SPECIALS (cat B), and a
        # per-source filter is one a fourth source can be added without. On the datacopy path
        # the LREG holds +0.0, so the probe there blames the kernel for a datum it never got.
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
    **kwargs,
) -> Optional[StimuliSpec]:
    """edge_values() as a StimuliSpec, or None if *op* has no edge worth probing.

    Returns None rather than an empty spec so a caller can fall back to the op's random
    domain: many unary SFPU ops are smooth everywhere, with no knee and no pole, and for
    those an edge sweep has nothing to add beyond cat B.

    ``custom`` with ``cycle=True`` tiles the values across every face instead of writing them
    at its head with a zero tail. That tail was not free: the median edge list is four values
    against a 256-element face, so ~98% of the tensor was 0.0, and it drove an unrecorded
    out-of-domain 0.0 into Acosh, Log and Rsqrt. ``cycle=False`` is still there for a probe
    that depends on the tail.

    ``custom`` is per-face only (generate_full_tensor raises), so the pattern repeats in every
    face; ``custom_faces`` is available when faces must differ.

    Integer formats: format_specials() returns the integer extremes, but INT_MIN cannot
    be delivered through any spec — CustomStrategy clamps through _get_integer_bounds,
    which returns info.min + 1. Deliver integer extremes as a raw override tensor
    instead; this raises rather than silently clamping.
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
    pow's base-zero probe against an exponent outside its registered domain cannot happen.
    Ops with no registry entry at all keep the full counterpart spread.
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
# The coverage ratchet
#
# Per class of input value, how many ops the stimulus machinery drives that class for.
#
# Coverage loss in this suite is silent: drop an op from an enrolment table and nothing fails,
# the sweep just collects fewer variants. Measured, removing three ops from the cat-F enrolment
# took 24 device variants out of the unary sweep without one failure.
#
# Counts rather than a per-op matrix, which would be a second copy of the tables that goes
# stale, and every number is asked of the machinery that builds the stimuli. Classes delivered
# from the test modules are floored there.
# ─────────────────────────────────────────────────────────────────────────────

#: The classes this module's own tables can account for, and where each comes from.
COVERAGE_CLASSES = {
    "A": "singularities -- _OP_SINGULARITIES, straddled by boundary_probes()",
    "B": "ieee_specials -- the three *_SPECIALS_READY_OPS gates",
    "D": "knees -- _OP_EDGE_POINTS / _OP_OPERAND_EDGE_POINTS",
    "F": "magnitude_extremes -- EXTREMES_READY_OPS",
    "G": "signed_zero_at_a_pole -- a -0.0 boundary_probes() emits at a registered pole",
}


def _emits_negative_zero_at_a_pole(op: MathOperation) -> bool:
    """Does boundary_probes() actually emit a -0.0 for *op*, on any operand?

    Asked of the probe builder, not of _OP_SINGULARITIES: reading the table would report
    coverage from the moment a pole was *registered*, true for years before anything drove it.
    """
    return any(
        _is_negative_zero(value)
        for operand in Operand
        for value in boundary_probes(op, operand, DataFormat.Float32)
    )


def coverage_counts() -> Dict[str, int]:
    """Per class in COVERAGE_CLASSES, the ops this module drives that class of value for."""
    singular = ops_with_singularity()
    return {
        "A": len(singular),
        "B": len(
            SPECIALS_READY_OPS | BINARY_SPECIALS_READY_OPS | TERNARY_SPECIALS_READY_OPS
        ),
        "D": len(set(_OP_EDGE_POINTS) | set(_OP_OPERAND_EDGE_POINTS)),
        "F": len(EXTREMES_READY_OPS),
        "G": sum(1 for op in singular if _emits_negative_zero_at_a_pole(op)),
    }


if __name__ == "__main__":  # pragma: no cover - developer entry point
    _counts = coverage_counts()
    for _cls, _what in COVERAGE_CLASSES.items():
        print(f"  {_cls} {_counts[_cls]:>3} ops   {_what}")

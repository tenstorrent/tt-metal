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
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

from .format_config import DataFormat
from .llk_params import MathOperation
from .stimuli_generator_v2 import DistributionKind, StimuliSpec

# ─────────────────────────────────────────────────────────────────────────────
# OperandSpecs
# ─────────────────────────────────────────────────────────────────────────────


class Operand(str, Enum):
    """Identifies which operand of an OperandSpecs a value refers to."""

    A = "spec_A"
    B = "spec_B"


@dataclass
class OperandSpecs:
    """Per-operand input domain specs returned by for_op.

    For binary ops where operands need different domains (e.g. divisor avoids
    zero), spec_A and spec_B differ; unary ops need only spec_A.
    spec_B defaults to a copy of spec_A when "None".
    """

    spec_A: StimuliSpec
    spec_B: Optional[StimuliSpec] = None

    def __post_init__(self) -> None:
        if self.spec_B is None:
            self.spec_B = copy.deepcopy(self.spec_A)


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific domain builders
# ─────────────────────────────────────────────────────────────────────────────


def _exp_spec(fmt: DataFormat) -> OperandSpecs:
    """Safe input range for exp(x) per format to avoid overflow."""
    if fmt == DataFormat.MxFp8P:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    elif fmt in (DataFormat.Float16, DataFormat.MxFp8R):
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    else:
        # the lower bound is intentionally pushed to -100.0 so we cross the SFPU's negative-side
        # sanitization boundary near x ≈ -88.5 (where InputClamping::ClampToNegative saturates inputs
        # in the fast/approx exp path).
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-100.0, high=80.0)
    return OperandSpecs(spec_A=spec)


def _exp2_spec(fmt: DataFormat) -> OperandSpecs:
    """Safe input range for exp2(x) = 2^x per format to avoid overflow."""
    if fmt == DataFormat.MxFp8P:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-7.0, high=7.0)
    elif fmt in (DataFormat.Float16, DataFormat.MxFp8R):
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-14.0, high=14.0)
    else:
        spec = StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-100.0, high=100.0
        )
    return OperandSpecs(spec_A=spec)


def _square_spec(fmt: DataFormat) -> OperandSpecs:
    """Safe input range for square(x) = x^2 per format to avoid overflow."""
    if fmt == DataFormat.MxFp8P:
        spec = StimuliSpec(distribution=DistributionKind.UNIFORM, low=-20.0, high=20.0)
    elif fmt in (DataFormat.Float16, DataFormat.MxFp8R):
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
    # atanh: domain |x| < 1; stay away from ±1 to avoid ±inf
    MathOperation.Atanh: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-0.95, high=0.95)
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
    # exp: format-specific overflow threshold
    MathOperation.Exp: _exp_spec,
    # exp2: format-specific overflow threshold
    MathOperation.Exp2: _exp2_spec,
    # fill: the hardware ignores the input value; any range is fine
    MathOperation.Fill: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=1.0)
    ),
    # gelu: Gaussian activation — gaussian distribution naturally exercises tails
    MathOperation.Gelu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.GAUSSIAN, mean=0.0, std=3.0)
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
    # log1p: domain x > -1; log1p(x) = log(1 + x)
    MathOperation.Log1p: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-0.99, high=10.0)
    ),
    # neg: all reals
    MathOperation.Neg: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-10.0, high=10.0)
    ),
    # reciprocal: domain x != 0; avoid a small band around 0 and cover both signs
    MathOperation.Reciprocal: OperandSpecs(
        spec_A=StimuliSpec.uniform(intervals=[(-100.0, -0.1), (0.1, 100.0)])
    ),
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
    MathOperation.Threshold: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # rsqrt: domain x > 0; log-uniform covers a wide positive range
    MathOperation.Rsqrt: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.LOG_UNIFORM, low=1e-4, high=100.0
        )
    ),
    # sigmoid: cover both saturation regions
    MathOperation.Sigmoid: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-8.0, high=8.0)
    ),
    # silu: silu(x) = x * sigmoid(x); cover saturation + linear regions
    MathOperation.Silu: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0)
    ),
    # sin: cover the full unit circle
    MathOperation.Sin: OperandSpecs(
        spec_A=StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-math.pi, high=math.pi
        )
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
    # pow: srcA is the base (must be non-negative for non-integer exponents);
    # srcB is the exponent (non-negative to keep output finite)
    MathOperation.SfpuElwpow: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=5.0),
        spec_B=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=5.0),
    ),
    # xlogy: computes x * log(y) element-wise
    # srcA (x): x >= 0 so xlogy(0, y) = 0 is well-defined
    # srcB (y): y > 0 so log(y) is finite; log-uniform spans several decades
    MathOperation.SfpuXlogy: OperandSpecs(
        spec_A=StimuliSpec(distribution=DistributionKind.UNIFORM, low=0.0, high=10.0),
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


def for_op(
    op: MathOperation,
    data_format: DataFormat = DataFormat.Float16_b,
) -> OperandSpecs:
    """Return OperandSpecs with safe input domains for *op* and *data_format*.

    Args:
        op: Target math operation.
        data_format: Input data format; controls the numeric range and
            precision used to choose safe per-op input domains (e.g. tighter
            ranges for narrower MX/BFP formats).

    Returns:
        OperandSpecs with per-operand domain specs.

    Raises:
        KeyError: If *op* is not in the registry.
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
        return copy.deepcopy(entry(data_format))
    return copy.deepcopy(entry)


# ─────────────────────────────────────────────────────────────────────────────
# Undefined-region subtraction
# ─────────────────────────────────────────────────────────────────────────────

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

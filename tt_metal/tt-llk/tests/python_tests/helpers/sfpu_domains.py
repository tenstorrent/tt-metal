# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where the SFPU drivers draw their inputs from, and which values they hit on purpose.

`for_op`/`for_op_pipeline` give every registered op's safe per-operand domain, which
`exclude_undefined` clips to where the op is defined; `edge_values`/`edge_spec`/
`edge_pair_values` list the singularities, knees and IEEE specials worth probing; and the
gates (`specials_safe`, `negative_zero_delivered`, `nan_survives_to_l1`) say where a
special or a signed zero survives the pipeline at all.
"""

import copy
import math
import struct
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

from .data_format_inference import _dest_acc_flag, _two_state_flag, unpacks_to_dest
from .format_config import MX_FORMAT_MAX_NORMAL, DataFormat
from .llk_params import ApproximationMode, DestAccumulation, MathOperation
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
from .stimuli_generator import StimuliSpec


class Operand(Enum):
    """Which operand of an OperandSpecs a value refers to."""

    A = auto()
    B = auto()
    C = auto()


@dataclass
class OperandSpecs:
    """Per-operand input domains. spec_B defaults to a copy of spec_A and spec_C to a
    copy of spec_B, so a unary entry resolves to three identical specs; a ternary op's
    divisor (addcdiv, snake_beta) is the operand C that carries the pole."""

    spec_A: StimuliSpec
    spec_B: StimuliSpec | None = None
    spec_C: StimuliSpec | None = None

    def __post_init__(self) -> None:
        if self.spec_B is None:
            self.spec_B = copy.deepcopy(self.spec_A)
        if self.spec_C is None:
            self.spec_C = copy.deepcopy(self.spec_B)

    def spec_for(self, operand: Operand) -> StimuliSpec:
        return {Operand.A: self.spec_A, Operand.B: self.spec_B, Operand.C: self.spec_C}[
            operand
        ]


# Largest finite magnitude of the formats with a narrower exponent field than bfloat16;
# every other format shares bfloat16's ceiling and is never the binding constraint.
_FORMAT_MAX_MAGNITUDE: dict[DataFormat, float] = {
    **MX_FORMAT_MAX_NORMAL,  # MxFp4 6, MxFp8P 448, MxFp8R 57344
    DataFormat.Float16: 65504.0,
    DataFormat.Fp8_e4m3: 448.0,
}
_BF16_MAX_MAGNITUDE = 3.3895314e38


def narrowest_range_format(
    input_format: DataFormat, output_format: DataFormat | None = None
) -> DataFormat:
    """Whichever of the two holds the smaller magnitude: a safe domain is bounded by the
    narrowest format anywhere in the pipeline (exp over (-100, 80) is fine into Float32
    and saturates Float16). A tie keeps the input."""
    if output_format is None:
        return input_format
    return min(
        (input_format, output_format),
        key=lambda fmt: _FORMAT_MAX_MAGNITUDE.get(fmt, _BF16_MAX_MAGNITUDE),
    )


# ── The domain registry ───────────────────────────────────────────────────────

# e4m3 (a ceiling of 448, the narrowest any entry narrows for); e5m2 and Float16 (57344
# and 65504: one tier); the block floats, whose shared exponent costs precision.
_E4M3_FORMATS = (DataFormat.MxFp8P, DataFormat.Fp8_e4m3)
_E5M2_AND_FLOAT16 = (DataFormat.Float16, DataFormat.MxFp8R)
_BLOCK_FLOAT_FORMATS = (DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Bfp2_b)


def _u(low: float, high: float) -> OperandSpecs:
    return OperandSpecs(StimuliSpec.uniform(low, high))


def _g(mean: float, std: float, **bounds: float) -> OperandSpecs:
    return OperandSpecs(StimuliSpec.gaussian(mean, std, **bounds))


def _lu(low: float, high: float) -> OperandSpecs:
    return OperandSpecs(StimuliSpec.log_uniform(low, high))


def _tiered(
    e4m3: tuple[float, float], e5m2: tuple[float, float], wide: tuple[float, float]
) -> Callable[[DataFormat], OperandSpecs]:
    """A format-sensitive entry: one uniform range per exponent tier."""

    def spec(fmt: DataFormat) -> OperandSpecs:
        if fmt in _E4M3_FORMATS:
            return _u(*e4m3)
        if fmt in _E5M2_AND_FLOAT16:
            return _u(*e5m2)
        return _u(*wide)

    return spec


def _reciprocal_spec(fmt: DataFormat) -> OperandSpecs:
    """1/x needs a tighter ratio on a block-float input: a 1000:1 spread inside a
    16-element block quantizes the smallest elements to zero (10:1 for Bfp8_b's 7 bits,
    4:1 for Bfp4_b's 3)."""
    if fmt == DataFormat.Bfp4_b:
        bands = [(-100.0, -25.0), (25.0, 100.0)]
    elif fmt in _BLOCK_FLOAT_FORMATS:
        bands = [(-100.0, -10.0), (10.0, 100.0)]
    else:
        bands = [(-100.0, -0.1), (0.1, 100.0)]
    return OperandSpecs(StimuliSpec.uniform(intervals=bands))


# Every op's safe input domain, as a spec or a per-format builder. A unary entry names
# spec_A only; a binary one with per-operand domains names spec_B too.
_OP_DOMAIN_REGISTRY: dict[
    MathOperation, OperandSpecs | Callable[[DataFormat], OperandSpecs]
] = {
    # ── SFPU unary ────────────────────────────────────────────────────────────
    MathOperation.Abs: _u(-10.0, 10.0),
    MathOperation.Acosh: _u(1.0, 10.0),
    MathOperation.Asinh: _u(-10.0, 10.0),
    MathOperation.Atanh: _u(-0.999, 0.999),  # |x| < 1; just inside the +-inf endpoints
    MathOperation.Celu: _u(-5.0, 5.0),
    MathOperation.Cos: _u(-math.pi, math.pi),
    MathOperation.Elu: _u(-5.0, 5.0),
    MathOperation.Erfinv: _u(-0.99, 0.99),
    MathOperation.Heaviside: _u(-5.0, 5.0),
    # exp: -100 crosses the kernel's negative clamp near -88.5; 80 stays under the fp32
    # overflow at 88.7. exp2 is 2**x, not range-bound on the positive side; exp_with_base
    # computes exp(0.5 x), so its positive side is double exp's.
    MathOperation.Exp: _tiered((-5.0, 5.0), (-10.0, 10.0), (-100.0, 80.0)),
    MathOperation.Exp2: _tiered((-7.0, 7.0), (-14.0, 14.0), (-100.0, 100.0)),
    MathOperation.ExpWithBase: _tiered((-5.0, 5.0), (-10.0, 10.0), (-100.0, 160.0)),
    MathOperation.Fill: _u(0.0, 1.0),  # the input is ignored
    # gelu family: most inputs near 0, some in the saturating tails.
    MathOperation.Gelu: _g(0.0, 3.0, low=-5.0, high=5.0),
    MathOperation.GeluAppx: _g(0.0, 3.0, low=-5.0, high=5.0),
    MathOperation.GeluTanh: _g(0.0, 3.0),
    MathOperation.GeluDerivative: _g(0.0, 3.0, low=-5.0, high=5.0),
    MathOperation.Hardsigmoid: _u(-4.0, 4.0),  # linear on [-3, 3], clipped outside
    MathOperation.Log: _lu(1e-4, 1e3),
    MathOperation.LogWithBase: _lu(1e-4, 1e3),
    MathOperation.Log1p: _u(-0.99, 10.0),  # defined for x > -1
    MathOperation.Neg: _u(-10.0, 10.0),
    MathOperation.Reciprocal: _reciprocal_spec,
    # relu_max clamps above at its threshold with a strict compare, and relu_min below at
    # its own, so each bound has to clear the threshold or every input clamps.
    MathOperation.ReluMax: _u(-5.0, 2.0 * RELU_MAX_THRESHOLD),
    MathOperation.ReluMin: _u(-5.0, 2.0 * RELU_MIN_THRESHOLD),
    MathOperation.Lrelu: _u(-5.0, 5.0),
    MathOperation.Threshold: _u(-5.0, 5.0),
    MathOperation.Rsqrt: _lu(1e-4, 100.0),
    MathOperation.Expm1Cw: _u(-5.0, 5.0),
    MathOperation.Sigmoid: _u(-8.0, 8.0),
    MathOperation.Silu: _u(-5.0, 5.0),
    MathOperation.Softshrink: _u(-5.0, 5.0),
    MathOperation.Softsign: _u(-5.0, 5.0),
    MathOperation.Mish: _u(-5.0, 5.0),
    MathOperation.Selu: _u(-5.0, 5.0),
    # i0 / i1: the kernels' polynomial fits hold on |x| <= 3.75.
    MathOperation.I0: _u(-3.75, 3.75),
    MathOperation.I1: _u(-3.75, 3.75),
    MathOperation.Erf: _u(-3.0, 3.0),
    MathOperation.Erfc: _u(-3.0, 3.0),
    MathOperation.Expm1: _u(-5.0, 5.0),
    MathOperation.Cbrt: _u(-27.0, 27.0),
    MathOperation.Sign: _u(-2.0, 2.0),
    MathOperation.Signbit: _u(-2.0, 2.0),
    MathOperation.TanhDerivative: _u(-5.0, 5.0),
    # The LUT tanh cancels catastrophically in 1 - tanh^2 past |x| ~ 3.4.
    MathOperation.TanhDerivativeLut: _u(-3.0, 3.0),
    MathOperation.Hardmish: _u(-4.0, 4.0),  # piecewise on [-2, 0]
    MathOperation.Lgamma: _u(1.0, 15.0),  # Stirling is accurate for x >= ~0.5
    MathOperation.Digamma: _u(0.1, 50.0),  # LUT fit on [0.01, 102]
    MathOperation.Identity: _u(-10.0, 10.0),
    MathOperation.Prelu: _u(-5.0, 5.0),
    MathOperation.Rpow: _u(-4.0, 4.0),  # 2**x
    MathOperation.UnaryPower: _u(-4.0, 4.0),  # x**2
    MathOperation.Fmod: _u(-5.0, 5.0),  # divisor fixed to 2.0
    MathOperation.Remainder: _u(-5.0, 5.0),
    # unary comparisons against 0.5, max/min against 0.0: span the threshold.
    MathOperation.UnaryGt: _u(-2.0, 2.0),
    MathOperation.UnaryLt: _u(-2.0, 2.0),
    MathOperation.UnaryGe: _u(-2.0, 2.0),
    MathOperation.UnaryLe: _u(-2.0, 2.0),
    MathOperation.UnaryMax: _u(-5.0, 5.0),
    MathOperation.UnaryMin: _u(-5.0, 5.0),
    MathOperation.Polygamma: _u(0.5, 10.0),  # poles at x <= 0
    MathOperation.Xielu: _u(-5.0, 5.0),
    MathOperation.Hardshrink: _u(-4.0, 4.0),
    MathOperation.Softplus: _u(-5.0, 30.0),  # past the linear threshold at 20
    MathOperation.SigmoidAppx: _u(-5.0, 5.0),
    MathOperation.SqrtCustom: _u(0.0, 100.0),
    MathOperation.Add1: _u(-10.0, 10.0),
    MathOperation.CastFp32ToFp16a: _u(-100000.0, 100000.0),
    MathOperation.EqualZero: _u(-2.0, 2.0),
    MathOperation.NotEqualZero: _u(-2.0, 2.0),
    MathOperation.LessThanZero: _u(-2.0, 2.0),
    MathOperation.GreaterThanZero: _u(-2.0, 2.0),
    MathOperation.LessThanEqualZero: _u(-2.0, 2.0),
    MathOperation.GreaterThanEqualZero: _u(-2.0, 2.0),
    MathOperation.Rdiv: _u(1.0, 8.0),  # value / x, away from the pole
    MathOperation.Clamp: _u(-2.0, 2.0),  # bounds fixed to [-1, 1]
    MathOperation.Hardtanh: _u(-2.0, 2.0),
    MathOperation.Sin: _u(-math.pi, math.pi),
    MathOperation.Tan: _u(-1.3, 1.3),  # inside the poles at +-pi/2
    MathOperation.Atan: _u(-10.0, 10.0),
    MathOperation.Asin: _u(-0.99, 0.99),
    MathOperation.Acos: _u(-0.99, 0.99),
    MathOperation.Sinh: _u(-5.0, 5.0),
    MathOperation.Cosh: _u(-5.0, 5.0),
    # round / floor / ceil / trunc / frac: both signs, so the four differ at all.
    MathOperation.Round: _u(-10.0, 10.0),
    MathOperation.Floor: _u(-10.0, 10.0),
    MathOperation.Ceil: _u(-10.0, 10.0),
    MathOperation.Trunc: _u(-10.0, 10.0),
    MathOperation.Frac: _u(-10.0, 10.0),
    MathOperation.Sqrt: _u(0.0, 100.0),
    MathOperation.Square: _tiered((-20.0, 20.0), (-200.0, 200.0), (-1000.0, 1000.0)),
    MathOperation.Tanh: _u(-5.0, 5.0),
    MathOperation.Tanhshrink: _u(-5.0, 5.0),
    MathOperation.TopKLocalSort: _u(-10.0, 10.0),
    MathOperation.TopKMerge: _u(-10.0, 10.0),
    MathOperation.TopKRebuild: _u(-10.0, 10.0),
    MathOperation.TopKDefuse: _u(-10.0, 10.0),
    # ── FPU binary ────────────────────────────────────────────────────────────
    MathOperation.Elwadd: _u(-1.0, 1.0),
    MathOperation.Elwmul: _u(-1.0, 1.0),
    MathOperation.Elwsub: _u(-1.0, 1.0),
    # ── SFPU binary ───────────────────────────────────────────────────────────
    MathOperation.SfpuElwadd: _u(-1.0, 1.0),
    MathOperation.SfpuElwmul: _u(-1.0, 1.0),
    MathOperation.SfpuElwsub: _u(-1.0, 1.0),
    # div: the divisor (B) spans both signs and stays off 0.
    MathOperation.SfpuElwdiv: OperandSpecs(
        StimuliSpec.uniform(-2.0, 2.0),
        StimuliSpec.uniform(intervals=[(-10.0, -0.1), (0.1, 10.0)]),
    ),
    MathOperation.SfpuElwrsub: _u(-1.0, 1.0),
    # pow: a**b as exp(b ln a), with a relative error roughly flat in the operands; the
    # bounds pair with the rtol SfpuElwpow declares. a <= 16 would make it an overflow test.
    MathOperation.SfpuElwpow: OperandSpecs(
        StimuliSpec.uniform(0.0, 8.0), StimuliSpec.uniform(0.0, 4.0)
    ),
    # xlogy: x * log(y); the error grows with x against a fixed atol, so x's ceiling is
    # an absolute-accuracy bound, and y keeps its log-uniform span.
    MathOperation.SfpuXlogy: OperandSpecs(
        StimuliSpec.uniform(0.0, 8.0), StimuliSpec.log_uniform(1e-4, 10.0)
    ),
    # logaddexp: finite for any finite pair, so the draw deliberately crosses the exp()
    # overflow (88.7; 127 for logaddexp2) where the naive composition returns inf.
    MathOperation.SfpuLogaddexp: OperandSpecs(
        StimuliSpec.uniform(-200.0, 200.0), StimuliSpec.uniform(-200.0, 200.0)
    ),
    MathOperation.SfpuLogaddexp2: OperandSpecs(
        StimuliSpec.uniform(-200.0, 200.0), StimuliSpec.uniform(-200.0, 200.0)
    ),
    MathOperation.SfpuAddTopRow: _u(-1.0, 1.0),
    # shifts operate on bit patterns; both operands in [0, 255].
    MathOperation.SfpuElwLeftShift: _u(0.0, 255.0),
    MathOperation.SfpuElwLogicalRightShift: _u(0.0, 255.0),
    MathOperation.SfpuElwRightShift: _u(0.0, 255.0),
    # ── Reduce ────────────────────────────────────────────────────────────────
    MathOperation.ReduceColumn: _u(-1.0, 1.0),
    MathOperation.ReduceRow: _u(-1.0, 1.0),
    MathOperation.ReduceScalar: _u(-1.0, 1.0),
}

# The approximating exp family's accuracy ceiling, applied on top of the range bound at
# ApproximationMode.Yes: exp overshoots past 16; exp2 is exp(x ln 2), so 16 / ln 2 ~ 23;
# exp_with_base computes exp(0.5 x), so double.
_APPROX_ACCURACY_MAX: dict[MathOperation, float] = {
    MathOperation.Exp: 16.0,
    MathOperation.Exp2: 23.0,
    MathOperation.ExpWithBase: 32.0,
}


def _domain_of(spec: StimuliSpec) -> tuple:
    """Which values *spec* draws, not how (`==` would also compare seed and faces)."""
    return (spec.low, spec.high, tuple(spec.intervals or ()))


def _apply_approx_ceiling(
    result: OperandSpecs, op: MathOperation, approx_mode: ApproximationMode | None
) -> None:
    """Lower every operand's upper bound to *op*'s approximation ceiling, in place, at
    ApproximationMode.Yes; a format branch already tighter is left alone."""
    if approx_mode is None or not _two_state_flag(
        approx_mode, "approx_mode", "ApproximationMode"
    ):
        return
    ceiling = _APPROX_ACCURACY_MAX.get(op)
    if ceiling is None:
        return
    specs = [s for s in (result.spec_A, result.spec_B, result.spec_C) if s is not None]
    # The table is written for the unary exp family: one domain, no intervals.
    assert all(_domain_of(s) == _domain_of(result.spec_A) for s in specs), op.name
    for spec in specs:
        assert not spec.intervals, op.name
        if spec.high is not None and spec.high > ceiling:
            spec.high = ceiling


def for_op(
    op: MathOperation,
    data_format: DataFormat = DataFormat.Float16_b,
    approx_mode: ApproximationMode | None = None,
) -> OperandSpecs:
    """*op*'s safe input domains for *data_format*, a fresh copy. At
    ApproximationMode.Yes the exp family is narrowed to its approximation's accuracy
    ceiling; None narrows nothing, for a harness that measures error rather than
    asserting a tolerance. KeyError for an op with no entry."""
    entry = _OP_DOMAIN_REGISTRY.get(op)
    if entry is None:
        raise KeyError(
            f"MathOperation.{op.name} has no entry in sfpu_domains._OP_DOMAIN_REGISTRY"
        )
    result = copy.deepcopy(entry(data_format) if callable(entry) else entry)
    _apply_approx_ceiling(result, op, approx_mode)
    return result


def _spec_span(spec: StimuliSpec) -> float:
    if spec.intervals:
        return sum(high - low for low, high in spec.intervals)
    return spec.high - spec.low


def _tighter_spec(a: StimuliSpec, b: StimuliSpec) -> StimuliSpec:
    """Whichever draws from the smaller domain; a tie keeps *a*."""
    if a == b:
        return a
    return a if _spec_span(a) <= _spec_span(b) else b


def for_op_pipeline(
    op: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat | None = None,
    approx_mode: ApproximationMode | None = None,
) -> OperandSpecs:
    """Domains safe over the whole input->output pipeline. Range is bounded by the
    narrowest format anywhere in it and precision by the input alone (a block-float
    input has spent its precision before the op runs), so resolve against both formats
    and keep the tighter spec per operand; both constraints only ever narrow."""
    by_input = for_op(op, input_format, approx_mode)
    range_format = narrowest_range_format(input_format, output_format)
    if range_format == input_format:
        return by_input
    by_range = for_op(op, range_format, approx_mode)
    return OperandSpecs(
        spec_A=_tighter_spec(by_input.spec_A, by_range.spec_A),
        spec_B=_tighter_spec(by_input.spec_B, by_range.spec_B),
        spec_C=_tighter_spec(by_input.spec_C, by_range.spec_C),
    )


# ── Which registered ops are unary SFPU ops ───────────────────────────────────

# The registered binary SFPU ops (test_eltwise_binary_sfpu.py also drives int, comparison
# and bitwise ops with no domain entry, declared in its own _UNREGISTERED_BINARY_OPS).
_SFPU_BINARY_OPS: frozenset[MathOperation] = frozenset(
    {
        MathOperation.SfpuAddTopRow,
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwdiv,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuElwrsub,
        MathOperation.SfpuXlogy,
        MathOperation.SfpuLogaddexp,
        MathOperation.SfpuLogaddexp2,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLogicalRightShift,
    }
)

# Everything registered that has no unary SFPU kernel: the binary SFPU ops, the FPU
# eltwise ops and the reduce family. A newly registered op lands in the unary sweep
# unless it is classified here, which is the safe direction.
_NON_SFPU_UNARY_OPS: frozenset[MathOperation] = _SFPU_BINARY_OPS | frozenset(
    {
        MathOperation.Elwadd,
        MathOperation.Elwmul,
        MathOperation.Elwsub,
        MathOperation.ReduceColumn,
        MathOperation.ReduceRow,
        MathOperation.ReduceScalar,
    }
)

# Registered unary ops the correctness sweeps leave out: the topk halves are perf-only,
# and whole-op topk is test_topk.py's.
_UNARY_OPS_NOT_SWEPT: frozenset[MathOperation] = frozenset(
    {
        MathOperation.TopKLocalSort,
        MathOperation.TopKMerge,
        MathOperation.TopKRebuild,
        MathOperation.TopKDefuse,
    }
)


def sfpu_unary_ops() -> frozenset[MathOperation]:
    """Every registered op with a unary SFPU kernel."""
    return frozenset(_OP_DOMAIN_REGISTRY) - _NON_SFPU_UNARY_OPS


def swept_unary_ops() -> frozenset[MathOperation]:
    """The unary ops the correctness sweeps drive."""
    return sfpu_unary_ops() - _UNARY_OPS_NOT_SWEPT


# ── Singularities, and the guard bands a random draw keeps off them ───────────


class SingularitySide(Enum):
    """Which side of a singular point the op is defined on, so a probe one ULP away is
    an edge case rather than a value the kernel never promised."""

    BOTH = auto()
    ABOVE = auto()
    BELOW = auto()


_BOTH, _ABOVE, _BELOW = SingularitySide

# Ops with a pole at every non-positive integer; `ulp_sweep.claimed_lanes` excludes the
# lattice, and every fp32 value of magnitude 2**23 or more is an integer.
_NONPOSITIVE_INTEGER_POLES = frozenset(
    {MathOperation.Digamma, MathOperation.Lgamma, MathOperation.Polygamma}
)

# Each op's exact singular points per operand, and the side the op is defined on.
# Lgamma, Digamma and Polygamma are held back: their kernels claim accuracy only inside a
# positive domain, so a probe at their poles tests nothing they promised.
_OP_SINGULARITIES: dict[
    MathOperation, dict[Operand, tuple[tuple[float, SingularitySide], ...]]
] = {
    MathOperation.Reciprocal: {Operand.A: ((0.0, _BOTH),)},
    MathOperation.Rdiv: {Operand.A: ((0.0, _BOTH),)},
    MathOperation.Log: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.LogWithBase: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.Log1p: {Operand.A: ((-1.0, _ABOVE),)},
    MathOperation.Sqrt: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.SqrtCustom: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.Rsqrt: {Operand.A: ((0.0, _ABOVE),)},
    # Inverse functions defined on (-1, 1) or [-1, 1]: -1 is probed upward, +1 downward.
    MathOperation.Atanh: {Operand.A: ((-1.0, _ABOVE), (1.0, _BELOW))},
    MathOperation.Erfinv: {Operand.A: ((-1.0, _ABOVE), (1.0, _BELOW))},
    MathOperation.Asin: {Operand.A: ((-1.0, _ABOVE), (1.0, _BELOW))},
    MathOperation.Acos: {Operand.A: ((-1.0, _ABOVE), (1.0, _BELOW))},
    MathOperation.Acosh: {Operand.A: ((1.0, _ABOVE),)},
    # Binary: the singularity sits on one operand. fmod and remainder divide by B.
    MathOperation.SfpuElwdiv: {Operand.B: ((0.0, _BOTH),)},
    MathOperation.SfpuXlogy: {Operand.B: ((0.0, _ABOVE),)},
    MathOperation.SfpuElwpow: {Operand.A: ((0.0, _ABOVE),)},
    MathOperation.SfpuBinaryFmod: {Operand.B: ((0.0, _BOTH),)},
    MathOperation.SfpuBinaryRemainder: {Operand.B: ((0.0, _BOTH),)},
    # atan2(y, x): x = 0 is a branch point (+-pi/2 by the sign of y, +-pi past it).
    MathOperation.SfpuAtan2: {Operand.B: ((0.0, _BOTH),)},
    # Ternary: a + value * b / c and a + sin(b a)^2 / c, so c = 0 is the pole.
    MathOperation.SfpuAddcdiv: {Operand.C: ((0.0, _BOTH),)},
    MathOperation.SfpuSnakeBeta: {Operand.C: ((0.0, _BOTH),)},
}

# How far off its singularities an op's random draw stays (`exclude_undefined`), for the
# ops whose registered domain reaches one: 1e-6 on the defined side, or nothing for sqrt
# and acosh, whose domains start on the point. Adding an entry re-draws that op's whole
# stimulus set (an interval spec consumes two random draws per element where a plain one
# consumes one), so an op absent here keeps its draw even where a singularity exists.
_GUARD_BAND: dict[tuple[MathOperation, Operand], float] = {
    (MathOperation.Reciprocal, Operand.A): 1e-6,
    (MathOperation.Log, Operand.A): 1e-6,
    (MathOperation.Sqrt, Operand.A): 0.0,
    (MathOperation.Atanh, Operand.A): 1e-6,
    (MathOperation.Log1p, Operand.A): 1e-6,
    (MathOperation.Rsqrt, Operand.A): 1e-6,
    (MathOperation.Acosh, Operand.A): 0.0,
    (MathOperation.Erfinv, Operand.A): 1e-6,
    (MathOperation.SfpuElwdiv, Operand.B): 1e-6,
    (MathOperation.SfpuXlogy, Operand.B): 1e-6,
    (MathOperation.SfpuElwpow, Operand.A): 1e-6,
}


def _hole(point: float, side: SingularitySide, band: float) -> tuple[float, float]:
    if side is _BOTH:
        return (point - band, point + band)
    if side is _ABOVE:
        return (-math.inf, point + band)
    return (point - band, math.inf)


def _undefined_ranges() -> (
    dict[MathOperation, dict[Operand, list[tuple[float, float]]]]
):
    ranges: dict[MathOperation, dict[Operand, list[tuple[float, float]]]] = {}
    for (op, operand), band in _GUARD_BAND.items():
        ranges.setdefault(op, {})[operand] = [
            _hole(point, side, band) for point, side in _OP_SINGULARITIES[op][operand]
        ]
    return ranges


# Per op and operand, the intervals its random draw must not enter.
_SFPU_UNDEFINED_RANGES = _undefined_ranges()


def ops_with_singularity() -> frozenset[MathOperation]:
    return frozenset(_OP_SINGULARITIES)


def _subtract_intervals(
    base: list[tuple[float, float]], holes: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """*base* with every hole cut out, as sorted non-overlapping intervals."""
    result: list[tuple[float, float]] = []
    for lo, hi in base:
        current = [(lo, hi)]
        for h_lo, h_hi in holes:
            next_segments: list[tuple[float, float]] = []
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
    spec: StimuliSpec, holes: list[tuple[float, float]]
) -> StimuliSpec:
    """A copy of *spec* with *holes* cut out of its domain (its intervals, or
    [low, high]); refused when nothing remains."""
    new_spec = copy.deepcopy(spec)
    base = new_spec.intervals or [(new_spec.low, new_spec.high)]
    defined = _subtract_intervals(base, holes)
    if not defined:
        raise ValueError(f"nothing left of {base} after removing {holes}")
    new_spec.intervals = defined
    return new_spec


def exclude_undefined(
    op: MathOperation, spec: StimuliSpec, operand: Operand = Operand.A
) -> StimuliSpec:
    """*spec* clipped to where *op* is defined for *operand*; unchanged where the op has
    no registered hole there."""
    undefined = _SFPU_UNDEFINED_RANGES.get(op, {}).get(operand)
    if not undefined:
        return spec
    return exclude_intervals(spec, undefined)


def exclude_undefined_pair(op: MathOperation, specs: OperandSpecs) -> OperandSpecs:
    """`exclude_undefined` over every operand, on a deep copy."""
    op_ranges = _SFPU_UNDEFINED_RANGES.get(op, {})
    new = copy.deepcopy(specs)
    if Operand.A in op_ranges:
        new.spec_A = exclude_intervals(new.spec_A, op_ranges[Operand.A])
    if Operand.B in op_ranges and new.spec_B is not None:
        new.spec_B = exclude_intervals(new.spec_B, op_ranges[Operand.B])
    if Operand.C in op_ranges and new.spec_C is not None:
        new.spec_C = exclude_intervals(new.spec_C, op_ranges[Operand.C])
    return new


# ── Probe spacing ─────────────────────────────────────────────────────────────

# Mantissa bits after the implicit one; for the block floats, the per-element width.
_FORMAT_MANTISSA_BITS: dict[DataFormat, int] = {
    DataFormat.Float32: 23,
    DataFormat.Tf32: 10,
    DataFormat.Float16: 10,
    DataFormat.Float16_b: 7,
    DataFormat.Bfp8: 7,
    DataFormat.Bfp8_b: 7,
    DataFormat.Bfp4_b: 3,
    DataFormat.Bfp2_b: 1,
    DataFormat.MxFp8R: 2,
    DataFormat.MxFp8P: 3,
    DataFormat.Fp8_e4m3: 3,
    DataFormat.MxFp4: 1,
}
# bfloat16's width for anything unlisted: the coarsest of the common float formats.
_DEFAULT_MANTISSA_BITS = 7
_FLOAT32_MANTISSA_BITS = _FORMAT_MANTISSA_BITS[DataFormat.Float32]

# How many ULPs off a boundary a probe sits.
_PROBE_ULPS = 2


def dest_truncation_mask(dst_format: DataFormat) -> int:
    """The bits a 32-bit datum keeps when it lands in a 16-bit *dst_format* Dest."""
    dropped = _FLOAT32_MANTISSA_BITS - _FORMAT_MANTISSA_BITS[dst_format]
    return (0xFFFFFFFF << dropped) & 0xFFFFFFFF


def format_ulp(fmt: DataFormat, magnitude: float = 1.0) -> float:
    """The distance to the next representable value of *fmt* near |*magnitude*| (the
    step at 1.0 for zero and non-finites). A lower bound for a block float, whose real
    step is the block's."""
    bits = _FORMAT_MANTISSA_BITS.get(fmt, _DEFAULT_MANTISSA_BITS)
    magnitude = abs(magnitude)
    if magnitude == 0.0 or not math.isfinite(magnitude):
        return 2.0**-bits
    return 2.0 ** (math.floor(math.log2(magnitude)) - bits)


def probe_spacing_format(
    fmt: DataFormat, dest_acc: DestAccumulation | None = None
) -> DataFormat:
    """Which format's ULP a probe has to step by to survive the datapath: at
    dest_acc=No a 32-bit float lands in a 16-bit Dest, which truncates an fp32-sized
    step straight back onto the boundary. Nothing else is coarsened."""
    if dest_acc is None or _dest_acc_flag(dest_acc):
        return fmt
    if fmt.is_integer() or not fmt.is_32_bit():
        return fmt
    return DataFormat.Float16_b


def _truncate_mantissa(value: float, fmt: DataFormat) -> float:
    """*value* with its mantissa truncated to *fmt*'s width (what a narrower Dest does to
    a wider datum; toward zero, the direction that can only move a probe onto its
    boundary, never off it)."""
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
    point: float, direction: int, range_fmt: DataFormat, step_fmt: DataFormat
) -> float:
    """The tightest probe *direction* (+1/-1) off *point* that still arrives distinct:
    `_PROBE_ULPS` of *range_fmt*, widened to *step_fmt*'s ULP only where the datapath
    would quantize that back onto *point*. Decided per side: at 1.0 under a 16-bit Dest
    stepping up truncates back, stepping down crosses a binade and survives."""
    fine = point + direction * _PROBE_ULPS * format_ulp(range_fmt, point)
    if _truncate_mantissa(fine, step_fmt) != _truncate_mantissa(point, step_fmt):
        return fine
    return point + direction * _PROBE_ULPS * format_ulp(step_fmt, point)


def _dedup_representable(values: list[float], fmt: DataFormat) -> list[float]:
    """*values* sorted, without those *fmt* cannot tell from the previous one;
    non-finites kept verbatim at the end. The two zeros are kept apart: for signbit,
    sign, heaviside and reciprocal they are the whole point of the probe."""
    finite = sorted(v for v in values if math.isfinite(v))
    non_finite = [v for v in values if not math.isfinite(v)]
    kept: list[float] = []
    seen_zeros = set()
    for v in finite:
        if v == 0.0:
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
    op: MathOperation, operand: Operand, fmt: DataFormat, step_fmt: DataFormat
) -> list[float]:
    """Values straddling every boundary of *op*'s defined region for *operand*: each
    singularity with a probe on its defined side(s). (Every guard band is built around
    a singularity, so the singularities are every boundary.) Not clipped to *fmt*;
    that is the caller's job against the pipeline's narrowest format."""
    probes: list[float] = []
    for point, side in _OP_SINGULARITIES.get(op, {}).get(operand, ()):
        probes.append(point)
        if side in (_BOTH, _BELOW):
            probes.append(probe_beside(point, -1, fmt, step_fmt))
        if side in (_BOTH, _ABOVE):
            probes.append(probe_beside(point, +1, fmt, step_fmt))
    # Against *fmt*, not *step_fmt*: probe_beside already kept every probe distinct, and
    # a step_fmt dedup would drop the tight below-boundary ones.
    return _dedup_representable(probes, fmt)


# ── Special values and discrete edges ─────────────────────────────────────────

# Both zeros, and both matter: signbit, sign, heaviside, reciprocal and the
# comparison-to-zero ops distinguish them.
FLOAT_SPECIALS: tuple[float, ...] = (
    float("inf"),
    float("-inf"),
    float("nan"),
    0.0,
    -0.0,
)


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0.0


# Shift amounts worth driving on purpose, shared by the unary and binary shift sweeps:
# the in-range ends, the first out-of-range value, larger ones, negatives. The amounts
# only: each golden states its own kernel's rule for what they produce (binary shifts and
# the unary left shift zero an out-of-range amount; the unary right shift clamps it to 31).
# fmt: off
SHIFT_EDGE_AMOUNTS: tuple[int, ...] = (
    0, 1, 2, 7, 15, 16, 30, 31,      # in range
    32, 33, 40, 63, 100, 1000,       # >= 32
    -1, -5, -32, -1000,              # < 0
)
# fmt: on

_UNSIGNED_FORMATS = frozenset({DataFormat.UInt32, DataFormat.UInt16, DataFormat.UInt8})


def integer_specials(fmt: DataFormat) -> tuple[int, ...]:
    """The extreme and near-extreme values of an integer *fmt*. The signed minimum is
    the one value no StimuliSpec delivers (CustomStrategy clamps to `info.min + 1`, as
    Dst stores integers sign-magnitude): hand it over as a raw override tensor."""
    if not fmt.is_integer():
        raise ValueError(f"{fmt.name} is not an integer format")
    bits = int(fmt.size) * 8
    if fmt in _UNSIGNED_FORMATS:
        return (0, 1, 2**bits - 1)
    signed_min = -(2 ** (bits - 1))
    return (signed_min, signed_min + 1, -1, 0, 1, 2 ** (bits - 1) - 1)


def format_specials(fmt: DataFormat) -> tuple[float, ...]:
    """IEEE specials for a float *fmt*, integer extremes for an integer one."""
    if fmt.is_integer():
        return integer_specials(fmt)
    return FLOAT_SPECIALS


@dataclass(frozen=True)
class Edges:
    """One operand's discrete edges that are not domain boundaries (knees, exact rounding
    ties, comparison thresholds), and the scalar a threshold op compares against, which
    is one of them; the entries straddle their cutoffs, so it is not `points[0]`."""

    points: tuple[float, ...]
    threshold: float | None = None


def _a(
    points: tuple[float, ...] = (), threshold: float | None = None
) -> dict[Operand, Edges]:
    """Operand A's edges; a threshold given alone is the one point."""
    if threshold is not None and not points:
        points = (threshold,)
    return {Operand.A: Edges(points, threshold)}


# Ops whose only interesting point is zero, where +0.0 and -0.0 may differ. Relu is
# absent: it is applied by the packer (STACC_RELU), so no SFPU probe reaches it.
_ZERO_EDGE_OPS = (
    MathOperation.EqualZero,
    MathOperation.NotEqualZero,
    MathOperation.LessThanZero,
    MathOperation.GreaterThanZero,
    MathOperation.LessThanEqualZero,
    MathOperation.GreaterThanEqualZero,
    MathOperation.Sign,
    MathOperation.Signbit,
    MathOperation.Heaviside,
    MathOperation.Lrelu,
    MathOperation.Prelu,
    MathOperation.Elu,
    MathOperation.Celu,
    MathOperation.Selu,
    MathOperation.Xielu,
)

_COMPARISON_EDGE_OPS = (
    MathOperation.UnaryGt,
    MathOperation.UnaryLt,
    MathOperation.UnaryGe,
    MathOperation.UnaryLe,
    MathOperation.UnaryEq,
    MathOperation.UnaryNe,
)

# Every dispatch constant probed here is imported from sfpu_dispatch_constants, which the
# goldens read too, so a threshold cannot move on one side alone. Literal values are the
# mathematics' (hardsigmoid's [-3, 3], the rounding ties, the integer knees).
_OP_EDGES: dict[MathOperation, dict[Operand, Edges]] = {
    **{op: _a((0.0, -0.0)) for op in _ZERO_EDGE_OPS},
    **{op: _a(threshold=UNARY_COMP_THRESHOLD) for op in _COMPARISON_EDGE_OPS},
    MathOperation.LogicalNot: _a((0.0, -0.0), threshold=0.0),  # (x == 0)
    MathOperation.UnaryMax: _a((UNARY_MAX_MIN_VALUE, -UNARY_MAX_MIN_VALUE)),
    MathOperation.UnaryMin: _a((UNARY_MAX_MIN_VALUE, -UNARY_MAX_MIN_VALUE)),
    MathOperation.Clamp: _a((CLAMP_MIN, CLAMP_MAX)),
    MathOperation.Hardtanh: _a((CLAMP_MIN, CLAMP_MAX)),
    MathOperation.Softshrink: _a((-SOFTSHRINK_LAMBDA, SOFTSHRINK_LAMBDA)),
    MathOperation.Hardshrink: _a((-HARDSHRINK_LAMBDA, HARDSHRINK_LAMBDA)),
    MathOperation.Hardsigmoid: _a((-3.0, 3.0)),
    MathOperation.Hardmish: _a((-2.0, 0.0)),  # x * clamp(0.5 x + 1, 0, 1)
    MathOperation.Threshold: _a(threshold=THRESHOLD_T),
    # relu_max clamps above at its threshold and keeps relu's knee at 0; both are strict
    # compares, so each wants a value on either side. -0.0 stays out: SFPSETCC is
    # unspecified for it.
    MathOperation.ReluMax: _a(
        (
            -1.0,
            0.0,
            1.0,
            RELU_MAX_THRESHOLD - 1.0,
            RELU_MAX_THRESHOLD,
            RELU_MAX_THRESHOLD + 1.0,
        ),
        threshold=RELU_MAX_THRESHOLD,
    ),
    MathOperation.ReluMin: _a(
        (RELU_MIN_THRESHOLD - 1.0, RELU_MIN_THRESHOLD, RELU_MIN_THRESHOLD + 1.0),
        threshold=RELU_MIN_THRESHOLD,
    ),
    MathOperation.Softplus: _a((SOFTPLUS_THRESHOLD,)),  # linear past it
    MathOperation.Erfinv: _a((-0.0999, -0.01, -1e-3, -1e-4, -1e-5, 1e-5, 1e-4, 1e-3, 0.01, 0.0999)),
    MathOperation.Softsign: _a((-(2.0**126), 2.0**126, -3.0e38, 3.0e38)),
    MathOperation.Round: _a((-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)),  # half-to-even ties
    MathOperation.Floor: _a((-2.0, -1.0, 0.0, 1.0, 2.0)),
    MathOperation.Ceil: _a((-2.0, -1.0, 0.0, 1.0, 2.0)),
    MathOperation.Trunc: _a((-1.0, 0.0, 1.0)),
    MathOperation.Frac: _a((-1.5, -1.0, 1.0, 1.5)),
    # The integer scalar comparisons' exact tie, read by the int unary sweep.
    MathOperation.UnaryMaxInt32: _a((INT_MAXMIN_SCALAR,)),
    MathOperation.UnaryMinInt32: _a((INT_MAXMIN_SCALAR,)),
    MathOperation.UnaryMaxUint32: _a((INT_MAXMIN_SCALAR,)),
    MathOperation.UnaryMinUint32: _a((INT_MAXMIN_SCALAR,)),
    # pow(x, 0) == 1 for every x, a negative base included; the kernel compares the
    # exponent on setsgn, so -0.0 needs its own probe.
    MathOperation.SfpuElwpow: {
        **_a((-2.0, 2.0)),
        Operand.B: Edges((-0.0, 0.0, 1.0, 2.0)),
    },
    # lerp is a + c (b - a): c = 0 gives a, c = 1 gives b, outside [0, 1] extrapolates.
    MathOperation.SfpuLerp: {Operand.C: Edges((-1.0, 0.0, 1.0, 2.0))},
}


def op_edge_points(
    op: MathOperation, operand: Operand = Operand.A
) -> tuple[float, ...]:
    """*op*'s discrete edges for *operand*, or ()."""
    edges = _OP_EDGES.get(op, {}).get(operand)
    return () if edges is None else edges.points


def op_threshold(op: MathOperation) -> float | None:
    """The scalar *op* compares its input against, or None."""
    edges = _OP_EDGES.get(op, {}).get(Operand.A)
    return None if edges is None else edges.threshold


# ── Where an IEEE special or a signed zero survives the pipeline ──────────────

# Formats whose unpack leaves +inf, -inf and NaN intact.
_SPECIALS_CARRYING_INPUTS: frozenset[DataFormat] = frozenset(
    {DataFormat.Float32, DataFormat.Float16, DataFormat.Float16_b}
)

# Ops whose golden agrees with the kernel at every special the pipeline delivers (not
# that the golden is independently right). The golden-side gate; `specials_safe` is the
# pipeline-side one, and both have to pass. Log stays out: the kernel clamps a non-finite
# input to the format maximum and logs that, with no ISA ruling on which is right.
SPECIALS_READY_OPS: frozenset[MathOperation] = frozenset(
    {
        MathOperation.Abs,
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
        MathOperation.Exp,
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
        MathOperation.Identity,
        MathOperation.LessThanEqualZero,
        MathOperation.LessThanZero,
        MathOperation.Log1p,
        MathOperation.Lrelu,
        MathOperation.Mish,
        MathOperation.Neg,
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
        # Each of these diverges somewhere and is xfailed per combination in the sweep.
        MathOperation.Reciprocal,
        MathOperation.Sqrt,
        MathOperation.Rsqrt,
        MathOperation.SqrtCustom,
        # Routed through torch: math.sin/cos/asin/acos/tan raise on a non-finite input.
        MathOperation.Sin,
        MathOperation.Cos,
        MathOperation.Acos,
        MathOperation.Asin,
        MathOperation.Tan,
        # The golden tests for a NaN before any finite-input question.
        MathOperation.Square,
        MathOperation.I0,
        MathOperation.Hardshrink,
        # The comparison family answers on the SFPU's total order, not IEEE's unordered
        # compare: +NaN ranks above every finite value, so a clamp lands on its bound.
        MathOperation.UnaryGt,
        MathOperation.UnaryGe,
        MathOperation.UnaryMin,
        MathOperation.Clamp,
        MathOperation.Hardtanh,
        MathOperation.ReluMax,
        MathOperation.ReluMin,
        MathOperation.Hardsigmoid,
        # The scalar binops: x (+|-|*|/) c for a compile-time c; cat B is their whole
        # edge story. ScalarDiv folds to x * (1/d), so divide-by-zero is unreachable.
        MathOperation.ScalarAdd,
        MathOperation.ScalarSub,
        MathOperation.ScalarMul,
        MathOperation.ScalarDiv,
        MathOperation.ScalarRsub,
        # fill writes a constant; the probe says only that a non-finite input does not
        # corrupt it.
        MathOperation.Fill,
    }
)


def specials_safe(
    input_format: DataFormat, output_format: DataFormat, dest_acc: DestAccumulation
) -> bool:
    """May FLOAT_SPECIALS be injected on this (input, output, dest_acc) triple? Measured
    with the isinf/isnan predicates, whose 0/1 output every format holds: a Float16 input
    never preserves them; a Float16 output loses them unless a 32-bit input lands in a
    32-bit Dest; a 16-bit input at dest_acc=Yes keeps +inf but loses -inf and NaN; a
    block-float or MX output cannot express the result. Anything unmeasured is False."""
    dest_acc = _dest_acc_flag(dest_acc)
    if input_format not in _SPECIALS_CARRYING_INPUTS:
        return False
    if input_format == DataFormat.Float16:
        return False
    if output_format == DataFormat.Float16 and not (
        input_format.is_32_bit() and dest_acc
    ):
        return False
    if not input_format.is_32_bit() and dest_acc:
        return False
    if output_format.is_block_float() or output_format.is_mx_format():
        return False
    return True


def negative_zero_delivered(
    input_format: DataFormat, dest_acc: DestAccumulation | None
) -> bool:
    """Does a -0.0 written to L1 still have its sign when the SFPU reads it? Only on the
    unpack-to-dest path; the datacopy hands on +0.0. None (pipeline unknown) keeps the
    probe."""
    if dest_acc is None:
        return True
    return unpacks_to_dest(input_format, dest_acc)


def nan_survives_to_l1(
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: DestAccumulation | None,
) -> bool:
    """Does a NaN the kernel produces reach L1 as a NaN, or as a signed infinity? Keyed
    on (Dest format, output) as the goldens are: a NaN survives {(Float16, Float16),
    (Float32, Float16), (Float32, Float32)}; everywhere else the narrowing store makes
    an infinity of its sign, and that sign is suddenly load-bearing. None (pipeline
    unknown) keeps the assertion."""
    if dest_acc is None:
        return True
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


# Ops whose NaN result the kernel invents rather than forwards. IEEE leaves the sign of
# an invalid-operation default unspecified (Blackhole emits the canonical NaN, Wormhole
# one of either sign), which shows the moment `nan_survives_to_l1` is False. Membership
# is by observed disagreement on Wormhole, not by kernel shape; add an op if it fails.
GENERATED_NAN_SIGN_OPS: frozenset[MathOperation] = frozenset(
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
    dest_acc: DestAccumulation | None,
) -> bool:
    """Would this variant assert the sign of a NaN the ISA leaves unspecified: an op
    that invents its NaN, on a pipeline that turns the sign into a +-inf?"""
    return mathop in GENERATED_NAN_SIGN_OPS and not nan_survives_to_l1(
        input_format, output_format, dest_acc
    )


def specials_after_nan_sign_gate(
    mathop: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: DestAccumulation | None,
    specials: bool,
    on_wormhole: bool,
) -> bool:
    """*specials*, switched off where the NaN sign would be unspecified on Wormhole
    (Blackhole's SFPMAD promises the canonical NaN). The probe is switched off rather
    than the variant skipped, so the pole and knee probes sharing its tensor stay."""
    if (
        specials
        and on_wormhole
        and nan_sign_is_unspecified(mathop, input_format, output_format, dest_acc)
    ):
        return False
    return specials


# Cat B for the binary SFPU family: the golden defines an answer for a non-finite
# operand on every specials-safe cell. The comparisons and max/min answer on the SFPU's
# total order rather than IEEE's unordered compare, which BinarySFPUGolden models.
BINARY_SPECIALS_READY_OPS: frozenset[MathOperation] = frozenset(
    {
        # SFPMAD arithmetic, IEEE for a non-finite input.
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuElwrsub,
        # A bare SFPSWAP with no NaN guard: +NaN is the maximum, -NaN the minimum.
        MathOperation.SfpuBinaryMax,
        MathOperation.SfpuBinaryMin,
        # These reject a NaN operand before comparing, leaving the pre-stored default.
        MathOperation.SfpuElwEq,
        MathOperation.SfpuElwNe,
        MathOperation.SfpuElwLt,
        MathOperation.SfpuElwGt,
        MathOperation.SfpuElwLe,
        MathOperation.SfpuElwGe,
        # Classify |a| and |b| against +inf first: 0 for any NaN (1 for two with
        # EQUAL_NAN), 1 for two equal infinities, else 0 -- torch.isclose. Passes the
        # (special, special) grid on Blackhole once the harness runs isclose_init.
        MathOperation.SfpuIsclose,
        MathOperation.SfpuIscloseEqualNan,
    }
)

# The binary ops that diverge at a special, each waiting on a decision: compositions
# through a reciprocal, log or exp (an LLK decision, not an ISA one); SfpuMask's
# SFPSETCC on a possibly-NaN operand, and SfpuMaskPosinf's, the same zero test;
# SfpuLogsigmoid, which reads B only on one branch; SfpuCopyDest, a bit copy.
_BINARY_SPECIALS_NOT_READY: frozenset[MathOperation] = frozenset(
    {
        MathOperation.SfpuElwdiv,
        MathOperation.SfpuXlogy,
        MathOperation.SfpuElwpow,
        MathOperation.SfpuBinaryFmod,
        MathOperation.SfpuBinaryRemainder,
        MathOperation.SfpuAtan2,
        MathOperation.SfpuLogaddexp,
        MathOperation.SfpuLogaddexp2,
        MathOperation.SfpuMask,
        MathOperation.SfpuMaskPosinf,
        MathOperation.SfpuLogsigmoid,
        MathOperation.SfpuCopyDest,
    }
)

assert not (BINARY_SPECIALS_READY_OPS & _BINARY_SPECIALS_NOT_READY)


def generated_nan_sign_is_asserted(
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: DestAccumulation | None,
    on_wormhole: bool,
) -> bool:
    """The binary family's twin of `nan_sign_is_unspecified`, with no op argument: the
    binary edge sweep partitions its probe by what the golden answers, so every NaN in
    its `both_zero` and `nan_golden` classes is generated by construction."""
    return on_wormhole and not nan_survives_to_l1(input_format, output_format, dest_acc)


# ── Edge values and specs ─────────────────────────────────────────────────────


def clip_to_format(values: list[float], fmt: DataFormat) -> list[float]:
    """Without the finite values *fmt* cannot represent; non-finites are the point of a
    cat-B probe and are never clipped."""
    limit = _FORMAT_MAX_MAGNITUDE.get(fmt, _BF16_MAX_MAGNITUDE)
    return [v for v in values if not math.isfinite(v) or abs(v) <= limit]


def edge_values(
    op: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat | None = None,
    operand: Operand = Operand.A,
    specials: bool = False,
    dest_acc: DestAccumulation | None = None,
) -> list[float]:
    """Every value worth hitting on purpose for (*op*, *operand*) in this pipeline: the
    singularities straddled (`boundary_probes`), the discrete edges (`op_edge_points`),
    and the format's specials when *specials* is on (the caller decides that with
    `specials_safe`). Clipped and spaced against the narrowest format in the pipeline;
    -0.0 is dropped where the datacopy path would hand the kernel +0.0."""
    range_fmt = narrowest_range_format(input_format, output_format)
    step_fmt = probe_spacing_format(range_fmt, dest_acc)
    drop_negative_zero = not range_fmt.is_integer() and not negative_zero_delivered(
        input_format, dest_acc
    )

    def delivered(values):
        return [v for v in values if not (drop_negative_zero and _is_negative_zero(v))]

    vals = boundary_probes(op, operand, range_fmt, step_fmt)
    vals += delivered(op_edge_points(op, operand))
    if specials:
        vals += delivered(format_specials(range_fmt))
    return _dedup_representable(clip_to_format(vals, range_fmt), range_fmt)


def edge_spec(
    op: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat | None = None,
    operand: Operand = Operand.A,
    specials: bool = False,
    dest_acc: DestAccumulation | None = None,
) -> StimuliSpec | None:
    """`edge_values` as a StimuliSpec, or None if *op* has no edge worth probing, so a
    caller can fall back to the random domain. The values head every face and the rest
    is zero-filled, itself a useful probe. Integer extremes cannot go through a spec
    (INT_MIN clamps to INT_MIN + 1): hand them over as a raw override tensor."""
    vals = edge_values(op, input_format, output_format, operand, specials, dest_acc)
    if not vals:
        return None
    if input_format.is_integer() and specials:
        raise ValueError(
            f"edge_spec(specials=True) cannot deliver {input_format.name}'s extremes: "
            "StimuliSpec.custom clamps INT_MIN to INT_MIN + 1; use a raw override tensor"
        )
    return StimuliSpec.custom(values=vals, seed=0)


# What the operand with no edge of its own contributes to a pair: a divisor-zero probe
# means something only against a positive, a negative and a zero numerator.
_EDGE_COUNTERPARTS: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _in_spec_domain(spec: StimuliSpec | None, value: float) -> bool:
    if spec is None:
        return True
    if spec.intervals:
        return any(lo <= value <= hi for lo, hi in spec.intervals)
    if spec.low is None or spec.high is None:
        return True
    return spec.low <= value <= spec.high


def edge_counterparts(
    op: MathOperation, fmt: DataFormat, operand: Operand = Operand.A
) -> list[float]:
    """The counterpart spread, clipped to *op*'s registered domain for *operand*."""
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
    output_format: DataFormat | None = None,
    specials: bool = False,
    dest_acc: DestAccumulation | None = None,
) -> list[tuple[float, float]]:
    """The product of both operands' edge values for a binary op, the operand with no
    edge of its own contributing `edge_counterparts`; [] when neither has anything to
    probe, the caller's cue to skip."""
    a = edge_values(op, input_format, output_format, Operand.A, specials, dest_acc)
    b = edge_values(op, input_format, output_format, Operand.B, specials, dest_acc)
    if not a and not b:
        return []
    if not a:
        a = edge_counterparts(op, input_format, Operand.A)
    if not b:
        b = edge_counterparts(op, input_format, Operand.B)
    if not a or not b:
        return []
    return [(x, y) for x in a for y in b]

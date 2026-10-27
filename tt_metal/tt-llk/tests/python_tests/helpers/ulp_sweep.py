# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive 16-bit ULP sweeps, and folding what they measure back into the table.

bfloat16 has 65,025 normal values (zero included) and float16 61,441, so each fits one
64-tile device run; Float32 is walked with a stride that gives every binade an equal
share. No subnormal is fed (the unpack flushes them), and only the formats the SFPU
computes in are swept.
"""

import math
import pickle
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import torch
from helpers.chip_architecture import get_chip_architecture
from helpers.data_format_inference import (
    _dest_acc_flag,
    effective_dest_acc,
    infer_unpack_out,
    unpacks_to_dest,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import truncate_to_dest_width
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.sfpu_domains import (
    _NONPOSITIVE_INTEGER_POLES,
    _OP_SINGULARITIES,
    Operand,
    SingularitySide,
    negative_zero_delivered,
)
from helpers.stimuli_generator import StimuliSpec
from helpers.stimuli_generator.strategies.structured import _enumerate_representable
from helpers.ulp import NEAR_ZERO_FRACTION, nonfinite_mismatches, ulp_elementwise_valid
from helpers.ulp_domains import (
    BINADE_OFFSET,
    Bin,
    Domain,
    entries_of,
    usable_budget_ceiling,
)
from helpers.ulp_table import (
    MEASURED_ARCH,
    BudgetKey,
    Metric,
    Reading,
    Row,
    load_table,
    write_table_file,
)

# The formats the sweep feeds and judges; Float32 is strided, the others enumerated.
SWEEP_FORMATS: tuple[DataFormat, ...] = (
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
)

# Float32's 2**32 values do not fit one run: one sample per normal bfloat16 cell, 65,024
# of them, each moved along its cell by 1/phi of a cell from the one before
# (`_enumerate_fp32_in_range`), so the low 16 mantissa bits are never zero and both
# halves of a cell are reached in every binade.
_FP32_STRIDE = 2**16

# One 64-tile run: the most values a sweep variant generates.
_SWEEP_TENSOR = 2**16

_INF = float("inf")


def sweep_cells(
    arch=None,
) -> list[tuple[DataFormat, DataFormat, ApproximationMode, DestAccumulation]]:
    """Every ``(input, output, approx_mode, dest_acc)`` cell the sweep runs on *arch*
    (default: the chip this session targets), less the cells TestConfig would run as
    another: an exponent-B input packed to Float16 needs a 32-bit Dest, so a
    ``dest_acc=No`` request runs the ``Yes`` kernel and must not be keyed as ``No``."""
    if arch is None:
        arch = get_chip_architecture()
    return [
        (in_fmt, out_fmt, approx, dest)
        for in_fmt in SWEEP_FORMATS
        for out_fmt in SWEEP_FORMATS
        for approx in ApproximationMode
        for dest in DestAccumulation
        if effective_dest_acc(in_fmt, out_fmt, dest, arch) == dest
    ]


def _stride(input_format: DataFormat) -> int:
    return _FP32_STRIDE if input_format == DataFormat.Float32 else 1


@lru_cache(maxsize=None)
def swept_value_count(input_format: DataFormat) -> int:
    """How many values the sweep generates for *input_format* (65,024 of Float32's
    2**32)."""
    return int(
        _enumerate_representable(
            input_format,
            -_INF,
            _INF,
            _SWEEP_TENSOR,
            stride=_stride(input_format),
            normal_only=True,
        ).numel()
    )


def sweep_spec(input_format: DataFormat) -> StimuliSpec:
    """Every normal value of *input_format* and its zero, once (every
    ``_FP32_STRIDE``-th for Float32). Not clipped to the op's domain: undefined inputs
    reach the hardware and are masked out of the statistics afterwards."""
    return StimuliSpec.ulp_sweep(
        low=-_INF, high=_INF, stride=_stride(input_format), normal_only=True
    )


def padding_lanes(src: torch.Tensor, input_format: DataFormat) -> torch.Tensor:
    """The zero tail ``generate_full_tensor`` pads the swept values with to reach the
    tile count (511 lanes for bf16, 4,095 for fp16); by position, since one legitimate
    0.0 is swept too."""
    flat = torch.zeros(src.numel(), dtype=torch.bool, device=src.device)
    flat[swept_value_count(input_format) :] = True
    return flat.reshape(src.shape)


def received_inputs(
    src: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: DestAccumulation,
) -> torch.Tensor:
    """*src* as the op receives it: itself, except a Float32 input at ``dest_acc=No``,
    which the unpack truncates to the 16-bit Dest (-8326436.5 arrives as the pole
    -8323072)."""
    if input_format != DataFormat.Float32 or _dest_acc_flag(dest_acc):
        return src
    width = _unpack_format(input_format, output_format, dest_acc)
    return truncate_to_dest_width(src.detach().to(torch.float32), width)


def golden_input(src: torch.Tensor, input_format: DataFormat, dest_acc) -> torch.Tensor:
    """*src* as the golden should see it: the walk's one data zero is -0.0, which the
    unpack hands on as +0.0 except on the unpack-to-dest path (``signbit`` would
    otherwise read 16129 steps on that one lane)."""
    if negative_zero_delivered(input_format, dest_acc):
        return src
    return torch.where(src == 0, torch.zeros_like(src), src)


def _unpack_format(
    input_format: DataFormat, output_format: DataFormat, dest_acc: DestAccumulation
) -> DataFormat:
    """The format whose values the unpacker writes for this cell (a Float32 input into
    a Float16 output at ``dest_acc=No`` lands in a Float16 Dest)."""
    return infer_unpack_out(
        input_format,
        output_format,
        dest_acc,
        unpacking_to_dest=unpacks_to_dest(input_format, dest_acc),
    )


def _dest_dtype(input_format, output_format, dest_acc) -> torch.dtype | None:
    """The float dtype of :func:`_unpack_format`'s format, or ``None``."""
    dtype = format_dict.get(_unpack_format(input_format, output_format, dest_acc))
    return dtype if dtype is not None and dtype.is_floating_point else None


def dest_holds(
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: DestAccumulation,
) -> torch.Tensor:
    """False on lanes an fp16 Dest (``dest_acc=No``) under a wider output lost: a golden
    past 65504 answered by NaN or by the golden's sign in the Dest's top binade, or a
    flush at the 2**-14 boundary. Any other answer is the op's."""
    holds = torch.ones(golden.shape, dtype=torch.bool, device=golden.device)
    if _dest_acc_flag(dest_acc):
        return holds
    if _dest_dtype(input_format, output_format, dest_acc) is not torch.float16:
        return holds
    dest = torch.finfo(torch.float16)
    out = torch.finfo(format_dict[output_format])
    if dest.max >= out.max and dest.tiny <= out.tiny:
        return holds
    g = golden.detach().to(torch.float32)
    r = result.detach().to(torch.float32)
    same_sign = torch.signbit(r) == torch.signbit(g)
    # Past 65504 the golden is a stand-in (131008 for an fp16 inf), so an answer near
    # the edge cannot be ranked against it.
    top_binade = 2.0 ** (math.frexp(dest.max)[1] - 1)
    overflowed = (g.abs() > dest.max) & (
        torch.isnan(r) | (same_sign & (r.abs() >= top_binade))
    )
    flushed = ((g == 0) & (r != 0) & (r.abs() <= dest.tiny)) | (
        (r == 0) & (g != 0) & (g.abs() <= dest.tiny)
    )
    return holds & ~overflowed & ~flushed


def _normal_input(
    src: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: DestAccumulation,
) -> torch.Tensor:
    """Lanes whose input survives the unpack: zero, or at least the smallest normal of
    the format the unpacker writes, and no larger than it can hold (a Float32 input
    into a Float16 Dest flushes below 2**-14 and saturates past 65504)."""
    fed = torch.finfo(format_dict[input_format])
    cutoff, ceiling = fed.smallest_normal, math.inf
    dtype = _dest_dtype(input_format, output_format, dest_acc)
    if dtype is not None:
        unpacked = torch.finfo(dtype)
        cutoff = max(cutoff, unpacked.smallest_normal)
        if math.frexp(unpacked.max)[1] < math.frexp(fed.max)[1]:
            ceiling = float(unpacked.max)
    # From `src` as generated, in float32: cast to a 16-bit golden dtype first, an fp16
    # subnormal rounds up past the threshold.
    magnitude = src.detach().to(torch.float32).abs()
    return ((magnitude >= cutoff) | (magnitude == 0)) & (magnitude <= ceiling)


def measurable_mask(
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: DestAccumulation,
) -> torch.Tensor:
    """Lanes a step count can describe: both sides finite and agreeing about it, the
    input neither flushed nor saturated by the unpack, the answer not lost by an fp16
    Dest, and not the sweep's zero padding. Op-agnostic: the claim is
    :func:`ranked_mask`'s, and the non-finite disagreements dropped here are
    :func:`nonfinite_failures`' to report."""
    return (
        ~(torch.isnan(golden) | torch.isnan(result))
        & ~nonfinite_mismatches(golden, result)
        & _normal_input(src, input_format, output_format, dest_acc)
        & dest_holds(golden, result, input_format, output_format, dest_acc)
        & ~padding_lanes(src, input_format)
    )


def ranked_mask(
    op, src, golden, result, input_format, output_format, dest_acc
) -> torch.Tensor:
    """The lanes a sweep ranks: :func:`measurable_mask` within *op*'s
    :func:`claimed_lanes`."""
    return measurable_mask(
        src, golden, result, input_format, output_format, dest_acc
    ) & claimed_lanes(op, src, input_format, output_format, dest_acc)


# Where Sin, Cos and Tan's range reduction breaks, per format the kernel is handed
# (`_unpack_format`); fp16 ends there, so an fp16 input has no limit.
_TRIG_LIMIT = {DataFormat.Float16_b: 65504.0, DataFormat.Float32: 65504.0}
_CLAIM_LIMIT = {
    MathOperation.Sin: _TRIG_LIMIT,
    MathOperation.Cos: _TRIG_LIMIT,
    MathOperation.Tan: _TRIG_LIMIT,
}


def claimed_lanes(
    op,
    src: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    dest_acc: DestAccumulation,
) -> torch.Tensor:
    """Lanes where *op* claims an answer, judged on the input as received: all but its
    singularities' undefined side, its poles, and anything past its claim limit."""
    value = received_inputs(src, input_format, output_format, dest_acc).to(
        torch.float32
    )
    claimed = torch.ones_like(value, dtype=torch.bool)
    for point, side in _OP_SINGULARITIES.get(op, {}).get(Operand.A, ()):
        # The defined side keeps the point: sqrt(0) and acosh(1) are answers.
        if side is SingularitySide.ABOVE:
            claimed &= value >= point
        elif side is SingularitySide.BELOW:
            claimed &= value <= point
        else:
            claimed &= value != point
    if op in _NONPOSITIVE_INTEGER_POLES:
        # Every fp32 value of magnitude 2**23 or more is an integer: the whole negative
        # tail past there is poles.
        claimed &= ~((value <= 0) & (value == torch.floor(value)))
    limit = _CLAIM_LIMIT.get(op, {}).get(
        _unpack_format(input_format, output_format, dest_acc)
    )
    if limit is not None:
        claimed &= value.abs() <= limit
    return claimed


def _at_a_singularity(op, src, input_format, output_format, dest_acc) -> torch.Tensor:
    """Lanes received exactly on one of *op*'s singularity points."""
    value = received_inputs(src, input_format, output_format, dest_acc).to(
        torch.float32
    )
    on_point = torch.zeros_like(value, dtype=torch.bool)
    for point, _side in _OP_SINGULARITIES.get(op, {}).get(Operand.A, ()):
        on_point |= value == point
    return on_point


@dataclass(frozen=True)
class KnownNonfiniteLanes:
    """Inputs on which an op answers on the wrong side of infinity on one cell, tracked
    by an issue. Only the non-finite disagreement is excused, on that cell alone, and
    :func:`stale_excuses` retires the entry once its lanes agree again."""

    issue: int
    input: DataFormat
    output: DataFormat
    low: float  # inclusive bounds on the input as received, on |x| with `magnitude`
    high: float
    approx: ApproximationMode | None = None
    dest: DestAccumulation | None = None
    magnitude: bool = False
    why: str = ""

    def applies_to(self, input_format, output_format, approx_mode, dest_acc) -> bool:
        return (
            input_format == self.input
            and output_format == self.output
            and (self.approx is None or approx_mode == self.approx)
            and (self.dest is None or dest_acc == self.dest)
        )

    def lanes(self, received: torch.Tensor) -> torch.Tensor:
        value = received.detach().to(torch.float32)
        if self.magnitude:
            value = value.abs()
        return (value >= self.low) & (value <= self.high)


# #58607: on a 16-bit Float16 Dest these ops answer inf where the answer is one of the
# four largest fp16 values, 65408..65504 (a 32-bit Dest reads 1-2 steps there).
_TOP_OF_FP16 = KnownNonfiniteLanes(
    issue=58607,
    input=DataFormat.Float16,
    output=DataFormat.Float16,
    dest=DestAccumulation.No,
    low=65408.0,
    high=65504.0,
    why="inf where the answer is x itself, in the top four fp16 values, on a 16-bit Dest",
)
_ABOVE_65408 = replace(_TOP_OF_FP16, low=65440.0)  # the op answers 65408 itself

_KNOWN_NONFINITE_LANES: dict[MathOperation, tuple[KnownNonfiniteLanes, ...]] = {
    MathOperation.Celu: (_TOP_OF_FP16,),
    MathOperation.Elu: (_TOP_OF_FP16,),
    MathOperation.Gelu: (replace(_TOP_OF_FP16, approx=ApproximationMode.No),),
    MathOperation.GeluTanh: (_TOP_OF_FP16,),
    MathOperation.Mish: (_TOP_OF_FP16,),
    MathOperation.Silu: (_ABOVE_65408,),
    MathOperation.Selu: (
        replace(
            _TOP_OF_FP16,
            low=62272.0,
            high=62336.0,
            why="inf where 1.0507 x is 65440..65504, on a 16-bit Dest",
        ),
    ),
    MathOperation.Xielu: (
        replace(
            _TOP_OF_FP16,
            low=255.5,
            high=255.625,
            why="inf where the answer is 65408 or 65472, on a 16-bit Dest",
        ),
    ),
    MathOperation.UnaryPower: (
        replace(
            _TOP_OF_FP16,
            low=255.625,
            high=255.875,
            magnitude=True,
            why="inf (NaN at |x|=255.875) where x*x is 65344..65472, on a 16-bit Dest",
        ),
    ),
    MathOperation.Square: (
        replace(
            _TOP_OF_FP16,
            low=255.75,
            high=255.875,
            magnitude=True,
            why="inf where x*x is 65408 or 65472, on a 16-bit Dest",
        ),
    ),
    MathOperation.Tanhshrink: (
        replace(
            _ABOVE_65408,
            magnitude=True,
            why="inf where the answer is x - tanh(x), in the top fp16 values, on a 16-bit Dest",
        ),
    ),
    MathOperation.Lgamma: (
        replace(
            _TOP_OF_FP16,
            low=8168.0,
            high=8180.0,
            why="inf where lgamma(x) is a top fp16 value, on a 16-bit Dest",
        ),
    ),
    # #50465: i0 is a bare Taylor series with no large-|x| branch, so where i0 first
    # overflows fp16 the truncated series still answers finite.
    MathOperation.I0: (
        KnownNonfiniteLanes(
            issue=50465,
            input=DataFormat.Float16,
            output=DataFormat.Float16,
            low=13.296875,
            high=13.296875,
            magnitude=True,
            why="finite where i0(x) has just overflowed fp16: the series has no large-|x| branch",
        ),
    ),
    # #57215: the Float16 store carries exactly 2**16 to inf, and approximate sqrt lands
    # on 2**16 where the answer rounds to 65504.
    MathOperation.Sqrt: (
        KnownNonfiniteLanes(
            issue=57215,
            input=DataFormat.Float32,
            output=DataFormat.Float16,
            approx=ApproximationMode.Yes,
            dest=DestAccumulation.Yes,
            low=65488.0**2,
            high=65520.0**2,
            why="inf where sqrt(x) rounds to 65504: the approximation lands on 2**16, the one value the store carries to inf",
        ),
    ),
}


def known_nonfinite_lanes(
    op, src, input_format, output_format, *, approx_mode, dest_acc
) -> torch.Tensor:
    """The lanes of *src* a :data:`_KNOWN_NONFINITE_LANES` entry names on this cell."""
    received = received_inputs(src, input_format, output_format, dest_acc)
    excused = torch.zeros(src.shape, dtype=torch.bool, device=src.device)
    for entry in _KNOWN_NONFINITE_LANES.get(op, ()):
        if entry.applies_to(input_format, output_format, approx_mode, dest_acc):
            excused |= entry.lanes(received)
    return excused


def stale_excuses(
    op, src, golden, result, input_format, output_format, *, approx_mode, dest_acc
) -> list[KnownNonfiniteLanes]:
    """The entries that apply to this cell but excuse nothing, because their lanes agree
    with the golden again or another exclusion already covers them."""
    without = nonfinite_failures(
        op,
        src,
        golden,
        result,
        input_format,
        output_format,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        known_lanes=False,
    )
    received = received_inputs(src, input_format, output_format, dest_acc)
    return [
        entry
        for entry in _KNOWN_NONFINITE_LANES.get(op, ())
        if entry.applies_to(input_format, output_format, approx_mode, dest_acc)
        and not bool((entry.lanes(received) & without).any())
    ]


def nonfinite_failures(
    op,
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    *,
    approx_mode,
    dest_acc,
    known_lanes: bool = True,
) -> torch.Tensor:
    """The lanes :func:`measurable_mask` drops that are a *failure*: the two sides
    disagreeing about being non-finite where the output could have held the answer.
    The sweep ranks a distance rather than calling ``passed_test``, so without this a
    hardware overflow would leave the statistics clean.

    Excused: a flushed or saturated input (:func:`_normal_input`); a NaN golden; a
    golden past the output range answered by a saturated store (NaN, an infinity of the
    golden's sign, or on Float16 the pack's clamp to +-65504); an infinite golden on a
    singularity point itself; the zero padding; an answer the Dest lost
    (:func:`dest_holds`); an input the op makes no claim on (:func:`claimed_lanes`);
    and, unless *known_lanes* is off, a lane a tracked issue names."""
    normal_input = _normal_input(src, input_format, output_format, dest_acc)
    output_max = torch.finfo(format_dict[output_format]).max
    past_range = golden.detach().to(torch.float32).abs() > output_max
    same_sign = torch.signbit(result) == torch.signbit(golden)
    saturated = torch.isnan(result) | (torch.isinf(result) & same_sign)
    if output_format == DataFormat.Float16:
        saturated |= same_sign & (result.detach().to(torch.float32).abs() == output_max)
    excused = torch.isnan(golden) | (
        past_range
        & (
            saturated
            | _at_a_singularity(op, src, input_format, output_format, dest_acc)
        )
    )
    if known_lanes:
        excused |= known_nonfinite_lanes(
            op,
            src,
            input_format,
            output_format,
            approx_mode=approx_mode,
            dest_acc=dest_acc,
        )
    return (
        nonfinite_mismatches(golden, result)
        & normal_input
        & dest_holds(golden, result, input_format, output_format, dest_acc)
        & ~excused
        & claimed_lanes(op, src, input_format, output_format, dest_acc)
        & ~padding_lanes(src, input_format)
    )


# How many offending lanes a not-measurable note names: the row's `nonfinite` count says
# how many there are, two lanes show the kind, and more would only lengthen every note.
NAMED_LANES = 2


@dataclass(frozen=True)
class Unmeasurable:
    """A cell the sweep could not describe with a step count: *nonfinite* lanes
    disagreed with the golden about being finite (0 for a cell with no rankable lane),
    *measured* is the worst of the lanes that could be ranked (``None`` if none)."""

    why: str
    nonfinite: int = 0
    measured: int | None = None


def nonfinite_verdict(
    overflowed: torch.Tensor,
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    stats: dict,
) -> Unmeasurable:
    """The verdict a cell with non-finite disagreements is written with."""
    named = "; ".join(
        f"x={float(src[i]):g}: {float(golden[i]):g} -> {float(result[i]):g}"
        for i in overflowed.nonzero().flatten()[:NAMED_LANES].tolist()
    )
    return Unmeasurable(
        why="lanes disagreeing with the golden about being finite "
        f"(golden -> result: {named})",
        nonfinite=int(overflowed.sum()),
        # Over no lane at all `ulp_stats` reads 0, which would claim the rest bit-exact.
        measured=int(stats["max"]) if stats["lanes"] else None,
    )


# ── Folding a sweep back into the table ───────────────────────────────────────

# Set by --ulp-emit: measure and rewrite the table instead of gating against it.
EMIT = False

# Where an xdist worker hands its measurements to the controller, in `workeroutput`.
WORKEROUTPUT_KEY = "ulp_measured"

# The widest near_zero_atol the emitter writes on its own; past it the floor would
# forgive a kernel's cut-off rather than its rounding, which is a decision for a person.
EMIT_MAX_NEAR_ZERO_ATOL = 1e-3


@dataclass(frozen=True)
class Floored:
    """A cell's measurement with the near-zero floor that makes it gateable: the worst
    lane over every ranked lane, the worst outside the floor's band, and the floor."""

    max_ulp: int
    residual: int
    atol: float


# What one cell recorded: a worst-lane step count, a floored measurement, or an
# unmeasurable verdict.
Recorded = int | Floored | Unmeasurable

# The cells measured this session, by op and by the variant's key, and the list each
# was recorded with (a cell read twice keeps one only while both readings agree; see
# `record`).
MEASURED: dict[MathOperation, dict[BudgetKey, Recorded]] = {}
DOMAINS: dict[MathOperation, dict[BudgetKey, tuple[Domain, ...]]] = {}


def near_zero_floor(
    golden: torch.Tensor,
    result: torch.Tensor,
    distance: torch.Tensor,
    mask: torch.Tensor,
    out_fmt: DataFormat,
) -> tuple[int, float] | None:
    """``(residual, atol)``: the worst lane outside the near-zero floor a demoted cell
    needs, and that floor, exactly the largest absolute error it rescues (capped at
    :data:`EMIT_MAX_NEAR_ZERO_ATOL`); ``None`` when the cell fits without one."""
    ceiling = usable_budget_ceiling(out_fmt)
    flat = distance.reshape(-1).to(torch.int64)
    selected = mask.reshape(-1) & (flat >= 0)
    if not selected.any() or int(flat[selected].max()) <= ceiling:
        return None
    g = golden.detach().reshape(-1).to(torch.float32)
    r = result.detach().reshape(-1).to(torch.float32)
    magnitude = g.abs()
    finite = selected & torch.isfinite(g)
    if not finite.any():
        return None
    band = finite & (magnitude < NEAR_ZERO_FRACTION * float(magnitude[finite].max()))
    absolute_error = (r - g).abs()
    demoting = band & (flat > ceiling)
    if not demoting.any():
        return None
    atol = float(absolute_error[demoting].max())
    if atol == 0:
        return None
    atol = min(atol, EMIT_MAX_NEAR_ZERO_ATOL)
    rescued = band & (magnitude <= atol / NEAR_ZERO_FRACTION) & (absolute_error <= atol)
    kept = selected & ~rescued
    residual = int(flat[kept].max()) if kept.any() else 0
    return residual, atol


def _max_of(value) -> int:
    return value.max_ulp if isinstance(value, Floored) else int(value)


def floor_covered(
    golden: torch.Tensor,
    result: torch.Tensor,
    mask: torch.Tensor,
    atol: float | None,
) -> torch.Tensor:
    """The lanes of *mask* a floor of *atol* judges on absolute error instead of steps:
    ``ulp_elementwise_valid``'s own band, so a list's entries and the cell's residual
    leave out the same lanes."""
    if atol is None:
        return torch.zeros_like(mask)
    # The covered mask does not depend on the budget.
    _, _, covered = ulp_elementwise_valid(
        golden, result, max_ulp=0, near_zero_atol=atol, selected=mask
    )
    return covered.reshape(mask.shape) & mask


def floored_profile(golden, result, x, distance, mask, present, atol):
    """``(judged, bins)`` for a cell held with a floor of *atol*, or none: the ranked
    lanes outside the floor, which a list's entries are figured and gated on, and the
    binade profile with the covered lanes ranked at 0 steps (held, so no edge of their
    own). The emitter passes the floor it measured, the gate the row's, so the gate
    judges exactly the lanes the emitter figured."""
    covered = floor_covered(golden, result, mask, atol)
    held = torch.where(
        covered.reshape(distance.shape), torch.zeros_like(distance), distance
    )
    return mask & ~covered, binade_maxima(x, held, mask, present)


def _binade_index(x: torch.Tensor) -> torch.Tensor:
    """``ulp_domains.binade`` over a tensor of received inputs."""
    flat = x.reshape(-1).to(torch.float64)
    mantissa, exponent = torch.frexp(flat)
    k = exponent.to(torch.int64) - 1 - ((flat < 0) & (mantissa.abs() == 0.5)).long()
    index = k + BINADE_OFFSET
    return torch.where(
        flat > 0, index, torch.where(flat < 0, -index, torch.zeros_like(index))
    )


def binade_maxima(
    x: torch.Tensor, distance: torch.Tensor, ranked: torch.Tensor, present: torch.Tensor
) -> dict[int, Bin]:
    """Per signed binade of the received input *x*: lanes *present* (data, not
    padding), lanes *ranked* and the worst of them by *distance*; the profile
    ``ulp_domains.partition`` draws a list on and ``--ulp-measure`` records as
    ``bins``."""
    index = _binade_index(x)
    dist = distance.reshape(-1).to(torch.int64)
    present = present.reshape(-1)
    ranked = ranked.reshape(-1)
    bins: dict[int, Bin] = {}
    for b in torch.unique(index[present]).tolist():
        here = index == b
        lanes = here & ranked
        n = int(lanes.sum())
        bins[int(b)] = (
            int((here & present).sum()),
            n,
            int(dist[lanes].max()) if n else -1,
        )
    return bins


def band_mask(low: float, high: float, x: torch.Tensor, mask: torch.Tensor):
    """The lanes of *mask* whose received input lies in ``[low, high)``."""
    flat = x.reshape(-1).to(torch.float64)
    return (mask.reshape(-1) & (flat >= low) & (flat < high)).reshape(mask.shape)


def _figure(low, high, x, distance, mask, judged) -> tuple[int | None, int]:
    """``(worst, lanes)`` over one entry: the worst *judged* lane (ranked, outside the
    floor), 0 when the floor covers every ranked lane, ``None`` when none is ranked; and
    the ranked lane count."""
    dist = distance.reshape(-1).to(torch.int64)
    ranked = band_mask(low, high, x, mask).reshape(-1)
    outside = band_mask(low, high, x, judged).reshape(-1)
    n = int(ranked.sum())
    if bool(outside.any()):
        return int(dist[outside].max()), n
    return (0 if n else None), n


def domain_figures(spans, x, distance, mask, judged) -> tuple[Domain, ...]:
    """Each span of ``ulp_domains.partition`` as the entry the table writes
    (``entries_of``), figured over exactly its lanes outside the floor."""
    return entries_of(
        spans, lambda low, high: _figure(low, high, x, distance, mask, judged)[0]
    )


def entry_readings(domains, x, distance, mask, judged) -> tuple[Reading, ...]:
    """What the sweep measured over each entry of a list, for the ``--ulp-measure``
    record."""
    return tuple(
        Reading(e.low, e.high, *_figure(e.low, e.high, x, distance, mask, judged))
        for e in domains
    )


def record(
    op: MathOperation,
    key: BudgetKey,
    max_ulp: int,
    floor: tuple[int, float] | None = None,
    domains: tuple[Domain, ...] = (),
) -> None:
    """Fold one measurement into the cell *key* names, keeping the worst lane; a repeat
    comes from :func:`merge_measured`. A cell already unmeasurable stays so (its worst
    lane still counts). Two readings that differ in figure, floor or list keep the worst
    lane with no floor and no list (either list holds only its own reading's worst lane
    per entry), and a cell that lost its list never gets one back."""
    cells = MEASURED.setdefault(op, {})
    lists = DOMAINS.setdefault(op, {})
    current = cells.get(key)
    if isinstance(current, Unmeasurable):
        cells[key] = _with_measured(current, max_ulp)
        return
    new = Floored(max_ulp, *floor) if floor else max_ulp
    if current is None:
        cells[key], lists[key] = new, tuple(domains)
    elif current != new or lists.get(key) != tuple(domains):
        cells[key] = max(_max_of(new), _max_of(current))
        lists.pop(key, None)


def record_unmeasurable(
    op: MathOperation, key: BudgetKey, verdict: Unmeasurable
) -> None:
    """Record that the cell could not be measured; read twice, the cell keeps the reason
    with more non-finite lanes and the larger finite-lane figure."""
    cells = MEASURED.setdefault(op, {})
    DOMAINS.setdefault(op, {}).pop(key, None)  # no list describes a parked cell
    current = cells.get(key)
    if isinstance(current, Unmeasurable):
        worse, other = sorted(
            (current, verdict), key=lambda v: (v.nonfinite, v.why), reverse=True
        )
        verdict = _with_measured(worse, other.measured)
    elif current is not None:
        verdict = _with_measured(verdict, _max_of(current))
    cells[key] = verdict


def _with_measured(verdict: Unmeasurable, measured: int | None) -> Unmeasurable:
    """*verdict* with *measured* as its finite-lane figure if that is worse. Kept on a
    verdict of no rankable lane too, though its row will not write it (:func:`_decide`):
    dropped here, a later non-finite verdict would win with only its own figure, and the
    cell's worst lane would depend on the order the workers' readings arrived in."""
    if measured is None or (
        verdict.measured is not None and verdict.measured >= measured
    ):
        return verdict
    return replace(verdict, measured=measured)


def export_measured() -> bytes:
    """This process's measurements, for an xdist worker to hand to the controller."""
    return pickle.dumps((MEASURED, DOMAINS))


def merge_measured(blob: bytes | None) -> None:
    """Fold a worker's :func:`export_measured` into this process, worst lane winning."""
    if blob is None:
        return
    measured, domains = pickle.loads(blob)
    for op, cells in measured.items():
        for key, value in cells.items():
            if isinstance(value, Unmeasurable):
                record_unmeasurable(op, key, value)
                continue
            drawn = domains.get(op, {}).get(key, ())
            if isinstance(value, Floored):
                record(op, key, value.max_ulp, (value.residual, value.atol), drawn)
            else:
                record(op, key, value, domains=drawn)


def cell_key(input_format, output_format, approx_mode, dest_acc) -> BudgetKey:
    """The key a swept cell is recorded under."""
    return BudgetKey(
        approx_mode=approx_mode,
        input_format=input_format,
        output_format=output_format,
        dest_acc=dest_acc,
    )


class EmitRefused(RuntimeError):
    """``--ulp-emit`` refused to write; nothing was written."""


def _incomplete_grids() -> list[tuple[MathOperation, DataFormat, DataFormat, int]]:
    """Each touched ``(op, input, output)`` missing a cell :func:`sweep_cells` runs, with
    how many: a write replaces every row of a touched pair."""
    expected: dict[tuple[DataFormat, DataFormat], set[BudgetKey]] = {}
    for cell in sweep_cells():
        expected.setdefault(cell[:2], set()).add(cell_key(*cell))
    gaps = []
    for op, cells in sorted(MEASURED.items(), key=lambda item: item[0].name):
        pairs = {(key.input_format, key.output_format) for key in cells}
        for pair in sorted(pairs, key=lambda p: (p[0].name, p[1].name)):
            missing = expected.get(pair, set()) - set(cells)
            if missing:
                gaps.append((op, pair[0], pair[1], len(missing)))
    return gaps


def finish_emit(arch, clean: bool, path: Path) -> dict[MathOperation, tuple[Row, ...]]:
    """Write this session's measurements into the table at *path*
    (:func:`write_table`) and return the rows each written op holds now, or raise
    :class:`EmitRefused`. :func:`write_table` can also raise
    :class:`~helpers.ulp_domains.TableError` (a ``ValueError``): the table on disk does
    not load, or the rewrite would not read back, or would not resolve, once written."""
    if arch != MEASURED_ARCH:
        raise EmitRefused(
            f"ran on {arch.name}, but the table's unkeyed rows are "
            f"{MEASURED_ARCH.name} measurements"
        )
    if not clean:
        raise EmitRefused(
            "the session did not run clean (a failure, an interrupt or an early exit), "
            "so it measured a subset"
        )
    gaps = _incomplete_grids()
    if gaps:
        raise EmitRefused(
            "measured only part of an op's grid, and a write would drop the rows of "
            f"the rest: {gaps}"
        )
    return write_table(path)


def _verdict(measured: int, output_format: DataFormat) -> Metric:
    """``ULP`` while the measurement fits ``usable_budget_ceiling`` (the measurement is
    the gate, with no headroom), else ``TOLERANCE``."""
    if measured > usable_budget_ceiling(output_format):
        return Metric.TOLERANCE
    return Metric.ULP


@dataclass(frozen=True)
class Decision:
    """What the table should say for one measured cell: its metric and figure, the
    floor it is written with, the non-finite lanes that park it, a note, and its list.
    """

    metric: Metric
    measured: int | None
    floor: float | None = None
    nonfinite: int | None = None
    note: str | None = None
    domains: tuple[Domain, ...] = ()

    def row(self, key: BudgetKey) -> Row:
        pinned = dict(
            input=key.input_format,
            output=key.output_format,
            approx=key.approx_mode,
            dest=key.dest_acc,
        )
        if self.domains:
            return Row(**pinned, near_zero_atol=self.floor, domains=self.domains)
        tolerance = self.metric is Metric.TOLERANCE
        return Row(
            **pinned,
            metric=Metric.TOLERANCE if tolerance else None,
            measured=self.measured,
            nonfinite=self.nonfinite,
            near_zero_atol=None if tolerance else self.floor,
            note=self.note,
        )


def _decide(
    cells: dict[BudgetKey, Recorded],
    domains: dict[BudgetKey, tuple[Domain, ...]] | None = None,
) -> dict[BudgetKey, Decision]:
    """Each measured cell's :class:`Decision`. A floored cell whose worst lane demotes it
    is judged on the lanes outside its floor: enrolled at the residual, or, when even
    those are past the ceiling, demoted with the floor it tried noted."""
    lists = domains or {}
    decided = {}
    for key, value in cells.items():
        drawn = tuple(lists.get(key, ()))
        out = key.output_format
        if isinstance(value, Unmeasurable):
            # A verdict that ranked no lane records no figure: one written here would
            # sit without `nonfinite`, and the row would read as an ordinary demotion.
            decided[key] = Decision(
                Metric.TOLERANCE,
                value.measured if value.nonfinite else None,
                nonfinite=value.nonfinite or None,
                note=f"not measurable: {value.why}",
            )
        elif isinstance(value, Floored):
            plain = _verdict(value.max_ulp, out)
            if plain is Metric.ULP:
                decided[key] = Decision(plain, value.max_ulp, domains=drawn)
            elif _verdict(value.residual, out) is Metric.ULP:
                decided[key] = Decision(
                    Metric.ULP, value.residual, floor=value.atol, domains=drawn
                )
            else:
                decided[key] = Decision(
                    Metric.TOLERANCE,
                    value.max_ulp,
                    floor=value.atol,
                    note=f"{value.residual} ULP outside a {value.atol:.2e} floor",
                    domains=drawn,
                )
        else:
            decided[key] = Decision(_verdict(value, out), value, domains=drawn)
    return decided


# The key fields a collapse may drop when every value of them decides the same.
_DROPPABLE = ("approx_mode", "dest_acc")


def _collapse(decided: dict[BudgetKey, Decision]) -> dict[BudgetKey, Decision]:
    """The decided cells as the fewest rows that reproduce them; the formats stay
    pinned, because a wildcard would answer for any a driver adds later."""
    dropped = [
        name
        for name in _DROPPABLE
        if all(
            len(group) == 1
            for group in _group(decided, lambda key: replace(key, **{name: None}))
        )
    ]
    merged: dict[BudgetKey, Decision] = {}
    for key, decision in decided.items():
        collapsed = replace(key, **{name: None for name in dropped})
        if merged.get(collapsed, decision) != decision:
            # Each field's droppability is decided alone, which is sound on a full 2x2
            # grid but merges two measurements on an anti-diagonal.
            raise EmitRefused(
                f"collapsing merges two different measurements onto "
                f"{collapsed.describe()}; emit from a complete run"
            )
        merged[collapsed] = decision
    return merged


def _group(decided: dict[BudgetKey, Decision], project) -> list[set[Decision]]:
    groups: dict[BudgetKey, set[Decision]] = {}
    for key, decision in decided.items():
        groups.setdefault(project(key), set()).add(decision)
    return list(groups.values())


def _kept(row: Row, measured: set[tuple[DataFormat, DataFormat]]) -> bool:
    """A row survives a write when it pins an ``arch``, declares ``atol``/``rtol`` (a
    judgement no measurement re-derives), or names a pair this run did not measure."""
    return (
        row.arch is not None
        or row.atol is not None
        or row.rtol is not None
        or (row.input, row.output) not in measured
    )


def write_table(path: Path) -> dict[MathOperation, tuple[Row, ...]]:
    """Replace, in the table at *path*, every row of each measured ``(input, output)``
    pair with what the sweep measured, write it back, and return the written ops'
    rows. A kept row with the key of an emitted row is refused (:class:`EmitRefused`),
    so an emitted row always outranks a kept row pinning its pair."""
    table = load_table(path)
    written: dict[MathOperation, tuple[Row, ...]] = {}
    for op, cells in MEASURED.items():
        pairs = {(key.input_format, key.output_format) for key in cells}
        kept = tuple(row for row in table.get(op, ()) if _kept(row, pairs))
        taken = {row.key for row in kept}
        emitted = tuple(
            decision.row(key)
            for key, decision in _collapse(_decide(cells, DOMAINS.get(op))).items()
        )
        for row in emitted:
            if row.key in taken:
                # A kept row never displaces a measurement, nor is silently displaced.
                raise EmitRefused(
                    f"{op.name} {row.key.describe()}: a kept row has the key this emit "
                    "writes; give it a key no measured cell collapses to"
                )
        written[op] = table[op] = kept + emitted
    write_table_file(table, Path(path))
    return written

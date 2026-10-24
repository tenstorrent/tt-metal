# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive 16-bit ULP sweeps, and folding what they measure back into the table.

The functional drivers sample a few thousand points from an op's *safe* domain. A budget
measured that way describes the sample, not the format, and cannot see a tail the sample
never reaches. This module measures the format's number.

Exhaustive is only honest for the 16-bit formats. bfloat16 has 65,025 normal values
(zero included) and float16 61,441, so either fits one 64-tile device run; Float32's 2**32
does not, so it is walked with a stride that gives every binade an equal share. The
subnormals are not fed at all: the unpack path flushes them, and a golden that does not
would be measuring the flush rather than the op. Only the formats the SFPU computes in
are swept: a block float's error is the unpacker's and packer's, not the op's.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec
from helpers.ulp_domains import (
    BINADE_OFFSET,
    Bin,
    Domain,
    entries_of,
    usable_budget_ceiling,
)
from helpers.ulp_table import (
    MEASURED_ARCH,
    TABLE_PATH,
    BudgetKey,
    Metric,
    Reading,
    Row,
    load_table,
    write_table_file,
)

#: The formats the sweep feeds and judges, in and out: the ones the SFPU computes in.
#: Each is enumerated whole as an input, except Float32, which is strided
#: (`is_exhaustive`).
SWEEP_FORMATS: Tuple[DataFormat, ...] = (
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
)

#: Float32 has 2**32 values and one run holds 2**16, so it is the one input the sweep
#: samples rather than enumerates. Striding the total order gives every binade an equal
#: share (each holds the same number of values); a consecutive walk of one run's 2**16
#: would cover 1/128 of one binade's 2**23. One sample per normal bfloat16 cell, 65,024
#: of them, but not at the cell's start: the walk starts on -inf, a multiple of 2**16, so
#: every sample would be a bfloat16 value -- the Float16_b sweep again -- and an fp32
#: path that reads the low 16 mantissa bits (a LUT index, a truncating convert) would go
#: untested. The walk moves each sample along its cell by 1/phi of a cell from the one
#: before (`_enumerate_fp32_in_range`), so the low half is never zero and lands in either
#: half of the cell about equally often within every binade. A stride
#: of 2**16 + 1 did not: it tied the low half to the high, and all 128 samples of a
#: binade sat at one point of their cells.
_FP32_STRIDE = 2**16

#: One 64-tile run: the most values a sweep variant generates.
_SWEEP_TENSOR = 2**16

_INF = float("inf")


def sweep_cells(arch=None) -> List[Tuple[DataFormat, DataFormat, object, object]]:
    """Every ``(input, output, approx_mode, dest_acc)`` cell the sweep runs on *arch*
    (default: the chip this session targets).

    Less the cells ``TestConfig`` would silently run as another: on Wormhole and
    Blackhole an exponent-B input packed to Float16 needs a 32-bit Dest, so a
    ``dest_acc=No`` request runs the ``Yes`` kernel. Measuring it under ``No`` would
    judge the hardware against a golden modelling a 16-bit Dest, and record the result
    under a key that kernel never ran with.

    A host check of the Wormhole table passes ``MEASURED_ARCH``, so its verdict does not
    depend on which ``CHIP_ARCH`` the host happens to set.

    The grid is Wormhole's. On another arch only the Dest promotion is applied, not that
    arch's own format support: Blackhole skips a ``Float16`` input with ``dest_acc=No``
    (``_skip_if_unsupported`` in accuracy/test_sfpu_accuracy.py). That is inert while
    every cell off Wormhole resolves to tolerance and skips, which holds because no ULP
    row names another ``arch``; test_ulp_sweep.py pins that, so the first such row has
    to bring those filters with it.
    """
    from helpers.chip_architecture import get_chip_architecture
    from helpers.data_format_inference import effective_dest_acc
    from helpers.llk_params import ApproximationMode, DestAccumulation

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


def is_exhaustive(input_format: DataFormat) -> bool:
    """Whether the sweep sees *every* normal value the input can take (and zero), or a
    stride of them."""
    return input_format != DataFormat.Float32


def _stride_for(input_format: DataFormat) -> int:
    return 1 if is_exhaustive(input_format) else _FP32_STRIDE


@lru_cache(maxsize=None)
def swept_value_count(input_format: DataFormat) -> int:
    """How many values the sweep generates for *input_format* -- not how many the
    format has, which for float32 is 2**32 against the 2**16 generated.

    Cached: finding it enumerates the whole format, and `padding_lanes` asks twice per
    variant.
    """
    from helpers.stimuli_generator.strategies.structured import (
        _enumerate_representable,
    )

    stride = _stride_for(input_format)
    return int(
        _enumerate_representable(
            input_format, -_INF, _INF, _SWEEP_TENSOR, stride=stride, normal_only=True
        ).numel()
    )


def sweep_spec(input_format: DataFormat) -> StimuliSpec:
    """Every normal representable value of the stimuli format and its zero, once -- or,
    for float32, every ``_FP32_STRIDE``-th, since 2**32 values do not fit one run. No
    subnormal is fed: the unpack path flushes it, so a lane there would measure the
    flush, not the op.

    Deliberately not clipped to the op's domain. ``exclude_undefined`` expresses a domain
    as ``intervals``, which ULP_SWEEP does not read -- and clipping would also stop the
    undefined inputs reaching hardware at all. They are swept and then masked out of the
    statistics by :func:`measurable_mask`, so the run still exercises them.
    """
    return StimuliSpec.ulp_sweep(
        low=-_INF, high=_INF, stride=_stride_for(input_format), normal_only=True
    )


def padding_lanes(src: torch.Tensor, input_format: DataFormat) -> torch.Tensor:
    """The tail ``generate_full_tensor`` fills with zeros to reach the tile count.

    The sweep enumerates every normal value of the stimuli format and its zero --
    65,025 for bfloat16, 61,441 for float16 -- into a fixed 65,536-lane tensor, so the
    last 511 (or 4,095) lanes are padding rather than data. They are not values the
    sweep chose to feed, and they are all the same one, so they belong in no
    statistic: they inflate every lane count, and on an op singular at zero they land
    on the pole: ``reciprocal`` returns ``Inf`` there, golden and hardware alike, so the
    lanes are not a mismatch but they are 511 copies of one value the sweep never chose.

    Identified by position rather than by value, because ``0.0`` is also a legitimate
    swept value: exactly one, in the middle of the sorted order. Confirmed on hardware
    that the padding is the contiguous tail.

    Counted by :func:`swept_value_count`, not by how many values the format has: a
    strided float32 walk generates 65,024 of 2**32, and asking the format would put the
    padding boundary past the end of the tensor and mask nothing.
    """
    swept = swept_value_count(input_format)
    # On *src*'s device: the mask is composed with tensors derived from it, and a
    # CPU-only mask would fail that composition for a device-resident sweep.
    flat = torch.zeros(src.numel(), dtype=torch.bool, device=src.device)
    flat[swept:] = True
    return flat.reshape(src.shape)


def received_inputs(
    src: torch.Tensor, input_format: DataFormat, output_format=None, dest_acc=None
) -> torch.Tensor:
    """*src* as the op receives it: the tensor itself, except, given the variant, a
    Float32 input at ``dest_acc=No``, which the unpack and the golden both truncate to
    the 16-bit Dest: past 256 every truncated bf16 value is an integer, so -8326436.5
    arrives as the pole -8323072. The lane checks that ask where an input *is* -- on a
    singularity, inside the op's claim, inside a tracked issue's lanes -- have to ask it
    of this value, not of the one generated.
    """
    from helpers.sfpu_domains import _dest_acc_flag

    if (
        input_format != DataFormat.Float32
        or dest_acc is None
        or _dest_acc_flag(dest_acc)
    ):
        return src
    from helpers.golden_generators import truncate_to_dest_width

    width = _unpack_format(
        input_format, output_format or DataFormat.Float16_b, dest_acc
    )
    return truncate_to_dest_width(src.detach().to(torch.float32), width)


def golden_input(src: torch.Tensor, input_format: DataFormat, dest_acc) -> torch.Tensor:
    """*src* as the golden should see it: with its zeros made +0.0 where the unpack
    drops the sign, so the golden is computed on what the kernel receives.

    The walk's one data zero is -0.0, and the kernel is handed +0.0 -- except on the
    unpack-to-dest path, which keeps it (``negative_zero_delivered``, the same rule the
    edge tests use rather than a second copy of it). Otherwise an op whose finite answer
    depends on the sign of zero reads one lane as a whole-cell error that is the
    unpack's, not the op's: ``signbit`` answers 1.0 against 0.0, 16129 bf16 steps. (A
    pole's ``-inf`` against ``+inf``, ``rsqrt(-0)``, was never a failure here: the
    singularity exclusion drops it.) The dedicated signed-zero tests in
    test_eltwise_unary_sfpu.py hold that path to account.
    """
    from helpers.sfpu_domains import negative_zero_delivered

    if negative_zero_delivered(input_format, dest_acc):
        return src
    return torch.where(src == 0, torch.zeros_like(src), src)


def _unpack_format(input_format: DataFormat, output_format=None, dest_acc=None):
    """The format whose values the unpacker writes for this cell: a Float32 input into
    a Float16 output at ``dest_acc=No`` lands in a Float16 Dest. Without
    *output_format* and *dest_acc*, the input's own."""
    if output_format is None or dest_acc is None:
        return input_format
    from helpers.data_format_inference import infer_unpack_out
    from helpers.sfpu_domains import unpacks_to_dest

    return infer_unpack_out(
        input_format,
        output_format,
        dest_acc,
        unpacking_to_dest=unpacks_to_dest(input_format, dest_acc),
    )


def _dest_dtype(input_format: DataFormat, output_format=None, dest_acc=None):
    """The torch dtype of the format the unpacker writes for this variant -- the lattice
    the SFPU reads its input from and stores its answer to (:func:`_unpack_format`) --
    or ``None`` when the caller did not name the variant, or that format has no float
    dtype."""
    from helpers.llk_params import format_dict

    if output_format is None or dest_acc is None:
        return None
    dtype = format_dict.get(_unpack_format(input_format, output_format, dest_acc))
    return dtype if dtype is not None and dtype.is_floating_point else None


def dest_holds(
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
    output_format=None,
    dest_acc=None,
) -> torch.Tensor:
    """False on lanes an fp16 Dest (``dest_acc=No``) under a wider output lost: a golden
    past 65504 answered by NaN or by the golden's sign in the Dest's top binade, or a
    flush at the 2**-14 boundary (one side zero, the other at most 2**-14). Any other
    answer -- wrong-signed, or more than 2x short of the edge -- is the op's."""
    from helpers.llk_params import format_dict
    from helpers.sfpu_domains import _dest_acc_flag

    holds = torch.ones(golden.shape, dtype=torch.bool, device=golden.device)
    # A 32-bit Dest at dest_acc=Yes, and a bf16 one (fp32's exponent range), hold
    # every answer the output can.
    if dest_acc is None or _dest_acc_flag(dest_acc):
        return holds
    dtype = _dest_dtype(input_format, output_format, dest_acc)
    if dtype is not torch.float16:
        return holds
    dest = torch.finfo(dtype)
    out = torch.finfo(format_dict[output_format])
    if dest.max >= out.max and dest.tiny <= out.tiny:
        return holds
    g = golden.detach().to(torch.float32)
    r = result.detach().to(torch.float32)
    same_sign = torch.signbit(r) == torch.signbit(g)
    # Past 65504 the golden is a stand-in (131008 for an fp16 inf), not the answer, so an
    # answer near the edge cannot be ranked against it: I0(13.297) is 65772 and the
    # kernel's 65440 is its usual 0.5%, which the stand-in reads as 8.4M fp32 steps.
    top_binade = 2.0 ** (math.frexp(dest.max)[1] - 1)
    overflowed = (g.abs() > dest.max) & (
        torch.isnan(r) | (same_sign & (r.abs() >= top_binade))
    )
    # The generator zeroes a golden under 2**-14 before its fp16 cast, so the flushes
    # the sweep sees are pairs across the boundary: 2**-14 against 0, or the reverse.
    flushed = ((g == 0) & (r != 0) & (r.abs() <= dest.tiny)) | (
        (r == 0) & (g != 0) & (g.abs() <= dest.tiny)
    )
    return holds & ~overflowed & ~flushed


def _normal_input(
    src: torch.Tensor, input_format: DataFormat, output_format=None, dest_acc=None
) -> torch.Tensor:
    """Lanes whose input survives the unpack: zero, or at least the smallest normal of
    the stimuli format *and* of the format the unpacker writes, and no larger than that
    format can hold. The sweep feeds no subnormal of the stimuli format
    (:func:`sweep_spec`), so only the unpack format's cutoff bites, and it matters once:
    a Float32 input into a Float16 output at ``dest_acc=No`` unpacks into a Float16
    Dest, which flushes everything below 2**-14 -- a Float32 input's own cutoff is
    1.18e-38, so the strided lanes in between scored ``f(x)`` against ``f(0)`` as op
    error (Floor's Float32 -> Float16 cell read 15360 steps, the distance to 1.0).
    Without *output_format* and *dest_acc*, only the stimuli format's cutoff applies.
    """
    from helpers.llk_params import format_dict

    fed = torch.finfo(format_dict[input_format])
    cutoff = fed.smallest_normal
    ceiling = math.inf
    dtype = format_dict.get(_unpack_format(input_format, output_format, dest_acc))
    if dtype is not None and dtype.is_floating_point:
        unpacked = torch.finfo(dtype)
        cutoff = max(cutoff, unpacked.smallest_normal)
        # The other end of the same unpack: a Float32 input past fp16's range
        # saturates to +-65504 on the way into a Float16 Dest, while the golden, which
        # takes the value as fed, sees an infinity (asinh(-3.4e38) read -inf against
        # the kernel's -11.8). Only where the unpack's exponent range is narrower: a
        # bf16 Dest has fp32's, and fp32's largest values reach it as bf16's largest
        # whether the unpack truncates or rounds -- nothing overflows there.
        if math.frexp(unpacked.max)[1] < math.frexp(fed.max)[1]:
            ceiling = float(unpacked.max)
    # In float32, and from `src` as generated. Casting to the golden's dtype first
    # rounds an fp16 subnormal *up* -- bf16 keeps 8 mantissa bits, so 6.09e-05 becomes
    # 6.10e-05 and clears a 6.10e-05 threshold. The whole subnormal band then passed
    # this filter while looking, in any printout, like the smallest normal.
    magnitude = src.detach().to(torch.float32).abs()
    return ((magnitude >= cutoff) | (magnitude == 0)) & (magnitude <= ceiling)


def measurable_mask(
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
    output_format=None,
    dest_acc=None,
) -> torch.Tensor:
    """Lanes of an all-finite-input sweep that a *step count* can describe.

    Op-agnostic on purpose: an op undefined at an input lands in the NaN kind on its
    own, so there is nothing per-op to look up.

    The sweep feeds every normal value of the format and its zero (a stride of them for
    Float32), with no per-op domain clipping -- an op is measured wherever its format can reach. Five lane kinds come
    back out, none of them a budget question (the fourth is two, on one boundary):

    * either side NaN. :func:`ulp_distance` returns ``UNMEASURABLE`` there, and an op
      undefined at an input (``log`` of a negative) lands here on its own.
    * the two sides disagreeing about being non-finite -- a reciprocal overflowing where
      the golden is still finite. ``passed_test`` rejects those positionally whatever the
      budget says, so ranking them would inflate the number without tightening the gate.
      One such lane is worth ~48,000 steps.
    * the sweep's own zero padding -- see :func:`padding_lanes`.
    * inputs the unpack flushes (:func:`_normal_input`). The sweep feeds no subnormal of
      its own -- the hardware flushes them on the way in and the golden does not, so
      ``ceil(5.69e-39)`` was 1 in the model and 0 on silicon, 16,129 bf16 steps for a
      difference that is the unpack path's flush, not the op's accuracy. The flush is
      covered on its own terms elsewhere; a step count is the wrong instrument for it.
      The same boundary has a top on a narrower unpack: a Float32
      input into a Float16 output at ``dest_acc=No`` lands in a Float16 Dest, which
      flushes below 2**-14 and *saturates* past 65504 -- about 44% of the strided lanes,
      whose golden sees the value as fed. Saturation, not a flush, and not the op's.

    * answers the Dest lost (:func:`dest_holds`): an fp16 Dest's overflow past 65504
      or its flush at 2**-14 under a wider output, which is the Dest's, not the op's.

    Subnormal *outputs* stay in. Where the golden underflows and the hardware writes
    zero the count is large but the lane is a real one the op produced -- Silu at
    ``x=-87.5`` is that case, and it is the op's own tail, not the unpack path.

    Op-agnostic, so the op's *claim* is not read here: :func:`ranked_mask` ANDs
    :func:`claimed_lanes` into this mask, so a lane past an argument-reduction limit
    or on a pole sets no budget, just as it is no non-finite failure.

    The second kind is a *failure*, not a non-question, and dropping it here is only
    sound because :func:`nonfinite_failures` reports it separately -- less the lanes it
    excuses: an input the op makes no claim on (``sin(2.6e28)``, past Sin's
    argument-reduction limit on a bfloat16 input) and a lane a tracked issue names
    (:data:`_KNOWN_NONFINITE_LANES`). A caller that ranks this mask and nothing else
    would let a hardware overflow produce a clean budget.
    """
    # The threshold is the stimuli (and unpack) format's, not the golden's. Taking it
    # from the golden dtype silently passed every fp16 subnormal through on a
    # Float16->Float16_b variant -- bf16's smallest normal is 1.18e-38 and fp16's is
    # 6.1e-05, so 2,046 flushed lanes read as a 14,337-step error on `Abs`, an op that
    # cannot be wrong. See `_normal_input`.
    from .ulp import nonfinite_mismatches

    normal_input = _normal_input(src, input_format, output_format, dest_acc)

    both_measurable = ~(torch.isnan(golden) | torch.isnan(result))
    return (
        both_measurable
        & ~nonfinite_mismatches(golden, result)
        & normal_input
        & dest_holds(golden, result, input_format, output_format, dest_acc)
        & ~padding_lanes(src, input_format)
    )


def ranked_mask(
    op,
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
    output_format=None,
    dest_acc=None,
) -> torch.Tensor:
    """The lanes a sweep ranks: :func:`measurable_mask` within *op*'s :func:`claimed_lanes`."""
    return measurable_mask(
        src, golden, result, input_format, output_format, dest_acc
    ) & claimed_lanes(op, src, input_format, output_format, dest_acc)


#: Where Sin, Cos and Tan's range reduction breaks (0-2 steps inside, inf/NaN past it).
#: It equals fp16's max, so an fp16 input -- or one unpacked into an fp16 Dest -- has none.
_TRIG_CLAIM_LIMIT = 65504.0

#: ``{op: {format the kernel is handed (:func:`_unpack_format`): limit}}``, bounding
#: both the ranking mask and the non-finite check.
_CLAIM_LIMIT: Dict = {}


def _claim_limits() -> Dict:
    from helpers.llk_params import MathOperation

    if not _CLAIM_LIMIT:
        wide_formats = {
            DataFormat.Float16_b: _TRIG_CLAIM_LIMIT,
            DataFormat.Float32: _TRIG_CLAIM_LIMIT,
        }
        _CLAIM_LIMIT.update(
            {
                MathOperation.Sin: wide_formats,
                MathOperation.Cos: wide_formats,
                MathOperation.Tan: wide_formats,
            }
        )
    return _CLAIM_LIMIT


def claimed_lanes(
    op,
    src: torch.Tensor,
    input_format: DataFormat,
    output_format=None,
    dest_acc=None,
) -> torch.Tensor:
    """Lanes where *op* claims an answer, judged on the input as received: all but its
    singularities' undefined side, its poles, and anything past its ``_CLAIM_LIMIT``."""
    from helpers.sfpu_domains import (
        _NONPOSITIVE_INTEGER_POLES,
        _OP_SINGULARITIES,
        Operand,
        SingularitySide,
    )

    value = received_inputs(src, input_format, output_format, dest_acc).to(
        torch.float32
    )
    claimed = torch.ones_like(value, dtype=torch.bool)
    for point, side in _OP_SINGULARITIES.get(op, {}).get(Operand.A, ()):
        # The defined side keeps the point: sqrt(0) and acosh(1) are answers. A golden
        # that is infinite there (log(0), atanh(1)) is out of range and dropped anyway.
        if side is SingularitySide.ABOVE:
            claimed &= value >= point
        elif side is SingularitySide.BELOW:
            claimed &= value <= point
        else:
            claimed &= value != point
    if op in _NONPOSITIVE_INTEGER_POLES:
        # Every fp32 value of magnitude 2**23 or more is an integer, so for lgamma the
        # whole negative tail past there is poles; on the pole itself the golden is a
        # limit (torch's polygamma returns 4e15 at -6.0), and the kernel's inf is not
        # what the cell measures.
        claimed &= ~((value <= 0) & (value == torch.floor(value)))
    limit = (
        _claim_limits()
        .get(op, {})
        .get(_unpack_format(input_format, output_format, dest_acc))
    )
    if limit is not None:
        claimed &= value.abs() <= limit
    return claimed


def _at_a_singularity(
    op, src: torch.Tensor, input_format: DataFormat, output_format=None, dest_acc=None
) -> torch.Tensor:
    """Lanes received (:func:`received_inputs`) exactly on one of *op*'s
    ``_OP_SINGULARITIES`` points, either side."""
    from helpers.sfpu_domains import _OP_SINGULARITIES, Operand

    value = received_inputs(src, input_format, output_format, dest_acc).to(
        torch.float32
    )
    on_point = torch.zeros_like(value, dtype=torch.bool)
    for point, _side in _OP_SINGULARITIES.get(op, {}).get(Operand.A, ()):
        on_point |= value == point
    return on_point


@dataclass(frozen=True)
class KnownNonfiniteLanes:
    """Inputs on which an op is known to answer on the wrong side of infinity on one
    cell, tracked by an issue.

    One such lane used to park the whole cell -- ~64,000 lanes -- as ``not measurable``
    on the tolerance metric, which the gate then skips outright: Celu lost its
    Float16->Float16 step gate over four inputs. Naming the inputs instead keeps the rest
    of the cell on its budget. Only the *non-finite disagreement* is excused: a named
    lane that agrees is ranked like any other, the entry applies to no cell but the ones
    it pins, and a gate run fails once no named lane of a cell disagrees any more
    (:func:`stale_excuses`), so an entry cannot outlive the defect it tracks.
    """

    #: The tracking issue, ``#NNNNN``.
    issue: str
    #: The input formats the entry holds on.
    inputs: Tuple[DataFormat, ...]
    output: DataFormat
    #: Inclusive bounds on the input as received (:func:`received_inputs`) -- on ``|x|``
    #: when *magnitude* is set.
    low: float
    high: float
    #: ``ApproximationMode`` / ``DestAccumulation``, or ``None`` for either value.
    approx: Optional[object] = None
    dest: Optional[object] = None
    magnitude: bool = False
    why: str = ""

    def applies_to(self, input_format, output_format, approx_mode, dest_acc) -> bool:
        return (
            input_format in self.inputs
            and output_format == self.output
            and (self.approx is None or approx_mode == self.approx)
            and (self.dest is None or dest_acc == self.dest)
        )

    def lanes(self, received: torch.Tensor) -> torch.Tensor:
        value = received.detach().to(torch.float32)
        if self.magnitude:
            value = value.abs()
        return (value >= self.low) & (value <= self.high)


#: ``{op: (KnownNonfiniteLanes, ...)}``. Read through :func:`_known_lanes`; the enum
#: imports are deferred like ``_CLAIM_LIMIT``'s.
_KNOWN_NONFINITE_LANES: Dict = {}


def _known_lanes() -> Dict:
    from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation

    if _KNOWN_NONFINITE_LANES:
        return _KNOWN_NONFINITE_LANES
    # #58607: on a 16-bit Float16 Dest these ops answer inf where the answer is one of
    # the four largest fp16 values, 65408..65504. The same inputs on a 32-bit Dest read
    # 1-2 steps, and Abs/Identity read 0 on the same cell, so it is neither the input
    # nor the store alone. Float16 inputs only: the strided Float32 walk has no lane in
    # the band, so a Float32 entry would excuse nothing and read as stale.
    top_of_fp16 = dict(
        issue="#58607",
        inputs=(DataFormat.Float16,),
        output=DataFormat.Float16,
        dest=DestAccumulation.No,
        low=65408.0,
        high=65504.0,
        why="inf where the answer is x itself, in the top four fp16 values, on a 16-bit Dest",
    )
    # The same top-of-fp16 window for an op that answers 65408 finite: #58607's lanes
    # start one fp16 step above it, at 65440.
    above_65408 = {**top_of_fp16, "low": 65440.0}
    _KNOWN_NONFINITE_LANES.update(
        {
            MathOperation.Celu: (KnownNonfiniteLanes(**top_of_fp16),),
            MathOperation.Elu: (KnownNonfiniteLanes(**top_of_fp16),),
            MathOperation.Gelu: (
                KnownNonfiniteLanes(**top_of_fp16, approx=ApproximationMode.No),
            ),
            MathOperation.GeluTanh: (KnownNonfiniteLanes(**top_of_fp16),),
            MathOperation.Mish: (KnownNonfiniteLanes(**top_of_fp16),),
            # Silu answers 65408 itself; #58607 lists only the three lanes above it.
            MathOperation.Silu: (KnownNonfiniteLanes(**above_65408),),
            # The same band reached through the op: selu(x) = 1.0507 x, xielu(x) ~ x*x.
            MathOperation.Selu: (
                KnownNonfiniteLanes(
                    **{
                        **top_of_fp16,
                        "low": 62272.0,
                        "high": 62336.0,
                        "why": "inf where 1.0507 x is 65440..65504, on a 16-bit Dest",
                    }
                ),
            ),
            MathOperation.Xielu: (
                KnownNonfiniteLanes(
                    **{
                        **top_of_fp16,
                        "low": 255.5,
                        "high": 255.625,
                        "why": "inf where the answer is 65408 or 65472, on a 16-bit Dest",
                    }
                ),
            ),
            MathOperation.UnaryPower: (
                KnownNonfiniteLanes(
                    **{
                        **top_of_fp16,
                        "low": 255.625,
                        "high": 255.875,
                        "magnitude": True,
                        "why": "inf (NaN at |x|=255.875) where x*x is 65344..65472, on a 16-bit Dest",
                    }
                ),
            ),
            MathOperation.Square: (
                KnownNonfiniteLanes(
                    **{
                        **top_of_fp16,
                        "low": 255.75,
                        "high": 255.875,
                        "magnitude": True,
                        "why": "inf where x*x is 65408 or 65472, on a 16-bit Dest",
                    }
                ),
            ),
            # Not in #58607's first table; recorded there since (issuecomment-5999436777):
            # +-65440..65504, and 65408 answers finite as it does for Silu.
            MathOperation.Tanhshrink: (
                KnownNonfiniteLanes(
                    **{
                        **above_65408,
                        "magnitude": True,
                        "why": "inf where the answer is x - tanh(x), in the top fp16 values, on a 16-bit Dest",
                    }
                ),
            ),
            # The same band reached through lgamma: lgamma(x) rounds to 65408..65504 in
            # fp16 for x in [8168, 8180], bounded by the answer as the band is; measured,
            # 8172 and 8176 read inf where the answer is 65440 and 65472.
            MathOperation.Lgamma: (
                KnownNonfiniteLanes(
                    **{
                        **top_of_fp16,
                        "low": 8168.0,
                        "high": 8180.0,
                        "why": "inf where lgamma(x) is a top fp16 value, on a 16-bit Dest",
                    }
                ),
            ),
            # #50465: i0 is a bare Taylor series with no large-|x| branch, so where i0
            # first overflows fp16 -- i0(13.296875) is 65772, the Float16 golden inf --
            # the truncated series answers finite. The one fp16 input in that gap; past
            # it the series overflows too.
            MathOperation.I0: (
                KnownNonfiniteLanes(
                    issue="#50465",
                    inputs=(DataFormat.Float16,),
                    output=DataFormat.Float16,
                    low=13.296875,
                    high=13.296875,
                    magnitude=True,
                    why="finite where i0(x) has just overflowed fp16: the series has no large-|x| branch",
                ),
            ),
            # #57215: the Float16 store carries exactly 2**16 to inf (and every larger
            # value to NaN). Approximate sqrt lands on 2**16 exactly where the answer
            # rounds to 65504. The store's *clamp* of (65504, 2**16) to 65504 is the
            # same issue but needs no entry: `nonfinite_failures` reads it as saturation.
            # Bounded by where sqrt(x) rounds to 65504 in fp16, [65488**2, 65520**2],
            # not by the one lane a walk happens to put there.
            MathOperation.Sqrt: (
                KnownNonfiniteLanes(
                    issue="#57215",
                    inputs=(DataFormat.Float32,),
                    output=DataFormat.Float16,
                    approx=ApproximationMode.Yes,
                    dest=DestAccumulation.Yes,
                    low=65488.0**2,
                    high=65520.0**2,
                    why="inf where sqrt(x) rounds to 65504: the approximation lands on 2**16, the one value the store carries to inf",
                ),
            ),
        }
    )
    return _KNOWN_NONFINITE_LANES


def known_nonfinite_lanes(
    op,
    src: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    *,
    approx_mode,
    dest_acc,
) -> torch.Tensor:
    """The lanes of *src* a :data:`_KNOWN_NONFINITE_LANES` entry names on this cell.

    The cell's two enums are keyword-only, here and in :func:`stale_excuses` and
    :func:`nonfinite_failures`: a swap raises nothing (the enums just compare unequal),
    and in ``stale_excuses`` it would silently turn the stale-entry gate off."""
    received = received_inputs(src, input_format)
    excused = torch.zeros(src.shape, dtype=torch.bool, device=src.device)
    for entry in _known_lanes().get(op, ()):
        if entry.applies_to(input_format, output_format, approx_mode, dest_acc):
            excused |= entry.lanes(received)
    return excused


def stale_excuses(
    op,
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    *,
    approx_mode,
    dest_acc,
) -> List[KnownNonfiniteLanes]:
    """The entries that apply to this cell but excuse nothing: without them, no lane
    they name would be a non-finite failure -- because the lanes agree with the golden
    again, or because another exclusion already covers them. The gate fails on one, so
    a defect being fixed, or a rule that subsumes the entry, retires it rather than
    leaving dead excuses in the list."""
    without = nonfinite_failures(
        op,
        src,
        golden,
        result,
        input_format,
        output_format,
        dest_acc=dest_acc,
        approx_mode=approx_mode,
        known_lanes=False,
    )
    received = received_inputs(src, input_format)
    return [
        entry
        for entry in _known_lanes().get(op, ())
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
    dest_acc=None,
    approx_mode=None,
    known_lanes: bool = True,
) -> torch.Tensor:
    """The lanes :func:`measurable_mask` drops that are a *failure* rather than a
    non-question: the two sides disagreeing about being non-finite where the output
    format could have held the answer.

    *dest_acc* also sets the unpack's cutoff and ceiling (:func:`_normal_input`): left
    unset, only the stimuli format's own cutoff applies. With *approx_mode* it names the
    cell for :data:`_KNOWN_NONFINITE_LANES`, where an unset one matches only an entry
    that pins neither. *known_lanes* False leaves those entries out altogether, which is
    how :func:`stale_excuses` asks what they buy.

    ``passed_test`` rejects these positionally whatever the budget says, but the sweep
    driver ranks a distance rather than calling it, so it has to ask separately -- a
    hardware overflow or an unexpected NaN would otherwise leave the statistics clean
    and both emit and gate would pass.

    The exclusions, and whose doing each one is:

    * **flushed inputs** (:func:`_normal_input`), on the same grounds as in the mask --
      the unpack path flushes a subnormal and the golden does not, so a disagreement
      there is the flush.
    * **a NaN golden**, and **a golden past the output format's range answered by a
      saturated store** -- ``NaN``, an infinity of the golden's sign, or on a Float16
      output the pack's clamp to +-65504 of that sign. A full-range
      sweep feeds every value of a 16-bit input, and ``relu_min`` passes most of them
      straight through, so a bf16 input against a Float16 output reaches magnitudes fp16
      cannot represent -- 14,334 lanes of it. Saturating there is the store doing what
      it must (on WH an fp16 destination overflow packs NaN, not Inf), not the kernel
      being wrong, and no budget on any op could be met. A *finite* or wrong-signed
      answer against such a golden is still judged: ``exp`` past overflow returning
      3.39e38 where the golden is ``+inf`` is the kernel being wrong, and the ranking
      mask drops that lane too.
    * **an infinite golden on a registered singularity point itself**, as received --
      ``log(0)``, ``rsqrt(0)``, ``atanh(-1)``. The value there is a limit, not a number,
      and what the pipeline makes of it is not the op's accuracy: the unpack drops the
      sign of ``-0.0``, so ``rsqrt(-0)`` answers ``+inf`` against ``-inf``, and a 16-bit
      fp16 Dest has no infinity, so ``log(0)`` answers -130560. Only the point: one step
      off it the op claims a finite answer again.
    * **the sweep's own zero padding**, which is not a value it chose to feed.
    * **an answer the Dest lost** (:func:`dest_holds`), on the same grounds as in the
      mask: an fp16 Dest's overflow under a wider output, answered with the golden's
      sign at the Dest's edge. A wrong-signed or finite small answer stays.
    * **an input the op makes no claim on** (:func:`claimed_lanes`): the op's own limit
      rather than the sweep's -- the undefined side of a registered singularity, or
      past an argument-reduction limit on a format that reaches it. ``Sin`` and ``Cos``
      disagree on bf16 lanes far past 65504. The sweep ranks the claimed lanes only,
      so the budget and this check agree on what the op answers for.
    * **a lane a tracked issue names** (:data:`_KNOWN_NONFINITE_LANES`): a defect
      already on the books, excused on its own cell and inputs so the rest of the cell
      keeps its step gate. The gate fails once the lanes stop disagreeing
      (:func:`stale_excuses`).

    What is left is the case the mask would otherwise hide: an op returning ``inf`` or
    ``NaN`` where it is defined, the input is normal, and the output could have held
    the answer -- or a finite answer where the answer is infinite.
    """
    from helpers.llk_params import format_dict

    from .ulp import nonfinite_mismatches

    normal_input = _normal_input(src, input_format, output_format, dest_acc)
    output_max = torch.finfo(format_dict[output_format]).max
    # `golden` is usually already in the output dtype, so "past the range" is mostly an
    # infinity; a wider golden can also be finite and past it. NaN compares false here.
    past_range = golden.detach().to(torch.float32).abs() > output_max
    same_sign = torch.signbit(result) == torch.signbit(golden)
    saturated = torch.isnan(result) | (torch.isinf(result) & same_sign)
    if output_format == DataFormat.Float16:
        # The fp16 pack clamps an out-of-range value from a wider Dest to +-65504 (or
        # packs NaN, above): x + 1 at -66048 reads -65504 against the golden's -inf.
        # A bfloat16 pack cannot overflow that way -- its range is the Dest's.
        saturated |= same_sign & (result.detach().to(torch.float32).abs() == output_max)
    excused = torch.isnan(golden) | (
        past_range
        & (
            saturated
            | _at_a_singularity(op, src, input_format, output_format, dest_acc)
        )
    )
    return (
        nonfinite_mismatches(golden, result)
        & normal_input
        & dest_holds(golden, result, input_format, output_format, dest_acc)
        & ~excused
        & claimed_lanes(op, src, input_format, output_format, dest_acc)
        & ~(
            known_nonfinite_lanes(
                op,
                src,
                input_format,
                output_format,
                approx_mode=approx_mode,
                dest_acc=dest_acc,
            )
            if known_lanes
            else torch.zeros_like(src, dtype=torch.bool)
        )
        & ~padding_lanes(src, input_format)
    )


#: How many offending lanes a non-finite verdict spells out. Two: the row's `nonfinite`
#: count says how many there are, two lanes show the kind (an overflow, a NaN, a wrong
#: sign), and more would only lengthen every not-measurable row's `note`.
NAMED_LANES = 2


@dataclass(frozen=True)
class Unmeasurable:
    """A cell the sweep could not describe with a step count, as the emitter writes it:
    a tolerance row with these figures and *why* in its ``note``.

    *nonfinite* is how many lanes disagreed with the golden about being finite -- what
    makes the cell unmeasurable, and the count the headroom report (from #57527) fails a
    run on when it grows; 0 for a cell with no rankable lane and no non-finite one
    either. *measured* is the worst lane over the lanes that *could* be ranked --
    usually tens of thousands of them, which a demotion used to throw away, so a cell
    parked by a handful of lanes recorded no step count for the rest of it; ``None``
    when there were none."""

    why: str
    nonfinite: int = 0
    measured: Optional[int] = None


def nonfinite_verdict(
    overflowed: torch.Tensor,
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    stats: Dict,
    named_lanes: int = NAMED_LANES,
) -> Unmeasurable:
    """The verdict a cell with non-finite disagreements is written with. *named_lanes*
    bounds how many offending inputs are spelled out."""
    named = "; ".join(
        f"x={float(src[i]):g}: {float(golden[i]):g} -> {float(result[i]):g}"
        for i in overflowed.nonzero().flatten()[:named_lanes].tolist()
    )
    return Unmeasurable(
        why="lanes disagreeing with the golden about being finite "
        f"(golden -> result: {named})",
        nonfinite=int(overflowed.sum()),
        # Over no lane at all `ulp_stats` reads 0, which would claim the rest bit-exact.
        measured=int(stats["max"]) if stats["lanes"] else None,
    )


def nonfinite_reason(verdict: Unmeasurable, lanes: int) -> str:
    """*verdict*, from :func:`nonfinite_verdict`, as one sentence for a failure message:
    the message and the recorded verdict are the one computation. A cell whose every
    lane is non-finite has no worst lane to name."""
    reason = f"{verdict.why}: {verdict.nonfinite} such lane(s)"
    if verdict.measured is None:
        return reason
    return f"{reason}; max {verdict.measured} ULP over the {lanes} measurable lane(s)"


# ─────────────────────────────────────────────────────────────────────────────
# Folding a sweep back into the table
# ─────────────────────────────────────────────────────────────────────────────

#: Set by ``--ulp-emit``. Measure and rewrite the table instead of gating against it.
EMIT = False

#: Where an xdist worker hands its measurements to the controller, in ``workeroutput``.
WORKEROUTPUT_KEY = "ulp_measured"

#: What one cell recorded: the worst lane's step count, a :class:`Floored` measurement,
#: or an :class:`Unmeasurable` verdict.
Recorded = Union[int, "Floored", Unmeasurable]

#: The cells measured this session, by op and by the variant's key (which pins
#: ``input_format``, ``output_format``, ``approx_mode`` and ``dest_acc``).
MEASURED: Dict[MathOperation, Dict[BudgetKey, Recorded]] = {}

#: The list each measured cell was recorded with, beside its figure in ``MEASURED``. A
#: cell read twice keeps its list only while both readings are the same; one read with
#: unequal readings or lists, or parked as unmeasurable, keeps none (:func:`record`).
DOMAINS: Dict[MathOperation, Dict[BudgetKey, Tuple[Domain, ...]]] = {}

#: The widest ``near_zero_atol`` the emitter writes on its own. A cell whose only
#: over-ceiling lanes sit in the near-zero band is enrolled with the floor that covers
#: them (:func:`near_zero_floor`); past this the floor would be forgiving a kernel's
#: cut-off rather than its rounding -- Softplus answers 0 below x = -5 where the answer
#: is up to 0.0065, the approximate Gelu is 0.024 off -- and that is a decision to make
#: by hand.
EMIT_MAX_NEAR_ZERO_ATOL = 1e-3


@dataclass(frozen=True)
class Floored:
    """A cell's measurement with the near-zero floor that makes it gateable: the worst
    lane over every ranked lane, the worst lane outside the floor's band, and the
    floor itself (see :func:`near_zero_floor`)."""

    max_ulp: int
    residual: int
    atol: float


def near_zero_floor(
    golden: torch.Tensor,
    result: torch.Tensor,
    distance: torch.Tensor,
    mask: torch.Tensor,
    out_fmt: DataFormat,
) -> Optional[Tuple[int, float]]:
    """``(residual, atol)``: the worst lane outside the near-zero floor a demoted cell
    needs, and that floor; ``None`` when the cell fits without one (see SFPU_ULP.md).
    The floor is exactly the largest absolute error it rescues, no margin and no
    rounding: like the step figure, it is the measurement."""
    from helpers.ulp import NEAR_ZERO_FRACTION

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
    dynamic_range = float(magnitude[finite].max())
    band = finite & (magnitude < NEAR_ZERO_FRACTION * dynamic_range)
    if not band.any():
        return None
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
    # Past the ceiling even so, the cell demotes; _decide records the floor it tried.
    return residual, atol


def _max_of(value) -> int:
    return value.max_ulp if isinstance(value, Floored) else int(value)


def floor_covered(
    golden: torch.Tensor,
    result: torch.Tensor,
    mask: torch.Tensor,
    atol: Optional[float],
) -> torch.Tensor:
    """The lanes of *mask* a ``near_zero_atol`` of *atol* judges on absolute error
    instead of steps: ``ulp_elementwise_valid``'s own band, so the entries of a list
    and the cell's residual leave out the same lanes. Nothing, for no floor."""
    if atol is None:
        return torch.zeros_like(mask)
    from helpers.ulp import ulp_elementwise_valid

    # The covered mask does not depend on the budget.
    _, _, covered = ulp_elementwise_valid(
        golden, result, max_ulp=0, near_zero_atol=atol, selected=mask
    )
    return covered.reshape(mask.shape) & mask


def floored_distance(distance: torch.Tensor, covered: torch.Tensor) -> torch.Tensor:
    """*distance* with the lanes a near-zero floor *covered* read as 0 steps: they are
    held, by the floor, so for the profile a list is drawn on they are in-budget lanes,
    not edges. The figures themselves are measured outside the floor
    (:func:`domain_figures`)."""
    return torch.where(
        covered.reshape(distance.shape), torch.zeros_like(distance), distance
    )


def floored_profile(
    golden: torch.Tensor,
    result: torch.Tensor,
    x: torch.Tensor,
    distance: torch.Tensor,
    mask: torch.Tensor,
    present: torch.Tensor,
    atol: Optional[float],
) -> Tuple[torch.Tensor, Dict[int, Bin]]:
    """``(judged, bins)`` for a cell held with a near-zero floor of *atol*, or none: the
    ranked lanes outside the floor, which a list's entries are figured and gated on, and
    the binade profile with the lanes the floor covers ranked at 0 steps. The emitter
    calls it with the floor it measured, the gate with the row's, so the gate judges
    exactly the lanes the emitter figured."""
    covered = floor_covered(golden, result, mask, atol)
    bins = binade_maxima(x, floored_distance(distance, covered), mask, present)
    return mask & ~covered, bins


def _binade_index(x: torch.Tensor) -> torch.Tensor:
    """``ulp_domains.binade`` over a tensor of received inputs: a negative power of two
    is the low edge of the binade above it."""
    flat = x.reshape(-1).to(torch.float64)
    mantissa, exponent = torch.frexp(flat)
    k = exponent.to(torch.int64) - 1 - ((flat < 0) & (mantissa.abs() == 0.5)).long()
    index = k + BINADE_OFFSET
    return torch.where(
        flat > 0, index, torch.where(flat < 0, -index, torch.zeros_like(index))
    )


def binade_maxima(
    x: torch.Tensor, distance: torch.Tensor, ranked: torch.Tensor, present: torch.Tensor
) -> Dict[int, Bin]:
    """Per signed binade of the received input *x*: lanes *present* (data, not
    padding), lanes *ranked* and the worst of them by *distance* -- the profile
    ``ulp_domains.partition`` draws a cell's list on, and what ``--ulp-measure``
    records as ``bins``. Both callers pass the floor-covered lanes as ranked at 0 steps
    (:func:`floored_distance`)."""
    index = _binade_index(x)
    dist = distance.reshape(-1).to(torch.int64)
    present = present.reshape(-1)
    ranked = ranked.reshape(-1)
    bins: Dict[int, Bin] = {}
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


def band_mask(entry: Domain, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The lanes of *mask* whose received input lies in *entry*."""
    flat = x.reshape(-1).to(torch.float64)
    return (mask.reshape(-1) & (flat >= entry.low) & (flat < entry.high)).reshape(
        mask.shape
    )


def _figure(entry: Domain, x, distance, mask, judged) -> Tuple[Optional[int], int]:
    """``(worst, lanes)`` over one entry: the worst *judged* lane (ranked, outside the
    floor), 0 when the floor covers every ranked lane there, ``None`` when none is
    ranked; and how many lanes are ranked in it."""
    dist = distance.reshape(-1).to(torch.int64)
    ranked = band_mask(entry, x, mask).reshape(-1)
    outside = band_mask(entry, x, judged).reshape(-1)
    n = int(ranked.sum())
    if bool(outside.any()):
        return int(dist[outside].max()), n
    return (0 if n else None), n


def domain_figures(
    spans,
    x: torch.Tensor,
    distance: torch.Tensor,
    mask: torch.Tensor,
    judged: torch.Tensor,
) -> Tuple[Domain, ...]:
    """Each ``(low, high, ranked)`` of ``ulp_domains.partition`` as the entry the table
    writes (``ulp_domains.entries_of``): the worst *judged* lane in it (ranked, outside
    the floor), 0 where the floor covers every ranked lane, ``None`` where the span is
    unranked and its stray ranked lanes fit the row's largest figure; ``()`` when no
    span measured anything. Measured over the final edges rather than summed from the
    binades, so the figure written is exactly what the gate will hold the same lanes
    to."""
    return entries_of(
        spans,
        lambda low, high: _figure(Domain(low, high, None), x, distance, mask, judged)[
            0
        ],
    )


def entry_readings(
    domains: Tuple[Domain, ...],
    x: torch.Tensor,
    distance: torch.Tensor,
    mask: torch.Tensor,
    judged: torch.Tensor,
) -> Tuple[Reading, ...]:
    """What the sweep measured over each entry of a list, for the ``--ulp-measure``
    record: the worst lane outside the floor (0 if the floor covers all of them, ``None``
    where none is ranked) and the ranked lanes. The headroom report holds each entry to
    its figure from these."""
    return tuple(
        Reading(entry.low, entry.high, *_figure(entry, x, distance, mask, judged))
        for entry in domains
    )


def record(
    op: MathOperation,
    key: BudgetKey,
    max_ulp: int,
    floor: Optional[Tuple[int, float]] = None,
    domains: Tuple[Domain, ...] = (),
) -> None:
    """Fold one measurement into the cell *key* names, keeping the *worst* lane.
    *floor* is :func:`near_zero_floor`'s ``(residual, atol)`` when the cell has one;
    *domains* the list drawn on the same reading, which rides with the reading that wins.

    A repeat comes from :func:`merge_measured` folding in another xdist worker's
    reading. ``max`` is the polarity that cannot hide error, and a cell already recorded
    unmeasurable stays so, though its worst lane still counts. Two readings that differ
    in figure, floor or list keep the worst lane with no floor and no list: either list
    holds only its own reading's worst lane per entry, and the other's may lie in
    another entry. A cell that lost its list keeps none, whatever reading comes next."""
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
    """Record that the cell *key* names could not be measured, and why. A cell read
    twice keeps the worst of both: the reason with more non-finite lanes, and the larger
    finite-lane figure."""
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


def _with_measured(verdict: Unmeasurable, measured: Optional[int]) -> Unmeasurable:
    """*verdict* with *measured* as its finite-lane figure, if that is worse. Kept on a
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


def merge_measured(blob: Optional[bytes]) -> None:
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


def _incomplete_grids() -> List[Tuple[MathOperation, DataFormat, DataFormat, int]]:
    """Each touched ``(op, input, output)`` missing a cell :func:`sweep_cells` runs, with
    how many it is missing: a write replaces every row of a touched pair, so a run
    narrowed within one would drop the rows of the cells it never reached."""
    expected: Dict[Tuple[DataFormat, DataFormat], Set[BudgetKey]] = {}
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


def finish_emit(arch, testsfailed: int, path=None, exitstatus=0) -> "WriteReport":
    """Write this session's measurements into the table (:func:`write_table`), or raise
    an :class:`EmitRefused` subclass naming why nothing was written. *exitstatus* is
    pytest's; *testsfailed* is read first to give a failed run its own message.

    :func:`write_table` can also raise :class:`~helpers.ulp_domains.TableError` (a
    ``ValueError``, not an :class:`EmitRefused`): the table on disk does not load, or
    the rewrite would not read back, or would not resolve, once written."""
    import pytest

    if arch != MEASURED_ARCH:
        raise WrongArch(arch)
    if testsfailed or exitstatus != pytest.ExitCode.OK:
        raise SessionNotClean(testsfailed, exitstatus)
    gaps = _incomplete_grids()
    if gaps:
        raise IncompleteGrid(gaps)
    return write_table(path or TABLE_PATH)


def _verdict(measured: int, output_format: DataFormat) -> Metric:
    """The metric the table should hold a measured worst lane to: ``ULP`` while it fits
    ``usable_budget_ceiling`` (the measurement is the gate, with no headroom), else
    ``TOLERANCE`` -- past the ceiling a step budget would no longer be *tighter* than the
    tolerance it replaces, and would be the whole gate."""
    if measured > usable_budget_ceiling(output_format):
        return Metric.TOLERANCE
    return Metric.ULP


@dataclass(frozen=True)
class Decision:
    """What the table should say for one measured cell: its metric and figure, the
    near-zero floor it is written with, the non-finite lanes that park it, a note for
    people, and its list."""

    metric: Metric
    measured: Optional[int]
    floor: Optional[float] = None
    nonfinite: Optional[int] = None
    note: Optional[str] = None
    domains: Tuple[Domain, ...] = ()

    def row(self, key: BudgetKey) -> Row:
        """The row the table writes for this decision under *key*. A cell with a list is
        written as the list, with the floor its entries were measured outside."""
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
    cells: Dict[BudgetKey, Recorded],
    domains: Optional[Dict[BudgetKey, Tuple[Domain, ...]]] = None,
) -> Dict[BudgetKey, Decision]:
    """Each measured cell's :class:`Decision`. A :class:`Floored` cell whose worst lane
    demotes it is judged on the lanes outside its floor: enrolled at the residual, or,
    when even those are past the ceiling, demoted with the floor it tried noted."""
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


#: The key fields a collapse may drop when every value of them decides the same.
_DROPPABLE = ("approx_mode", "dest_acc")


def _collapse(decided: Dict[BudgetKey, Decision]) -> Dict[BudgetKey, Decision]:
    """The decided cells as the fewest rows that reproduce them. Only ``approx_mode``
    and ``dest_acc`` may be dropped; the formats stay pinned, because the sweep drives a
    fixed set of them and a wildcard would answer for any a driver adds later."""
    dropped = [
        name
        for name in _DROPPABLE
        if all(
            len(group) == 1
            for group in _group(decided, lambda key: replace(key, **{name: None}))
        )
    ]
    merged: Dict[BudgetKey, Decision] = {}
    for key, decision in decided.items():
        collapsed = replace(key, **{name: None for name in dropped})
        if merged.get(collapsed, decision) != decision:
            # Each field's droppability is decided alone, which is sound on a full 2x2
            # grid but merges two measurements on an anti-diagonal.
            raise AmbiguousCollapse(collapsed, (merged[collapsed], decision))
        merged[collapsed] = decision
    return merged


def _group(decided: Dict[BudgetKey, Decision], project) -> List[Set[Decision]]:
    groups: Dict[BudgetKey, Set[Decision]] = {}
    for key, decision in decided.items():
        groups.setdefault(project(key), set()).add(decision)
    return list(groups.values())


# ─────────────────────────────────────────────────────────────────────────────
# Writing the table
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WriteReport:
    """What :func:`write_table` did: the rows each written op holds now."""

    path: Path
    written: Dict[MathOperation, Tuple[Row, ...]]

    def summary(self) -> str:
        return f"--ulp-emit: rewrote {len(self.written)} op(s) in {self.path.name}"


class EmitRefused(RuntimeError):
    """``--ulp-emit`` refused to write. Each subclass carries the facts as attributes;
    the message is for the terminal."""


class WrongArch(EmitRefused):
    def __init__(self, arch):
        self.arch = arch
        super().__init__(
            f"ran on {arch.name}, but the table's unkeyed rows are measurements on the "
            "measured architecture. Nothing written."
        )


class SessionNotClean(EmitRefused):
    """The session failed tests, or did not run to its end."""

    def __init__(self, failures: int, exitstatus):
        self.failures, self.exitstatus = failures, exitstatus
        super().__init__(
            f"the session saw {failures} failure(s) and ended with exit status "
            f"{exitstatus!r}, so it measured a subset. Nothing written."
        )


class IncompleteGrid(EmitRefused):
    """*gaps* is ``(op, input, output, unmeasured cells)`` for each touched pair missing
    a cell :func:`sweep_cells` runs."""

    def __init__(self, gaps):
        self.gaps = gaps
        super().__init__(
            "measured only part of an op's grid, and a write would drop the rows of the "
            f"rest. Nothing written: {gaps}"
        )


class KeptRowCollides(EmitRefused):
    """A row the write keeps (:func:`_kept`) has the key of a row the emit writes for
    *op*: one would silently replace the other, and whether the cells collapsed onto
    that key decided which."""

    def __init__(self, op: MathOperation, key: BudgetKey):
        self.op, self.key = op, key
        super().__init__(
            f"{op.name} {key.describe()}: a kept row has the key this emit writes. "
            "Nothing written -- give the kept row a key no measured cell collapses to."
        )


class AmbiguousCollapse(EmitRefused):
    """Collapsing the recorded cells would merge two different decisions onto *key*."""

    def __init__(self, key: BudgetKey, decisions: Tuple[Decision, Decision]):
        self.key, self.decisions = key, decisions
        super().__init__(
            f"collapsing merges two different measurements onto {key.describe()}. "
            "Nothing written -- emit from a complete run."
        )


def _kept(row: Row, measured: Set[Tuple[DataFormat, DataFormat]]) -> bool:
    """Whether *row* survives a write of the cells *measured*: it pins an ``arch`` (this
    sweep runs on one), declares ``atol``/``rtol`` (a judgement no measurement
    re-derives), or names a pair of formats this run did not measure."""
    return (
        row.arch is not None
        or row.atol is not None
        or row.rtol is not None
        or (row.input, row.output) not in measured
    )


def write_table(path=TABLE_PATH) -> WriteReport:
    """Replace, in the table at *path*, every row of each measured ``(input, output)``
    pair with what the sweep measured, and write it back. A kept row never displaces a
    measurement: one with the key of an emitted row is refused (:class:`KeptRowCollides`),
    so an emitted row always outranks a kept row pinning its pair."""
    table = load_table(path)
    written: Dict[MathOperation, Tuple[Row, ...]] = {}
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
                raise KeptRowCollides(op, row.key)
        written[op] = table[op] = kept + emitted
    write_table_file(table, path)
    return WriteReport(Path(path), written)

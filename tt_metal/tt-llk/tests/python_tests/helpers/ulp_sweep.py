# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive 16-bit ULP sweeps, and folding what they measure back into the table.

The functional drivers sample a few thousand points from an op's *safe* domain. A budget
measured that way describes the sample, not the format, and cannot see a tail the sample
never reaches. This module measures the format's number.

Exhaustive is only honest for the 16-bit formats. bfloat16 has 65,279 finite values and
float16 63,487, so either fits one 64-tile device run; Float32's 2**32 does not, and is
left to a stratified sweep. ``Bfp8_b`` is swept in bfloat16 and packed on the way in --
it has no enumerable value set of its own.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from helpers.format_config import DataFormat
from helpers.stimuli_generator import StimuliSpec

#: The formats this harness can enumerate. Bfp8_b rides on the bfloat16 value set: the
#: sweep generates bf16 and the pipeline packs it, which is the only sense in which a
#: block format has "every value".
SWEEP_FORMATS: Tuple[DataFormat, ...] = (
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Bfp8_b,
)

#: What a Bfp8_b sweep is actually enumerated in.
_STIMULI_FORMAT: Dict[DataFormat, DataFormat] = {
    DataFormat.Bfp8_b: DataFormat.Float16_b,
}

#: A top-level op key in the table.
_OP_KEY = re.compile(r"^([A-Za-z_]\w*):")

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
    arch's own format support: Blackhole's functional drivers skip ``Float16`` with
    ``dest_acc=No``, and Quasar has no ``Bfp8_b``. That is inert while every cell off
    Wormhole resolves to tolerance and skips, which holds because no ULP row names
    another ``arch``; test_ulp_sweep.py pins that, so the first such row has to bring
    those filters with it.
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


def stimuli_format_for(fmt: DataFormat) -> DataFormat:
    """The format whose values the sweep enumerates in order to drive *fmt*."""
    return _STIMULI_FORMAT.get(fmt, fmt)


def sweep_spec() -> StimuliSpec:
    """Every finite representable value of the stimuli format, once.

    Deliberately not clipped to the op's domain. ``exclude_undefined`` expresses a domain
    as ``intervals``, which ULP_SWEEP does not read -- and clipping would also stop the
    undefined inputs reaching hardware at all. They are swept and then masked out of the
    statistics by :func:`measurable_mask`, so the run still exercises them.
    """
    return StimuliSpec.ulp_sweep(low=-_INF, high=_INF)


@lru_cache(maxsize=None)
def swept_value_count(input_format: DataFormat) -> int:
    """How many values the sweep actually generates for *input_format*.

    Cached because the answer is a property of the format, while finding it walks the
    whole format, and `padding_lanes` asks twice per variant.
    """
    from helpers.stimuli_generator.strategies.structured import ulp_sweep_value_count

    return int(ulp_sweep_value_count(stimuli_format_for(input_format), -_INF, _INF))


def padding_lanes(src: torch.Tensor, input_format: DataFormat) -> torch.Tensor:
    """The tail ``generate_full_tensor`` fills with zeros to reach the tile count.

    The sweep enumerates every finite value of the stimuli format -- 65,279 for
    bfloat16, 63,487 for float16 -- into a fixed 65,536-lane tensor, so the last 257
    (or 2,049) lanes are padding rather than data. They are not values the sweep chose
    to feed, and they are all the same one, so they belong in no statistic: they
    inflate every lane count, and on an op singular at zero they land on the pole:
    ``reciprocal`` returns ``Inf`` there, golden and hardware alike, so the lanes are
    not a mismatch but they are 257 copies of one value the sweep never chose.

    Identified by position rather than by value, because ``0.0`` is also a legitimate
    swept value: exactly one, in the middle of the sorted order. Confirmed on hardware
    that the padding is the contiguous tail.
    """
    swept = swept_value_count(input_format)
    # On *src*'s device: the mask is composed with tensors derived from it, and a
    # CPU-only mask would fail that composition for a device-resident sweep.
    flat = torch.zeros(src.numel(), dtype=torch.bool, device=src.device)
    flat[swept:] = True
    return flat.reshape(src.shape)


def received_inputs(src: torch.Tensor, input_format: DataFormat) -> torch.Tensor:
    """*src* as the op receives it: what ``quantize_input_to_unpack_format`` hands the
    golden, and the unpack hands the kernel. The same tensor on every format but a block
    float.

    The lane checks that ask where an input *is* -- on a singularity, inside the op's
    claim, inside a tracked issue's lanes -- have to ask it of this value, not of the one
    generated. On Bfp8_b the shared exponent moves a value across those boundaries: in
    the sorted sweep 1.0 is the last lane of the block ``0x3F71..0x3F80``, so 0.992 and
    0.996 both arrive as exactly 1.0, and ``atanh`` of them is the pole.
    """
    from helpers.bfp_format_utils import BFP_BLOCK
    from helpers.golden_generators import quantize_input_to_unpack_format

    # The block quantizer works on whole BFP_BLOCK-lane blocks. A device sweep is 65,536
    # lanes; a host test may hand in a fragment, so pad it with zeros, which never raise
    # a block's exponent, and drop the padding again.
    flat = src.detach().flatten()
    short = (-flat.numel()) % BFP_BLOCK
    padded = torch.cat([flat, torch.zeros(short, dtype=flat.dtype, device=flat.device)])
    received = quantize_input_to_unpack_format(padded, input_format)[: flat.numel()]
    return received.reshape(src.shape)


def flushed_inputs(src: torch.Tensor, input_format: DataFormat) -> torch.Tensor:
    """Lanes whose input the unpack path flushes to zero and the golden does not.

    Two values are tested against the *stimuli* format's smallest normal: the input as
    generated, and as received (:func:`received_inputs`). They differ on a block float.
    The sweep's one ``-0.0`` shares a Bfp8_b block with the bf16 subnormals
    ``0x8001..0x800F``, the shared exponent is 0, and the quantizer's forced hidden bit
    gives the golden ``-2**-127``; ``floor`` of that is -1
    against the 0 silicon sees. That one lane was 16,129 steps on every Bfp8_b-input
    cell of Floor, and the reason Ceil and Trunc read 0 on the same cells.

    The threshold is the stimuli format's, not the golden's: taking it from the golden
    dtype silently passed every fp16 subnormal through on a Float16->Float16_b variant --
    bf16's smallest normal is 1.18e-38 and fp16's is 6.1e-05, so 2,046 flushed lanes
    read as a 14,337-step error on ``Abs``, an op that cannot be wrong.
    """
    from helpers.llk_params import format_dict

    stimuli_dtype = format_dict[stimuli_format_for(input_format)]
    smallest_normal = torch.finfo(stimuli_dtype).smallest_normal

    def subnormal(values: torch.Tensor) -> torch.Tensor:
        # In float32, from the values as they are. Casting to the golden's dtype first
        # rounds an fp16 subnormal *up* -- bf16 keeps 8 mantissa bits, so 6.09e-05
        # becomes 6.10e-05 and clears a 6.10e-05 threshold. The whole subnormal band
        # then passed this filter while looking, in any printout, like the smallest
        # normal.
        magnitude = values.detach().to(torch.float32).abs()
        return (magnitude < smallest_normal) & (magnitude != 0)

    return subnormal(src) | subnormal(received_inputs(src, input_format))


def measurable_mask(
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
) -> torch.Tensor:
    """Lanes of an all-finite-input sweep that a *step count* can describe.

    Op-agnostic on purpose: an op undefined at an input lands in the NaN kind on its
    own, so there is nothing per-op to look up.

    The sweep feeds every non-special value of the format, with no per-op domain
    clipping -- an op is measured wherever its format can reach. Four lane kinds come
    back out, none of them a budget question:

    * either side NaN. :func:`ulp_distance` returns ``UNMEASURABLE`` there, and an op
      undefined at an input (``log`` of a negative) lands here on its own.
    * the two sides disagreeing about being non-finite -- a reciprocal overflowing where
      the golden is still finite. ``passed_test`` rejects those positionally whatever the
      budget says, so ranking them would inflate the number without tightening the gate.
      One such lane is worth ~48,000 steps.
    * the sweep's own zero padding -- see :func:`padding_lanes`.
    * subnormal inputs, as generated or as the block-float quantizer hands them to the
      golden (:func:`flushed_inputs`). The hardware flushes them on the way in and the
      golden does not, so ``ceil(5.69e-39)`` is 1 in the model and 0 on silicon --
      16,129 bf16 steps for a difference that is the unpack path's flush, not the op's
      accuracy. Measured, it is the whole of Ceil's, Floor's and Sqrt's apparent error:
      excluding it returns all three to the 0 their exactness claims, and moves nothing
      else. The flush is covered on its own terms elsewhere; a step count is the wrong
      instrument for it.

    Subnormal *outputs* stay in. Where the golden underflows and the hardware writes
    zero the count is large but the lane is a real one the op produced -- Silu at
    ``x=-87.5`` is that case, and it is the op's own tail, not the unpack path.

    The second kind is a *failure*, not a non-question, and dropping it here is only
    sound because :func:`nonfinite_failures` reports it separately -- less the lanes it
    excuses: an input the op makes no claim on (``sin(2.6e28)``, past Sin's
    argument-reduction limit on a bfloat16 input) and a lane a tracked issue names
    (:data:`_KNOWN_NONFINITE_LANES`). A caller that ranks this mask and nothing else
    would let a hardware overflow produce a clean budget.
    """
    from .ulp import nonfinite_mismatches

    both_measurable = ~(torch.isnan(golden) | torch.isnan(result))
    return (
        both_measurable
        & ~nonfinite_mismatches(golden, result)
        & ~flushed_inputs(src, input_format)
        & ~padding_lanes(src, input_format)
    )


#: The magnitude past which an op's argument reduction stops claiming a finite answer,
#: per op and per *stimuli* format: ``{op: {stimuli_format: limit}}``. Only ops whose
#: kernel reduces its argument belong here; anywhere else a non-finite answer against a
#: finite golden is a failure over the whole format. Sin and Cos give up far outside
#: [-pi, pi] -- `sin(2.6e28)` returns inf against a golden of -1 -- and pi is the widest
#: bound measured so far, so it is the claim until one is wider.
#:
#: Keyed on the stimuli format because the claim is about what the format can reach.
#: 2.6e28 is a bfloat16 value (and a Float32 one); float16 ends at 65504, and Sin and
#: Cos measure 1-4 steps over the whole float16 format, so an fp16 input carries no
#: limit and a non-finite answer anywhere in it is a failure. Keyed on the op alone, the
#: pi claim silently covered the fp16 cells too -- the only Sin/Cos cells the table
#: step-gates -- and a range-reduction regression on the ~46% of fp16 lanes past pi
#: would have passed the gate and been emitted as a clean budget.
_CLAIM_LIMIT: Dict = {}


def _claim_limits() -> Dict:
    from helpers.llk_params import MathOperation

    if not _CLAIM_LIMIT:
        wide_formats = {DataFormat.Float16_b: math.pi, DataFormat.Float32: math.pi}
        _CLAIM_LIMIT.update(
            {MathOperation.Sin: wide_formats, MathOperation.Cos: wide_formats}
        )
    return _CLAIM_LIMIT


def _claimed(op, src: torch.Tensor, input_format: DataFormat) -> torch.Tensor:
    """Lanes where *op* claims a finite, accurate answer: the whole format, less the
    side of each ``_OP_SINGULARITIES`` point the op is undefined on, the point itself
    for a pole, and any ``_CLAIM_LIMIT`` for the format *input_format* is swept in.
    Judged on the input as received (:func:`received_inputs`): a Bfp8_b 0.996 that
    arrives as 1.0 is ``acosh``'s defined side.

    Deliberately not the functional driver's sampling window, which is where a few
    thousand points are drawn, not where the op stops being defined: Abs is sampled on
    (-10, 10) and defined everywhere. Nor ``_SFPU_UNDEFINED_RANGES``, whose holes are
    guard bands that keep a random draw off a singularity: Reciprocal's is
    (-1e-6, 1e-6), and ``1/1e-7`` is a finite bf16 answer an inf must not be excused on.
    """
    from helpers.sfpu_domains import _OP_SINGULARITIES, Operand, SingularitySide

    value = received_inputs(src, input_format).to(torch.float32)
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
    limit = _claim_limits().get(op, {}).get(stimuli_format_for(input_format))
    if limit is not None:
        claimed &= value.abs() <= limit
    return claimed


def _at_a_singularity(op, src: torch.Tensor, input_format: DataFormat) -> torch.Tensor:
    """Lanes received (:func:`received_inputs`) exactly on one of *op*'s
    ``_OP_SINGULARITIES`` points, either side."""
    from helpers.sfpu_domains import _OP_SINGULARITIES, Operand

    value = received_inputs(src, input_format).to(torch.float32)
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
    # nor the store alone.
    top_of_fp16 = dict(
        issue="#58607",
        inputs=(DataFormat.Float16,),
        output=DataFormat.Float16,
        dest=DestAccumulation.No,
        low=65408.0,
        high=65504.0,
        why="inf where the answer is x itself, in the top four fp16 values, on a 16-bit Dest",
    )

    def store_saturates(float16_b: float, bfp8_b: Tuple[float, float], **fields):
        """#57215: the Float16 store saturates a value just past 65504 to 65504 rather
        than to an infinity, on a 32-bit Dest packed to Float16, where the golden's fp16
        rounding answers inf. One entry per input, because the inputs reaching that band
        differ: a Float16_b input reaches the kernel as it is, so only the one value
        whose answer is 2**16 lands there; a Bfp8_b input is block-quantized first, so
        its window is every bf16 value the quantizer maps onto that one -- the
        quantization preimage, not a wider defect."""
        shared = dict(
            issue="#57215", output=DataFormat.Float16, dest=DestAccumulation.Yes
        )
        return tuple(
            KnownNonfiniteLanes(
                **shared, inputs=(fmt,), low=bounds[0], high=bounds[1], **fields
            )
            for fmt, bounds in (
                (DataFormat.Float16_b, (float16_b, float16_b)),
                (DataFormat.Bfp8_b, bfp8_b),
            )
        )

    _KNOWN_NONFINITE_LANES.update(
        {
            MathOperation.Celu: (KnownNonfiniteLanes(**top_of_fp16),),
            MathOperation.Elu: (KnownNonfiniteLanes(**top_of_fp16),),
            MathOperation.Gelu: (
                KnownNonfiniteLanes(**top_of_fp16, approx=ApproximationMode.No),
            ),
            MathOperation.GeluTanh: (KnownNonfiniteLanes(**top_of_fp16),),
            # Silu answers 65408 itself; #58607 lists only the three lanes above it.
            MathOperation.Silu: (
                KnownNonfiniteLanes(**{**top_of_fp16, "low": 65440.0}),
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
                        **top_of_fp16,
                        "low": 65440.0,
                        "magnitude": True,
                        "why": "inf where the answer is x - tanh(x), in the top fp16 values, on a 16-bit Dest",
                    }
                ),
                # x - tanh(x) for |x| at 2**16 is just past fp16's range.
                *store_saturates(
                    float16_b=2.0**16,
                    bfp8_b=(65024.0, 66048.0),
                    magnitude=True,
                    why="+-65504 where x - tanh(x) is just past fp16's range and the store saturates instead of overflowing",
                ),
            ),
            # sqrt(x) for x at 2**32 is 2**16.
            MathOperation.SqrtCustom: store_saturates(
                float16_b=2.0**32,
                bfp8_b=(4.26e9, 4.33e9),
                why="65504 where sqrt(x) is just past fp16's range and the store saturates instead of overflowing",
            ),
            # 1/x is 2**16 at |x| = 2**-16 and ~65793-66052 a step or two under it: past
            # fp16's range, which the store overflows to inf as the golden does. The
            # approximate reciprocal falls a fraction of a percent short, into
            # (65504, 2**16), where the store saturates instead -- hence approx=Yes; the
            # exact kernel agrees with the golden on these lanes.
            MathOperation.Reciprocal: store_saturates(
                float16_b=2.0**-16,
                bfp8_b=(1.51e-5, 1.54e-5),
                approx=ApproximationMode.Yes,
                magnitude=True,
                why="+-65504 where 1/x is past fp16's range and the store saturates instead of overflowing",
            ),
        }
    )
    return _KNOWN_NONFINITE_LANES


def known_nonfinite_lanes(
    op,
    src: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
    approx_mode,
    dest_acc,
) -> torch.Tensor:
    """The lanes of *src* a :data:`_KNOWN_NONFINITE_LANES` entry names on this cell."""
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
        approx_mode,
        dest_acc,
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
    approx_mode=None,
    dest_acc=None,
    known_lanes: bool = True,
) -> torch.Tensor:
    """The lanes :func:`measurable_mask` drops that are a *failure* rather than a
    non-question: the two sides disagreeing about being non-finite where the output
    format could have held the answer.

    *approx_mode* and *dest_acc* name the cell for :data:`_KNOWN_NONFINITE_LANES`; left
    unset, only an entry that pins neither can apply. *known_lanes* False leaves those
    entries out altogether, which is how :func:`stale_excuses` asks what they buy.

    ``passed_test`` rejects these positionally whatever the budget says, but the sweep
    driver ranks a distance rather than calling it, so it has to ask separately -- a
    hardware overflow or an unexpected NaN would otherwise leave the statistics clean
    and both emit and gate would pass.

    The exclusions, and whose doing each one is:

    * **flushed inputs** (:func:`flushed_inputs`), on the same grounds as in the mask --
      the unpack path flushes a subnormal and the golden does not, so a disagreement
      there is the flush. As generated or as the block-float quantizer hands it to the
      golden: the sweep's ``-0.0`` becomes ``-2**-127`` in a Bfp8_b block of subnormals.
    * **a NaN golden**, and **a golden past the output format's range answered by a
      saturated store** -- ``NaN`` or an infinity of the golden's sign. A full-range
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
    * **an input the op makes no claim on** (:func:`_claimed`): the op's own limit
      rather than the sweep's -- the undefined side of a registered singularity, or
      past an argument-reduction limit on a format that reaches it. ``Sin`` and ``Cos``
      disagree on ~21,000 bf16 lanes far outside ``[-pi, pi]``. The budget is still
      measured over the whole format; it is only the *non-finite* answer that needs the
      op to have been claiming something.
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

    output_max = torch.finfo(format_dict[stimuli_format_for(output_format)]).max
    # `golden` is usually already in the output dtype, so "past the range" is mostly an
    # infinity; a wider golden can also be finite and past it. NaN compares false here.
    past_range = golden.detach().to(torch.float32).abs() > output_max
    saturated = torch.isnan(result) | (
        torch.isinf(result) & (torch.signbit(result) == torch.signbit(golden))
    )
    excused = torch.isnan(golden) | (
        past_range & (saturated | _at_a_singularity(op, src, input_format))
    )
    return (
        nonfinite_mismatches(golden, result)
        & ~flushed_inputs(src, input_format)
        & ~excused
        & _claimed(op, src, input_format)
        & ~(
            known_nonfinite_lanes(
                op, src, input_format, output_format, approx_mode, dest_acc
            )
            if known_lanes
            else torch.zeros_like(src, dtype=torch.bool)
        )
        & ~padding_lanes(src, input_format)
    )


#: How many offending lanes a non-finite verdict spells out.
NAMED_LANES = 4


def nonfinite_reason(
    overflowed: torch.Tensor,
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    stats: Dict,
    lanes: int,
    named_lanes: int = NAMED_LANES,
) -> str:
    """The verdict a cell with non-finite disagreements is written with.

    Two numbers, because two readers want them. The lane count is what makes the cell
    unmeasurable, and the headroom report (from #57527) fails a run in which it grows.
    The maximum over the *measurable* lanes -- the rest of the cell, usually tens of
    thousands of them -- is what the demotion used to throw away: a cell parked by a
    handful of lanes recorded no step count for the rest of it. It is written as
    ``max N ULP``, the same shape a measured row's figure has, so the same reader finds
    it. *named_lanes* bounds how many offending inputs are spelled out.
    """
    named = "; ".join(
        f"x={float(src[i]):g}: {float(golden[i]):g} -> {float(result[i]):g}"
        for i in overflowed.nonzero().flatten()[:named_lanes].tolist()
    )
    return (
        f"{int(overflowed.sum())} lane(s) disagreeing with the golden about being "
        f"finite (golden -> result: {named}); max {int(stats['max'])} ULP over the "
        f"{lanes} measurable lanes"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Folding a sweep back into the table
# ─────────────────────────────────────────────────────────────────────────────

#: Set by ``--ulp-emit``. Measure and rewrite the table instead of gating against it.
EMIT = False

#: Where an xdist worker hands its ``MEASURED`` to the controller, in ``workeroutput``.
#: One name for both ends: a mismatch reads as an empty worker, and the controller then
#: refuses with "nothing was measured".
WORKEROUTPUT_KEY = "ulp_measured"

#: The axes of a ``MEASURED`` key, in key order, and the row fields they are written as.
KEY_AXES = ("in", "out", "approx", "dest")

#: {op_name: {(in, out, approx, dest): max_ulp}}, filled during an emitting session.
MEASURED: Dict[str, Dict[Tuple[str, str, str, str], Union[int, str]]] = {}

#: Headroom over the measured worst lane. The sweep is exhaustive, so unlike a sampled
#: measurement there is no unseen tail to leave room for -- but a budget at exactly the
#: maximum fails on any movement at all, including a golden that gets more accurate.
EMIT_HEADROOM = 1.1


def record(op_name: str, key: Tuple[str, str, str, str], max_ulp: int) -> None:
    """Fold one measurement into the cell *key* names, keeping the *worst* lane.

    The sweep driver records each cell once per process, so a repeat comes from
    :func:`merge_measured` folding in another xdist worker's reading of it -- or from a
    future driver that enumerates an axis the key does not have. Last-write-wins would
    keep whichever arrived last, which is the polarity that can hide error; ``max`` is
    the one that cannot. For the same reason a cell already recorded unmeasurable
    (:func:`record_unmeasurable`) stays so: a number arriving later does not rescue it.
    """
    cells = MEASURED.setdefault(op_name, {})
    if isinstance(cells.get(key), str):
        return  # already unmeasurable; a reading elsewhere does not rescue it
    cells[key] = max(cells.get(key, 0), max_ulp)


def record_unmeasurable(op_name: str, key: Tuple[str, str, str, str], why: str) -> None:
    """Record that the cell *key* names could not be measured, and why.

    Written into the table as a tolerance row naming the reason, rather than left out:
    a hole in an op's grid would let ``write_table`` drop the cell's old row with
    nothing to replace it, and ``_collapse`` could stretch a neighbour's budget over it.
    """
    MEASURED.setdefault(op_name, {})[key] = why


def export_measured() -> List[list]:
    """``MEASURED`` as plain lists, for an xdist worker to hand to the controller."""
    return [
        [op, list(key), max_ulp]
        for op, cells in MEASURED.items()
        for key, max_ulp in cells.items()
    ]


def merge_measured(rows) -> None:
    """Fold a worker's :func:`export_measured` into this process, worst lane winning."""
    for op, key, value in rows:
        if isinstance(value, str):
            record_unmeasurable(op, tuple(key), value)
        else:
            record(op, tuple(key), value)


def _incomplete_grids() -> List[str]:
    """Each touched ``(op, in, out)`` missing a cell :func:`sweep_cells` runs.

    ``write_table`` replaces every row of a touched ``(in, out)``, so a run narrowed
    within one -- ``-k``, ``--maxfail``, an interrupt -- would drop the rows of the
    cells it never reached.
    """
    expected: Dict[Tuple[str, str], Set[Tuple[str, str, str, str]]] = {}
    for in_fmt, out_fmt, approx, dest in sweep_cells():
        cell = (in_fmt.name, out_fmt.name, approx.name, dest.name)
        expected.setdefault(cell[:2], set()).add(cell)
    gaps = []
    for op, cells in sorted(MEASURED.items()):
        for pair in sorted({key[:2] for key in cells}):
            missing = expected.get(pair, set()) - set(cells)
            if missing:
                gaps.append(
                    f"{op} {pair[0]}->{pair[1]}: {len(missing)} cell(s) unmeasured"
                )
    return gaps


def finish_emit(arch, testsfailed: int, path=None, exitstatus=0) -> str:
    """Write this session's measurements into the table, and say what was written.

    * **nothing written, ``RuntimeError``** when the session cannot vouch for them: off
      ``MEASURED_ARCH``, where unkeyed rows would carry another arch's numbers under
      Wormhole's name; after a failure, when only a subset was measured; when
      *exitstatus* says the session did not run to the end (pytest calls
      ``pytest_sessionfinish`` after a Ctrl-C too, and an interrupt between two ops
      leaves every touched grid complete and nothing counted as failed); or when an
      op's ``(in, out)`` grid is incomplete, when a write would drop the rest's rows.
    * **written, then ``RuntimeError``** when an op was measured with no key line to
      write into. Every *other* op's block has already been rewritten, so a red emit is
      not an untouched table.
    * **written**, returning the summary line. An op kept verbatim -- a row the run
      covers carries a field ``_render`` cannot put back, the ``near_zero_atol`` floors
      and ``atol``/``rtol`` anchors -- is named in it: such a block is maintained by
      hand by design, and failing on it would turn every whole-table emit red.

    *exitstatus* is pytest's ``session.exitstatus``, an ``ExitCode`` or a plain int.
    pytest has already set it to ``TESTS_FAILED`` when a test failed, so the failure
    count is read first, to give that run its own message.
    """
    from datetime import date

    import pytest
    from helpers.sfpu_accuracy_budget import _TABLE_PATH, MEASURED_ARCH

    if arch != MEASURED_ARCH:
        raise RuntimeError(
            f"ran on {arch.value}, but the table's unkeyed rows are read as "
            f"{MEASURED_ARCH.value} measurements and `_render` does not emit `arch`. "
            "Nothing written."
        )
    if testsfailed:
        raise RuntimeError(
            f"saw {testsfailed} failure(s), so the session measured a subset. Nothing "
            "written -- emit from a clean run."
        )
    if exitstatus != pytest.ExitCode.OK:
        try:
            name = pytest.ExitCode(exitstatus).name
        except ValueError:  # pytest.exit(returncode=...) can carry any int
            name = str(exitstatus)
        raise RuntimeError(
            f"the session ended with exit status {name}, not OK, so what it measured "
            "is whatever it got to. Nothing written -- emit from a run that ends on "
            "its own."
        )
    gaps = _incomplete_grids()
    if gaps:
        raise RuntimeError(
            "measured only part of an op's grid, and a write would drop the rows of "
            "the rest. Nothing written -- emit whole (in, out) grids:\n  "
            + "\n  ".join(gaps)
        )
    path = path or _TABLE_PATH
    suffix = (
        f"exhaustive {'/'.join(f.name for f in SWEEP_FORMATS)} sweep, "
        f"{arch.value}, {date.today().isoformat()}"
    )
    try:
        n, kept = write_table(path, suffix)
        unplaced = []
    except UnplacedMeasurements as exc:
        n, kept, unplaced = exc.written, exc.kept, exc.missing
    message = f"rewrote {n} op block(s) in {path.name}"
    if kept:
        message += (
            f"; kept {', '.join(kept)} verbatim: a row this sweep covers carries a "
            f"field it cannot regenerate (beyond {sorted(_RENDERABLE_FIELDS)}), so "
            "their measurements were not written. Settle those cells by hand"
        )
    if unplaced:
        raise RuntimeError(
            f"{message}; but measured {', '.join(unplaced)} and the table has no key "
            "line for them, so they were not written. Give an op its block first "
            "(SFPU_ULP.md, step 2) to enrol it."
        )
    return f"--ulp-emit: {message}"


def _verdict(measured: int, out_fmt: str) -> Tuple[str, int]:
    """What the table should say for a measured worst lane on *out_fmt*.

    ``("ulp", budget)`` while a step budget is still *stronger* than the tolerance it
    replaces, and ``("tolerance", measured)`` once it is not. The bound is the table's
    own ``usable_budget_ceiling``: ~419,430 steps for fp32, 51 for fp16, 6 for bf16, 25
    for Bfp8_b. Decided per cell and before collapsing, because it depends on the output
    format and collapsing may drop it.

    Without this the sweep enrols what it should not. ``Abs`` measures 393 steps on a
    Bfp8_b output from a bf16 input -- the block exponent quantizing a small element, not
    the op -- and a 433-step budget on a format whose ceiling is 25 gates nothing.

    Headroom is 1.1x, except at zero: the sweep saw every value, so a measured 0 means
    the op is exactly rounded on this format, and widening it to 1 retires that claim.
    """
    from helpers.format_config import DataFormat
    from helpers.sfpu_accuracy_budget import usable_budget_ceiling
    from helpers.ulp import _ULP_PROXY_DTYPES

    if DataFormat[out_fmt] in _ULP_PROXY_DTYPES:
        # reason: block composition, not the size of the number
        # A block float never enrols from *this* sweep, whatever it measures. The sweep
        # enumerates a format in value order, so sixteen adjacent values share a block
        # and the exponent fits all of them -- which is the best case for quantization,
        # not a representative one. Measured the two ways: `Abs` reads 15616 steps on a
        # Bfp8_b output from random mixed-magnitude blocks (the table's Bfp8_b note) and
        # 393 from the sorted sweep. Enrolling the second would hide the first.
        return ("block", measured)
    budget = 0 if measured == 0 else math.ceil(measured * EMIT_HEADROOM)
    if budget > usable_budget_ceiling(DataFormat[out_fmt]):
        return ("tolerance", measured)
    return ("ulp", budget)


def _decide(
    cells: Dict[Tuple[str, str, str, str], Union[int, str]],
) -> Dict[Tuple, Tuple]:
    """Each measured cell as ``(verdict, measured)``, verdict decided per output format;
    an unmeasurable cell as ``(("unmeasurable", why), None)``."""
    return {
        key: (
            (("unmeasurable", v), None)
            if isinstance(v, str)
            else (_verdict(v, key[1]), v)
        )
        for key, v in cells.items()
    }


def _collapse(decided: Dict[Tuple, Tuple]) -> List[dict]:
    """The decided cells as the fewest rows that reproduce them.

    Only ``approx`` and ``dest`` may be dropped. ``in`` and ``out`` stay pinned even
    when every value agrees, because this sweep covers the 16-bit formats only: a row
    that wildcards the output would, by most-specific-wins, also answer for Float32 and
    the block floats below Bfp8_b, which nothing here measured. `Abs` losing its
    Float32 row that way is what the registry's unswept-architecture guard caught.
    """
    axes = KEY_AXES
    keep = [0, 1]
    for i in (2, 3):
        seen: Dict[Tuple, set] = {}
        for key, value in decided.items():
            seen.setdefault(key[:i] + key[i + 1 :], set()).add(value)
        if any(len(v) > 1 for v in seen.values()):
            keep.append(i)
    keep.sort()

    merged: Dict[Tuple, Tuple] = {}
    for key, value in decided.items():
        collapsed = tuple(key[i] for i in keep)
        if merged.get(collapsed, value) != value:
            # `approx` and `dest` droppability is decided independently above, which is
            # sound on a full 2x2 grid -- both dropped implies all four agree. On an
            # anti-diagonal (only (No,No) and (Yes,Yes) recorded) each axis sees only
            # singletons, both get dropped, and one measurement would silently overwrite
            # the other. `_render` writes the survivor's own figure into the provenance
            # comment, so the budget audit could not catch it either.
            raise ValueError(
                f"collapsing {axes} to {[axes[i] for i in keep]} merges two different "
                f"measurements onto {collapsed}: {merged[collapsed]} and {value}. The "
                "recorded cells do not form a full grid -- emit from a complete run."
            )
        merged[collapsed] = value

    rows = []
    for key, (verdict, measured) in sorted(merged.items()):
        row = {axes[i]: v for i, v in zip(keep, key)}
        row["verdict"] = verdict
        row["measured"] = measured
        rows.append(row)
    return rows


def _render(key_line: str, rows: List[dict], suffix: str) -> List[str]:
    """One op's block: each row with its verdict, and the measurement behind it.

    *key_line* is passed through verbatim. Several ops carry their measurement as a
    header comment on that line -- `Fill:  # 0 ULP, 115 variants` -- and it is the
    provenance for every row of theirs this sweep does not reach. Rewriting the key as
    a bare `Fill:` dropped it, and the guard that every budget names its measurement
    then failed on rows that had one all along.
    """
    out = [key_line]
    for row in rows:
        metric, value = row["verdict"]
        body = ", ".join(
            f'{k}: "{row[k]}"' if k in ("approx", "dest") else f"{k}: {row[k]}"
            for k in KEY_AXES
            if k in row
        )
        decided = (
            "max_ulp: {}".format(value) if metric == "ulp" else "metric: tolerance"
        )
        pairs = f"{body}, {decided}" if body else decided
        note = f"max {row['measured']} ULP"
        if metric == "unmeasurable":
            note = f"not measurable: {value}"
        elif metric == "tolerance":
            note += ", past this output's usable ceiling, so tolerance"
        elif metric == "block":
            note += ", but a sorted sweep flatters a block format, so tolerance"
        out.append(f"  - {{{pairs}}}  # {note}, {suffix}\n")
    return out


#: A ``name: value`` pair inside an inline row, with the value unquoted.
_ROW_FIELD = re.compile(r'([A-Za-z_]\w*)\s*:\s*"?([^,}"]*)"?')

#: What `_render` can put back. A row carrying anything else -- a `near_zero_atol`
#: floor, an `atol`/`rtol` pair -- cannot be regenerated from a measurement, so it is
#: preserved rather than replaced even when this sweep covers its cell.
_RENDERABLE_FIELDS = frozenset({"in", "out", "approx", "dest", "max_ulp", "metric"})


def _row_fields(line: str) -> Dict[str, str]:
    """The inline row's fields, parsed. Substring matching is not enough: ``in:
    Float16`` is a substring of ``in: Float16_b``."""
    body = line.split("#", 1)[0]
    if "{" not in body:
        return {}
    return dict(_ROW_FIELD.findall(body[body.index("{") + 1 : body.rindex("}")]))


def _pins_a_measured_cell(line: str, measured: Set[Tuple[str, str, str, str]]) -> bool:
    """Whether some cell this run measured resolves through *line*'s key: every key
    field *line* pins agrees with it. A wildcard row -- an op-wide ``{metric:
    tolerance, atol: 0.13}``, or a YAML alias whose fields are not inline -- pins
    nothing, so it answers for every cell.

    Wider than :func:`_covered`, which asks whether the run measured the row's own
    ``(in, out)``: a rendered row is more specific than a wildcard, so it would shadow
    the wildcard's ``atol``/``rtol`` on every cell it names.
    """
    if line.strip().startswith("- *"):
        return True  # an alias: its fields live on the anchor, so assume the widest
    fields = _row_fields(line)
    pinned = [(i, fields[axis]) for i, axis in enumerate(KEY_AXES) if axis in fields]
    return any(all(key[i] == value for i, value in pinned) for key in measured)


def _unrenderable(line: str) -> bool:
    """A row carrying something ``_render`` cannot put back: a field beyond
    ``_RENDERABLE_FIELDS``, or an alias whose fields are not inline. Arch-keyed rows are
    never touched, so they are not counted."""
    if line.strip().startswith("- *"):
        return True
    fields = _row_fields(line)
    return "arch" not in fields and bool(set(fields) - _RENDERABLE_FIELDS)


class UnplacedMeasurements(ValueError):
    """``write_table`` wrote every op it could, and measured some it had no key line
    for. Carries what a normal return would, plus the ops it could not place."""

    def __init__(self, message: str, written: int, kept: List[str], missing: List[str]):
        super().__init__(message)
        self.written, self.kept, self.missing = written, kept, missing


def _covered(line: str, emitted_cells: Set[Tuple[str, str]]) -> bool:
    """Whether this run measured the ``(in, out)`` cell *line* declares.

    Only the cells actually in ``MEASURED`` for this op, never the static
    ``SWEEP_FORMATS`` cross-product: a partial run -- ``-k``, an interrupt, a driver
    skip -- must replace what it measured and leave the rest alone, rather than
    rendering a whole op block from an incomplete session.
    """
    fields = _row_fields(line)
    return (
        bool(fields)
        and (
            fields.get("in", ""),
            fields.get("out", ""),
        )
        in emitted_cells
    )


def _replaceable(line: str, emitted_cells: Set[Tuple[str, str]]) -> bool:
    """Whether this run's output supersedes *line*.

    A row that pins ``arch`` is never replaceable. This sweep runs on one architecture,
    and the table's own header says to re-measure on Blackhole; specificity lets the two
    rows coexist, so regenerating one arch must not erase the other's contract.

    Nor is a row carrying a field ``_render`` cannot put back -- a ``near_zero_atol``
    floor, an ``atol``/``rtol`` pair. :func:`write_table` keeps such an op's block
    verbatim and names it in its return, rather than quietly replacing or quietly
    duplicating the row.
    """
    fields = _row_fields(line)
    if "arch" in fields or set(fields) - _RENDERABLE_FIELDS:
        return False
    return _covered(line, emitted_cells)


def write_table(path, suffix: str) -> Tuple[int, List[str]]:
    """Replace every swept op's block in the YAML with what the sweep measured.

    Returns how many blocks were rewritten, and the ops kept verbatim because a row the
    sweep covers carries a field it cannot regenerate (a ``near_zero_atol`` floor).

    Line-oriented on purpose. The table's comments *are* its provenance, and a load and
    re-dump through PyYAML would drop every one of them, including for the ops this
    sweep never touched.

    Raises :class:`UnplacedMeasurements` if an op in ``MEASURED`` has no key line to
    write into -- *after* writing every op that has one, so one unenrolled op does not
    throw away the rest of a whole-table emit. The measurement would otherwise be
    dropped in silence, and the sampled rows it was meant to replace would stay in place
    looking measured -- the failure the ``_OP_KEY`` comment below records biting once
    already.
    """
    import pathlib as _pathlib

    path = _pathlib.Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out, i, written, kept_verbatim = [], 0, set(), []
    while i < len(lines):
        line = lines[i]
        # A top-level key by shape, not by trailing colon: an op whose key line carries a
        # header comment -- `Signbit:  # 0 ULP, 16 variants` -- does not end with one.
        # Matching on that silently skipped 17 ops, every one of them exact or predicate,
        # and left their sampled rows in place looking measured. Signbit and Threshold
        # measure 16,129 steps exhaustively against the 0 those rows claimed.
        head = _OP_KEY.match(line)
        name = head.group(1) if head else ""
        if head and name in MEASURED:
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j][0].isspace()):
                j += 1
            # The blank lines after the block are left for the loop to copy through.
            while j - 1 > i and not lines[j - 1].strip():
                j -= 1
            emitted_cells = {(k[0], k[1]) for k in MEASURED[name]}
            rows = [l for l in lines[i + 1 : j] if l.strip().startswith("- ")]
            # Any row a measured cell resolves through, not only one pinning the
            # measured (in, out): an op-wide `{metric: tolerance, atol: 0.13}` would
            # otherwise be shadowed by the bare rows rendered below it, and every cell
            # they name would silently lose its declared atol.
            unrenderable = [
                l
                for l in rows
                if _unrenderable(l) and _pins_a_measured_cell(l, set(MEASURED[name]))
            ]
            if unrenderable:
                # Emitting over it would drop the floor; keeping it as well would give
                # the cell two equally specific keys, which `_load_table` refuses. So
                # the op's block stays as it is, and the caller is told: the row is a
                # judgement a measurement cannot re-derive. Every other op is written.
                out.extend(lines[i:j])
                kept_verbatim.append(name)
                written.add(name)
                i = j
                continue
            kept = [l for l in rows if not _replaceable(l, emitted_cells)]
            out.extend(_render(line, _collapse(_decide(MEASURED[name])), suffix))
            # Rows this run did not supersede -- a format it does not reach, an
            # arch-keyed entry, a floor `_render` cannot re-derive -- are the
            # measurement of a different run and stay as they are. Replacing a whole op
            # block deleted them.
            out.extend(kept)
            written.add(name)
            i = j
            continue
        out.append(line)
        i += 1
    # Exactly one trailing newline: an op block carries its own trailing blank lines,
    # and the last block's leave the file ending in several. `end-of-file-fixer` then
    # rewrites the table on every commit.
    path.write_text("".join(out).rstrip("\n") + "\n", encoding="utf-8")
    missing = sorted(set(MEASURED) - written)
    if missing:
        raise UnplacedMeasurements(
            f"{path.name}: measured {', '.join(missing)} but found no key line to "
            "write into, so those were not written; every other op was. Add the op's "
            "block to the table first -- the key line is passed through verbatim so a "
            "header comment survives, and cannot be generated here.",
            written=len(written) - len(kept_verbatim),
            kept=kept_verbatim,
            missing=missing,
        )
    return len(written) - len(kept_verbatim), kept_verbatim

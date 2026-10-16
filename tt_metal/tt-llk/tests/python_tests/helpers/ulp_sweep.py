# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive 16-bit ULP sweeps, and folding what they measure back into the table.

The functional drivers sample a few thousand points from an op's *safe* domain. A budget
measured that way describes the sample, not the format, and cannot see a tail the sample
never reaches. This module measures the format's number.

Exhaustive is only honest for the 16-bit formats. bfloat16 has 65,279 finite values and
float16 63,487, so either fits one 64-tile device run; Float32's 2**32 does not, so it is
walked with a stride that gives every binade an equal share. ``Bfp8_b`` and ``Bfp4_b``
are swept in bfloat16 and packed on the way in -- neither has an enumerable value set of
its own.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
from helpers.format_config import DataFormat
from helpers.stimuli_generator import StimuliSpec
from helpers.ulp_provenance import (
    EMITTER_KINDS,
    BudgetTable,
    KeyLine,
    Kind,
    Provenance,
    Row,
    RunIdentity,
    render_row,
)

#: The formats this harness can judge a result in -- enumerated whole as inputs, except
#: Float32, which is strided (`is_exhaustive`). Bfp8_b rides on the bfloat16 value set: the
#: sweep generates bf16 and the pipeline packs it, which is the only sense in which a
#: block format has "every value".
SWEEP_FORMATS: Tuple[DataFormat, ...] = (
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Bfp8_b,
    DataFormat.Float32,
)

#: Fed as an *input* but never judged as an output: Bfp4_b keeps 2 fractional bits, so
#: a bf16 step count would read each legal quantization of its output as 32 steps.
SWEEP_INPUT_ONLY_FORMATS: Tuple[DataFormat, ...] = (DataFormat.Bfp4_b,)

#: Every format the sweep feeds, whether or not it can judge a result in it.
SWEEP_INPUT_FORMATS: Tuple[DataFormat, ...] = SWEEP_FORMATS + SWEEP_INPUT_ONLY_FORMATS

#: What a block float's sweep is actually enumerated in: they have no enumerable value
#: set of their own, so the sweep generates bfloat16 and the pipeline packs it on the
#: way in.
_STIMULI_FORMAT: Dict[DataFormat, DataFormat] = {
    DataFormat.Bfp8_b: DataFormat.Float16_b,
    DataFormat.Bfp4_b: DataFormat.Float16_b,
}

#: Float32 has 2**32 values and one run holds 2**16, so it is the one input the sweep
#: samples rather than enumerates. Striding the total order gives every binade an equal
#: share (each holds the same number of values); a consecutive walk of one run's 2**16
#: would cover 1/128 of one binade's 2**23. One sample per bfloat16 cell, 65,280 of
#: them, but not at the cell's start: the walk starts on -inf, a multiple of 2**16, so
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
        for in_fmt in SWEEP_INPUT_FORMATS
        for out_fmt in SWEEP_FORMATS
        for approx in ApproximationMode
        for dest in DestAccumulation
        if effective_dest_acc(in_fmt, out_fmt, dest, arch) == dest
    ]


def stimuli_format_for(fmt: DataFormat) -> DataFormat:
    """The format whose values the sweep enumerates in order to drive *fmt*."""
    return _STIMULI_FORMAT.get(fmt, fmt)


def is_exhaustive(input_format: DataFormat) -> bool:
    """Whether the sweep sees *every* value the input can take, or a stride of them."""
    return stimuli_format_for(input_format) != DataFormat.Float32


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

    fmt = stimuli_format_for(input_format)
    stride = _stride_for(input_format)
    return int(
        _enumerate_representable(fmt, -_INF, _INF, _SWEEP_TENSOR, stride=stride).numel()
    )


def sweep_spec(input_format: DataFormat) -> StimuliSpec:
    """Every finite representable value of the stimuli format, once -- or, for float32,
    every ``_FP32_STRIDE``-th, since 2**32 values do not fit one run.

    Deliberately not clipped to the op's domain. ``exclude_undefined`` expresses a domain
    as ``intervals``, which ULP_SWEEP does not read -- and clipping would also stop the
    undefined inputs reaching hardware at all. They are swept and then masked out of the
    statistics by :func:`measurable_mask`, so the run still exercises them.
    """
    return StimuliSpec.ulp_sweep(low=-_INF, high=_INF, stride=_stride_for(input_format))


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

    Counted by :func:`swept_value_count`, not by how many values the format has: a
    strided float32 walk generates 65,280 of 2**32, and asking the format would put the
    padding boundary past the end of the tensor and mask nothing.
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
    float -- and, not modelled here, a Float32 input at ``dest_acc=No``, which the unpack
    and the golden both truncate to the 16-bit Dest. The lane checks below then judge the
    value before truncation; harmless while truncation only moves toward zero, so at worst
    a lane near a boundary is excused that the truncated value would not be.

    The lane checks that ask where an input *is* -- on a singularity, inside the op's
    claim, inside a tracked issue's lanes -- have to ask it of this value, not of the one
    generated. On Bfp8_b the shared exponent moves a value across those boundaries: in
    the sorted sweep 1.0 is the last lane of the block ``0x3F71..0x3F80``, so 0.992 and
    0.996 both arrive as exactly 1.0, and ``atanh`` of them is the pole.
    """
    from helpers.bfp_format_utils import BFP_BLOCK
    from helpers.golden_generators import quantize_input_to_unpack_format

    # The block quantizer works on whole BFP_BLOCK-lane blocks. A device sweep is a
    # multiple of that; a host test may hand in a fragment, so pad it with zeros, which
    # never raise a block's exponent, and drop the padding again.
    flat = src.detach().flatten()
    short = (-flat.numel()) % BFP_BLOCK
    padded = torch.cat([flat, torch.zeros(short, dtype=flat.dtype, device=flat.device)])
    received = quantize_input_to_unpack_format(padded, input_format)[: flat.numel()]
    return received.reshape(src.shape)


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

    Not for a block-float input: the golden quantizes that itself, and in the
    shared-exponent-0 block the zero lane sits in, the forced hidden bit turns it into
    -2**-127 as -0.0 and +2**-127 (~6e-39) as +0.0 -- neither of them zero, so
    canonicalizing moved Ceil/Sqrt/Log/Rsqrt's Bfp8_b cells rather than fixing them.
    """
    from helpers.sfpu_domains import negative_zero_delivered

    if stimuli_format_for(input_format) != input_format or negative_zero_delivered(
        input_format, dest_acc
    ):
        return src
    return torch.where(src == 0, torch.zeros_like(src), src)


def _unpack_format(input_format: DataFormat, output_format=None, dest_acc=None):
    """The format whose values the unpacker writes for this cell, as the stimuli format
    that carries them: a Float32 input into a Float16 output at ``dest_acc=No`` lands
    in a Float16 Dest. Without *output_format* and *dest_acc*, the input's own."""
    if output_format is None or dest_acc is None:
        return stimuli_format_for(input_format)
    from helpers.data_format_inference import infer_unpack_out
    from helpers.sfpu_domains import unpacks_to_dest

    return stimuli_format_for(
        infer_unpack_out(
            input_format,
            output_format,
            dest_acc,
            unpacking_to_dest=unpacks_to_dest(input_format, dest_acc),
        )
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
    golden: torch.Tensor, input_format: DataFormat, output_format=None, dest_acc=None
) -> torch.Tensor:
    """Lanes whose answer the Dest can hold at all.

    A ``Float16`` input runs on an fp16 Dest, which tops out at 65504 and flushes below
    6.1e-5, whatever the output format. Packed to Float16_b or Float32 the hardware
    hands back 65536 or 66048 where the answer was 70000, and 0 where it was 3e-5 --
    the Dest's doing, not the op's -- while the golden holds the answer in the wider
    output and, for an overflow, substitutes a finite 130560/131008 that reads as 8
    million fp32 steps and is not even a non-finite disagreement. Every ``Float16 ->
    Float32`` cell of Exp, Cosh, Square, Selu, I0, ... was on tolerance for that one
    lane class. Such a lane is a question the variant cannot be asked: neither ranked
    nor a failure.

    Only where the Dest is narrower than the output. An fp16 Dest under an fp16 output
    overflows and flushes exactly as the output does, so a disagreement there is the
    output's edge, which :func:`nonfinite_failures` judges; a bf16 Dest has fp32's
    range. Without *output_format* and *dest_acc* nothing is excluded.
    """
    from helpers.llk_params import format_dict

    holds = torch.ones(golden.shape, dtype=torch.bool, device=golden.device)
    dtype = _dest_dtype(input_format, output_format, dest_acc)
    if dtype is None:
        return holds
    dest = torch.finfo(dtype)
    out = torch.finfo(format_dict[stimuli_format_for(output_format)])
    if dest.max >= out.max and dest.tiny <= out.tiny:
        return holds
    magnitude = golden.detach().to(torch.float32).abs()
    overflowed = magnitude > dest.max
    flushed = (magnitude > 0) & (magnitude < dest.tiny)
    return holds & ~overflowed & ~flushed


def _normal_input(
    src: torch.Tensor, input_format: DataFormat, output_format=None, dest_acc=None
) -> torch.Tensor:
    """Lanes whose input survives the unpack: zero, or at least the smallest normal of
    the stimuli format *and* of the format the unpacker writes, and no larger than that
    format can hold. The unpack format matters once:
    a Float32 input into a Float16 output at ``dest_acc=No`` unpacks into a Float16
    Dest, which flushes everything below 2**-14 -- a Float32 input's own cutoff is
    1.18e-38, so the strided lanes in between scored ``f(x)`` against ``f(0)`` as op
    error (Floor's Float32 -> Float16 cell read 15360 steps, the distance to 1.0).
    Without *output_format* and *dest_acc*, only the stimuli format's cutoff applies.

    Judged on the input as generated *and* as received (:func:`received_inputs`). They
    differ on a block float: the sweep's one ``-0.0`` shares a Bfp8_b block with the
    bf16 subnormals ``0x8001..0x800F``, the shared exponent is 0, and the quantizer's
    forced hidden bit gives the golden ``-2**-127``; ``floor`` of that is -1 against
    the 0 silicon sees. That one lane was 16,129 steps on every
    Bfp8_b-input cell of Floor and Signbit, and why Ceil and Trunc read 0 there.
    """
    from helpers.llk_params import format_dict

    fed = torch.finfo(format_dict[stimuli_format_for(input_format)])
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
    survives = ((magnitude >= cutoff) | (magnitude == 0)) & (magnitude <= ceiling)
    quantized_magnitude = received_inputs(src, input_format).to(torch.float32).abs()
    quantized_subnormal = (quantized_magnitude < cutoff) & (quantized_magnitude != 0)
    return survives & ~quantized_subnormal


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

    The sweep feeds every non-special value of the format, with no per-op domain
    clipping -- an op is measured wherever its format can reach. Four lane kinds come
    back out, none of them a budget question (the last is two, on one boundary):

    * either side NaN. :func:`ulp_distance` returns ``UNMEASURABLE`` there, and an op
      undefined at an input (``log`` of a negative) lands here on its own.
    * the two sides disagreeing about being non-finite -- a reciprocal overflowing where
      the golden is still finite. ``passed_test`` rejects those positionally whatever the
      budget says, so ranking them would inflate the number without tightening the gate.
      One such lane is worth ~48,000 steps.
    * the sweep's own zero padding -- see :func:`padding_lanes`.
    * subnormal inputs, as generated or as the block-float quantizer hands them to the
      golden (:func:`_normal_input`). The hardware flushes them on the way in and the
      golden does not, so ``ceil(5.69e-39)`` is 1 in the model and 0 on silicon --
      16,129 bf16 steps for a difference that is the unpack path's flush, not the op's
      accuracy. Measured, it is the whole of Ceil's, Floor's and Sqrt's apparent error:
      excluding it returns all three to the 0 their exactness claims, and moves nothing
      else. The flush is covered on its own terms elsewhere; a step count is the wrong
      instrument for it. The same boundary has a top on a narrower unpack: a Float32
      input into a Float16 output at ``dest_acc=No`` lands in a Float16 Dest, which
      flushes below 2**-14 and *saturates* past 65504 -- about 44% of the strided lanes,
      whose golden sees the value as fed. Saturation, not a flush, and not the op's.

    * answers the Dest cannot hold (:func:`dest_holds`): past an fp16 Dest's 65504 or
      below its 6.1e-5 under a wider output, where what comes back is the Dest's
      overflow or flush, not the op.

    Subnormal *outputs* stay in. Where the golden underflows and the hardware writes
    zero the count is large but the lane is a real one the op produced -- Silu at
    ``x=-87.5`` is that case, and it is the op's own tail, not the unpack path.

    Op-agnostic, so the op's *claim* is not read here: the sweep driver ANDs
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
        & dest_holds(golden, input_format, output_format, dest_acc)
        & ~padding_lanes(src, input_format)
    )


#: The magnitude past which an op's argument reduction stops claiming a finite answer,
#: per op and per *stimuli* format: ``{op: {stimuli_format: limit}}``. Only ops whose
#: kernel reduces its argument belong here; anywhere else a non-finite answer against a
#: finite golden is a failure over the whole format. Sin, Cos and Tan give up far
#: outside [-pi, pi] -- `sin(2.6e28)` returns inf against a golden of -1, `tan(-3e38)`
#: NaN against -2.4 -- and pi is the widest bound measured so far, so it is the claim
#: until one is wider.
#:
#: Keyed on the format the kernel is handed (:func:`_unpack_format`), because the claim
#: is about what that format can reach. 2.6e28 is a bfloat16 value (and a Float32 one);
#: float16 ends at 65504, and Sin and Cos measure 1-4 steps over the whole float16
#: format, so an fp16 input carries no limit and a non-finite answer anywhere in it is
#: a failure. Keyed on the op alone, the pi claim silently covered the fp16 cells too,
#: and a range-reduction regression on the ~46% of fp16 lanes past pi would have passed
#: the gate and been emitted as a clean budget. Keyed on the input alone, it covered a
#: Float32 input into a Float16 output at ``dest_acc=No`` too, which unpacks into a
#: Float16 Dest and so reaches the kernel as fp16 values only.
_CLAIM_LIMIT: Dict = {}


def _claim_limits() -> Dict:
    from helpers.llk_params import MathOperation

    if not _CLAIM_LIMIT:
        wide_formats = {DataFormat.Float16_b: math.pi, DataFormat.Float32: math.pi}
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
    """Lanes where *op* claims a finite, accurate answer: the whole format, less the
    side of each ``_OP_SINGULARITIES`` point the op is undefined on, the point itself
    for a pole, every non-positive integer for the gamma family
    (``_NONPOSITIVE_INTEGER_POLES``), and any ``_CLAIM_LIMIT`` for the format the kernel
    is handed (:func:`_unpack_format`; *input_format*'s own without the other two).
    Judged on the input as received (:func:`received_inputs`): a Bfp8_b 0.996 that
    arrives as 1.0 is ``acosh``'s defined side. Read by :func:`nonfinite_failures` and
    ANDed into the ranking mask by the sweep driver, so the two agree on what the op
    answers for.

    Deliberately not the functional driver's sampling window, which is where a few
    thousand points are drawn, not where the op stops being defined: Abs is sampled on
    (-10, 10) and defined everywhere. Nor ``_SFPU_UNDEFINED_RANGES``, whose holes are
    guard bands that keep a random draw off a singularity: Reciprocal's is
    (-1e-6, 1e-6), and ``1/1e-7`` is a finite bf16 answer an inf must not be excused on.
    """
    from helpers.sfpu_domains import (
        _NONPOSITIVE_INTEGER_POLES,
        _OP_SINGULARITIES,
        Operand,
        SingularitySide,
    )

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


_claimed = claimed_lanes


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
      there is the flush. As generated or as the block-float quantizer hands it to the
      golden: the sweep's ``-0.0`` becomes ``-2**-127`` in a Bfp8_b block of subnormals.
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
    * **an answer the Dest cannot hold** (:func:`dest_holds`), on the same grounds as
      in the mask: past an fp16 Dest under a wider output, what comes back is the
      Dest's overflow, not the op's.
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

    normal_input = _normal_input(src, input_format, output_format, dest_acc)
    output_max = torch.finfo(format_dict[stimuli_format_for(output_format)]).max
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
        past_range & (saturated | _at_a_singularity(op, src, input_format))
    )
    return (
        nonfinite_mismatches(golden, result)
        & normal_input
        & dest_holds(golden, input_format, output_format, dest_acc)
        & ~excused
        & _claimed(op, src, input_format, output_format, dest_acc)
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


#: How many offending lanes a non-finite verdict spells out. Two: the verdict is written
#: on each of the table's ~400 not-measurable rows, and at four the named lanes were a
#: tenth of the file and took it past the repo's 500 KB `check-large-files` limit
#: (test_the_table_fits_the_repos_file_size_limit). The count beside them says how many
#: there are; two show the kind.
NAMED_LANES = 2


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

#: {op_name: {(in, out, approx, dest): max_ulp}}, filled during an emitting session. A
#: value is the worst lane's step count, a :class:`Floored` measurement, or the reason
#: a cell could not be measured.
MEASURED: Dict[str, Dict[Tuple[str, str, str, str], Union[int, str, "Floored"]]] = {}

#: Headroom over the measured worst lane. A 16-bit input is swept exhaustively, so unlike a
#: sampled measurement there is no unseen tail to leave room for -- but a budget at
#: exactly the maximum fails on any movement at all, including a golden that gets more
#: accurate. A strided Float32 input does leave a tail unseen; the headroom is the same,
#: and only its 0 is treated as a sample (:func:`_verdict`).
EMIT_HEADROOM = 1.1

#: The widest ``near_zero_atol`` the emitter writes on its own. A cell whose only
#: over-ceiling lanes sit in the near-zero band is enrolled with the floor that covers
#: them (:func:`near_zero_floor`); past this the floor would be forgiving a kernel's
#: cut-off rather than its rounding -- Softplus answers 0 below x = -5 where the answer
#: is up to 0.0065, the approximate Gelu is 0.024 off -- and that is a decision to make
#: by hand, with the figure the demotion note records.
EMIT_MAX_NEAR_ZERO_ATOL = 1e-3


@dataclass(frozen=True)
class Floored:
    """A cell's measurement with the near-zero floor that makes it gateable: the worst
    lane over every ranked lane, the worst lane outside the floor's band, and the
    floor itself (see :func:`near_zero_floor`)."""

    max_ulp: int
    residual: int
    atol: float


def _round_up_sig(x: float, digits: int = 2) -> float:
    """*x* rounded up to *digits* significant figures, so a printed floor is never below
    the error it was derived from."""
    if x == 0:
        return 0.0
    exponent = math.floor(math.log10(x))
    scale = 10.0 ** (exponent - digits + 1)
    return math.ceil(x / scale - 1e-9) * scale


def near_zero_floor(
    golden: torch.Tensor,
    result: torch.Tensor,
    distance: torch.Tensor,
    mask: torch.Tensor,
    out_fmt: DataFormat,
) -> Optional[Tuple[int, float]]:
    """The ``(residual, atol)`` a demoted cell would be enrolled with, or ``None``.

    The step count is a difference of bit-pattern ranks, so a lane where the hardware
    answers 0 for a 2e-9 the golden still holds, or where two values of 1e-24 differ in
    sign, reads as the whole exponent range -- 14,337 bf16 steps, 2**29 fp32 steps --
    and demotes the cell, although the error is nothing a consumer could measure.
    815 of the 943 non-block cells on tolerance at P9 were demoted by such a lane. The
    table already answers that with ``near_zero_atol``, the floor under which a lane is
    judged on absolute error (:func:`ulp.ulp_elementwise_valid`); this derives it from
    the measurement so the emitter can write it.

    The band is the floor rule's own: lanes under ``NEAR_ZERO_FRACTION`` of the cell's
    largest finite golden, capped at ``atol / NEAR_ZERO_FRACTION``. The floor is the
    smallest that does the job: the largest absolute error among the band lanes that
    are *past the ceiling* -- a band lane within budget needs no rescue, and taking the
    band's largest error instead widened Gelu's 7.6e-5 floor to 1e-3 for lanes its
    step budget already covered -- with ``EMIT_HEADROOM``, rounded up to two figures.
    It is granted only up to ``EMIT_MAX_NEAR_ZERO_ATOL``, and only when the lanes it
    does not rescue then fit the output's usable ceiling. A cell that fits without it
    gets none.
    """
    from helpers.sfpu_accuracy_budget import usable_budget_ceiling
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
    atol = _round_up_sig(float(absolute_error[demoting].max()) * EMIT_HEADROOM)
    if atol == 0:
        return None
    atol = min(atol, EMIT_MAX_NEAR_ZERO_ATOL)
    rescued = band & (magnitude <= atol / NEAR_ZERO_FRACTION) & (absolute_error <= atol)
    kept = selected & ~rescued
    residual = int(flat[kept].max()) if kept.any() else 0
    if residual > ceiling:
        return None
    return residual, atol


def _max_of(value) -> int:
    return value.max_ulp if isinstance(value, Floored) else int(value)


def record(
    op_name: str,
    key: Tuple[str, str, str, str],
    max_ulp: int,
    floor: Optional[Tuple[int, float]] = None,
) -> None:
    """Fold one measurement into the cell *key* names, keeping the *worst* lane.
    *floor* is :func:`near_zero_floor`'s ``(residual, atol)`` when the cell has one.

    The sweep driver records each cell once per process, so a repeat comes from
    :func:`merge_measured` folding in another xdist worker's reading of it -- or from a
    future driver that enumerates an axis the key does not have. Last-write-wins would
    keep whichever arrived last, which is the polarity that can hide error; ``max`` is
    the one that cannot. For the same reason a cell already recorded unmeasurable
    (:func:`record_unmeasurable`) stays so: a number arriving later does not rescue it.
    """
    cells = MEASURED.setdefault(op_name, {})
    current = cells.get(key)
    if isinstance(current, str):
        return  # already unmeasurable; a reading elsewhere does not rescue it
    new = Floored(max_ulp, *floor) if floor else max_ulp
    if current is None or _max_of(new) > _max_of(current):
        cells[key] = new


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
        [op, list(key), asdict(value) if isinstance(value, Floored) else value]
        for op, cells in MEASURED.items()
        for key, value in cells.items()
    ]


def merge_measured(rows) -> None:
    """Fold a worker's :func:`export_measured` into this process, worst lane winning."""
    for op, key, value in rows:
        if isinstance(value, str):
            record_unmeasurable(op, tuple(key), value)
        elif isinstance(value, dict):
            record(op, tuple(key), value["max_ulp"], (value["residual"], value["atol"]))
        else:
            record(op, tuple(key), value)


def _incomplete_grids() -> List[Tuple[str, str, str, int]]:
    """Each touched ``(op, in, out)`` missing a cell :func:`sweep_cells` runs, with how
    many it is missing.

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
                gaps.append((op, pair[0], pair[1], len(missing)))
    return gaps


def sweep_run(arch) -> RunIdentity:
    """This session's run identity, as the key lines it writes name it. Float32 has
    2^32 values and one run holds 2^16, so its input is strided, and calling it
    exhaustive would overstate every row keyed on it."""
    from datetime import date

    walked = "/".join(f.name for f in SWEEP_INPUT_FORMATS if is_exhaustive(f))
    strided = "/".join(f.name for f in SWEEP_INPUT_FORMATS if not is_exhaustive(f))
    return RunIdentity(
        sweep=f"exhaustive {walked}"
        + (f" + strided {strided}" if strided else "")
        + " sweep",
        arch=arch.value,
        date=date.today().isoformat(),
    )


def finish_emit(arch, testsfailed: int, path=None, exitstatus=0) -> WriteReport:
    """Write this session's measurements into the table, and report what was written.

    * **nothing written**, :class:`WrongArch` off ``MEASURED_ARCH``, where unkeyed rows
      would carry another arch's numbers under Wormhole's name; :class:`SessionNotClean`
      after a failure, when only a subset was measured, or when *exitstatus* says the
      session did not run to the end (pytest calls ``pytest_sessionfinish`` after a
      Ctrl-C too, and an interrupt between two ops leaves every touched grid complete
      and nothing counted as failed); :class:`IncompleteGrid` when an op's ``(in,
      out)`` grid is incomplete, when a write would drop the rest's rows.
    * **written, then** :class:`UnplacedMeasurements` when an op was measured with no
      key line to write into. Every *other* op's block has already been rewritten, so a
      red emit is not an untouched table.
    * **written**, returning the :class:`WriteReport`. An op kept verbatim -- a row the
      run covers carries a field ``_render`` cannot put back, the ``atol``/``rtol``
      anchors -- is named in it: such a block is maintained by hand by design, and
      failing on it would turn every whole-table emit red.

    *exitstatus* is pytest's ``session.exitstatus``, an ``ExitCode`` or a plain int.
    pytest has already set it to ``TESTS_FAILED`` when a test failed, so the failure
    count is read first, to give that run its own message.
    """
    import pytest
    from helpers.sfpu_accuracy_budget import _TABLE_PATH, MEASURED_ARCH

    if arch != MEASURED_ARCH:
        raise WrongArch(arch)
    if testsfailed or exitstatus != pytest.ExitCode.OK:
        raise SessionNotClean(testsfailed, exitstatus)
    gaps = _incomplete_grids()
    if gaps:
        raise IncompleteGrid(gaps)
    return write_table(path or _TABLE_PATH, sweep_run(arch))


def _verdict(
    measured: int, out_fmt: str, in_fmt: Optional[str] = None, exact: bool = False
) -> Tuple[str, int]:
    """What the table should say for a measured worst lane on *out_fmt*.

    ``("ulp", budget)`` while the measurement fits the table's ``usable_budget_ceiling``
    -- 419,430 steps for fp32, 51 for fp16, 6 for bf16, 25 for Bfp8_b -- with the
    budget the 1.1x headroom gives, capped at the ceiling. ``("tolerance", budget)``
    once the measurement itself is past it; the budget it would have needed goes on the
    row beside the measurement. Decided per cell and before collapsing, because it
    depends on the output format and collapsing may drop it.

    Without this the sweep enrols what it should not. ``Abs`` measures 393 steps on a
    Bfp8_b output from a bf16 input -- the block exponent quantizing a small element, not
    the op -- and a 433-step budget on a format whose ceiling is 25 gates nothing.

    Headroom is 1.1x, except at zero on an input the sweep enumerates (*in_fmt*, see
    :func:`is_exhaustive`): it saw every value, so a measured 0 means the op is exactly
    rounded on this format, and widening it to 1 retires that claim. A strided Float32
    input saw 65,280 of 2**32, and a finite sample cannot assert exactness, so its 0 is
    written as 1 -- the table's rule for sampled rows -- unless the op is *exact* on the
    cell (:func:`_is_exact`), whose 0 is the construction's, not the sample's.
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
    # In exact arithmetic: `100 * 1.1` is 110.00000000000001 in binary floating point,
    # so a float ceil wrote 111, one step past the rule. And from `str`: `Fraction(1.1)`
    # is the binary double, which gives 12 for a measured 10 where the rule says 11.
    budget = math.ceil(Fraction(measured) * Fraction(str(EMIT_HEADROOM)))
    if (
        budget == 0
        and not exact
        and in_fmt is not None
        and not is_exhaustive(DataFormat[in_fmt])
    ):
        budget = 1
    ceiling = usable_budget_ceiling(DataFormat[out_fmt])
    if budget > ceiling:
        if measured <= ceiling:
            # The kernel meets the gate and only the headroom does not (a bf16 cell
            # measuring 6, ceiling 6). Cap at the ceiling: still stronger than the tolerance
            # it replaces, with zero slack so any drift fails.
            return ("ulp", int(ceiling))
        # Only reached with the measurement itself past the ceiling: no budget would be
        # tighter than the tolerance. The row names both numbers.
        return ("tolerance", budget)
    return ("ulp", budget)


def _decide(
    cells: Dict[Tuple[str, str, str, str], Union[int, str]],
    op_name: Optional[str] = None,
) -> Dict[Tuple, Tuple]:
    """Each measured cell as ``(verdict, measured, extra)``; an unmeasurable one as
    ``(("unmeasurable", why), None, None)``. The verdict is :func:`_verdict`'s for the
    cell's output *and* input format -- a strided Float32 0 is written as 1 -- and for
    whether *op_name* is exact on that cell (:func:`_is_exact`).

    A :class:`Floored` cell whose worst lane demotes it is judged on the lanes outside
    its floor: enrolled at the residual's budget with ``extra = ("floor", atol,
    max_ulp)``, or, when even those are past the ceiling, demoted with the floor that
    was tried in ``extra = ("floor_short", atol, residual)`` so the note can say so."""
    decided = {}
    for key, v in cells.items():
        exact = op_name is not None and _is_exact(op_name, key[0], key[1], key[3])
        if isinstance(v, str):
            decided[key] = (("unmeasurable", v), None, None)
        elif isinstance(v, Floored):
            plain = _verdict(v.max_ulp, key[1], key[0], exact)
            floored = _verdict(v.residual, key[1], key[0], exact)
            if plain[0] == "tolerance" and floored[0] == "ulp":
                decided[key] = (floored, v.residual, ("floor", v.atol, v.max_ulp))
            elif plain[0] == "tolerance":
                decided[key] = (plain, v.max_ulp, ("floor_short", v.atol, v.residual))
            else:
                decided[key] = (plain, v.max_ulp, None)
        else:
            decided[key] = (_verdict(v, key[1], key[0], exact), v, None)
    return decided


def _collapse(decided: Dict[Tuple, Tuple]) -> List[dict]:
    """The decided cells as the fewest rows that reproduce them.

    Only ``approx`` and ``dest`` may be dropped. ``in`` and ``out`` stay pinned even
    when every value agrees, because the sweep drives a fixed set of formats: a row that
    wildcards the output would, by most-specific-wins, also answer for the block floats
    below Bfp8_b and for any format a driver adds later, which nothing here measured. `Abs` losing its
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
            raise AmbiguousCollapse(
                axes, [axes[i] for i in keep], collapsed, (merged[collapsed], value)
            )
        merged[collapsed] = value

    rows = []
    for key, (verdict, measured, extra) in sorted(merged.items()):
        row = {axes[i]: v for i, v in zip(keep, key)}
        row["verdict"] = verdict
        row["measured"] = measured
        row["extra"] = extra
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Writing the table
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WriteReport:
    """What :func:`write_table` did, as data: tests assert on these fields, and the
    terminal line is rendered from them (:meth:`summary`)."""

    path: Path
    #: op -> the rows its block now has.
    written: Dict[str, List[Row]]
    #: op -> why its block was kept verbatim: the field a covered row carries that
    #: ``_render`` cannot put back (``atol``, ``near_zero_atol`` before P10, ...), or
    #: ``alias`` for a row whose fields live on an anchor.
    kept: Dict[str, str]
    #: Ops measured with no key line to write into.
    unplaced: List[str]
    #: Rows this run did not supersede that were given the run identity they had.
    stamped: List[Row]

    def summary(self) -> str:
        message = (
            f"--ulp-emit: rewrote {len(self.written)} op block(s) in {self.path.name}"
        )
        if self.kept:
            named = ", ".join(f"{op} ({why})" for op, why in sorted(self.kept.items()))
            message += (
                f"; kept {named} verbatim: a row this sweep covers carries a field it "
                f"cannot regenerate (beyond {sorted(_RENDERABLE_FIELDS)}), so their "
                "measurements were not written. Settle those cells by hand"
            )
        return message


class EmitRefused(RuntimeError):
    """``--ulp-emit`` refused to write, or wrote only part of what it measured. Each
    subclass carries the facts as attributes; the message is for the terminal."""


class WrongArch(EmitRefused):
    def __init__(self, arch):
        from helpers.sfpu_accuracy_budget import MEASURED_ARCH

        self.arch = arch
        super().__init__(
            f"ran on {arch.value}, but the table's unkeyed rows are read as "
            f"{MEASURED_ARCH.value} measurements and `_render` does not emit `arch`. "
            "Nothing written."
        )


class SessionNotClean(EmitRefused):
    """The session failed tests, or did not run to its end."""

    def __init__(self, failures: int, exitstatus):
        import pytest

        self.failures, self.exitstatus = failures, exitstatus
        if failures:
            message = (
                f"saw {failures} failure(s), so the session measured a subset. Nothing "
                "written -- emit from a clean run."
            )
        else:
            try:
                name = pytest.ExitCode(exitstatus).name
            except ValueError:  # pytest.exit(returncode=...) can carry any int
                name = str(exitstatus)
            message = (
                f"the session ended with exit status {name}, not OK, so what it "
                "measured is whatever it got to. Nothing written -- emit from a run "
                "that ends on its own."
            )
        super().__init__(message)


class IncompleteGrid(EmitRefused):
    """*gaps* is ``(op, in, out, unmeasured cells)`` for each touched ``(in, out)``
    missing a cell :func:`sweep_cells` runs."""

    def __init__(self, gaps: List[Tuple[str, str, str, int]]):
        self.gaps = gaps
        super().__init__(
            "measured only part of an op's grid, and a write would drop the rows of "
            "the rest. Nothing written -- emit whole (in, out) grids:\n  "
            + "\n  ".join(
                f"{op} {i}->{o}: {missing} cell(s) unmeasured"
                for op, i, o, missing in gaps
            )
        )


class AmbiguousCollapse(EmitRefused):
    """Collapsing the recorded cells would merge two different measurements onto one
    row: *cells* is the collapsed key, and *values* the two decisions it would hold."""

    def __init__(self, axes, kept_axes, cells, values):
        self.cells, self.values = cells, values
        super().__init__(
            f"collapsing {list(axes)} to {list(kept_axes)} merges two different "
            f"measurements onto {cells}: {values[0]} and {values[1]}. The recorded "
            "cells do not form a full grid -- emit from a complete run."
        )


class UnplacedMeasurements(EmitRefused):
    """:func:`write_table` wrote every op it could, and measured some it had no key line
    for. *report* is what a normal return would have been; *missing* the ops it could
    not place."""

    def __init__(self, report: WriteReport):
        self.report, self.missing = report, report.unplaced
        super().__init__(
            f"{report.summary()}; but measured {', '.join(report.unplaced)} and the "
            "table has no key line for them, so they were not written. Give an op its "
            "block first (SFPU_ULP.md, step 2) to enrol it."
        )


#: What `_render` can put back. A row carrying anything else -- an `atol`/`rtol` pair --
#: cannot be regenerated from a measurement, so it is preserved rather than replaced
#: even when this sweep covers its cell.
_RENDERABLE_FIELDS = frozenset(
    {"in", "out", "approx", "dest", "max_ulp", "metric", "near_zero_atol"}
)


def _provenance_of(row: dict) -> Provenance:
    """The note :func:`_render` writes beside a decided row. The first figure is the one
    the budget was derived from, which is what the audits and the headroom report read.
    """
    from helpers.sfpu_accuracy_budget import usable_budget_ceiling

    metric, value = row["verdict"]
    extra = row.get("extra")
    if metric == "unmeasurable":
        return Provenance.parse(f"not measurable: {value}")
    if metric == "tolerance":
        # Just the two numbers: the reason is in the table header, and this pair keeps
        # the claim checkable against `usable_budget_ceiling`.
        floor = {}
        if extra and extra[0] == "floor_short":
            floor = dict(floor=extra[1], residual=extra[2])
        return Provenance(
            Kind.DEMOTED,
            measured=row["measured"],
            budget_needed=value,
            ceiling=round(usable_budget_ceiling(DataFormat[row["out"]])),
            **floor,
        )
    if metric == "block":
        return Provenance(Kind.BLOCK, measured=row["measured"])
    if extra and extra[0] == "floor":
        return Provenance(
            Kind.FLOORED,
            measured=row["measured"],
            floor=extra[1],
            measured_all=extra[2],
        )
    return Provenance(Kind.EMITTED, measured=row["measured"])


def _render(
    op: str, key_line: KeyLine, rows: List[dict], run: RunIdentity
) -> List[str]:
    """One op's block: the key line credited to *run*, and each row with its verdict
    and the measurement behind it.

    The key line keeps whatever header it already had: `Fill:  # 0 ULP, 115 variants`
    is the provenance for every row this sweep does not reach."""
    out = [f"{op}:  # {key_line.with_run(run).render()}\n"]
    for row in rows:
        metric, value = row["verdict"]
        fields = {k: row[k] for k in KEY_AXES if k in row}
        if metric == "ulp":
            fields["max_ulp"] = value
        else:
            fields["metric"] = "tolerance"
        extra = row.get("extra")
        if metric == "ulp" and extra and extra[0] == "floor":
            fields["near_zero_atol"] = f"{extra[1]:.2e}"
        out.append(render_row(fields, _provenance_of(row)))
    return out


def _stamp(row: Row, key_line: KeyLine) -> Optional[Provenance]:
    """The provenance *row* needs once its key line names this run instead of the one
    it had, or ``None`` if it needs none.

    ``_render`` replaces the key line's run, and a row this run did not supersede would
    then be credited to it -- a pair-narrowed re-emit, or a sampled ``{out: Float32,
    max_ulp: 0}`` under an exhaustive run the sweep cannot produce. So the run goes onto
    those rows first: an emitter-written note naming no run (``max 1 ULP``, from an
    earlier emit) gets the outgoing run, or the key line's header when it had none, and a
    bare row -- hand-authored, never emitted -- the header, which is the provenance it
    was written against.

    Left alone: a row naming its own run, a hand-written note (``see the Bfp8_b note
    above``, a sample's figures) since no run on the key line measured it and crediting
    one would be a guess, and an ``arch:`` row, which a run on this arch never measures.
    """
    if not row.alias and "arch" in row.pinned:
        return None
    outgoing, header = key_line.run, key_line.header
    p = row.provenance
    if p is None:
        origin = header or (outgoing.render() if outgoing else "")
        return Provenance.parse(origin) if origin else None
    if p.kind not in EMITTER_KINDS or p.names_its_run:
        return None
    if outgoing is not None:
        return p.with_run(outgoing)
    return Provenance.parse(f"{p.render()}, {header}") if header else None


def _pins_a_measured_cell(row: Row, measured: Set[Tuple[str, str, str, str]]) -> bool:
    """Whether some cell this run measured resolves through *row*'s key: every key
    field *row* pins agrees with it. A wildcard row -- an op-wide ``{metric:
    tolerance, atol: 0.13}``, or a YAML alias whose fields are not inline -- pins
    nothing, so it answers for every cell.

    Wider than :func:`_covered`, which asks whether the run measured the row's own
    ``(in, out)``: a rendered row is more specific than a wildcard, so it would shadow
    the wildcard's ``atol``/``rtol`` on every cell it names.
    """
    if row.alias:
        return True  # its fields live on the anchor, so assume the widest
    pinned = [
        (i, row.pinned[axis]) for i, axis in enumerate(KEY_AXES) if axis in row.pinned
    ]
    return any(all(key[i] == value for i, value in pinned) for key in measured)


def _unrenderable(row: Row) -> Optional[str]:
    """What *row* carries that ``_render`` cannot put back -- the fields beyond
    ``_RENDERABLE_FIELDS``, or ``alias`` for a row whose fields are not inline -- or
    ``None``. Arch-keyed rows are never touched, so they are not counted."""
    if row.alias:
        return "alias"
    if "arch" in row.pinned:
        return None
    extra = sorted(set(row.values) - _RENDERABLE_FIELDS)
    return ", ".join(extra) or None


def _covered(row: Row, emitted_cells: Set[Tuple[str, str]]) -> bool:
    """Whether this run measured the ``(in, out)`` cell *row* declares.

    Only the cells actually in ``MEASURED`` for this op, never the static
    ``SWEEP_FORMATS`` cross-product: a partial run -- ``-k``, an interrupt, a driver
    skip -- must replace what it measured and leave the rest alone, rather than
    rendering a whole op block from an incomplete session.
    """
    if row.alias:
        return False
    return (row.pinned.get("in", ""), row.pinned.get("out", "")) in emitted_cells


def _replaceable(row: Row, emitted_cells: Set[Tuple[str, str]]) -> bool:
    """Whether this run's output supersedes *row*.

    A row that pins ``arch`` is never replaceable. This sweep runs on one architecture,
    and the table's own header says to re-measure on Blackhole; specificity lets the two
    rows coexist, so regenerating one arch must not erase the other's contract.

    Nor is a row carrying a field ``_render`` cannot put back -- an ``atol``/``rtol``
    pair. :func:`write_table` keeps such an op's block verbatim and names it in its
    report, rather than quietly replacing or quietly duplicating the row.
    """
    if row.alias or "arch" in row.pinned or _unrenderable(row):
        return False
    return _covered(row, emitted_cells)


def rounds_at_pack(in_fmt: str, out_fmt: str, dest: Optional[str]) -> bool:
    """Whether a cell's output holds fewer mantissa bits than its input still has in
    Dest: there the operand an exact op returns -- a selected value, an integer -- is
    rounded at pack, as Abs, Neg and Identity's 1-step Float32 -> Float16_b cells record.
    Dest keeps the input's precision except where a 32-bit input meets a 16-bit Dest
    (*dest* ``"No"``), which the unpack has narrowed already. A 16-bit input keeps it in
    a 16-bit Dest too: Float16 -> Float16_b at ``dest: "No"`` reads 1 step on Abs and
    ReluMin. Format names and the ``dest`` value as the table writes them; *dest* None
    means either.
    """
    from helpers.ulp import _ULP_DTYPES, has_ulp_gate, ulp_dtype

    def mantissa_bits(name: str) -> int:
        fmt = DataFormat[name]
        return _ULP_DTYPES[ulp_dtype(fmt)].mantissa_bits if has_ulp_gate(fmt) else 0

    if dest == "No" and DataFormat[in_fmt].is_32_bit():
        return False
    return mantissa_bits(out_fmt) < mantissa_bits(in_fmt)


def _is_exact(
    op_name: str,
    in_fmt: Optional[str] = None,
    out_fmt: Optional[str] = None,
    dest: Optional[str] = None,
) -> bool:
    """Whether a measured 0 on this cell is *op_name*'s construction rather than its
    sample: an op exact in every format, or an exact op on a cell the pack cannot round
    (:func:`rounds_at_pack`). On a cell that does round, a passed-through value is
    rounded, so a strided 0 says only that no sample hit a tie: the walk holds one
    fp32 -> bf16 tie, which ReluMin and UnaryMin send to a constant (0 steps) and Abs
    passes through (1 step). A cell that leaves `in` or `out` open cannot be judged
    and is taken as exact, as the table's op-wide canary rows are."""
    from helpers.sfpu_accuracy_budget import (
        EXACT_BY_CONSTRUCTION_OPS,
        EXACT_IN_EVERY_FORMAT_OPS,
    )

    if op_name in {op.name for op in EXACT_IN_EVERY_FORMAT_OPS}:
        return True
    if op_name not in {op.name for op in EXACT_BY_CONSTRUCTION_OPS}:
        return False
    if in_fmt is None or out_fmt is None:
        return True
    return not rounds_at_pack(in_fmt, out_fmt, dest)


def _block_end(lines: List[str], start: int) -> int:
    """The index past an op block starting at *start*: its indented and blank lines,
    less the blank lines after it, which are left for the next block."""
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0].isspace()):
        end += 1
    while end - 1 > start and not lines[end - 1].strip():
        end -= 1
    return end


def _restamped(table: BudgetTable, row: Row, provenance: Provenance) -> List[str]:
    """*row*'s lines with its comment replaced by *provenance*."""
    lines = table.lines[row.first_line : row.last_line + 1]
    body = lines[-1][: row.end_column].rstrip()
    return lines[:-1] + [f"{body}  # {provenance.render()}\n"]


def write_table(path, run: RunIdentity) -> WriteReport:
    """Replace every swept op's block in the YAML with what the sweep measured, credited
    to *run*.

    Line-oriented on purpose. The table's comments carry its provenance, and a load and
    re-dump through PyYAML would drop every one of them, including for the ops this
    sweep never touched. Which lines are an op's key line and rows is read from
    :class:`BudgetTable`, never from the shape of a line: matching key lines by a
    trailing colon once skipped 17 ops whose key line carries a header comment, and left
    their sampled rows in place looking measured.

    Raises :class:`UnplacedMeasurements` if an op in ``MEASURED`` has no key line to
    write into -- *after* writing every op that has one, so one unenrolled op does not
    throw away the rest of a whole-table emit. The measurement would otherwise be
    dropped in silence, and the sampled rows it was meant to replace would stay in place
    looking measured.
    """
    path = Path(path)
    table = BudgetTable.load(path)
    lines = table.lines
    starts = {block.line: block for block in table.blocks.values()}
    out: List[str] = []
    written: List[str] = []
    kept_verbatim: Dict[str, str] = {}
    stamped_at: List[int] = []
    i = 0
    while i < len(lines):
        block = starts.get(i)
        if block is None or block.op not in MEASURED:
            out.append(lines[i])
            i += 1
            continue
        name, end = block.op, _block_end(lines, i)
        cells = MEASURED[name]
        emitted_cells = {(k[0], k[1]) for k in cells}
        # Any row a measured cell resolves through, not only one pinning the measured
        # (in, out): an op-wide `{metric: tolerance, atol: 0.13}` would otherwise be
        # shadowed by the bare rows rendered below it, and every cell they name would
        # silently lose its declared atol.
        why = sorted(
            {
                reason
                for row in block.rows
                if _pins_a_measured_cell(row, set(cells))
                for reason in [_unrenderable(row)]
                if reason
            }
        )
        if why:
            # Emitting over it would drop the row; keeping it as well would give the
            # cell two equally specific keys, which `_load_table` refuses. So the op's
            # block stays as it is, and the caller is told: the row is a judgement a
            # measurement cannot re-derive. Every other op is written.
            out.extend(lines[i:end])
            kept_verbatim[name] = ", ".join(why)
            i = end
            continue
        out.extend(
            _render(
                name,
                block.key_line,
                _collapse(_decide(cells, name)),
                run,
            )
        )
        # Rows this run did not supersede -- a format it does not reach, an arch-keyed
        # entry -- are the measurement of a different run and stay as they are.
        # Replacing a whole op block deleted them.
        for row in block.rows:
            if _replaceable(row, emitted_cells):
                continue
            stamp = _stamp(row, block.key_line)
            if stamp is None:
                out.extend(lines[row.first_line : row.last_line + 1])
            else:
                stamped_at.append(len(out) + row.last_line - row.first_line)
                out.extend(_restamped(table, row, stamp))
        written.append(name)
        i = end
    # Exactly one trailing newline: an op block carries its own trailing blank lines,
    # and the last block's leave the file ending in several. `end-of-file-fixer` then
    # rewrites the table on every commit.
    path.write_text("".join(out).rstrip("\n") + "\n", encoding="utf-8")
    after = BudgetTable.load(path)
    report = WriteReport(
        path=path,
        written={op: after.rows_of(op) for op in written},
        kept=kept_verbatim,
        unplaced=sorted(set(MEASURED) - set(written) - set(kept_verbatim)),
        stamped=[row for row in after.rows if row.last_line in set(stamped_at)],
    )
    if report.unplaced:
        raise UnplacedMeasurements(report)
    return report

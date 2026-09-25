# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive 16-bit ULP sweeps, and folding what they measure back into the table.

The functional drivers sample a few thousand points from an op's *safe* domain. A budget
measured that way describes the sample, not the format: approximate ``Reciprocal`` reads
1 ULP on ``uniform(0.1, 1.1)`` and 128 ULP over every bf16 value there is. This module
measures the second number.

Exhaustive is only honest for the 16-bit formats. bfloat16 has 65,279 finite values and
float16 63,487, so either fits one 64-tile device run; Float32's 2**32 does not, and is
left to a stratified sweep. ``Bfp8_b`` is swept in bfloat16 and packed on the way in --
it has no enumerable value set of its own.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Dict, List, Set, Tuple

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
    DataFormat.Float32,
)

#: Formats the sweep drives as an *input* but never judges as an output. Bfp4_b keeps
#: 2 fractional bits, so a bf16 step count would read every legal quantization of its
#: output as a 32-step error -- but it is a perfectly good thing to feed, and gated
#: cells do take it.
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

#: Float32 has 2**32 values and one device run holds 2**16, so it cannot be enumerated.
#: Striding the total order by this much samples it evenly instead: every binade holds
#: the same number of representable values, so each gets an equal share, and one run
#: reaches 261 binades from 0 to 3.4e38. A consecutive walk covers a millionth of one
#: binade and would call that a measurement.
#:
#: This is the one place the sweep is a *sample* rather than exhaustive.
_FP32_STRIDE = 2**16

#: A top-level op key in the table.
_OP_KEY = re.compile(r"^([A-Za-z_]\w*):")

_INF = float("inf")


def stimuli_format_for(fmt: DataFormat) -> DataFormat:
    """The format whose values the sweep enumerates in order to drive *fmt*."""
    return _STIMULI_FORMAT.get(fmt, fmt)


def is_exhaustive(input_format: DataFormat) -> bool:
    """Whether the sweep sees *every* value the input can take, or a stride of them."""
    return stimuli_format_for(input_format) != DataFormat.Float32


@lru_cache(maxsize=None)
def swept_value_count(input_format: DataFormat) -> int:
    """How many values the sweep actually generates for *input_format*.

    Not ``ulp_sweep_value_count``, which answers how many the format *has*: 2**32 for
    float32, where the sweep generates 2**16 of them.

    Cached because the answer is a property of the format and the walk, while finding
    it enumerates the whole format: 2.4 ms a call, and `padding_lanes` asks twice per
    variant, which is ~37 s of recomputation across an emit run over five formats.
    """
    from helpers.stimuli_generator.strategies.structured import (
        _enumerate_representable,
    )

    fmt = stimuli_format_for(input_format)
    stride = 1 if is_exhaustive(input_format) else _FP32_STRIDE
    return int(_enumerate_representable(fmt, -_INF, _INF, 2**16, stride).numel())


def sweep_spec(input_format: DataFormat = DataFormat.Float16_b) -> StimuliSpec:
    """Every finite representable value of the stimuli format, once -- or, for float32,
    every ``_FP32_STRIDE``-th, since 2**32 values do not fit one run.

    Deliberately not clipped to the op's domain. ``exclude_undefined`` expresses a domain
    as ``intervals``, which ULP_SWEEP does not read -- and clipping would also stop the
    undefined inputs reaching hardware at all. They are swept and then masked out of the
    statistics by :func:`measurable_mask`, so the run still exercises them.
    """
    stride = 1 if is_exhaustive(input_format) else _FP32_STRIDE
    return StimuliSpec.ulp_sweep(low=-_INF, high=_INF, stride=stride)


def padding_lanes(src: torch.Tensor, input_format: DataFormat) -> torch.Tensor:
    """The tail ``generate_full_tensor`` fills with zeros to reach the tile count.

    The sweep enumerates every finite value of the stimuli format -- 65,279 for
    bfloat16, 63,487 for float16 -- into a fixed 65,536-lane tensor, so the last 257
    (or 2,049) lanes are padding rather than data. They are not values the sweep chose
    to feed, and they are all the same one, so they belong in no statistic: they
    inflate every lane count, and on an op singular at zero whose registered domain
    includes it they would read as a real failure. ``reciprocal`` is the near miss --
    the hardware returns ``Inf`` there against a finite golden clamp, and only its
    registered domain excluding zero keeps those 257 lanes out of the verdict.

    Identified by position rather than by value, because ``0.0`` is also a legitimate
    swept value: exactly one, in the middle of the sorted order. Confirmed on hardware
    that the padding is the contiguous tail.

    Counted by :func:`swept_value_count`, not by how many values the format has: a
    strided float32 walk generates 65,279 of 2**32, and asking the format would put the
    padding boundary past the end of the tensor and mask nothing.
    """
    swept = swept_value_count(input_format)
    # On *src*'s device: the mask is composed with tensors derived from it, and a
    # CPU-only mask would fail that composition for a device-resident sweep.
    flat = torch.zeros(src.numel(), dtype=torch.bool, device=src.device)
    flat[swept:] = True
    return flat.reshape(src.shape)


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
    clipping -- an op is measured wherever its format can reach. Three lane kinds come
    back out, none of them a budget question:

    * either side NaN. :func:`ulp_distance` returns ``UNMEASURABLE`` there, and an op
      undefined at an input (``log`` of a negative) lands here on its own.
    * the two sides disagreeing about being non-finite -- ``sin(2.6e28)`` returning
      ``inf`` against a golden of ``-1``, or a reciprocal overflowing where the golden
      is still finite. ``passed_test`` rejects those positionally whatever the budget
      says, so ranking them would inflate the number without tightening the gate. One
      such lane is worth ~48,000 steps.
    * the sweep's own zero padding -- see :func:`padding_lanes`.
    * subnormal inputs. The hardware flushes them on the way in and the golden does not,
      so ``ceil(5.69e-39)`` is 1 in the model and 0 on silicon -- 16,129 bf16 steps for
      a difference that is the unpack path's flush, not the op's accuracy. Measured, it
      is the whole of Ceil's, Floor's and Sqrt's apparent error: excluding it returns
      all three to the 0 their exactness claims, and moves nothing else. The flush is
      covered on its own terms elsewhere; a step count is the wrong instrument for it.

    Subnormal *outputs* stay in. Where the golden underflows and the hardware writes
    zero the count is large but the lane is a real one the op produced -- Silu at
    ``x=-87.5`` is that case, and it is the op's own tail, not the unpack path.

    The second kind is a *failure*, not a non-question, and dropping it here is only
    sound because :func:`nonfinite_failures` reports it separately: a caller that ranks
    this mask and nothing else would let a hardware overflow produce a clean budget.
    """
    # The threshold is the *stimuli* format's, not the golden's. Taking it from the
    # golden dtype silently passed every fp16 subnormal through on a Float16->Float16_b
    # variant -- bf16's smallest normal is 1.18e-38 and fp16's is 6.1e-05, so 2,046
    # flushed lanes read as a 14,337-step error on `Abs`, an op that cannot be wrong.
    from helpers.llk_params import format_dict

    from .ulp import nonfinite_mismatches

    stimuli_dtype = format_dict[stimuli_format_for(input_format)]
    smallest_normal = torch.finfo(stimuli_dtype).smallest_normal
    # In float32, and from `src` as generated. Casting to the golden's dtype first
    # rounds an fp16 subnormal *up* -- bf16 keeps 8 mantissa bits, so 6.09e-05 becomes
    # 6.10e-05 and clears a 6.10e-05 threshold. The whole subnormal band then passed
    # this filter while looking, in any printout, like the smallest normal.
    magnitude = src.detach().to(torch.float32).abs()
    normal_input = (magnitude >= smallest_normal) | (magnitude == 0)

    both_measurable = ~(torch.isnan(golden) | torch.isnan(result))
    return (
        both_measurable
        & ~nonfinite_mismatches(golden, result)
        & normal_input
        & ~padding_lanes(src, input_format)
    )


def _within_safe_domain(
    op, src: torch.Tensor, input_format: DataFormat
) -> torch.Tensor:
    """Lanes whose input is inside the domain ``sfpu_domains`` registers for *op*.

    The sweep deliberately runs outside it -- that is the whole point, and the budget is
    measured over everything. This is used only to judge *non-finite* answers, where the
    distinction matters: outside the registered domain an op is not claiming anything,
    and ``sin(2.6e28)`` returning ``inf`` against a golden of ``-1`` is an argument
    reduction giving up, not a regression.
    """
    from helpers.sfpu_domains import exclude_undefined, for_op

    spec = exclude_undefined(op, for_op(op, input_format).spec_A)
    magnitude = src.detach().to(torch.float32)
    inside = torch.ones_like(magnitude, dtype=torch.bool)
    if spec.low is not None:
        inside &= magnitude >= spec.low
    if spec.high is not None:
        inside &= magnitude <= spec.high
    intervals = getattr(spec, "intervals", None)
    if intervals:
        covered = torch.zeros_like(inside)
        for low, high in intervals:
            covered |= (magnitude >= low) & (magnitude <= high)
        inside &= covered
    return inside


def nonfinite_failures(
    op,
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """The lanes :func:`measurable_mask` drops that are a *failure* rather than a
    non-question: the two sides disagreeing about being non-finite where the output
    format could have held the answer.

    ``passed_test`` rejects these positionally whatever the budget says, but the sweep
    driver ranks a distance rather than calling it, so it has to ask separately -- a
    hardware overflow or an unexpected NaN would otherwise leave the statistics clean
    and both emit and gate would pass.

    Four exclusions, all of them the sweep's own doing rather than the op's:

    * **subnormal inputs**, on the same grounds as in the mask -- the unpack path
      flushes them and the golden does not, so a disagreement there is the flush.
    * **a golden past the output format's finite range.** A full-range sweep feeds
      every value of a 16-bit input, and ``relu_min`` passes most of them straight
      through, so a bf16 input against a Float16 output reaches magnitudes fp16 cannot
      represent -- 14,334 lanes of it. Saturating there is the store doing what it must
      (on WH an fp16 destination overflow packs NaN, not Inf), not the kernel being
      wrong, and no budget on any op could be met.
    * **the sweep's own zero padding**, which is not a value it chose to feed.
    * **an input outside the op's registered safe domain.** ``Sin`` and ``Cos`` are
      registered over ``[-pi, pi]`` and disagree on ~21,000 bf16 lanes far outside it,
      which is the case ``measurable_mask``'s own docstring cites. The budget is still
      measured over the whole format; it is only the *non-finite* answer that needs the
      op to have been claiming something.

    What is left is the case the mask would otherwise hide: an op returning ``inf`` or
    ``NaN`` where it is defined, the input is normal, and the output could have held
    the answer.
    """
    from helpers.llk_params import format_dict

    from .ulp import nonfinite_mismatches

    stimuli_dtype = format_dict[stimuli_format_for(input_format)]
    magnitude = src.detach().to(torch.float32).abs()
    normal_input = (magnitude >= torch.finfo(stimuli_dtype).smallest_normal) | (
        magnitude == 0
    )
    output_max = torch.finfo(format_dict[stimuli_format_for(output_format)]).max
    in_range = golden.detach().to(torch.float32).abs() <= output_max
    return (
        nonfinite_mismatches(golden, result)
        & normal_input
        & in_range
        & _within_safe_domain(op, src, input_format)
        & ~padding_lanes(src, input_format)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Folding a sweep back into the table
# ─────────────────────────────────────────────────────────────────────────────

#: Set by ``--ulp-emit``. Measure and rewrite the table instead of gating against it.
EMIT = False

#: {op_name: {(in, out, approx, dest): max_ulp}}, filled during an emitting session.
MEASURED: Dict[str, Dict[Tuple[str, str, str, str], int]] = {}

#: Headroom over the measured worst lane. The sweep is exhaustive, so unlike a sampled
#: measurement there is no unseen tail to leave room for -- but a budget at exactly the
#: maximum fails on any movement at all, including a golden that gets more accurate.
EMIT_HEADROOM = 1.1


def record(op_name: str, key: Tuple[str, str, str, str], max_ulp: int) -> None:
    """Fold one measurement into the cell *key* names, keeping the *worst* lane.

    The key is four-dimensional and a driver enumerates more than four axes --
    ``fast_mode`` and ``input_dimensions`` are both multi-valued -- so one cell is
    recorded several times per emit run. Last-write-wins would keep whichever variant
    ran last, which is the polarity that can hide error; ``max`` is the one that cannot.
    """
    cells = MEASURED.setdefault(op_name, {})
    cells[key] = max(cells.get(key, 0), max_ulp)


def _verdict(measured: int, out_fmt: str) -> Tuple[str, int]:
    """What the table should say for a measured worst lane on *out_fmt*.

    ``("ulp", budget)`` while a step budget is still *stronger* than the tolerance it
    replaces, and ``("tolerance", budget)`` once it is not -- the budget either way, so
    the row's comment can name the number that actually crossed the line. The bound is the table's
    own ``usable_budget_ceiling``: 419,431 steps for fp32, 52 for fp16, 7 for bf16, 26
    for Bfp8_b. Decided per cell and before collapsing, because it depends on the output
    format and collapsing may drop it.

    Without this the sweep enrols what it should not. ``Abs`` measures 393 steps on a
    Bfp8_b output -- the block exponent quantizing a small element, not the op -- and a
    433-step budget on a format whose ceiling is 25 gates nothing at all.

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
        # not a representative one. Measured the two ways: `Abs` reads 393 steps on a
        # Bfp8_b output from random mixed-magnitude blocks and 3 from the sorted sweep.
        # Enrolling the second number would gate nothing and hide the first.
        return ("block", measured)
    budget = 0 if measured == 0 else math.ceil(measured * EMIT_HEADROOM)
    ceiling = usable_budget_ceiling(DataFormat[out_fmt])
    if budget > ceiling:
        if measured <= ceiling:
            # The kernel meets the gate; only the headroom does not. Exp measures 7 on
            # a bf16 output whose ceiling is 7, and 1.1x made that 8 -- refusing the
            # only step gate Exp could have on bf16 over rounding. Cap at the ceiling:
            # a budget sitting exactly on it is still stronger than the tolerance it
            # replaces, and zero slack means any drift fails, which is what a gate is
            # for. 14 cells on the 2026-09-25 sweep, all Exp.
            return ("ulp", int(ceiling))
        # The *budget* is what crosses the line, not the measurement, so the row's
        # comment names both and the claim stays checkable.
        return ("tolerance", budget)
    return ("ulp", budget)


def _decide(cells: Dict[Tuple[str, str, str, str], int]) -> Dict[Tuple, Tuple]:
    """Each measured cell as ``(verdict, measured)``, verdict decided per output format."""
    return {key: (_verdict(v, key[1]), v) for key, v in cells.items()}


def _collapse(decided: Dict[Tuple, Tuple]) -> List[dict]:
    """The decided cells as the fewest rows that reproduce them.

    Only ``approx`` and ``dest`` may be dropped. ``in`` and ``out`` stay pinned even
    when every value agrees, because this sweep covers the 16-bit formats only: a row
    that wildcards the output would, by most-specific-wins, also answer for Float32 and
    the block floats below Bfp8_b, which nothing here measured. `Abs` losing its
    Float32 row that way is what the registry's unswept-architecture guard caught.
    """
    axes = ("in", "out", "approx", "dest")
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


#: What a key line says about the run every emitted row below it came from. The sweep
#: identity is identical on every one of them -- ~63 characters times ~2,000 rows, a
#: quarter of the file -- so it is stated once per op instead. "except where a row says
#: otherwise" is not hedging: rows this run did not supersede keep their own suffix.
_MEASURED_BY = "measured by: {suffix}, except where a row says otherwise"

#: The same clause, for stripping a previous run's before writing this one's. Without
#: it a second `--ulp-emit` appends rather than replaces, and the key line accumulates
#: one stale run identity per regeneration.
_MEASURED_BY_RE = re.compile(
    r";?\s*measured by: .*?, except where a row says otherwise"
)


def _render(key_line: str, rows: List[dict], suffix: str) -> List[str]:
    """One op's block: each row with its verdict, and the measurement behind it.

    *key_line* keeps whatever it already said. Several ops carry their measurement as a
    header comment on that line -- `Fill:  # 0 ULP, 115 variants` -- and it is the
    provenance for every row of theirs this sweep does not reach. Rewriting the key as
    a bare `Fill:` dropped it, and the guard that every budget names its measurement
    then failed on rows that had one all along. The run identity is *appended* to it.

    Each row still carries its own number, which is what the provenance audit reads and
    what a budget may only be raised against. What moves to the key line is the part
    that is the same on every row: which sweep, on which arch, on which day.
    """
    order = ("in", "out", "approx", "dest")
    head, sep, comment = key_line.rstrip("\n").partition("#")
    measured_by = _MEASURED_BY.format(suffix=suffix)
    existing = _MEASURED_BY_RE.sub("", comment).strip().rstrip(";").strip()
    out = [f"{head.rstrip()}  # {existing + '; ' if existing else ''}{measured_by}\n"]
    for row in rows:
        metric, value = row["verdict"]
        body = ", ".join(
            f'{k}: "{row[k]}"' if k in ("approx", "dest") else f"{k}: {row[k]}"
            for k in order
            if k in row
        )
        decided = (
            "metric: tolerance"
            if metric in ("tolerance", "block")
            else f"max_ulp: {value}"
        )
        pairs = f"{body}, {decided}" if body else decided
        note = f"max {row['measured']} ULP"
        if metric == "tolerance":
            from helpers.sfpu_accuracy_budget import usable_budget_ceiling

            # Terse on purpose: the clause is repeated on every demoted row, and the
            # reason it names is stated once in the table header. What has to be *here*
            # is the pair of numbers, so the claim stays checkable against
            # `usable_budget_ceiling`.
            ceiling = usable_budget_ceiling(DataFormat[row["out"]])
            note += f", budget {value} > ceiling {ceiling:.0f}"
        elif metric == "block":
            note += ", block-quantized"
        out.append(f"  - {{{pairs}}}  # {note}\n")
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
    floor, an ``atol``/``rtol`` pair. Those are refused at :func:`write_table` rather
    than quietly replaced or quietly duplicated.
    """
    fields = _row_fields(line)
    if "arch" in fields or set(fields) - _RENDERABLE_FIELDS:
        return False
    return _covered(line, emitted_cells)


def write_table(path, suffix: str) -> int:
    """Replace every swept op's block in the YAML with what the sweep measured.

    Line-oriented on purpose. The table's comments *are* its provenance, and a load and
    re-dump through PyYAML would drop every one of them, including for the ops this
    sweep never touched.

    Raises if an op in ``MEASURED`` has no key line to write into: the measurement would
    otherwise be dropped in silence, and the sampled rows it was meant to replace would
    stay in place looking measured -- the failure the ``_OP_KEY`` comment below records
    biting once already.
    """
    import pathlib as _pathlib

    path = _pathlib.Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out, i, written = [], 0, set()
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
            trailing = []
            while j - 1 > i and not lines[j - 1].strip():
                trailing.insert(0, lines[j - 1])
                j -= 1
            emitted_cells = {(k[0], k[1]) for k in MEASURED[name]}
            rows = [l for l in lines[i + 1 : j] if l.strip().startswith("- ")]
            unrenderable = [
                l
                for l in rows
                if "arch" not in _row_fields(l)
                and set(_row_fields(l)) - _RENDERABLE_FIELDS
                and _covered(l, emitted_cells)
            ]
            if unrenderable:
                # Emitting over it would drop the floor; keeping it as well would give
                # the cell two equally specific keys, which `_load_table` refuses. So
                # neither, loudly, here: the row is a judgement a measurement cannot
                # re-derive, and the cell has to be settled by hand.
                raise ValueError(
                    f"{path.name}: {name} has {len(unrenderable)} row(s) this sweep "
                    "covers but cannot regenerate -- they carry a field beyond "
                    f"{sorted(_RENDERABLE_FIELDS)}:\n"
                    + "".join(unrenderable)
                    + "Settle the cell by hand, or drop the extra field, before "
                    "emitting."
                )
            kept = [l for l in rows if not _replaceable(l, emitted_cells)]
            out.extend(_render(line, _collapse(_decide(MEASURED[name])), suffix))
            # Rows this run did not supersede -- a format it does not reach, an
            # arch-keyed entry, a floor `_render` cannot re-derive -- are the
            # measurement of a different run and stay as they are. Replacing a whole op
            # block deleted them.
            out.extend(kept)
            out.extend(trailing)
            written.add(name)
            i = j
            continue
        out.append(line)
        i += 1
    missing = sorted(set(MEASURED) - written)
    if missing:
        raise ValueError(
            f"{path.name}: measured {', '.join(missing)} but found no key line to "
            "write into. Add the op's block to the table first -- the key line is "
            "passed through verbatim so a header comment survives, and cannot be "
            "generated here."
        )
    # Exactly one trailing newline: an op block carries its own trailing blank lines,
    # and the last block's leave the file ending in several. `end-of-file-fixer` then
    # rewrites the table on every commit.
    path.write_text("".join(out).rstrip("\n") + "\n", encoding="utf-8")
    return len(written)

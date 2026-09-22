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

import re
from typing import Dict, List, Optional, Tuple

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import MathOperation
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


def stimuli_format_for(fmt: DataFormat) -> DataFormat:
    """The format whose values the sweep enumerates in order to drive *fmt*."""
    return _STIMULI_FORMAT.get(fmt, fmt)


def sweep_spec() -> StimuliSpec:
    """Every finite representable value of the stimuli format, once.

    Deliberately not clipped to the op's domain. ``exclude_undefined`` expresses a domain
    as ``intervals``, which ULP_SWEEP does not read -- and clipping would also stop the
    undefined inputs reaching hardware at all. They are swept and then masked out of the
    statistics by :func:`defined_mask`, so the run still exercises them.
    """
    return StimuliSpec.ulp_sweep(low=-_INF, high=_INF)


def measurable_mask(
    op: MathOperation,
    src: torch.Tensor,
    golden: torch.Tensor,
    result: torch.Tensor,
    input_format: DataFormat,
) -> torch.Tensor:
    """Lanes of an all-finite-input sweep that a *step count* can describe.

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
    * subnormal inputs. The hardware flushes them on the way in and the golden does not,
      so ``ceil(5.69e-39)`` is 1 in the model and 0 on silicon -- 16,129 bf16 steps for
      a difference that is the unpack path's flush, not the op's accuracy. Measured, it
      is the whole of Ceil's, Floor's and Sqrt's apparent error: excluding it returns
      all three to the 0 their exactness claims, and moves nothing else. The flush is
      covered on its own terms elsewhere; a step count is the wrong instrument for it.

    Subnormal *outputs* stay in. Where the golden underflows and the hardware writes
    zero the count is large but the lane is a real one the op produced -- Silu at
    ``x=-87.5`` is that case, and it is the op's own tail, not the unpack path.
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
    return both_measurable & ~nonfinite_mismatches(golden, result) & normal_input


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
    MEASURED.setdefault(op_name, {})[key] = max_ulp


def _verdict(measured: int, out_fmt: str) -> Tuple[str, int]:
    """What the table should say for a measured worst lane on *out_fmt*.

    ``("ulp", budget)`` while a step budget is still *stronger* than the tolerance it
    replaces, and ``("tolerance", measured)`` once it is not. The bound is the table's
    own ``usable_budget_ceiling``: ~419,430 steps for fp32, 51 for fp16, 6 for bf16, 25
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
    budget = 0 if measured == 0 else -(-measured * 11 // 10)
    if budget > usable_budget_ceiling(DataFormat[out_fmt]):
        return ("tolerance", measured)
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
        merged[tuple(key[i] for i in keep)] = value

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
    order = ("in", "out", "approx", "dest")
    out = [key_line]
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
            note += ", past this output's usable ceiling, so tolerance"
        elif metric == "block":
            note += ", but a sorted sweep flatters a block format, so tolerance"
        out.append(f"  - {{{pairs}}}  # {note}, {suffix}\n")
    return out


def write_table(path, suffix: str) -> int:
    """Replace every swept op's block in the YAML with what the sweep measured.

    Line-oriented on purpose. The table's comments *are* its provenance, and a load and
    re-dump through PyYAML would drop every one of them, including for the ops this
    sweep never touched.
    """
    import pathlib as _pathlib

    path = _pathlib.Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out, i, rewritten = [], 0, 0
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
            swept = {f.name for f in SWEEP_FORMATS}
            kept = [
                l
                for l in lines[i + 1 : j]
                if l.strip().startswith("- ")
                and not (
                    any(f"in: {f}" in l for f in swept)
                    and any(f"out: {f}" in l for f in swept)
                )
            ]
            out.extend(_render(line, _collapse(_decide(MEASURED[name])), suffix))
            # Rows for formats this sweep does not reach -- Float32, Bfp4_b, and any
            # arch-keyed entry -- are the measurement of a different run and stay as
            # they are. Replacing a whole op block deleted them.
            out.extend(kept)
            out.extend(trailing)
            rewritten += 1
            i = j
            continue
        out.append(line)
        i += 1
    path.write_text("".join(out), encoding="utf-8")
    return rewritten

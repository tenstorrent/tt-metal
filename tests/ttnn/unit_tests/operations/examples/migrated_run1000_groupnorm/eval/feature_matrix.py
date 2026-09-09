# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Feature-matrix primitives for golden-test parameterization.

The registry model has three pieces of data per op:

- TARGET: per-axis lists describing the universe of values we *want* the op
  to eventually support. Authored upfront, ambition.
- SUPPORTED: per-axis lists describing what the op accepts *now*. Lives next
  to the op file. Edited by the implementer as features get added.
- EXCLUSIONS: list of cell-shaped dicts ({"axis": "value", ...}) marking
  combinations inside SUPPORTED that the op explicitly refuses anyway.
  Same dict shape used by EXCLUSIONS entries is matched against every
  generated case.

Shape is not directly in any of these. Instead, each op declares
INPUT_TAGGERS — a dict of {axis_name: function(inputs) -> categorical_value}
— and the cartesian generator runs every tagger over each entry from the
op's INPUTS list to project shape facets onto categorical axes that look
exactly like every other axis.

This module is pure functions over dicts/lists. No I/O, no pytest, no
ttnn. The test_golden.py per op imports these to decide xfail decoration;
validate() on the op side reuses the same is_supported function.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable


# A "values dict" is a flat {axis_name: value} mapping. value can be any
# python object — categorical strings, ttnn enums, ints, etc. is_supported
# uses `==` and `in` to compare against SUPPORTED lists, so any hashable
# value works.
ValuesDict = dict[str, object]

# A "feature axes" dict maps axis_name -> list of allowed values. Both
# TARGET and SUPPORTED have this shape.
FeatureAxes = dict[str, list]

# An input tagger maps (inputs tuple, partial axes dict) onto a single
# categorical value for one axis. Per-op. The `axes` argument contains:
# - The finite-axis values from the current cartesian combination (e.g.
#   `num_groups=4` when iterating over a `num_groups` target axis).
# - Outputs of earlier-declared taggers (declaration order in INPUT_TAGGERS).
# Taggers that only depend on shape ignore the second argument.
InputTagger = Callable[[tuple, "ValuesDict"], object]

# INPUT_TAGGERS is {axis_name: tagger}.
InputTaggers = dict[str, InputTagger]

# An exclusion is a partial values dict — every key present must match
# the candidate case for the exclusion to fire.
Exclusion = dict[str, object]


def apply_input_taggers(
    taggers: InputTaggers,
    inputs: tuple,
    axes: ValuesDict | None = None,
) -> ValuesDict:
    """Run every tagger over `(inputs, axes)` and return the resulting values.

    `inputs` is the per-case input shapes tuple — a 1-tuple for single-input
    ops, multi-tuple for matmul-style.

    `axes` is the partial axis dict the taggers can read — typically the
    cartesian combination of finite axes for the current case. Defaults to
    empty for callers that only have purely-shape-dependent taggers.

    Taggers are evaluated in declaration order; a later tagger sees earlier
    taggers' outputs in its `axes` argument. Useful when one shape facet
    depends on another (e.g. groupnorm's `groups_alignment` depending on
    `num_groups` and the channel count).
    """
    ctx: ValuesDict = {} if axes is None else dict(axes)
    out: ValuesDict = {}
    for axis, tagger in taggers.items():
        value = tagger(inputs, ctx)
        out[axis] = value
        ctx[axis] = value
    return out


def cartesian(
    target: FeatureAxes,
    input_taggers: InputTaggers,
    inputs: tuple,
) -> Iterable[ValuesDict]:
    """Yield {axis: value} dicts for one inputs tuple × every finite-axis combo.

    Finite axes (from `target`) cartesian-multiply. For each combo, shape-
    derived axes (from `input_taggers`) are computed against that combo —
    so taggers can read sibling axis values via their `axes` argument.

    Per-shape coupled values (e.g. `num_groups` for groupnorm, `kernel_size`
    for conv) live as positional slots inside the `inputs` tuple and get
    projected onto axes by INPUT_TAGGERS that read them by index. The op's
    runner unpacks the same positions from `inputs`. Nothing special-cases
    them at the harness level.

    If an axis is both in `target` (declaring the universe) and in
    `input_taggers` (computing the value for this shape+combo), the tagger
    wins for that case — the axis isn't iterated. The target entry still
    serves as the universe declaration; SUPPORTED lookups go through the
    same axis name.
    """
    tagger_keys = set(input_taggers.keys())
    finite_keys = [k for k in target.keys() if k not in tagger_keys]
    finite_values = [target[k] for k in finite_keys]
    combos = itertools.product(*finite_values) if finite_keys else [()]
    for combo in combos:
        axes: ValuesDict = dict(zip(finite_keys, combo))
        tagger_axes = apply_input_taggers(input_taggers, inputs, axes)
        axes.update(tagger_axes)
        yield axes


def matches_exclusion(exclusion: Exclusion, values: ValuesDict) -> bool:
    """True iff every key in `exclusion` is present in `values` and equal."""
    for k, expected in exclusion.items():
        if k not in values or values[k] != expected:
            return False
    return True


def is_supported(
    values: ValuesDict,
    supported: FeatureAxes,
    exclusions: list[Exclusion],
) -> bool:
    """True iff every axis in `supported` lists `values[axis]`, and no
    exclusion fires.

    `values` may carry axes not in `supported` (op-specific kwargs not
    gated by the support contract). Those are ignored.
    """
    for axis, allowed in supported.items():
        if axis not in values:
            return False
        if values[axis] not in allowed:
            return False
    for exc in exclusions:
        if matches_exclusion(exc, values):
            return False
    return True


def unsupported_reason(
    values: ValuesDict,
    supported: FeatureAxes,
    exclusions: list[Exclusion],
) -> str | None:
    """None if supported. Otherwise a short string describing why not.

    Used as the `reason=` argument to pytest.mark.xfail. Always paired
    with `strict=True, raises=NotImplementedError` by the caller.

    This is the "could be supported in the future" case. For cells that
    are structurally impossible (bf8b + ROW_MAJOR, etc.), see
    `invalid_reason` — those get pytest.mark.skip instead.
    """
    for axis, allowed in supported.items():
        if axis not in values:
            return f"axis {axis!r} missing from values dict"
        if values[axis] not in allowed:
            return f"{axis}={values[axis]!r} not in SUPPORTED {allowed}"
    for exc in exclusions:
        if matches_exclusion(exc, values):
            return f"matches EXCLUSIONS entry {exc!r}"
    return None


def invalid_reason(
    values: ValuesDict,
    invalid: list[Exclusion],
) -> str | None:
    """None if `values` is not invalid; else a short string describing why.

    `invalid` is the list of cell patterns that are *structurally* impossible
    for this op — combinations that will never be supported no matter how
    much work is done (e.g. `bfloat8_b` is a block-quantized format that
    only makes sense in `TILE` layout, so `bf8b + ROW_MAJOR` is invalid by
    design).

    The test harness applies `pytest.mark.skip(reason=...)` to cells that
    match an INVALID entry. Compare with `unsupported_reason`, which gates
    cells that *could* be supported in the future — those get xfail-strict.

    INVALID takes precedence over unsupported: a cell that is both invalid
    and unsupported is skipped, not xfailed. Skip is the stronger statement
    (won't ever work) and the more honest signal.
    """
    for inv in invalid:
        if matches_exclusion(inv, values):
            return f"matches INVALID entry {inv!r}"
    return None


def case_id(values: ValuesDict) -> str:
    """Render a stable, human-readable pytest case id from a values dict.

    Axes are sorted alphabetically so ids are deterministic across runs.
    """
    parts = []
    for k in sorted(values):
        v = values[k]
        # Prefer enum.name / __name__ for ttnn enums; fall back to repr-ish.
        if hasattr(v, "name"):
            vs = v.name
        elif hasattr(v, "__name__"):
            vs = v.__name__
        else:
            vs = str(v)
        parts.append(f"{k}={vs}")
    return "-".join(parts)

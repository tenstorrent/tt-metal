# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared parametrize logic for registry-model golden tests.

Each op's test_golden.py used to be ~80 LoC of identical scaffolding —
the cartesian and loose parametrize loops differ across ops only by
which `run_<op>` helper gets called and which shape-id format is used.
This module hosts the shared logic; per-op test_golden.py files shrink
to imports + two `@pytest.mark.parametrize` decorators.

Two entry points, mirroring the two test functions every op exposes:

  parametrize_cases(...)        — cartesian over TARGET × INPUTS, with
                                  skip/xfail decoration from INVALID /
                                  SUPPORTED / EXCLUSIONS
  parametrize_loose_cases(...)  — hand-authored LOOSE_CASES entries
                                  decorated the same way, with extras
                                  dict threaded through

Both return lists of `pytest.param(...)` ready to feed into
`@pytest.mark.parametrize`. Both accept an optional `shape_id` callable
for ops whose case-id format isn't just "join the first shape's dims".
"""

from __future__ import annotations

from typing import Callable

import pytest

from eval.feature_matrix import (
    apply_input_taggers,
    case_id,
    cartesian,
    invalid_reason,
    unsupported_reason,
)


def default_shape_id(inputs) -> str:
    """Join the dims of the first tensor shape with 'x'.

    Works for any op whose INPUTS entry starts with the input shape:
      - single-tensor:        ((H, W),)               → "HxW"
      - per-shape scalars:    ((N, C, H, W), G)       → "NxCxHxW"  (G in axes)
      - multi-tensor heads:   ((1, 8, 64), (1, 8, 64))→ "1x8x64"   (Q only)

    Ops that need a richer id (conv2d's full knob set, SDPA's Q+KV,
    group_norm's `_gN` suffix) pass their own callable via `shape_id=`.
    """
    return "x".join(str(d) for d in inputs[0])


def _decorate(axes, supported, exclusions, invalid) -> list:
    """Build the marks list for one (axes) row.

    INVALID takes precedence (structural impossibility → skip). Otherwise
    cells outside SUPPORTED or hitting EXCLUSIONS get xfail-strict with
    raises=NotImplementedError, matching the validate() contract.
    """
    skip = invalid_reason(axes, invalid)
    if skip is not None:
        return [pytest.mark.skip(reason=skip)]
    xfail = unsupported_reason(axes, supported, exclusions)
    if xfail is not None:
        return [
            pytest.mark.xfail(
                reason=xfail,
                strict=True,
                raises=NotImplementedError,
            )
        ]
    return []


def parametrize_cases(
    target, inputs, input_taggers, supported, exclusions, invalid, *, shape_id: Callable = default_shape_id
):
    """Cartesian TARGET × INPUTS → list of pytest.param entries.

    Each (inputs_entry × axes_combination) becomes one parametrize row.
    Marks come from `_decorate`. Case ids: `{shape_id}-{case_id(axes)}`.
    """
    cases = []
    for inputs_entry in inputs:
        for axes in cartesian(target, input_taggers, inputs_entry):
            cases.append(
                pytest.param(
                    inputs_entry,
                    axes,
                    marks=_decorate(axes, supported, exclusions, invalid),
                    id=f"{shape_id(inputs_entry)}-{case_id(axes)}",
                )
            )
    return cases


def parametrize_loose_cases(
    loose_cases, input_taggers, supported, exclusions, invalid, *, shape_id: Callable = default_shape_id
):
    """LOOSE_CASES → list of pytest.param entries with three values
    (inputs, axes, extras).

    Each entry pins every finite axis; shape-derived axes are filled in
    by INPUT_TAGGERS so the resulting axes dict is the same shape as a
    cartesian-generated one. The optional `extras` field is a runner-
    override dict (defaults to `{}`), passed through as the third
    parametrize value.
    """
    cases = []
    for case in loose_cases:
        inputs_entry = case["inputs"]
        extras = case.get("extras", {})
        pinned = {k: v for k, v in case.items() if k not in ("inputs", "extras")}
        axes = {
            **pinned,
            **apply_input_taggers(input_taggers, inputs_entry, pinned),
        }
        cases.append(
            pytest.param(
                inputs_entry,
                axes,
                extras,
                marks=_decorate(axes, supported, exclusions, invalid),
                id=f"{shape_id(inputs_entry)}-{case_id(axes)}",
            )
        )
    return cases

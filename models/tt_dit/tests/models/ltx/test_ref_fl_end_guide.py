# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the ref+fl end guide (LTX_REF_FL_END_GUIDE, sulphur task #3360). The latent frames before a
pinned last frame get it as an annealed x0 target; they stay unpinned (free AdaLN timestep). Off, nothing
changes.

The helpers are AST-extracted from the pipeline source and exec'd, like ``test_ref_fl_end_hold``, so the
test needs no device build. Run: python -m pytest <file> -v -p no:cacheprovider
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

DISTILLED_PATH = Path(__file__).resolve().parents[3] / "pipelines" / "ltx" / "pipeline_ltx_distilled.py"
NAMES = (
    "build_conditioning_tensors",
    "_ref_fl_end_guide_strengths",
    "apply_ref_fl_end_guide",
    "ref_fl_end_guide_step_mask",
    "_ref_fl_end_softpin",
    "apply_ref_fl_end_softpin",
)
C, H, W, T = 4, 2, 3, 6
HW = H * W


def _ns():
    src = DISTILLED_PATH.read_text()
    lines = src.splitlines(keepends=True)
    blocks = [
        "".join(lines[n.lineno - 1 : n.end_lineno])
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name in NAMES
    ]
    assert len(blocks) == len(NAMES)
    ns = {"torch": torch, "REF_FL_END_GUIDE_DEFAULT": (0.5, 0.25)}
    exec(compile("from __future__ import annotations\n\n" + "\n\n".join(blocks), str(DISTILLED_PATH), "exec"), ns)
    return ns


def _conds(*idx):
    return [(i, torch.full((1, C, 1, H, W), float(i + 1)), 1.0) for i in idx]


def _build(ns, conds, strengths):
    clean, mask, pin_rows, _, _ = ns["build_conditioning_tensors"](conds, T, H, W, C)
    before = (clean.clone(), mask.clone(), pin_rows.clone())
    guide = ns["apply_ref_fl_end_guide"](clean, mask, pin_rows, conds, T, HW, strengths)
    return before, clean, mask, pin_rows, guide


@pytest.mark.parametrize(
    "raw, want",
    [(None, ()), ("", ()), ("0", ()), ("1", (0.5, 0.25)), ("0.4, 2", (0.4, 1.0)), ("pin:0.8", ()), ("pin:0.8,0.5", (0.5,))],
)
def test_strengths_from_env(raw, want):
    assert _ns()["_ref_fl_end_guide_strengths"](raw) == want


def test_guide_targets_the_frames_before_the_last():
    ns = _ns()
    (clean0, mask0, pins0), clean, mask, pins, guide = _build(ns, _conds(0, T - 1), (0.5, 0.25))
    assert torch.equal(pins, pins0)  # guide rows stay unpinned: AdaLN timestep untouched
    want = torch.zeros(T * HW, dtype=torch.bool)
    want[(T - 3) * HW : (T - 1) * HW] = True
    assert torch.equal(guide, want)
    assert torch.allclose(mask[0, (T - 2) * HW : (T - 1) * HW], torch.full((HW, 1), 0.5))
    assert torch.allclose(mask[0, (T - 3) * HW : (T - 2) * HW], torch.full((HW, 1), 0.75))
    assert torch.equal(clean[0, (T - 3) * HW : (T - 1) * HW], torch.full((2 * HW, C), float(T)))
    rest = ~guide
    assert torch.equal(mask[:, rest], mask0[:, rest]) and torch.equal(clean[:, rest], clean0[:, rest])


def test_no_last_frame_cond_or_no_strength_changes_nothing():
    ns = _ns()
    for conds, strengths in ((_conds(0), (0.5, 0.25)), (_conds(0, T - 1), (0.0, 0.0))):
        (clean0, mask0, _), clean, mask, _, guide = _build(ns, conds, strengths)
        assert guide is None and torch.equal(clean, clean0) and torch.equal(mask, mask0)


def test_an_interior_keyframe_is_left_alone():
    ns = _ns()
    (_, mask0, _), _, mask, _, guide = _build(ns, _conds(0, T - 2, T - 1), (0.5, 0.25))
    assert torch.equal(mask[0, (T - 2) * HW : (T - 1) * HW], mask0[0, (T - 2) * HW : (T - 1) * HW])
    assert guide[(T - 3) * HW : (T - 2) * HW].all() and not guide[(T - 2) * HW : (T - 1) * HW].any()


def test_step_mask_anneals_to_free_on_the_last_step():
    ns = _ns()
    _, _, mask, _, guide = _build(ns, _conds(0, T - 1), (0.5, 0.25))
    step = ns["ref_fl_end_guide_step_mask"]
    assert torch.equal(step(mask, guide, 1.0, 1.0), mask)
    half = step(mask, guide, 0.5, 1.0)
    assert torch.allclose(half[0, (T - 2) * HW : (T - 1) * HW], torch.full((HW, 1), 0.75))
    last = step(mask, guide, 0.0, 1.0)
    assert torch.equal(last[:, guide], torch.ones_like(last[:, guide]))
    assert torch.equal(last[:, ~guide], mask[:, ~guide])  # pins and free rows unchanged


@pytest.mark.parametrize("raw, want", [(None, None), ("1", None), ("0.5,0.25", None), ("pin:0.8", 0.8), ("0.5,pin:2", 0.999)])
def test_softpin_from_env(raw, want):
    assert _ns()["_ref_fl_end_softpin"](raw) == want


def test_softpin_unpins_the_last_frame_at_its_strength():
    ns = _ns()
    conds = _conds(0, T - 1)
    clean, mask, pins, _, _ = ns["build_conditioning_tensors"](conds, T, H, W, C)
    clean0, mask0, pins0 = clean.clone(), mask.clone(), pins.clone()
    soft = ns["apply_ref_fl_end_softpin"](clean, mask, pins, conds, T, HW, 0.8)
    want = torch.zeros(T * HW, dtype=torch.bool)
    want[(T - 1) * HW :] = True
    assert torch.equal(soft, want)
    assert not pins[soft].any() and torch.equal(pins[~soft], pins0[~soft])  # frame 0 stays hard-pinned
    assert torch.allclose(mask[0, soft], torch.full((HW, 1), 0.2))
    assert torch.equal(mask[:, ~soft], mask0[:, ~soft]) and torch.equal(clean, clean0)


def test_softpin_without_a_last_frame_cond_changes_nothing():
    ns = _ns()
    conds = _conds(0)
    clean, mask, pins, _, _ = ns["build_conditioning_tensors"](conds, T, H, W, C)
    mask0, pins0 = mask.clone(), pins.clone()
    assert ns["apply_ref_fl_end_softpin"](clean, mask, pins, conds, T, HW, 0.8) is None
    assert torch.equal(mask, mask0) and torch.equal(pins, pins0)

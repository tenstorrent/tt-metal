# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the ref+fl end-frame hold (LTX_REF_FL_END_HOLD, task #3075). On the aligned-node keyframe
path a non-frame-0 stage-1 pin softens; with the hold the LAST latent frame keeps frame 0's strength, so its
denoise mask matches the pinned AdaLN timestep. Off, every pin is what it was.

The helpers are AST-extracted from the pipeline source and exec'd, like ``test_ltx_seeded_noise``, so the
test needs no device build. Run: python -m pytest <file> -v -p no:cacheprovider
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

DISTILLED_PATH = Path(__file__).resolve().parents[3] / "pipelines" / "ltx" / "pipeline_ltx_distilled.py"
NAMES = ("_keyframe_s1_strength", "_s1_pin_strength", "build_conditioning_tensors")


def _ns():
    src = DISTILLED_PATH.read_text()
    lines = src.splitlines(keepends=True)
    blocks = [
        "".join(lines[n.lineno - 1 : n.end_lineno])
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name in NAMES
    ]
    assert len(blocks) == len(NAMES)
    ns = {"torch": torch, "KEYFRAME_S1_STRENGTH": 0.5, "KEYFRAME_S1_STRENGTH_COARSE": 0.25}
    exec(compile("from __future__ import annotations\n\n" + "\n\n".join(blocks), str(DISTILLED_PATH), "exec"), ns)
    return ns


def _strengths(ns, end_hold, s1_height=352, latent_frames=19, kf_fix=True, append_token=False):
    return [
        ns["_s1_pin_strength"](
            i, latent_frames, 1.0, s1_height, kf_fix=kf_fix, append_token=append_token, end_hold=end_hold
        )
        for i in (0, 9, latent_frames - 1)
    ]


@pytest.mark.parametrize("s1_height, soft", [(352, 0.25), (544, 0.5)])
def test_hold_off_softens_the_last_frame_as_before(s1_height, soft):
    ns = _ns()
    assert _strengths(ns, False, s1_height) == [1.0, soft, soft]


@pytest.mark.parametrize("s1_height, soft", [(352, 0.25), (544, 0.5)])
def test_hold_keeps_the_last_frame_at_frame0_strength(s1_height, soft):
    ns = _ns()
    # Only the last frame changes: an interior keyframe still softens.
    assert _strengths(ns, True, s1_height) == [1.0, soft, 1.0]


def test_hold_changes_nothing_off_the_aligned_path_or_on_append_token():
    ns = _ns()
    for hold in (False, True):
        assert _strengths(ns, hold, kf_fix=False) == [1.0, 1.0, 1.0]
        assert _strengths(ns, hold, append_token=True) == [1.0, 1.0, 1.0]


def test_hold_mask_matches_frame0_and_off_is_bit_identical():
    ns = _ns()
    lf, h, w, c = 4, 2, 3, 8
    gen = torch.Generator().manual_seed(7)
    first, last = (torch.randn(1, c, 1, h, w, generator=gen) for _ in range(2))
    hw = h * w

    def build(hold):
        conds = [
            (i, lat, ns["_s1_pin_strength"](i, lf, 1.0, 352, kf_fix=True, append_token=False, end_hold=hold))
            for i, lat in ((0, first), (lf - 1, last))
        ]
        return ns["build_conditioning_tensors"](conds, lf, h, w, c)

    clean_on, mask_on, pins_on, _, _ = build(True)
    clean_off, mask_off, pins_off, _, _ = build(False)
    end = slice((lf - 1) * hw, lf * hw)
    # Held: the last frame's mask is frame 0's (fully pinned), the AdaLN pinned timestep's level.
    assert torch.equal(mask_on[:, end], mask_on[:, :hw])
    assert torch.all(mask_on[:, end] == 0.0)
    assert torch.all(mask_off[:, end] == 0.75)
    # Same clean latent and pinned rows either way; only the end frame's mask differs.
    assert torch.equal(clean_on, clean_off) and torch.equal(pins_on, pins_off)
    assert torch.equal(mask_on[:, : (lf - 1) * hw], mask_off[:, : (lf - 1) * hw])
    # Off is bit-identical to the softening the pipeline did before the hold existed.
    legacy = [(0, first, 1.0), (lf - 1, last, min(1.0, 0.25))]
    ref = ns["build_conditioning_tensors"](legacy, lf, h, w, c)
    assert all(torch.equal(a, b) for a, b in zip(ref[:3], (clean_off, mask_off, pins_off)))

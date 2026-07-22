# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the denoise-trajectory comparison harness itself (#47468)."""

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block
from models.experimental.diffusion_gemma.tests.trajectory_pcc import (
    assert_trajectory_matches,
    compare_trajectories,
)


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cfg(**kw):
    return DiffusionConfig(max_denoise_steps=8, entropy_stop_threshold=0.1, stable_steps_to_halt=1, **kw)


def _peaked_traj(target=5, seed=1):
    batch, length, vocab = 1, 8, 32
    logits = torch.full((batch, length, vocab), -1e4)
    logits[..., target] = 1e4
    init = S.random_canvas((batch, length), vocab, generator=_gen(seed))
    return denoise_block(lambda canvas, step: logits, init, _cfg(), vocab)


def _random_traj(seed):
    batch, length, vocab = 1, 12, 40

    def logits_fn(canvas, step):
        return torch.randn(batch, length, vocab, generator=_gen(seed * 1000 + step))

    init = S.random_canvas((batch, length), vocab, generator=_gen(seed))
    return denoise_block(logits_fn, init, _cfg(), vocab)


def test_self_comparison_passes():
    traj = _peaked_traj()
    cmp = compare_trajectories(traj, traj)
    assert cmp.passed
    assert cmp.min_argmax_agreement == 1.0
    assert cmp.committed_match == 1.0
    assert cmp.entropy_trajectory_pcc == pytest.approx(1.0)
    # Decision-level fields (#47468): every per-step diff is perfect on self-compare.
    assert cmp.min_sampled_agreement == 1.0  # Gumbel-sampled ids
    assert cmp.min_accept_iou == 1.0  # accept-mask IoU
    assert cmp.min_canvas_agreement == 1.0  # renoised canvas
    assert cmp.min_entropy_pcc == pytest.approx(1.0)  # per-token entropy PCC


def test_entropy_abs_gate_catches_affine_error_pcc_misses():
    """PCC is invariant to a constant offset/scale, so a systematic entropy error
    (wrong log base / missing temperature) would pass PCC≈1 — the absolute gate
    must still catch it (finding #3). Only the entropy abs-err should fail here."""
    ref = _random_traj(seed=7)
    # candidate identical to ref on every decision, but each per-token entropy is offset by +0.5
    shifted_steps = [r._replace(entropy=r.entropy + 0.5) for r in ref.per_step]
    cand = ref._replace(per_step=shifted_steps)

    cmp = compare_trajectories(ref, cand)
    assert min(cmp.per_step_entropy_pcc) > 0.99  # PCC blind to the constant offset
    assert cmp.max_entropy_abs_err >= 0.49  # but the absolute gate sees it
    assert not cmp.passed  # so the comparison fails
    # every other decision class still matches (the failure is isolated to entropy magnitude)
    assert cmp.min_argmax_agreement == 1.0 and cmp.min_canvas_agreement == 1.0 and cmp.min_accept_iou == 1.0


def test_decision_level_fields_distinguish_drifted_trajectories():
    """Distinct trajectories must fail on EVERY decision class — sampled, accept,
    canvas, per-token entropy — not just the clean argmax."""
    ref = _random_traj(seed=11)
    cand = _random_traj(seed=42)  # different logits AND different RNG -> different decisions
    cmp = compare_trajectories(ref, cand)
    assert not cmp.passed
    assert cmp.min_sampled_agreement < 1.0
    assert cmp.min_accept_iou < 1.0
    assert cmp.min_canvas_agreement < 1.0
    # entropy is a real-valued vector → some PCC is plausible by chance, but it should be far from 1.0
    # for genuinely independent random logits. Don't assert a hard bound on it here — the harness's
    # min_per_step_entropy_pcc threshold (0.99) catches drift in real use.


def test_run_to_cap_self_comparison_passes():
    # constant near-uniform logits -> run to cap; constant entropy must not break PCC
    traj = _random_traj(seed=7)
    assert compare_trajectories(traj, traj).passed


def test_distinct_trajectories_fail():
    ref = _random_traj(seed=2)
    cand = _random_traj(seed=99)  # different logits -> different argmax decisions
    cmp = compare_trajectories(ref, cand)
    assert not cmp.passed
    assert cmp.min_argmax_agreement < 1.0


def test_assert_trajectory_matches_raises_on_mismatch(expect_error):
    ref = _random_traj(seed=3)
    cand = _random_traj(seed=88)
    with expect_error(AssertionError):
        assert_trajectory_matches(ref, cand)


def test_assert_trajectory_matches_returns_comparison_on_match():
    traj = _peaked_traj(target=9)
    cmp = assert_trajectory_matches(traj, traj)
    assert cmp.passed

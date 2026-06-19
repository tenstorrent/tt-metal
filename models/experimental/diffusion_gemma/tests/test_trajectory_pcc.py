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


def test_assert_trajectory_matches_raises_on_mismatch():
    ref = _random_traj(seed=3)
    cand = _random_traj(seed=88)
    with pytest.raises(AssertionError):
        assert_trajectory_matches(ref, cand)


def test_assert_trajectory_matches_returns_comparison_on_match():
    traj = _peaked_traj(target=9)
    cmp = assert_trajectory_matches(traj, traj)
    assert cmp.passed

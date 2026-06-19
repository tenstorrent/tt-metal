# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HF reference adapter (#47468).

The real HF model is environment-gated, so these verify (a) the guard behaves,
and (b) the adapter + trajectory + PCC harness compose correctly using a mock
canvas-logits model that stands in for the HF / device model.
"""

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.hf_reference import (
    is_hf_reference_available,
    load_hf_reference,
    run_reference_trajectory,
)
from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cfg(**kw):
    return DiffusionConfig(max_denoise_steps=8, entropy_stop_threshold=0.1, stable_steps_to_halt=1, **kw)


class _MockCanvasModel:
    """Deterministic canvas-logits model: fixed per-position logits, optional drift."""

    def __init__(self, batch, length, vocab, seed, drift=0.0):
        self._logits = torch.randn(batch, length, vocab, generator=_gen(seed))
        self._drift = drift

    def __call__(self, canvas, **kw):
        out = self._logits.clone()
        if self._drift:
            out[..., 0] += self._drift  # perturb one logit -> can flip argmax/accept
        return out


def test_guard_reports_unavailable_and_loader_raises():
    # In this environment diffusion_gemma is not installed.
    if not is_hf_reference_available():
        with pytest.raises(ImportError, match="diffusion_gemma"):
            load_hf_reference("google/diffusiongemma-26B-A4B-it")
    else:  # pragma: no cover - only when a diffusion_gemma build is present
        pytest.skip("diffusion_gemma is available; loader guard not exercised")


def test_adapter_drives_trajectory_and_is_deterministic():
    batch, length, vocab = 1, 8, 32
    model = _MockCanvasModel(batch, length, vocab, seed=1)
    init = S.random_canvas((batch, length), vocab, generator=_gen(2))
    a = run_reference_trajectory(model, init.clone(), _cfg(), vocab)
    b = run_reference_trajectory(model, init.clone(), _cfg(), vocab)
    assert torch.equal(a.committed, b.committed)
    assert compare_trajectories(a, b).passed  # faithful candidate matches oracle


def test_harness_rejects_drifted_candidate():
    batch, length, vocab = 1, 8, 32
    init = S.random_canvas((batch, length), vocab, generator=_gen(3))
    ref = run_reference_trajectory(_MockCanvasModel(batch, length, vocab, seed=5), init.clone(), _cfg(), vocab)
    drifted = run_reference_trajectory(
        _MockCanvasModel(batch, length, vocab, seed=5, drift=5.0), init.clone(), _cfg(), vocab
    )
    assert not compare_trajectories(ref, drifted).passed  # decision drift is caught

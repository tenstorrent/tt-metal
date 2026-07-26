# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DG_DEGENERACY_POLICY=retry: re-denoise a collapsed block instead of losing the request (#48291).

`stop` prevents degenerate output but ends the request; `retry` re-runs the block with different
Gumbel noise and commits the first attempt that is not degenerate.

The property these tests exist to protect is the one that makes retry worth having at all: a retry
must draw DIFFERENT noise. The per-step noise functions are pure in (block, step), so a factory
without attempt awareness would reproduce the trajectory exactly and the retry would be a silent
no-op -- worse than not retrying, because the log would claim a retry happened. So `retry` refuses
to run unless the factory is marked `supports_retry`.
"""

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt import degeneracy as DG
from models.experimental.diffusion_gemma.tt.generate import denoise_and_commit_block, make_seeded_gumbel_noise_fn

CANVAS = 8
CONTENT_ID = 236770
EOS_ID = 1


def _canvas(ids):
    return torch.tensor([ids], dtype=torch.long)


COLLAPSED = _canvas([CONTENT_ID] * CANVAS)
HEALTHY = _canvas([5, 9, 17, 33, 41, 58, 64, 77])


def _trajectory(tokens):
    return DenoiseTrajectory(committed=tokens, num_steps=4, halted=True, per_step=[])


class _Denoiser:
    """Returns a scripted canvas per attempt, and records the noise each attempt was given."""

    def __init__(self, canvases):
        self.canvases = list(canvases)
        self.noises = []
        self.canvas_args = []

    def __call__(self, logits_fn, init_canvas, config, *, gumbel_noise_fn=None, noise_tokens_fn=None):
        self.noises.append(gumbel_noise_fn)
        self.canvas_args.append(init_canvas)
        return _trajectory(self.canvases[min(len(self.noises) - 1, len(self.canvases) - 1)])


def _retry_noise_fn(marker="attempt-aware"):
    def factory(attempt):
        return f"{marker}-{attempt}"

    factory.supports_retry = True
    return factory


def _run(denoiser, *, retry_noise=None, retry_canvas=None, commits=None):
    return denoise_and_commit_block(
        "model",
        object(),
        "canvas-0",
        DiffusionConfig(canvas_length=CANVAS),
        start_pos=0,
        gumbel_noise_fn="noise-0",
        denoise_block_fn=denoiser,
        commit_fn=(commits if commits is not None else (lambda *a, **k: None)),
        stop_token_ids=[EOS_ID],
        retry_noise_fn=retry_noise,
        retry_init_canvas_fn=retry_canvas,
    )


def test_retry_commits_the_first_clean_attempt(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "retry")
    monkeypatch.setenv("DG_DEGENERACY_RETRIES", "2")
    denoiser = _Denoiser([COLLAPSED, HEALTHY])
    committed = []
    out = _run(
        denoiser,
        retry_noise=_retry_noise_fn(),
        retry_canvas=lambda: "canvas-retry",
        commits=lambda model, tokens, **kwargs: committed.append(tokens),
    )
    assert torch.equal(out.committed, HEALTHY), "the clean attempt must be the one committed"
    assert len(committed) == 1, "only the accepted attempt may reach the KV cache"
    assert denoiser.noises == ["noise-0", "attempt-aware-1"], "the retry must use DIFFERENT noise"
    assert denoiser.canvas_args == ["canvas-0", "canvas-retry"], "the retry needs a fresh canvas"


def test_retry_exhausted_still_refuses_to_commit(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "retry")
    monkeypatch.setenv("DG_DEGENERACY_RETRIES", "2")
    denoiser = _Denoiser([COLLAPSED])
    committed = []
    with expect_error(DG.DegenerateBlockError, match="degenerate committed canvas"):
        _run(
            denoiser,
            retry_noise=_retry_noise_fn(),
            retry_canvas=lambda: "canvas-retry",
            commits=lambda model, tokens, **kwargs: committed.append(tokens),
        )
    assert len(denoiser.noises) == 3, "one initial attempt plus two retries"
    assert committed == [], "nothing degenerate may be committed even after exhausting retries"


def test_retry_without_an_attempt_aware_factory_is_refused(monkeypatch, expect_error):
    """The whole point: a factory that cannot vary its draw would make retry a silent no-op."""
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "retry")

    def plain_factory(attempt):
        return "same-noise"

    with expect_error(ValueError, match="attempt-aware noise factory"):
        _run(_Denoiser([COLLAPSED]), retry_noise=plain_factory, retry_canvas=lambda: "canvas-retry")


def test_retry_without_the_hooks_is_refused(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "retry")
    with expect_error(ValueError, match="retry_noise_fn and retry_init_canvas_fn"):
        _run(_Denoiser([COLLAPSED]))


def test_retry_does_not_fire_on_a_healthy_block(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "retry")
    denoiser = _Denoiser([HEALTHY])
    _run(denoiser, retry_noise=_retry_noise_fn(), retry_canvas=lambda: "canvas-retry")
    assert denoiser.noises == ["noise-0"], "a healthy block must not be re-denoised"


def test_retry_does_not_fire_on_a_terminating_canvas(monkeypatch):
    """A wall of <eos> is termination, not degeneration -- retrying it would be wrong."""
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "retry")
    denoiser = _Denoiser([_canvas([EOS_ID] * CANVAS)])
    _run(denoiser, retry_noise=_retry_noise_fn(), retry_canvas=lambda: "canvas-retry")
    assert denoiser.noises == ["noise-0"]


def test_stop_and_warn_policies_never_retry(monkeypatch):
    for policy, expect_raise in (("warn", False), ("stop", True)):
        monkeypatch.setenv("DG_DEGENERACY_POLICY", policy)
        denoiser = _Denoiser([COLLAPSED, HEALTHY])
        if expect_raise:
            with pytest.raises(DG.DegenerateBlockError):  # allow-pytest.raises: parametrised in-loop
                _run(denoiser, retry_noise=_retry_noise_fn(), retry_canvas=lambda: "canvas-retry")
        else:
            _run(denoiser, retry_noise=_retry_noise_fn(), retry_canvas=lambda: "canvas-retry")
        assert denoiser.noises == ["noise-0"], f"{policy} must not re-denoise"


def test_retries_env_is_validated(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_RETRIES", "0")
    with expect_error(ValueError, match="DG_DEGENERACY_RETRIES"):
        DG.resolve_retries()
    monkeypatch.delenv("DG_DEGENERACY_RETRIES")
    assert DG.resolve_retries() == DG.DEFAULT_RETRIES


def test_seeded_noise_factory_varies_by_attempt():
    """Attempt 0 and attempt 1 must not produce the same per-step seeds."""

    class _Recorder:
        def __getattr__(self, name):
            raise AssertionError("the factory must not touch the device in this test")

    factory = make_seeded_gumbel_noise_fn(_Recorder(), batch=1, canvas_len=CANVAS, vocab_size=64, seed=11)
    assert getattr(factory, "supports_retry", False), "the device factory must advertise retry support"

    import models.experimental.diffusion_gemma.tt.sampling as TS

    original = TS.sample_gumbel_noise
    try:
        TS.sample_gumbel_noise = lambda shape, *, device, seed: seed
        first = [factory(0, 0)(step) for step in range(3)]
        retried = [factory(0, 1)(step) for step in range(3)]
    finally:
        TS.sample_gumbel_noise = original

    assert first != retried, "a retry must change the seeds"
    assert len(set(first) | set(retried)) == 6, "no seed may be reused between attempts"

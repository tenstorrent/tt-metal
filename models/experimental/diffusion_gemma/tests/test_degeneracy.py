# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical degenerate-canvas, stop-policy, and retry regressions."""

import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt import degeneracy as DG
from models.experimental.diffusion_gemma.tt.generate import denoise_and_commit_block

CANVAS = 256
EOS_ID = 1
CONTENT_ID = 236770


def _prose_like(seed: int = 0, length: int = CANVAS) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    weights = 1.0 / torch.arange(1, 4001, dtype=torch.float32)
    ids = torch.multinomial(weights, length, replacement=True, generator=generator)
    return ids.reshape(1, length)


def test_wall_of_one_token_is_maximally_degenerate():
    stats = DG.block_degeneracy(torch.full((1, CANVAS), 1, dtype=torch.long))
    assert stats["top_frac"] == 1.0
    assert stats["distinct"] == 1
    assert stats["max_run"] == CANVAS
    assert DG.is_degenerate(stats)


def test_realistic_prose_is_not_flagged():
    for seed in range(8):
        stats = DG.block_degeneracy(_prose_like(seed))
        assert not DG.is_degenerate(stats), f"seed {seed} false-positived: {stats}"
        assert stats["top_frac"] < DG.DEFAULT_TOP_FRAC
        assert stats["max_run"] < DG.DEFAULT_MAX_RUN


def test_stop_policy_raises_and_carries_the_evidence(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    tokens = torch.full((1, CANVAS), 42, dtype=torch.long)
    with expect_error(DG.DegenerateBlockError, match="degenerate committed canvas at block 5"):
        DG.check_committed_block(tokens, block_idx=5)


def test_measured_healthy_block_shapes_are_not_flagged():
    healthy = [
        {"top_frac": 0.0703, "max_run": 1, "top_id": 621},
        {"top_frac": 0.0820, "max_run": 1, "top_id": CONTENT_ID},
        {"top_frac": 0.0664, "max_run": 2, "top_id": CONTENT_ID},
        {"top_frac": 0.0625, "max_run": 2, "top_id": 621},
        {"top_frac": 0.1836, "max_run": 18, "top_id": 236772},
        {"top_frac": 0.1719, "max_run": 1, "top_id": 236772},
    ]
    degenerate = [
        {"top_frac": 0.9492, "max_run": 243, "top_id": CONTENT_ID},
        {"top_frac": 0.9375, "max_run": 240, "top_id": CONTENT_ID},
        {"top_frac": 0.8516, "max_run": 86, "top_id": CONTENT_ID},
    ]
    for stats in healthy:
        assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats
    for stats in degenerate:
        assert DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats


def _answer_then_padding(content_len: int, *, seed: int = 0) -> torch.Tensor:
    ids = _prose_like(seed, length=content_len).flatten()
    padding = torch.full((CANVAS - content_len,), EOS_ID, dtype=torch.long)
    return torch.cat([ids, padding]).reshape(1, CANVAS)


def test_the_five_shapes_the_served_run_rejected(monkeypatch):
    """Real terminal-padding shapes from the served run must not be rejected."""
    monkeypatch.delenv("DG_DEGENERACY_TOP_FRAC", raising=False)
    for content_len, recorded_top_frac in ((18, 0.930), (98, 0.617), (68, 0.734), (73, 0.715), (107, 0.582)):
        tokens = _answer_then_padding(content_len, seed=content_len)
        whole = DG.block_degeneracy(tokens)
        assert abs(whole["top_frac"] - recorded_top_frac) < 0.02
        assert DG.is_degenerate(whole), "reproduces the pre-fix rejection"
        content_stats = DG.block_degeneracy(tokens, stop_token_ids=[EOS_ID])
        assert not DG.is_degenerate(content_stats, stop_token_ids=[EOS_ID]), (content_len, content_stats)


def test_content_collapse_with_an_eos_tail_is_still_degenerate():
    collapsed = torch.cat(
        [
            _prose_like(9, length=39).flatten(),
            torch.full((161,), CONTENT_ID, dtype=torch.long),
            torch.full((56,), EOS_ID, dtype=torch.long),
        ]
    ).reshape(1, CANVAS)
    stats = DG.block_degeneracy(collapsed, stop_token_ids=[EOS_ID])
    assert stats["stop_tail"] == 56
    assert DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats


RETRY_CANVAS_LEN = 8


def _canvas(ids):
    return torch.tensor([ids], dtype=torch.long)


COLLAPSED = _canvas([CONTENT_ID] * RETRY_CANVAS_LEN)
HEALTHY = _canvas([5, 9, 17, 33, 41, 58, 64, 77])


def _trajectory(tokens):
    return DenoiseTrajectory(committed=tokens, num_steps=4, halted=True, per_step=[])


class _Denoiser:
    def __init__(self, canvases):
        self.canvases = list(canvases)
        self.noises = []
        self.canvas_args = []

    def __call__(self, logits_fn, init_canvas, config, *, gumbel_noise_fn=None, noise_tokens_fn=None):
        self.noises.append(gumbel_noise_fn)
        self.canvas_args.append(init_canvas)
        return _trajectory(self.canvases[min(len(self.noises) - 1, len(self.canvases) - 1)])


def _retry_noise_fn():
    def factory(attempt):
        return f"attempt-aware-{attempt}"

    factory.supports_retry = True
    return factory


def _run(denoiser, *, commits):
    return denoise_and_commit_block(
        "model",
        object(),
        "canvas-0",
        DiffusionConfig(canvas_length=RETRY_CANVAS_LEN),
        start_pos=0,
        gumbel_noise_fn="noise-0",
        denoise_block_fn=denoiser,
        commit_fn=commits,
        stop_token_ids=[EOS_ID],
        retry_noise_fn=_retry_noise_fn(),
        retry_init_canvas_fn=lambda: "canvas-retry",
    )


def test_retry_commits_the_first_clean_attempt(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "retry")
    monkeypatch.setenv("DG_DEGENERACY_RETRIES", "2")
    denoiser = _Denoiser([COLLAPSED, HEALTHY])
    committed = []
    out = _run(denoiser, commits=lambda model, tokens, **kwargs: committed.append(tokens))
    assert torch.equal(out.committed, HEALTHY)
    assert len(committed) == 1 and torch.equal(committed[0], HEALTHY)
    assert denoiser.noises == ["noise-0", "attempt-aware-1"]
    assert denoiser.canvas_args == ["canvas-0", "canvas-retry"]


def test_retry_exhausted_still_refuses_to_commit(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "retry")
    monkeypatch.setenv("DG_DEGENERACY_RETRIES", "2")
    denoiser = _Denoiser([COLLAPSED])
    committed = []
    with expect_error(DG.DegenerateBlockError, match="degenerate committed canvas"):
        _run(denoiser, commits=lambda model, tokens, **kwargs: committed.append(tokens))
    assert len(denoiser.noises) == 3
    assert committed == []

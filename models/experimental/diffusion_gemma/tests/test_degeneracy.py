# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Degenerate-canvas detection, the commit-path policies, and the output-quality gates (#48291).

Block diffusion commits a whole 256-token canvas at once, so degeneration is directly visible in
the committed tensor -- the observed GPQA failure emitted a canvas of ``\\ \\ \\ ...`` and then a
solid wall of one token id. These CPU tests pin the measurement, the calibration bounds against
realistic prose, the four commit policies, the thinking prompt contract every offline measurement
was missing, and the LLM-judge gate tool. Nothing here opens a device or talks to the network.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory
from models.experimental.diffusion_gemma.tt import degeneracy as DG
from models.experimental.diffusion_gemma.tt.generate import (
    denoise_and_commit_block,
    make_seeded_gumbel_noise_fn,
    tokenize_prompt,
)

CANVAS = 256

# The distinction calibration forced: on the real device run, a finished answer fills the canvas
# with <eos> (id 1) and scores top_frac 1.0 / max_run 256 -- numerically identical to the
# degenerate walls, opposite meaning. The GPQA degenerate canvases collapse onto CONTENT ids
# instead (id 621 " \\" and id 236770 "1"), which is what keeps the two separable.
EOS_ID = 1
CONTENT_ID = 236770


def _prose_like(seed: int = 0, length: int = CANVAS) -> torch.Tensor:
    """A canvas with a realistic id distribution: Zipf-ish, with repeated common tokens."""
    generator = torch.Generator().manual_seed(seed)
    weights = 1.0 / torch.arange(1, 4001, dtype=torch.float32)
    ids = torch.multinomial(weights, length, replacement=True, generator=generator)
    return ids.reshape(1, length)


# --- canvas degeneracy measurement ------------------------------------------------------------


def test_wall_of_one_token_is_maximally_degenerate():
    stats = DG.block_degeneracy(torch.full((1, CANVAS), 1, dtype=torch.long))
    assert stats["top_frac"] == 1.0
    assert stats["distinct"] == 1
    assert stats["max_run"] == CANVAS
    assert DG.is_degenerate(stats)


def test_short_cycle_is_caught_even_with_a_small_max_run():
    """The `\\ \\ \\ \\` shape: two ids alternating, so no long run but a huge top_frac."""
    ids = torch.tensor([[7, 9] * (CANVAS // 2)], dtype=torch.long)
    stats = DG.block_degeneracy(ids)
    assert stats["max_run"] == 1
    assert stats["top_frac"] == 0.5
    assert DG.is_degenerate(stats), "a 2-cycle must be caught by top_frac even with max_run == 1"


def test_realistic_prose_is_not_flagged():
    """The bounds must be far from healthy text or the gate is useless."""
    for seed in range(8):
        stats = DG.block_degeneracy(_prose_like(seed))
        assert not DG.is_degenerate(stats), f"seed {seed} false-positived: {stats}"
        assert stats["top_frac"] < DG.DEFAULT_TOP_FRAC
        assert stats["max_run"] < DG.DEFAULT_MAX_RUN


def test_long_run_of_one_token_inside_otherwise_healthy_text():
    ids = _prose_like(3).clone()
    ids[0, 100 : 100 + DG.DEFAULT_MAX_RUN] = 12345
    stats = DG.block_degeneracy(ids)
    assert stats["max_run"] >= DG.DEFAULT_MAX_RUN
    assert DG.is_degenerate(stats)


def test_longest_run_edges():
    assert DG.longest_run(torch.tensor([], dtype=torch.long)) == 0
    assert DG.longest_run(torch.tensor([5], dtype=torch.long)) == 1
    assert DG.longest_run(torch.tensor([1, 1, 2, 3, 3, 3, 4], dtype=torch.long)) == 3
    assert DG.longest_run(torch.tensor([1, 2, 3], dtype=torch.long)) == 1


def test_empty_canvas_is_not_degenerate():
    stats = DG.block_degeneracy(torch.zeros((1, 0), dtype=torch.long))
    assert stats["tokens"] == 0
    assert not DG.is_degenerate(stats)


# --- commit-path policy -----------------------------------------------------------------------


def test_default_policy_is_stop(monkeypatch, expect_error):
    """The default must actually prevent the emission, not just log it."""
    monkeypatch.delenv("DG_DEGENERACY_POLICY", raising=False)
    assert DG.resolve_policy() == "stop"
    with expect_error(DG.DegenerateBlockError):
        DG.check_committed_block(torch.full((1, CANVAS), 42, dtype=torch.long))


def test_off_policy_measures_nothing(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "off")
    assert DG.check_committed_block(torch.full((1, CANVAS), 1, dtype=torch.long)) == {}


@pytest.mark.parametrize(
    "token_id,kwargs",
    [
        pytest.param(42, {}, id="no-block-idx"),
    ],
)
def test_warn_policy_returns_stats_without_raising(monkeypatch, token_id, kwargs):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "warn")
    stats = DG.check_committed_block(torch.full((1, CANVAS), token_id, dtype=torch.long), **kwargs)
    assert stats["top_frac"] == 1.0


def test_stop_policy_raises_and_carries_the_evidence(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    tokens = torch.full((1, CANVAS), 42, dtype=torch.long)
    with expect_error(DG.DegenerateBlockError, match="degenerate committed canvas at block 5"):
        DG.check_committed_block(tokens, block_idx=5)


def test_stop_policy_leaves_healthy_blocks_alone(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    stats = DG.check_committed_block(_prose_like(1), block_idx=0)
    assert stats["distinct"] > 1


def test_invalid_policy_fails_loudly(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "halt")
    with expect_error(ValueError, match="DG_DEGENERACY_POLICY"):
        DG.resolve_policy()


def test_the_top_frac_threshold_is_consulted_on_every_call(monkeypatch, expect_error):
    """Tightening DEFAULT_TOP_FRAC must change the verdict on prose that is healthy at 0.5.

    This is what the deleted ``DG_DEGENERACY_TOP_FRAC`` env override used to prove. The threshold is
    read from the module global inside :func:`is_degenerate`, not bound as a default argument, so
    ``monkeypatch.setattr`` reaches it — that is the property this test pins.
    """
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    assert not DG.is_degenerate(DG.block_degeneracy(_prose_like(0))), "prose is healthy at 0.5"
    monkeypatch.setattr(DG, "DEFAULT_TOP_FRAC", 0.02)
    with expect_error(DG.DegenerateBlockError):
        DG.check_committed_block(_prose_like(0))


# --- termination versus degeneration: the declared stop set -----------------------------------


def test_all_eos_canvas_is_termination_not_degeneration():
    stats = DG.block_degeneracy(torch.full((1, CANVAS), EOS_ID, dtype=torch.long))
    assert DG.is_degenerate(stats), "with no stop set declared it still looks degenerate"
    assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), "a terminating canvas must not be flagged"


def test_stop_policy_lets_a_terminating_block_through(monkeypatch):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    stats = DG.check_committed_block(
        torch.full((1, CANVAS), EOS_ID, dtype=torch.long), block_idx=9, stop_token_ids=[EOS_ID]
    )
    assert stats["top_id"] == EOS_ID


def test_stop_policy_still_catches_the_content_wall(monkeypatch, expect_error):
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    with expect_error(DG.DegenerateBlockError, match="degenerate committed canvas"):
        DG.check_committed_block(
            torch.full((1, CANVAS), CONTENT_ID, dtype=torch.long), block_idx=7, stop_token_ids=[EOS_ID]
        )


def test_measured_healthy_block_shapes_are_not_flagged():
    """Real per-block statistics from the traced device sweep (host Gumbel, seed 0).

    Healthy blocks of a LaTeX-heavy physics answer: 54-106 distinct ids, top_frac 0.06-0.08,
    max_run 1-2. The degenerate ones from the same run: 1-16 distinct, top_frac 0.94-1.00,
    max_run 240-256. The bounds must sit in that gap.
    """
    healthy = [
        {"top_frac": 0.0703, "max_run": 1, "top_id": 621},
        {"top_frac": 0.0820, "max_run": 1, "top_id": CONTENT_ID},
        {"top_frac": 0.0664, "max_run": 2, "top_id": CONTENT_ID},
        {"top_frac": 0.0625, "max_run": 2, "top_id": 621},
        # the two most extreme HEALTHY canvases across all 136 measured: the bounds must clear both
        {"top_frac": 0.1836, "max_run": 18, "top_id": 236772},
        {"top_frac": 0.1719, "max_run": 1, "top_id": 236772},
    ]
    degenerate = [
        {"top_frac": 0.9492, "max_run": 243, "top_id": CONTENT_ID},
        {"top_frac": 0.9375, "max_run": 240, "top_id": CONTENT_ID},
        # GPQA doc7 under the host Gumbel default -- the case that proves the root-cause fix
        # alone is not sufficient and the guard has to be the default.
        {"top_frac": 0.8516, "max_run": 86, "top_id": CONTENT_ID},
    ]
    for stats in healthy:
        assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats
    for stats in degenerate:
        assert DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats


def test_scalar_stop_token_id_is_accepted():
    """Sessions initialised from a bare `eos_token_id` pass a scalar, not a collection."""
    stats = DG.block_degeneracy(torch.full((1, CANVAS), EOS_ID, dtype=torch.long))
    assert not DG.is_degenerate(stats, stop_token_ids=EOS_ID)
    assert DG.is_degenerate(stats, stop_token_ids=999)


# --- the terminal-padding shape ---------------------------------------------------------------
# What the whole-canvas rule missed: the canvases the served run actually rejected were not walls
# of <eos>, they were ANSWERS followed by <eos> padding. A block that finishes at position 149
# pads the remaining 107 -- top_frac 0.58 and max_run 107 over the whole canvas, both past the
# gate, while the content region is ordinary prose. On tt-shield run 30285823000 that ended 110 of
# 198 requests and threw away the block holding the answer. Shapes below are taken from that run's
# `ending request at block N` lines.


def _answer_then_padding(content_len: int, *, seed: int = 0) -> torch.Tensor:
    """A finished answer: healthy prose in the first ``content_len`` positions, then padding."""
    ids = _prose_like(seed, length=content_len).flatten()
    pad = torch.full((CANVAS - content_len,), EOS_ID, dtype=torch.long)
    return torch.cat([ids, pad]).reshape(1, CANVAS)


def test_answer_followed_by_eos_padding_is_not_degenerate():
    tokens = _answer_then_padding(149)
    whole = DG.block_degeneracy(tokens)
    assert whole["top_frac"] >= DG.DEFAULT_TOP_FRAC or whole["max_run"] >= DG.DEFAULT_MAX_RUN
    assert DG.is_degenerate(whole), "the whole-canvas view is what mis-rejected it"

    stats = DG.block_degeneracy(tokens, stop_token_ids=[EOS_ID])
    assert stats["stop_tail"] == CANVAS - 149
    assert stats["content_tokens"] == 149
    assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), stats


def test_the_five_shapes_the_served_run_rejected():
    """Real (distinct, top_frac, max_run) tuples the 07-27 server log printed as degenerate."""
    for content_len, recorded_top_frac in ((18, 0.930), (98, 0.617), (68, 0.734), (73, 0.715), (107, 0.582)):
        tokens = _answer_then_padding(content_len, seed=content_len)
        whole = DG.block_degeneracy(tokens)
        assert abs(whole["top_frac"] - recorded_top_frac) < 0.02, (content_len, whole["top_frac"])
        assert DG.is_degenerate(whole), "reproduces the pre-fix rejection"
        stats = DG.block_degeneracy(tokens, stop_token_ids=[EOS_ID])
        assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID]), (content_len, stats)


def test_mixed_stop_ids_in_the_tail_are_one_run():
    """<end_of_turn> (106) tails must count as padding too, not just <eos>."""
    ids = _prose_like(5, length=200).flatten()
    tail = torch.tensor([EOS_ID] * 30 + [106] * 26, dtype=torch.long)
    stats = DG.block_degeneracy(torch.cat([ids, tail]).reshape(1, CANVAS), stop_token_ids=[EOS_ID, 106])
    assert stats["stop_tail"] == 56, "the tail is padding regardless of which stop id fills it"
    assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID, 106])


def test_content_collapse_with_an_eos_tail_is_still_degenerate():
    """The 6 real trips on content ids: a short prefix, then a wall of content, then padding."""
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


def test_full_content_wall_with_a_stop_set_is_still_degenerate():
    """`尼` x 256 -- the 07-24 shape. Stripping must not empty the content region here."""
    stats = DG.block_degeneracy(torch.full((1, CANVAS), 239054, dtype=torch.long), stop_token_ids=[EOS_ID])
    assert stats["stop_tail"] == 0
    assert stats["content_tokens"] == CANVAS
    assert DG.is_degenerate(stats, stop_token_ids=[EOS_ID])


def test_all_stop_token_canvas_has_no_content_region():
    stats = DG.block_degeneracy(torch.full((1, CANVAS), EOS_ID, dtype=torch.long), stop_token_ids=[EOS_ID])
    assert stats["stop_tail"] == CANVAS
    assert stats["content_tokens"] == 0
    assert not DG.is_degenerate(stats, stop_token_ids=[EOS_ID])


def test_terminal_stop_run_only_counts_the_tail():
    ids = torch.tensor([5, EOS_ID, EOS_ID, 7, EOS_ID, EOS_ID, EOS_ID], dtype=torch.long)
    assert DG.terminal_stop_run(ids, {EOS_ID}) == 3, "a mid-canvas stop token is not padding"
    assert DG.terminal_stop_run(torch.tensor([5, 7], dtype=torch.long), {EOS_ID}) == 0
    assert DG.terminal_stop_run(torch.tensor([], dtype=torch.long), {EOS_ID}) == 0
    assert DG.terminal_stop_run(ids, set()) == 0, "no declared stop set means nothing is padding"


def test_whole_canvas_rule_is_unchanged_without_stop_ids():
    """Callers that declare no stop set keep the old behaviour rather than a weakened gate."""
    tokens = _answer_then_padding(149)
    stats = DG.block_degeneracy(tokens)
    assert "content_tokens" not in stats
    assert DG.is_degenerate(stats)


def test_stop_policy_commits_a_finished_answer(monkeypatch):
    """End to end through the policy: the shape that lost 110 answers must now pass."""
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "stop")
    stats = DG.check_committed_block(_answer_then_padding(149), block_idx=6, stop_token_ids=[EOS_ID])
    assert stats["content_tokens"] == 149


def test_describe_reports_the_content_region():
    stats = DG.block_degeneracy(
        torch.cat(
            [
                torch.full((200,), CONTENT_ID, dtype=torch.long),
                torch.full((56,), EOS_ID, dtype=torch.long),
            ]
        ).reshape(1, CANVAS),
        stop_token_ids=[EOS_ID],
    )
    message = DG.describe(stats, block_idx=3)
    assert "content 200/256" in message and "stop tail 56" in message


# --- DG_DEGENERACY_POLICY=retry ---------------------------------------------------------------
# `stop` prevents degenerate output but ends the request; `retry` re-runs the block with different
# Gumbel noise and commits the first attempt that is not degenerate.
#
# The property these tests exist to protect is the one that makes retry worth having at all: a
# retry must draw DIFFERENT noise. The per-step noise functions are pure in (block, step), so a
# factory without attempt awareness would reproduce the trajectory exactly and the retry would be
# a silent no-op -- worse than not retrying, because the log would claim a retry happened. So
# `retry` refuses to run unless the factory is marked `supports_retry`.

RETRY_CANVAS_LEN = 8


def _canvas(ids):
    return torch.tensor([ids], dtype=torch.long)


COLLAPSED = _canvas([CONTENT_ID] * RETRY_CANVAS_LEN)
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
        DiffusionConfig(canvas_length=RETRY_CANVAS_LEN),
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


def test_retry_does_not_fire_on_a_terminating_canvas(monkeypatch):
    """A wall of <eos> is termination, not degeneration -- retrying it would be wrong."""
    monkeypatch.setenv("DG_DEGENERACY_POLICY", "retry")
    denoiser = _Denoiser([_canvas([EOS_ID] * RETRY_CANVAS_LEN)])
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


def test_seeded_noise_factory_varies_by_attempt():
    """Attempt 0 and attempt 1 must not produce the same per-step seeds."""

    class _Recorder:
        def __getattr__(self, name):
            raise AssertionError("the factory must not touch the device in this test")

    factory = make_seeded_gumbel_noise_fn(_Recorder(), batch=1, canvas_len=RETRY_CANVAS_LEN, vocab_size=64, seed=11)
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


# --- thinking prompt contract -----------------------------------------------------------------
# `enable_thinking` was absent from the whole DiffusionGemma tree, so every offline measurement --
# including the early-halt probe whose "no-op" conclusion rests on entropy floors of
# 0.155/0.138/0.506 nats -- ran the model against the non-thinking prompt even though the released
# checkpoint is post-trained with a `<|think|>` turn. Tokenizer only: no device, no weights.

DG_CKPT = os.getenv("DG_CKPT", "google/diffusiongemma-26B-A4B-it")
PROMPT = "What is 17*23?"


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(DG_CKPT, local_files_only=True)
    except Exception as load_error:  # gated / not cached in this environment
        pytest.skip(f"DiffusionGemma tokenizer unavailable ({type(load_error).__name__}): {load_error}")


class _IgnoresThinkingTokenizer:
    """A chat tokenizer whose template drops unknown kwargs -- the silent-downgrade shape."""

    def apply_chat_template(self, messages, *, add_generation_prompt=True, tokenize=False, **ignored):
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if not tokenize:
            return rendered
        return {"input_ids": list(range(len(rendered.split())))}


class _NoChatTemplateTokenizer:
    def encode(self, text):
        return list(range(len(text.split())))


def test_default_render_is_unchanged_by_the_new_parameter(tokenizer):
    """enable_thinking=None must pass NOTHING to the template, byte-for-byte as before."""
    baseline = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=True
    )
    got = tokenize_prompt(tokenizer, PROMPT)
    assert got.flatten().tolist() == list(baseline["input_ids"])


def test_thinking_changes_the_prompt_and_is_longer(tokenizer):
    plain = tokenize_prompt(tokenizer, PROMPT)
    thinking = tokenize_prompt(tokenizer, PROMPT, enable_thinking=True)
    assert thinking.shape[1] > plain.shape[1], "the <|think|> turn must add tokens"
    assert thinking.flatten().tolist() != plain.flatten().tolist()
    # The thinking turn is rendered as a system turn, so the user turn is pushed to the right.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False, enable_thinking=True
    )
    assert "<|think|>" in rendered


def test_explicit_false_matches_the_template_default(tokenizer):
    assert (
        tokenize_prompt(tokenizer, PROMPT, enable_thinking=False).flatten().tolist()
        == tokenize_prompt(tokenizer, PROMPT).flatten().tolist()
    )


@pytest.mark.parametrize(
    "target,prompt,match",
    [
        # the regression that produced the malformed contract: a silently ignored request
        pytest.param(_IgnoresThinkingTokenizer(), PROMPT, "being ignored", id="template-ignores-the-flag"),
        pytest.param(
            object(), torch.tensor([[2, 105, 2364]], dtype=torch.long), "pre-tokenized", id="pre-tokenized-prompt"
        ),
        pytest.param(_NoChatTemplateTokenizer(), PROMPT, "apply_chat_template", id="no-chat-template"),
    ],
)
def test_thinking_request_is_refused_instead_of_downgraded(target, prompt, match, expect_error):
    with expect_error(ValueError, match=match):
        tokenize_prompt(target, prompt, enable_thinking=True)


def test_ignored_flag_is_allowed_when_thinking_is_not_requested():
    """enable_thinking=False on a template without thinking support is not an error."""
    assert tokenize_prompt(_IgnoresThinkingTokenizer(), PROMPT, enable_thinking=False).shape[0] == 1


# --- llm-judge gate tool ----------------------------------------------------------------------

SRC = Path(__file__).resolve().parents[1] / "doc" / "decision_fidelity" / "gate" / "llm_judge.py"


@pytest.fixture(scope="module")
def judge():
    spec = importlib.util.spec_from_file_location("dg_llm_judge", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = {"model": "claude-opus-5", "effort": "low", "votes": 1, "max_chars": 24000}
ITEM = {
    "question": "Which one?",
    "choices": ["alpha", "beta", "gamma particle", "delta"],
    "gold_letter": "C",
    "text": "Reasoning. The answer is (C).",
}
VERDICT = {
    "meaningful": True,
    "failure_mode": "none",
    "language": "en",
    "answered": True,
    "selected_letter": "C",
    "selected_answer_text": "gamma particle",
    "notes": "states an answer",
}


def _resp(payload=None):
    """A Messages response: thinking block first (thinking is on by default), then the verdict text.

    Every judge_one test drives this shape, so a caller that went back to indexing ``content[0]``
    fails all of them -- the thinking block carries no ``.text``.
    """
    blocks = [SimpleNamespace(type="thinking", thinking="")]
    blocks.append(SimpleNamespace(type="text", text=json.dumps(payload if payload is not None else VERDICT)))
    return SimpleNamespace(content=blocks, stop_reason="end_turn", stop_details=None)


def _client(response=None, capture=None):
    def create(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        return response if response is not None else _resp()

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def _raising_client(exc):
    def create(**_kwargs):
        raise exc

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def test_truncate_keeps_the_tail_where_the_answer_lives(judge):
    got = judge.truncate("H" * 5000 + "ANSWER IS (B)", 1000)
    assert "ANSWER IS (B)" in got and "elided" in got and len(got) < 1200


def test_prompt_never_reveals_which_choice_is_correct(judge):
    prompt = judge.prompt_for(ITEM, 24000)
    assert "gold" not in prompt.lower()
    assert "(C) gamma particle" in prompt  # offered as a plain option, unmarked


def test_request_omits_temperature_and_sets_the_schema(judge):
    """Opus 5 returns a 400 for temperature/top_p/top_k, and the verdict must be schema-constrained."""
    sent = {}
    judge.judge_one(_client(capture=sent), ITEM, CFG)
    assert not {"temperature", "top_p", "top_k"} & sent.keys()
    assert sent["output_config"] == {"effort": "low", "format": judge.SCHEMA}
    assert sent["fallbacks"] == "default" and sent["model"] == "claude-opus-5"


def test_a_refusal_is_not_treated_as_a_verdict(judge, expect_error):
    """A refusal is a successful HTTP 200 -- indexing content blindly is what breaks."""
    refused = SimpleNamespace(content=[], stop_reason="refusal", stop_details=SimpleNamespace(category="bio"))
    with expect_error(RuntimeError, match="declined"):
        judge.judge_one(_client(refused), ITEM, CFG)


def test_a_bogus_letter_is_dropped(judge):
    got = judge.judge_one(_client(_resp({**VERDICT, "selected_letter": "Z"})), ITEM, CFG)
    assert got["selected_letter"] is None


def test_empty_text_is_settled_locally_with_no_call(judge):
    client = _raising_client(AssertionError("should not call the API for empty text"))
    got = judge.judge_item(client, {"text": "  \n "}, CFG)
    assert got["failure_mode"] == "empty" and not got["meaningful"]


def test_a_failing_item_does_not_lose_the_run(judge):
    got = judge.judge_item(_raising_client(RuntimeError("500 boom")), ITEM, CFG)
    assert "500 boom" in got["error"]


def test_votes_take_the_majority_and_flag_the_split(judge):
    dissent = {**VERDICT, "meaningful": False, "failure_mode": "repetition", "answered": False, "selected_letter": None}
    got = judge.majority([VERDICT, VERDICT, dissent])
    assert got["meaningful"] and got["selected_letter"] == "C" and got["failure_mode"] == "none"
    assert got["split"] is True and got["votes"] == 3


def test_judge_letter_falls_back_to_matching_the_answer_text(judge):
    """The text path is what makes grading shuffle-independent."""
    verdict = {"selected_letter": None, "selected_answer_text": "the gamma particle"}
    assert judge.judge_letter(verdict, ITEM) == "C"


def test_judge_letter_credits_nothing_when_the_response_never_answered(judge):
    """Deriving a letter from a non-answer is the regex extractor's stage-3 mistake."""
    verdict = {"answered": False, "selected_letter": None, "selected_answer_text": "gamma particle"}
    assert judge.judge_letter(verdict, ITEM) is None


def test_judge_letter_refuses_an_ambiguous_text_match(judge):
    item = {**ITEM, "choices": ["alpha", "alpha decay", "gamma", "delta"]}
    assert judge.judge_letter({"selected_letter": None, "selected_answer_text": "alpha"}, item) is None


def test_load_detects_lm_eval_samples_by_content_not_filename(judge, tmp_path):
    """A renamed copy used to fall through and get judged as ONE giant response."""
    path = tmp_path / "renamed_run.jsonl"  # deliberately lacks "samples_"
    doc = {"Record ID": "rec-1", "Question": "Q?", "choices": ["a", "b", "c", "d"], "answer": "(A)"}
    path.write_text(json.dumps({"filter": "flexible-extract", "doc": doc, "resps": [["ans"]]}) + "\n")
    src, items = judge.load(path, "full")
    assert "1 questions" in src and items[0]["id"] == "rec-1"


def test_load_survives_a_u2028_inside_a_response(judge, tmp_path):
    """str.splitlines() breaks on U+2028 but JSON does not escape it, so it cuts the line in half.
    The real 198-question file has four of them."""
    path = tmp_path / "samples_x.jsonl"
    doc = {"Record ID": "rec-1", "Question": "Q?", "choices": ["a", "b"], "answer": "(A)"}
    text = "before\u2028after the separator"  # a real U+2028
    row = {"filter": "flexible-extract", "doc": doc, "resps": [[text]]}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    _src, items = judge.load(path, "full")
    assert len(items) == 1 and items[0]["text"] == text


def test_load_reads_one_row_per_question_not_per_filter(judge, tmp_path):
    path = tmp_path / "samples_gpqa.jsonl"
    doc = {"Record ID": "rec-9", "Question": "Q?", "choices": ["a", "b", "c", "d"], "answer": "(B)"}
    rows = [
        {"filter": "flexible-extract", "doc": doc, "resps": [["The answer is (B)."]], "exact_match": 1.0},
        {"filter": "strict-match", "doc": doc, "resps": [["The answer is (B)."]], "exact_match": 0.0},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _src, items = judge.load(path, "full")
    assert len(items) == 1 and items[0]["gold_letter"] == "B" and items[0]["regex_correct"] is True


def test_report_counts_what_the_regex_laundered(judge, capsys):
    items = [{**ITEM, "text": "(C) appears in prose", "regex_correct": True}, {**ITEM, "regex_correct": True}]
    verdicts = [{**VERDICT, "answered": False, "selected_letter": None}, VERDICT]
    judge.report("src", items, verdicts, CFG)
    out = capsys.readouterr().out
    assert "1 response(s) it scored correct that never answered" in out
    assert "correct: 1/2" in out
    assert "1 disagreement(s)" in out

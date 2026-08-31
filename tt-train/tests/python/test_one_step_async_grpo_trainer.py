# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-less loop-shape tests for :class:`OneStepAsyncGRPOTrainer`.

Focuses on what the async subclass adds on top of ``GRPOTrainer``:

* the ``_make_batch_iterator`` / ``_async_gen_next_batch`` primitives,
* the verl-shape ``train()`` composer (prime, then ``await`` -> ``push+submit``
  -> train per iteration),
* the completer-contract preflight check.

Bypasses the real ``GRPOTrainer.__init__`` and the phase helpers so the test
can run without a Tenstorrent device attached: the loop is what we're testing,
not any single-step correctness (that is covered end-to-end by the live
gsm8k_onestep example).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import numpy as np

from ttml.trainers import OneStepAsyncGRPOTrainer


class _StubCompleter:
    """Records the exact sequence of push / submit / await calls.

    Returns a deterministic completion list per submit so the trainer's
    downstream expansion has something concrete to look at even though the
    phase helpers are stubbed away.
    """

    def __init__(self, completions_per_prompt: int) -> None:
        self.completions_per_prompt = completions_per_prompt
        self.calls: List[str] = []
        self._last_submit_size = 0

    def push_weights(self) -> None:
        self.calls.append("push")

    def submit_generate(self, prompts: List[List[int]]) -> None:
        self.calls.append("submit")
        self._last_submit_size = len(prompts) * self.completions_per_prompt

    def await_generate(self) -> List[List[int]]:
        self.calls.append("await")
        return [[0, 1, 2] for _ in range(self._last_submit_size)]


class _RecordingCallback:
    def __init__(self) -> None:
        self.train_begin = 0
        self.train_end = 0
        self.step_end = 0
        self.before_step = 0

    def on_train_begin(self, trainer: Any) -> None:
        self.train_begin += 1

    def on_train_end(self, trainer: Any) -> None:
        self.train_end += 1

    def on_before_optimizer_step(self, trainer: Any) -> None:
        self.before_step += 1

    def on_step_end(self, trainer: Any, step: int, **kwargs: Any) -> None:
        self.step_end += 1


def _make_trainer(
    *,
    num_prompts: int,
    generation_batch_prompts: int,
    num_generations: int,
    num_iterations: int = 1,
    completer: Any,
    callbacks: List[Any],
) -> OneStepAsyncGRPOTrainer:
    """Bypass ``GRPOTrainer.__init__`` and pre-populate only the state the
    async ``train()`` composer actually reads."""

    trainer = OneStepAsyncGRPOTrainer.__new__(OneStepAsyncGRPOTrainer)
    trainer.completer = completer
    trainer.callbacks = callbacks
    trainer.config = SimpleNamespace(
        num_iterations=num_iterations,
        num_generations=num_generations,
    )
    trainer._prompts = [[i, i + 1] for i in range(num_prompts)]
    trainer._extra_columns = {"answer": [f"a{i}" for i in range(num_prompts)]}
    trainer._generation_batch_prompts = generation_batch_prompts

    def _stub_setup_training(self=trainer) -> None:
        return None

    def _stub_rewards(self, prompts_expanded, completions_batch, columns_expanded):
        rewards = np.zeros(len(completions_batch), dtype=np.float32)
        advantages = np.zeros_like(rewards)
        prompts_strs = [str(p) for p in prompts_expanded]
        completions_strs = [str(c) for c in completions_batch]
        return rewards, advantages, prompts_strs, completions_strs

    def _stub_ref(self, prompts_expanded, completions_batch):
        return []

    def _stub_optimizer_step(self, *args, **kwargs):
        return 1.0

    def _stub_finalize(self, *args, **kwargs):
        return None

    trainer._setup_training = _stub_setup_training  # type: ignore[assignment]
    trainer._run_rewards_and_advantages = _stub_rewards.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._run_ref_logprobs = _stub_ref.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._run_optimizer_step = _stub_optimizer_step.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._finalize_step_and_fire_callbacks = _stub_finalize.__get__(trainer, type(trainer))  # type: ignore[assignment]

    return trainer


def test_make_batch_iterator_yields_unexpanded_batches():
    trainer = _make_trainer(
        num_prompts=6,
        generation_batch_prompts=2,
        num_generations=4,
        completer=_StubCompleter(completions_per_prompt=4),
        callbacks=[],
    )
    batches = list(trainer._make_batch_iterator())

    assert len(batches) == 3, "6 prompts / gbp=2 -> 3 batches"
    for prompts_batch, columns_batch in batches:
        assert len(prompts_batch) == 2, "iterator must NOT pre-expand by num_generations"
        assert list(columns_batch.keys()) == ["answer"]
        assert len(columns_batch["answer"]) == 2


def test_async_gen_next_batch_returns_none_without_touching_completer():
    completer = _StubCompleter(completions_per_prompt=2)
    trainer = _make_trainer(
        num_prompts=0,
        generation_batch_prompts=2,
        num_generations=2,
        completer=completer,
        callbacks=[],
    )
    result = trainer._async_gen_next_batch(iter([]))

    assert result is None
    assert completer.calls == [], "empty iterator must NOT push or submit"


def test_train_verl_shape_call_sequence_and_counts():
    """3 generation batches -> exact interleaving of push / submit / await."""
    completer = _StubCompleter(completions_per_prompt=2)
    recorder = _RecordingCallback()
    trainer = _make_trainer(
        num_prompts=6,
        generation_batch_prompts=2,
        num_generations=2,
        completer=completer,
        callbacks=[recorder],
    )

    trainer.train()

    expected = [
        # prime: push theta_0, submit gen_0
        "push",
        "submit",
        # iter 0: await R_0, push theta_0, submit gen_1
        "await",
        "push",
        "submit",
        # iter 1: await R_1, push theta_1, submit gen_2
        "await",
        "push",
        "submit",
        # iter 2 (last): await R_2, _async_gen_next_batch -> None (no push, no submit)
        "await",
    ]
    assert (
        completer.calls == expected
    ), f"verl-shape interleaving broken.\nexpected={expected}\ngot     ={completer.calls}"

    assert completer.calls.count("push") == 3
    assert completer.calls.count("submit") == 3
    assert completer.calls.count("await") == 3

    assert recorder.train_begin == 1
    assert recorder.train_end == 1


def test_train_rejects_num_iterations_greater_than_one(expect_error):
    completer = _StubCompleter(completions_per_prompt=2)
    trainer = _make_trainer(
        num_prompts=2,
        generation_batch_prompts=2,
        num_generations=2,
        num_iterations=2,
        completer=completer,
        callbacks=[],
    )
    with expect_error(ValueError, "num_iterations == 1"):
        trainer.train()


def test_train_rejects_sync_only_completer(expect_error):
    """A completer missing the async surface must fail fast, before any device work."""

    class _SyncOnlyCompleter:
        def generate(self, prompts):
            return prompts

    trainer = _make_trainer(
        num_prompts=2,
        generation_batch_prompts=2,
        num_generations=2,
        completer=_SyncOnlyCompleter(),
        callbacks=[],
    )
    with expect_error(TypeError, "submit_generate"):
        trainer.train()

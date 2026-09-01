# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-less loop-shape tests for :class:`OneStepAsyncGRPOTrainer`.

Focuses on what the async subclass adds on top of ``GRPOTrainer``:

* the inherited ``_iter_prompt_batches`` / ``_async_gen_next_batch`` primitives,
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
    trainer.reward_funcs = []
    trainer._reward_func_names = []
    trainer._prompts = [[i, i + 1] for i in range(num_prompts)]
    trainer._extra_dataset_columns = {"answer": [f"a{i}" for i in range(num_prompts)]}
    trainer._generation_batch_prompts = generation_batch_prompts
    trainer.metrics = {}

    def _stub_setup(self=trainer) -> None:
        return None

    def _stub_expand(self, prompts, extra_cols):
        g = self.config.num_generations
        prompts_x = [p for p in prompts for _ in range(g)]
        cols_x = {k: [v for v in col for _ in range(g)] for k, col in extra_cols.items()}
        return prompts_x, cols_x

    def _stub_rewards(self, prompts_x, completions, extra_cols_x):
        import numpy as np

        return np.zeros(len(completions), dtype=np.float32)

    def _stub_advantages(self, rewards_np):
        import numpy as np

        return np.zeros_like(rewards_np)

    def _stub_noop(self, *args, **kwargs):
        return None

    trainer._setup = _stub_setup  # type: ignore[assignment]
    trainer._expand_prompts_and_columns = _stub_expand.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._compute_rewards = _stub_rewards.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._compute_advantages = _stub_advantages.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._optimize = _stub_noop.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._apply_gradients = _stub_noop.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._publish_step_metrics = _stub_noop.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._maybe_checkpoint = _stub_noop.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._reset_step_metrics = _stub_noop.__get__(trainer, type(trainer))  # type: ignore[assignment]

    return trainer


def test_iter_prompt_batches_yields_unexpanded_batches():
    trainer = _make_trainer(
        num_prompts=6,
        generation_batch_prompts=2,
        num_generations=4,
        completer=_StubCompleter(completions_per_prompt=4),
        callbacks=[],
    )
    batches = list(trainer._iter_prompt_batches())

    assert len(batches) == 3, "6 prompts / gbp=2 -> 3 batches"
    for prompts, extra_cols in batches:
        assert len(prompts) == 2, "iterator must NOT pre-expand by num_generations"
        assert list(extra_cols.keys()) == ["answer"]
        assert len(extra_cols["answer"]) == 2


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


def test_publish_step_metrics_handles_step_in_self_metrics():
    """Regression: ``_publish_step_metrics`` reads ``step`` from
    ``self.metrics["step"]`` and passes it positionally to callbacks while
    also splatting the metrics dict. It MUST filter ``step`` out of the
    splat, otherwise Python raises ``TypeError: got multiple values for
    argument 'step'`` at ``cb.on_step_end(trainer, step, step=step, ...)``.
    """
    import time

    from ttml.trainers.grpo_trainer import GRPOTrainer

    recorder = _RecordingCallback()
    trainer = _make_trainer(
        num_prompts=2,
        generation_batch_prompts=2,
        num_generations=2,
        completer=_StubCompleter(completions_per_prompt=2),
        callbacks=[recorder],
    )
    # Un-stub the two methods this exercises so we hit the real code path.
    trainer._publish_step_metrics = GRPOTrainer._publish_step_metrics.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer._time_callback = GRPOTrainer._time_callback.__get__(trainer, type(trainer))  # type: ignore[assignment]
    trainer.metrics = {"step": 5, "reward_mean": 0.3}
    trainer._step_start_time = time.perf_counter()

    trainer._publish_step_metrics()

    assert recorder.step_end == 1
    assert trainer.metrics["step"] == 5
    assert "step_time_s" in trainer.metrics


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

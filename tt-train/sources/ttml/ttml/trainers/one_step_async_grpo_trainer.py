# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""One-step off-policy GRPO trainer (verl-shape loop).

Overlaps generation of batch ``t+1`` (on the remote rollout worker) with the
optimizer step on batch ``t``. See
``tt-train/sources/examples/grpo_remote_rollout/gsm8k_onestep/`` for a live
example wiring.

Off-policy staleness is exactly 1: rollouts ``R_t`` were generated with
``theta_{t-1}`` at the moment we consume them against actor ``theta_t`` to
produce ``theta_{t+1}``.
"""

from __future__ import annotations

import time
from typing import Any, Iterator, List, Optional, Tuple

from .grpo_trainer import GRPOTrainer


class OneStepAsyncGRPOTrainer(GRPOTrainer):
    """One-step off-policy GRPO trainer, verl-shape loop.

    Overrides :meth:`train` only; every phase helper on :class:`GRPOTrainer`
    (``_setup``, ``_iter_prompt_batches``, ``_expand_prompts_and_columns``,
    ``_compute_rewards``, ``_compute_advantages``, ``_optimize``,
    ``_apply_gradients``, ``_publish_step_metrics``, ``_maybe_checkpoint``,
    ``_reset_step_metrics``) is inherited unchanged. Two new small helpers
    (``_await_rollout``, ``_async_gen_next_batch``) encapsulate the
    async-rollout-specific work.

    Requires ``config.num_iterations == 1`` (fundamental to the 1:1 rollout to
    optimizer-step mapping). ``config.gradient_accumulation_steps > 1`` is
    supported: the outer batch is ``_generation_batch_prompts =
    prompts_per_microbatch * grad_accum`` and the inherited ``_optimize``
    handles micro-batching internally.

    Completer contract (extends :class:`GRPOCompleter`):
      * ``submit_generate(prompts) -> None`` -- non-blocking submit.
      * ``await_generate() -> List[List[int]]`` -- blocks for the last submit.
      * ``push_weights() -> None`` -- blocks until the receiver installs.

    Timing schema (differs from the sync trainer): writes
    ``generation_wait_s`` (wall clock blocked in ``await_generate``) instead
    of ``generation_time_s``. Near-zero when inference successfully overlaps
    training; positive when the trainer stalls waiting on the inference
    engine.
    """

    def _await_rollout(self) -> List[List[int]]:
        """Block on the pending rollout submit; write ``generation_wait_s``
        to ``self.metrics``. Mirrors :meth:`GRPOTrainer._rollout` in shape.
        """
        wait_t0 = time.perf_counter()
        completions = self.completer.await_generate()
        self.metrics["generation_wait_s"] = time.perf_counter() - wait_t0
        return completions

    def _async_gen_next_batch(
        self,
        iterator: Iterator[Tuple[List[List[int]], dict]],
    ) -> Optional[Tuple[List[List[int]], dict]]:
        """Push the current actor weights, then submit the NEXT generation batch.

        Returns ``(prompts, extra_cols)`` so a later :py:meth:`_await_rollout`
        pairing knows which prompts / columns line up with the completions it
        drains. Returns ``None`` when the iterator is exhausted, WITHOUT
        touching the completer (so end-of-loop does not waste a push or a
        submit whose response nobody would ever await).
        """
        try:
            prompts, extra_cols = next(iterator)
        except StopIteration:
            return None
        self.completer.push_weights()
        self.completer.submit_generate(prompts)
        return prompts, extra_cols

    def _require_async_completer(self) -> None:
        """Preflight the completer surface + config invariants.

        Fails fast (before ``_setup`` opens the device) if the trainer is
        wired against a completer that does not expose the async surface, or
        if the config asks for ``num_iterations > 1`` (which violates the
        1:1 rollout to optimizer-step invariant).
        """
        if self.config.num_iterations != 1:
            raise ValueError(
                f"OneStepAsyncGRPOTrainer requires num_iterations == 1 (got {self.config.num_iterations})."
            )
        for name in ("submit_generate", "await_generate", "push_weights"):
            if not hasattr(self.completer, name):
                raise TypeError(
                    f"OneStepAsyncGRPOTrainer requires the completer to expose "
                    f"{name}(); {type(self.completer).__name__} does not."
                )

    def train(self) -> None:
        """One-step async GRPO training loop (verl-shape).

        Per iteration:

          1. ``_await_rollout`` -> rollout submitted last iteration (gen'd
             with ``theta_{t-1}``).
          2. ``_async_gen_next_batch`` -> push ``theta_t`` + submit
             ``gen_{t+1}`` (server generates with ``theta_t`` while the
             trainer trains on ``R_t``).
          3. Inherited phase helpers train on the rollout to produce
             ``theta_{t+1}``.

        Priming (before the loop): one ``_async_gen_next_batch`` call pushes
        ``theta_0`` and submits ``gen_0`` so the first ``_await_rollout``
        returns ``R_0``.
        """
        self._require_async_completer()
        self._setup()
        for cb in self.callbacks:
            cb.on_train_begin(self)
        self.metrics = {"step": 0}
        self._reset_step_metrics()

        iterator = self._iter_prompt_batches()
        pending = self._async_gen_next_batch(iterator)  # push theta_0, submit gen_0

        while pending is not None:
            prompts, extra_cols = pending
            completions = self._await_rollout()  # rollout generated with theta_{t-1}
            pending = self._async_gen_next_batch(iterator)  # push theta_t, submit gen_{t+1}

            prompts_x, cols_x = self._expand_prompts_and_columns(prompts, extra_cols)
            rewards_np = self._compute_rewards(prompts_x, completions, cols_x)
            advantages_np = self._compute_advantages(rewards_np)
            self._optimize(prompts_x, completions, advantages_np)
            self._apply_gradients()
            self.metrics["step"] += 1
            self._publish_step_metrics()
            self._maybe_checkpoint()
            self._reset_step_metrics()

        for cb in self.callbacks:
            cb.on_train_end(self)


__all__ = ["OneStepAsyncGRPOTrainer"]

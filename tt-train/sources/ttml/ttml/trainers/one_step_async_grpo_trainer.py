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

from .grpo_trainer import GRPOTrainer, _deallocate_tensors


class OneStepAsyncGRPOTrainer(GRPOTrainer):
    """One-step off-policy GRPO trainer, verl-shape loop.

    Overrides :meth:`train` only; every phase helper on :class:`GRPOTrainer`
    (``_setup_training``, ``_run_rewards_and_advantages``, ``_run_ref_logprobs``,
    ``_run_optimizer_step``, ``_finalize_step_and_fire_callbacks``) is inherited
    unchanged.

    Requires ``config.num_iterations == 1`` (fundamental to 1:1 rollout to
    optimizer-step mapping). ``config.grad_accum > 1`` is supported: the outer
    batch is ``_generation_batch_prompts = prompts_per_microbatch * grad_accum``
    and the inherited ``_run_optimizer_step`` handles micro-batching + the
    single ``optimizer.step()`` internally.

    Completer contract (extends :class:`GRPOCompleter`):
      * ``submit_generate(prompts) -> None`` -- non-blocking submit.
      * ``await_generate() -> List[List[int]]`` -- blocks for the last submit.
      * ``push_weights() -> None`` -- blocks until the receiver installs.
    """

    def _make_batch_iterator(self) -> Iterator[Tuple[List[List[int]], dict]]:
        """Yield ``(prompts_batch, columns_batch)`` UNEXPANDED per generation batch.

        Mirrors ``iter_batched_completions`` in :mod:`grpo_trainer` -- the
        completer's ``submit_generate`` expands internally by
        ``completions_per_prompt``. Terminates on ``StopIteration``;
        :meth:`_async_gen_next_batch` translates that to the loop-exit signal.
        """
        gbp = self._generation_batch_prompts
        prompts = self._prompts
        for start in range(0, len(prompts), gbp):
            end = min(start + gbp, len(prompts))
            yield (
                list(prompts[start:end]),
                {k: list(col[start:end]) for k, col in self._extra_columns.items()},
            )

    def _async_gen_next_batch(
        self,
        iterator: Iterator[Tuple[List[List[int]], dict]],
    ) -> Optional[Tuple[List[List[int]], dict]]:
        """Push the current actor weights, then submit the NEXT generation batch.

        Returns ``(prompts_batch, columns_batch)`` so a later
        :py:meth:`await_generate` pairing knows which prompts / columns line up
        with the completions it drains. Returns ``None`` when the iterator is
        exhausted, WITHOUT touching the completer (so end-of-loop does not waste
        a push or a submit whose response nobody would ever await).
        """
        try:
            prompts_batch, columns_batch = next(iterator)
        except StopIteration:
            return None
        self.completer.push_weights()
        self.completer.submit_generate(prompts_batch)
        return prompts_batch, columns_batch

    def train(self) -> None:
        """One-step async GRPO training loop (verl-shape).

        Per iteration:

          1. ``await_generate`` -> rollout submitted last iteration (gen'd with
             ``theta_{t-1}``).
          2. ``_async_gen_next_batch`` -> push ``theta_t`` + submit ``gen_{t+1}``
             (server generates with ``theta_t`` while the trainer works).
          3. Inherited phase helpers train on the rollout to produce
             ``theta_{t+1}``.

        Priming (before the loop): one ``_async_gen_next_batch`` call pushes
        ``theta_0`` and submits ``gen_0`` so the first ``await_generate``
        returns ``R_0``.
        """
        self._setup_training()

        if self.config.num_iterations != 1:
            raise ValueError(
                f"OneStepAsyncGRPOTrainer requires num_iterations == 1 " f"(got {self.config.num_iterations})."
            )
        for name in ("submit_generate", "await_generate", "push_weights"):
            if not hasattr(self.completer, name):
                raise TypeError(
                    f"OneStepAsyncGRPOTrainer requires the completer to expose "
                    f"{name}(); {type(self.completer).__name__} does not."
                )

        for cb in self.callbacks:
            cb.on_train_begin(self)

        iterator = self._make_batch_iterator()
        pending = self._async_gen_next_batch(iterator)

        self._step_t0 = time.perf_counter()
        num_steps = 0
        npp = self.config.num_generations

        while pending is not None:
            prompts_batch, columns_batch = pending
            completions_batch = self.completer.await_generate()
            pending = self._async_gen_next_batch(iterator)

            # Expand prompts + columns to line up 1:1 with completions
            # (mirrors iter_batched_completions in the sync trainer).
            prompts_expanded = [p for p in prompts_batch for _ in range(npp)]
            columns_expanded = {k: [v for v in col for _ in range(npp)] for k, col in columns_batch.items()}

            rewards_np, advantages_np, prompts_strs, completions_strs = self._run_rewards_and_advantages(
                prompts_expanded, completions_batch, columns_expanded
            )
            rewards = rewards_np.tolist()
            completion_lens = [len(c) for c in completions_batch]

            probs_old_list = self._run_ref_logprobs(prompts_expanded, completions_batch)

            warmup_factor = self._run_optimizer_step(
                prompts_expanded,
                completions_batch,
                advantages_np,
                probs_old_list,
                num_steps,
            )
            num_steps += 1

            self._finalize_step_and_fire_callbacks(
                num_steps,
                warmup_factor,
                rewards_np,
                completion_lens,
                prompts_strs,
                completions_strs,
                rewards,
                # No explicit gen-time attribution in the first pass -- the
                # generation happens in parallel with training, and step_time_s
                # already covers total wallclock.
                0.0,
            )

            for nlog_old, mask_old in probs_old_list:
                _deallocate_tensors([nlog_old, mask_old])

        for cb in self.callbacks:
            cb.on_train_end(self)


__all__ = ["OneStepAsyncGRPOTrainer"]

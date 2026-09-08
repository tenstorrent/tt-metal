# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fully-async off-policy GRPO trainer.

Rank 0 pulls :class:`RolloutBatch` objects from a :class:`RolloutQueue`
consumer, uses the batch's own sampled log-probs as the ratio denominator
in the clipped surrogate (proper off-policy PPO), runs one optimizer step,
and fires the updated weights through a
:class:`ThreadedWeightBridge` sender. No prompt iteration on rank 0 --
prompts + gold columns come from the queue.

Staleness handling follows the aReal formulation: for step ``s``
(1-indexed) with a rollout generated at weight version ``vgen_s``, the
batch is dropped when ``(s - 1) - vgen_s > max_staleness``.

Requires ``config.num_iterations == 1`` (fundamental to the 1-consumed-batch
== 1-optimizer-step mapping).

Completer contract: the completer only needs the sync surface
(``model`` / ``tokenizer`` / ``compute_nlog_probs`` / ``_dp_mapper`` /
``_num_devices``) already implemented by e.g. ``Qwen3CompleterRemoteRollout``
(pass ``inference_client=None`` -- the generate / push_weights paths are
never called by this trainer).

Live example wiring in
``tt-train/sources/examples/grpo_remote_rollout/gsm8k_fully_async/``.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import ttml
import ttnn

from ttml.common.utils import no_grad, round_up_to_tile

from .grpo_trainer import GRPOTrainer, _deallocate_tensors, iter_micro_batch


WeightsExportFn = Callable[[], Dict[str, "ttnn.Tensor"]]


class FullyAsyncGRPOTrainer(GRPOTrainer):
    """Fully-async off-policy GRPO trainer (aReal-shape loop).

    Overrides only :meth:`train` plus a few small helpers; every phase
    helper on :class:`GRPOTrainer` (``_setup``, ``_compute_rewards``,
    ``_compute_advantages``, ``_iter_micro_batches``,
    ``_compute_loss_and_backward``, ``_apply_gradients``,
    ``_publish_step_metrics``, ``_maybe_checkpoint``,
    ``_reset_step_metrics``) is inherited unchanged.

    ``RolloutBatch.prompts`` and ``RolloutBatch.extra`` are already
    expanded by ``num_generations`` on the producer side (rank 1), so
    ``_expand_prompts_and_columns`` is NOT called on the trainer side.

    Metrics written per consumed step:

      * ``rollout_wait_s`` -- wall clock spent in :meth:`RolloutQueue.pop`.
      * ``rollout_weight_version`` -- ``vgen_s`` of the consumed batch.
      * ``rollout_staleness`` -- ``(s - 1) - vgen_s``.
      * ``rollout_dropped`` -- cumulative count of stale batches dropped.
    """

    def __init__(
        self,
        completer: Any,
        config: Any,
        *,
        rollout_queue: Any,
        weight_bridge: Any,
        weights_export_fn: WeightsExportFn,
        max_staleness: int,
        dataset: Any = None,
        reward_func: Optional[Callable[..., List[float]]] = None,
        reward_funcs: Optional[List[Callable[..., List[float]]]] = None,
        optimizer_dict: Optional[dict] = None,
        callbacks: Optional[List[Any]] = None,
        model_source: Optional[str] = None,
    ) -> None:
        # Fully-async pulls prompts + gold columns from the queue, so the
        # trainer has no use for a dataset. GRPOTrainer._setup still touches
        # ``self.dataset`` (len + column_names + row iteration) though, so we
        # feed it an empty stub HF Dataset with the columns the reward
        # functions ask for. ``_setup`` treats total_prompts = 0 as
        # "nothing to pre-tokenize" and no batching check fires.
        if dataset is None:
            from datasets import Dataset as _HfDataset

            dataset = _HfDataset.from_dict({"prompt": [], "answer": []})

        super().__init__(
            completer=completer,
            dataset=dataset,
            config=config,
            reward_func=reward_func,
            reward_funcs=reward_funcs,
            optimizer_dict=optimizer_dict,
            callbacks=callbacks,
            model_source=model_source,
        )
        self._rollout_queue: Any = rollout_queue
        self._weight_bridge: Any = weight_bridge
        self._weights_export_fn: WeightsExportFn = weights_export_fn
        self._max_staleness: int = int(max_staleness)

    # -- preflight -----------------------------------------------------------

    def _require_fully_async_surface(self) -> None:
        """Fail fast on completer / config invariants before ``_setup`` opens
        the device."""
        if self.config.num_iterations != 1:
            raise ValueError(f"FullyAsyncGRPOTrainer requires num_iterations == 1 (got {self.config.num_iterations}).")
        for name in ("model", "tokenizer", "compute_nlog_probs"):
            if not hasattr(self.completer, name):
                raise TypeError(
                    f"FullyAsyncGRPOTrainer requires the completer to expose "
                    f"{name}; {type(self.completer).__name__} does not."
                )
        if self._max_staleness < 0:
            raise ValueError(f"max_staleness must be non-negative (got {self._max_staleness})")

    # -- rollout pop with stale-drop ----------------------------------------

    def _pop_fresh_batch(self) -> Tuple[Optional[Any], int]:
        """Blocking pop-with-stale-drop.

        aReal check: keep popping until ``self.metrics["step"] -
        batch.weight_version <= self._max_staleness`` (which is
        ``(s - 1) - vgen_s <= max_staleness``). Batches that fail the
        check are dropped and counted in ``rollout_dropped``.

        Returns ``(batch, staleness)`` on success or ``(None, 0)`` when
        the queue producer has closed with no fresh batch available.
        Sets ``rollout_wait_s`` on ``self.metrics``.
        """
        wait_t0 = time.perf_counter()
        while True:
            batch = self._rollout_queue.pop()
            if batch is None:
                self.metrics["rollout_wait_s"] = time.perf_counter() - wait_t0
                return None, 0
            stale = int(self.metrics["step"]) - int(batch.weight_version)
            if stale > self._max_staleness:
                self.metrics["rollout_dropped"] = int(self.metrics.get("rollout_dropped", 0)) + 1
                # Loop -- the wait_s clock keeps running.
                continue
            self.metrics["rollout_wait_s"] = time.perf_counter() - wait_t0
            return batch, stale

    # -- optimize with batch-supplied old logprobs --------------------------

    def _optimize_from_batch(
        self,
        prompts_x: List[List[int]],
        completions: List[List[int]],
        batch_logprobs: torch.Tensor,
        advantages_np: np.ndarray,
    ) -> None:
        """Mirrors :meth:`GRPOTrainer._optimize` but supplies the ratio
        denominator ``(nlog, mask)`` pairs from the rollout batch's own
        sampled log-probs instead of running a ``theta_t`` forward pass.
        """
        ref_logprobs = self._build_old_ref_logprobs(prompts_x, completions, batch_logprobs)
        try:
            self.model.train()
            self._optimizer.zero_grad()
            global_len = len(prompts_x)
            mb = self._completions_per_microbatch
            for i, (p, c, ref_nlog, ref_mask) in enumerate(
                self._iter_micro_batches(prompts_x, completions, ref_logprobs),
            ):
                adv_slice = advantages_np[i * mb : i * mb + len(c)]
                self._compute_loss_and_backward(p, c, adv_slice, ref_nlog, ref_mask, global_len)
        finally:
            for nlog, mask in ref_logprobs:
                _deallocate_tensors([nlog, mask])

    def _build_old_ref_logprobs(
        self,
        prompts_x: List[List[int]],
        completions: List[List[int]],
        batch_logprobs: torch.Tensor,
    ) -> List[Tuple[Any, Any]]:
        """Build one ``(nlog, mask)`` pair per micro-batch from the batch's
        own sampled log-probs.

        Layout matches :meth:`Qwen3CompleterRemoteRollout.compute_nlog_probs`
        exactly so the mask lines up slot-for-slot with the new-policy
        forward pass in :meth:`_compute_loss_and_backward`:

        * ``sequence = p + c``; ``L = len(sequence) - 1``.
        * ``start = len(p) - 1``; ``end = min(start + len(c), L)``.
        * ``nlog[u, start:end] = -log_pi_old[u, :len(c)]``; zero elsewhere.
        * ``mask[u, start:end] = 1.0``; zero elsewhere.
        * ``Tp = round_up_to_tile(max(len(p) + len(c) - 1))``.
        """
        mb = self._completions_per_microbatch
        out: List[Tuple[Any, Any]] = []
        with no_grad():
            for i, (p_mb, c_mb) in enumerate(
                iter_micro_batch(prompts_x, completions, mb),
            ):
                lp_mb = batch_logprobs[i * mb : i * mb + len(c_mb)]
                nlog, mask = self._build_old_ref_pair_one_mb(p_mb, c_mb, lp_mb)
                nlog.set_requires_grad(False)
                mask.set_requires_grad(False)
                out.append((nlog, mask))
        return out

    def _build_old_ref_pair_one_mb(
        self,
        prompts: List[List[int]],
        completions: List[List[int]],
        old_logprobs_mb: torch.Tensor,
    ) -> Tuple[Any, Any]:
        """Build a single micro-batch ``(nlog, mask)`` pair, matching the
        exact tile-padded left-aligned layout produced by
        ``Qwen3CompleterRemoteRollout.compute_nlog_probs``.
        """
        assert len(prompts) == len(
            completions
        ), f"prompts / completions length mismatch: {len(prompts)} vs {len(completions)}"
        B = len(completions)

        # Same divisibility invariant as compute_nlog_probs.
        total_devices = self._num_devices  # populated by GRPOTrainer._setup.
        assert B % total_devices == 0, f"batch {B} must be divisible by num_devices {total_devices}"

        lengths = [len(p) + len(c) - 1 for p, c in zip(prompts, completions)]
        T = max(lengths) if lengths else 1
        Tp = round_up_to_tile(T)

        nlog_np = np.zeros((B, Tp), dtype=np.float32)
        mask_np = np.zeros((B, Tp), dtype=np.float32)

        old_logprobs_cpu = old_logprobs_mb.detach().to(dtype=torch.float32).cpu().numpy()

        for u, (p, c) in enumerate(zip(prompts, completions)):
            if len(p) < 2 or (len(p) + len(c)) < 2:
                # Same guard as the completer.
                raise ValueError("Prompt/sequence too short for GRPO nlog layout")
            L = len(p) + len(c) - 1
            start = len(p) - 1
            end = min(start + len(c), L)
            if start < end:
                k = end - start
                mask_np[u, start:end] = 1.0
                nlog_np[u, start:end] = -old_logprobs_cpu[u, :k]

        dp_mapper = self._dp_mapper  # populated by GRPOTrainer._setup.
        nlog_tt = ttml.autograd.Tensor.from_numpy(nlog_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, dp_mapper)
        mask_tt = ttml.autograd.Tensor.from_numpy(mask_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, dp_mapper)
        return nlog_tt, mask_tt

    # -- training loop ------------------------------------------------------

    def publish_current_weights(self) -> None:
        """Publish the currently-parametrized policy through the weight
        bridge. Callers use this to seed ``theta_0`` BEFORE ``train()`` (so
        the initial handshake with rank 1 does not deadlock on the trainer
        waiting for INFERENCE_READY while rank 1 waits for theta_0).

        The trainer itself calls this internally after every
        ``optimizer.step()``.
        """
        self._weight_bridge.send_weights(self._weights_export_fn())

    def train(self) -> None:
        """Fully-async off-policy GRPO training loop.

        Per iteration:

          1. Pop the next rollout batch. Drop-and-repeat if the batch is
             stale under the aReal bound.
          2. Compute rewards + advantages using the (already-expanded)
             prompts, completions, and ``extra`` columns from the batch.
          3. Optimize using ``batch.logprobs`` as the ratio denominator.
          4. Apply gradients + publish per-step metrics.
          5. Publish the new ``theta_s`` through the weight bridge.

        The caller is responsible for publishing ``theta_0`` via
        :meth:`publish_current_weights` BEFORE the rank-1 handshake so the
        trainer does not deadlock waiting for ``INFERENCE_READY`` while
        rank 1 waits for the initial weights.
        """
        self._require_fully_async_surface()
        self._setup()
        for cb in self.callbacks:
            cb.on_train_begin(self)
        self.metrics = {"step": 0}
        self._reset_step_metrics()

        while self.metrics["step"] < self._total_optimizer_steps:
            batch, stale = self._pop_fresh_batch()
            if batch is None:
                # Producer closed with no fresh batch left.
                break
            self.metrics["rollout_weight_version"] = int(batch.weight_version)
            self.metrics["rollout_staleness"] = int(stale)
            # Keep the cumulative drop counter visible even when nothing was dropped.
            self.metrics.setdefault("rollout_dropped", int(self.metrics.get("rollout_dropped", 0)))

            # RolloutBatch.prompts / .extra are ALREADY expanded by
            # num_generations on the producer side.
            prompts_x = list(batch.prompts)
            completions_x = list(batch.completions)
            cols_x = dict(batch.extra)

            rewards_np = self._compute_rewards(prompts_x, completions_x, cols_x)
            advantages_np = self._compute_advantages(rewards_np)

            self._optimize_from_batch(prompts_x, completions_x, batch.logprobs, advantages_np)
            self._apply_gradients()
            self.metrics["step"] += 1
            self._publish_step_metrics()
            self._maybe_checkpoint()
            self._reset_step_metrics()

            # Publish theta_s for the next rollout.
            self.publish_current_weights()

        for cb in self.callbacks:
            cb.on_train_end(self)


__all__ = ["FullyAsyncGRPOTrainer"]

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Memory-profiling variant of :class:`ttml.trainers.GRPOTrainer`.

This is a thin subclass of ``ttml.trainers.grpo_trainer.GRPOTrainer`` that
overrides :meth:`train` to insert ``MemoryUsageTracker.snapshot(...)`` calls
at the natural training-loop phase boundaries:

    - ``OPTIMIZER_CREATION``     after ``create_optimizer``
    - ``GENERATION``             after the first ``completer.generate``
    - ``ADVANTAGES``             after ``compute_advantages_ttnn``
    - ``NLOG_OLD_COMPUTED``      after the old-policy nlog-prob pass
    - ``FIRST_FWD_BWD``          after the first ``loss.backward()``
    - ``FIRST_OPTIMIZER_STEP``   after the first ``optimizer.step()``

When ``track_memory=True`` (the only mode this trainer is intended for):

    - ``gradient_accumulation_steps`` is forced to 1 so the first outer batch
      maps to exactly one optimizer step, and
    - after that single step we call ``MemoryUsageTracker.end_capture(...)``
      + ``print_memory_usage()`` + ``clear()`` and return immediately.

The caller is responsible for ``MemoryUsageTracker.begin_capture()`` (and any
``snapshot("MODEL_LOAD")`` before calling :meth:`train`).
"""

from __future__ import annotations

import os
import time
from typing import Any, List

import numpy as np
import ttml
import ttnn
from ttml.common.utils import create_optimizer, no_grad
from ttml.trainers.grpo_trainer import (
    GRPOCompleter,
    GRPOConfig,
    GRPOTrainer as _BaseGRPOTrainer,
    _deallocate_tensors,
    compute_advantages_ttnn,
    dispatch_reward,
    get_grpo_config,
    iter_batched_completions,
    iter_micro_batch,
    save_checkpoint,
)

MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker

__all__ = [
    "GRPOCompleter",
    "GRPOConfig",
    "GRPOTrainer",
    "get_grpo_config",
    "save_checkpoint",
]


class GRPOTrainer(_BaseGRPOTrainer):
    """Memory-profiling GRPO trainer.

    Same constructor as the base trainer plus a ``track_memory`` flag. When
    ``track_memory`` is set, :meth:`train` inserts phase snapshots and exits
    after the first optimizer step.
    """

    def __init__(
        self,
        completer: GRPOCompleter,
        dataset: Any,
        config: GRPOConfig,
        reward_func: Any,
        optimizer_dict: dict,
        callbacks: Any = None,
        model_source: Any = None,
        track_memory: bool = False,
    ) -> None:
        super().__init__(
            completer=completer,
            dataset=dataset,
            config=config,
            reward_func=reward_func,
            optimizer_dict=optimizer_dict,
            callbacks=callbacks,
            model_source=model_source,
        )
        self.track_memory = track_memory

    def train(self) -> None:
        if not self.track_memory:
            return super().train()

        grpo_cfg = self.config
        completer = self.completer
        tt_model = completer.model
        tokenizer = completer.tokenizer
        self.model = tt_model

        first_iter_done = False

        def snap(name: str) -> None:
            if not first_iter_done:
                MemoryUsageTracker.snapshot(name)

        optimizer = create_optimizer(tt_model, self.optimizer_dict)
        snap("OPTIMIZER_CREATION")
        base_lr = optimizer.get_lr()

        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        device = autograd_ctx.get_device()
        num_devices: int = device.get_num_devices()
        ddp_enabled: bool = (
            autograd_ctx.is_parallelism_context_initialized()
            and autograd_ctx.get_parallelism_context().is_ddp_enabled()
        )
        dp_mapper: Any = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0) if ddp_enabled else None
        dp_composer: Any = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0) if ddp_enabled else None

        dataset = self.dataset.select(range(min(grpo_cfg.prompts_to_train, len(self.dataset))))
        prompts = [tokenizer.encode(row["prompt"]) for row in dataset]
        extra_columns = {k: list(dataset[k]) for k in dataset.column_names if k != "prompt"}

        num_steps = 0
        accum_count = 0
        # In profiling mode we want exactly one optimizer step so the report
        # flushes promptly; ignore whatever the YAML asks for.
        grad_accum = 1
        accum_rewards: List[np.ndarray] = []
        accum_completion_lens: List[int] = []
        accum_generation_time_s = 0.0
        step_t0 = time.perf_counter()

        optimizer.zero_grad()

        for cb in self.callbacks:
            cb.on_train_begin(self)

        for prompts_batch, completions_batch, dataset_columns_dict, generation_time_s in iter_batched_completions(
            completer, prompts, extra_columns, grpo_cfg.batch_size, grpo_cfg.num_generations
        ):
            snap("GENERATION")
            accum_generation_time_s += generation_time_s

            completions_strs = [tokenizer.decode(c, skip_special_tokens=True) for c in completions_batch]
            prompts_strs = [tokenizer.decode(p) for p in prompts_batch]
            rewards = dispatch_reward(self.reward_func, completions_strs, prompts_strs, dataset_columns_dict)
            rewards_np = np.array(rewards, dtype=np.float32)

            advantages_tt = compute_advantages_ttnn(
                rewards_np,
                grpo_cfg.num_generations,
                dp_mapper,
                num_devices,
            )
            snap("ADVANTAGES")
            accum_rewards.append(rewards_np)
            accum_completion_lens.extend(len(c) for c in completions_batch)

            probs_old_list = []
            tt_model.eval()
            with no_grad():
                for p, c in iter_micro_batch(prompts_batch, completions_batch, grpo_cfg.micro_batch_size):
                    nlog_old, mask = completer.compute_nlog_probs(p, c)
                    nlog_old.set_requires_grad(False)
                    mask.set_requires_grad(False)
                    probs_old_list.append((nlog_old, mask))
            snap("NLOG_OLD_COMPUTED")

            for _mini_epoch in range(grpo_cfg.num_iterations):
                tt_model.train()

                for i, (p, c) in enumerate(
                    iter_micro_batch(prompts_batch, completions_batch, grpo_cfg.micro_batch_size),
                ):
                    B = len(c)
                    start_local = (i * grpo_cfg.micro_batch_size) // num_devices
                    end_local = start_local + B // num_devices

                    adv_slice_val = ttnn.slice(advantages_tt, [start_local, 0], [end_local, 1])
                    adv_ttml = ttml.autograd.create_tensor(adv_slice_val, requires_grad=False)

                    nlog_old, mask_old = probs_old_list[i]
                    nlog_probs_new, mask_new = completer.compute_nlog_probs(p, c)

                    loss = self._compute_grpo_loss(
                        nlog_old,
                        nlog_probs_new,
                        mask_old,
                        adv_ttml,
                        len(prompts_batch) * grad_accum,
                        grpo_cfg.epsilon,
                    )

                    loss.backward(retain_graph=False)
                    ttml.autograd.AutoContext.get_instance().reset_graph()

                    _deallocate_tensors([nlog_probs_new, mask_new, adv_ttml, loss])

                    if i == 0:
                        snap("FIRST_FWD_BWD")

                accum_count += 1

                if accum_count == grad_accum:
                    warmup_factor = (
                        1.0 if grpo_cfg.warmup_steps == 0 else min(1.0, (num_steps + 1) / grpo_cfg.warmup_steps)
                    )
                    optimizer.set_lr(base_lr * warmup_factor)

                    if ddp_enabled:
                        ttml.core.distributed.synchronize_gradients(tt_model.parameters())

                    for cb in self.callbacks:
                        cb.on_before_optimizer_step(self)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0

                    snap("FIRST_OPTIMIZER_STEP")

                    step_time_s = time.perf_counter() - step_t0
                    num_steps += 1
                    all_rewards = np.concatenate(accum_rewards)
                    mean_reward = float(all_rewards.mean())
                    if accum_completion_lens:
                        mean_completion_len = sum(accum_completion_lens) / len(accum_completion_lens)
                    else:
                        mean_completion_len = 0.0
                    print(
                        f"[memprof] step={num_steps} reward_mean={mean_reward:.4f} "
                        f"mean_completion_len={mean_completion_len:.1f} "
                        f"step_time_s={step_time_s:.2f} gen_time_s={accum_generation_time_s:.2f}"
                    )

                    if grpo_cfg.checkpointing and num_steps % grpo_cfg.checkpoint_interval == 0:
                        ckpt_dir = os.path.join(grpo_cfg.output_dir, "checkpoints", f"grpo_step_{num_steps}")
                        save_checkpoint(
                            tt_model,
                            num_steps,
                            grpo_cfg.output_dir,
                            dp_composer=dp_composer,
                            tokenizer=tokenizer,
                            grpo_config=grpo_cfg,
                            optimizer=optimizer,
                            model_source=self.model_source,
                        )
                        for cb in self.callbacks:
                            cb.on_save(self, num_steps, ckpt_dir)

                    if not first_iter_done:
                        first_iter_done = True
                        MemoryUsageTracker.end_capture("TRAIN_STEP_COMPLETE")
                        print("\n========== Memory Usage Report (training, 1 step) ==========")
                        MemoryUsageTracker.print_memory_usage()
                        print("============================================================")
                        MemoryUsageTracker.clear()

                        for nlog_old, mask_old in probs_old_list:
                            _deallocate_tensors([nlog_old, mask_old])
                        _deallocate_tensors(advantages_tt)

                        for cb in self.callbacks:
                            cb.on_train_end(self)
                        return

            for nlog_old, mask_old in probs_old_list:
                _deallocate_tensors([nlog_old, mask_old])
            _deallocate_tensors(advantages_tt)

        for cb in self.callbacks:
            cb.on_train_end(self)

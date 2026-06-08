# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import inspect
import json
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import os
import numpy as np
import torch
import ttml
import ttnn
from safetensors.numpy import save_file
from ttml.common.utils import create_optimizer, no_grad


class GRPOCompleter(ABC):
    """Abstract base for model-specific completion engines used in GRPO training.

    Subclass this for each model architecture (Llama, Qwen, etc.).
    """

    @abstractmethod
    def generate(self, prompts: List[List[int]]) -> List[List[int]]:
        """Generate completions for a batch of tokenised prompts.

        For N prompts returns N * completions_per_prompt completions.
        """

    @abstractmethod
    def generate_str(self, prompt_strs: List[str]) -> List[str]:
        """Generate completions from string prompts, returning decoded strings.
        For N strs returns N * completions_per_prompt strs.
        """

    @abstractmethod
    def compute_nlog_probs(self, prompts: List[List[int]], completions: List[List[int]]) -> tuple:
        """Compute per-token negative log probabilities for prompt+completion pairs.

        Each prompt[i] and completion[i] are concatenated, and the standard
        next-token-prediction shift is applied (input = seq[:-1],
        target = seq[1:]).  The model runs a forward pass and returns
        cross-entropy at every position.

        Dimension glossary:
            B: Global batch size (number of prompt+completion pairs).
            B_local: Per-device batch size (``B // total_devices``).
                On a single device B_local == B.
            T: ``max(len(prompt[i]) + len(completion[i])) - 1`` across the
                batch — the sequence length after the next-token shift.
            T_padded: ``T`` rounded up to the tile boundary (multiple of 32).

        Args:
            prompts: B lists of token IDs (the original prompts).
            completions: B lists of token IDs (the generated completions).

        Returns:
            nlog_probs: Tensor [B_local, T_padded] — negative log-probability
                of each target token.  Prompt and padding positions contain
                meaningless values; use ``mask`` to ignore them.
            mask: Tensor [B_local, T_padded] — binary mask where 1.0 marks
                completion-token positions and 0.0 marks prompt tokens,
                left-padding, and tile-padding.
        """

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """The tokenizer used by this completion engine."""

    @property
    @abstractmethod
    def model(self) -> Any:
        """The underlying tt model used for forward passes and optimization."""


@dataclass
class GRPOConfig:
    epsilon: float
    batch_size: int
    micro_batch_size: int
    num_iterations: int
    gradient_accumulation_steps: int
    logging_steps: int
    output_dir: str
    checkpointing: bool
    checkpoint_interval: int
    prompts_to_train: int
    temperature: float
    max_completion_length: int
    num_generations: int
    warmup_steps: int


def get_grpo_config(yaml_config: dict, output_dir: str = "") -> GRPOConfig:
    """Build a :class:`GRPOConfig` from a top-level YAML config dict.

    Looks for ``training_config.grpo_config`` in ``yaml_config`` and constructs
    a :class:`GRPOConfig` from it. ``output_dir`` defaults to an empty string so
    callers can fill it in once they have picked a run directory.
    """
    tc = yaml_config.get("training_config", {})
    grpo_section = tc.get("grpo_config")
    if grpo_section is None:
        raise ValueError("training_config must contain a 'grpo_config' section")
    fields = dict(grpo_section)
    fields.setdefault("output_dir", output_dir)
    return GRPOConfig(**fields)


def _deallocate_tensors(tensors: Any) -> None:
    if tensors is None:
        return
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, ttml.autograd.Tensor):
            ttnn.deallocate(t.get_value(), force=True)
        elif isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t, force=True)


def dispatch_reward(
    reward_func: Callable[..., List[float]],
    completions: List[str],
    prompts: List[str],
    batch_columns: dict,
) -> List[float]:
    sig = inspect.signature(reward_func)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    data_pool = {"completions": completions, "prompts": prompts, **batch_columns}

    if has_kwargs:
        return reward_func(**data_pool)

    call_kwargs = {name: data_pool[name] for name in params if name in data_pool}
    return reward_func(**call_kwargs)


def compute_advantages_ttnn(
    rewards_np: np.ndarray,
    group_size: int,
    mapper: Any,
    num_devices: int,
) -> Any:
    """Compute group-relative advantages on device. Returns a ``ttnn.Tensor``.

    Uploads ``rewards_np`` (shape ``[B]`` with ``B = num_groups * group_size``
    and contiguous groups of length ``group_size``) to the device, subtracts
    the per-group mean, and returns a ``ttnn.Tensor`` of global shape
    ``[B, 1]`` (sharded along axis 0 across the mesh by ``mapper``).

    Each device must hold whole groups, which requires
    ``num_groups % num_devices == 0``. Group means are then a purely local
    reduction along the last axis of a ``[num_groups, 1, 1, group_size]``
    tensor, so no cross-device communication is needed.
    """
    B = rewards_np.shape[0]
    assert B % group_size == 0, "rewards length must be divisible by group_size"
    num_groups = B // group_size
    assert num_groups % num_devices == 0, "num_groups must be divisible by num_devices so groups don't straddle devices"

    rewards_grouped = rewards_np.reshape(num_groups, 1, 1, group_size).astype(np.float32)
    rewards_ttml = ttml.autograd.Tensor.from_numpy(rewards_grouped, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    rewards_val = rewards_ttml.get_value()

    group_mean = ttnn.mean(rewards_val, dim=-1, keepdim=True)  # [num_groups, 1, 1, 1]
    advantages_4d = ttnn.subtract(rewards_val, group_mean)  # [num_groups, 1, 1, G]
    ttnn.deallocate(group_mean, force=True)
    ttnn.deallocate(rewards_val, force=True)

    B_local = B // num_devices
    advantages_rm = ttnn.to_layout(advantages_4d, ttnn.Layout.ROW_MAJOR)
    return ttnn.reshape(advantages_rm, [B_local, 1])


def iter_batched_completions(
    completer: GRPOCompleter,
    prompts: Sequence[List[int]],
    batch_columns: dict,
    batch_size: int = 32,
    num_generations: int = 1,
) -> Iterator[Tuple[List[List[int]], List[List[int]], dict, float]]:
    completions_per_prompt = num_generations
    n = len(prompts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        prompt_batch = list(prompts[start:end])

        gen_t0 = time.perf_counter()
        completions_batch = completer.generate(prompt_batch)
        generation_time_s = time.perf_counter() - gen_t0

        prompt_batch_expanded = [item for item in prompt_batch for _ in range(completions_per_prompt)]
        columns_expanded = {
            k: [v for v in col[start:end] for _ in range(completions_per_prompt)] for k, col in batch_columns.items()
        }

        assert len(prompt_batch_expanded) == len(completions_batch)
        yield prompt_batch_expanded, completions_batch, columns_expanded, generation_time_s


def iter_micro_batch(
    prompts: List[List[int]],
    completions: List[List[int]],
    micro_batch_size: int = 16,
) -> Iterator[Tuple[List[List[int]], List[List[int]]]]:
    for start in range(0, len(completions), micro_batch_size):
        end = min(start + micro_batch_size, len(completions))

        yield prompts[start:end], completions[start:end]


def save_checkpoint(
    model: Any,
    step: int,
    output_dir: str,
    dp_composer: Any = None,
    tokenizer: Any = None,
    grpo_config: Optional[GRPOConfig] = None,
    optimizer: Any = None,
    model_source: Optional[str] = None,
) -> None:
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"grpo_step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    tensors = {name: param.to_numpy(ttnn.DataType.FLOAT32, dp_composer) for name, param in model.parameters().items()}
    save_file(tensors, os.path.join(ckpt_dir, "model.safetensors"))

    if model_source:
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(model_source)
            hf_config.save_pretrained(ckpt_dir)
        except Exception as exc:
            logging.warning("Could not save HF config for %s: %s", model_source, exc)

    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    if grpo_config is not None:
        gen_config = {
            "temperature": grpo_config.temperature,
            "max_new_tokens": grpo_config.max_completion_length,
        }
        if tokenizer is not None:
            gen_config["eos_token_id"] = tokenizer.eos_token_id
            gen_config["pad_token_id"] = tokenizer.pad_token_id
        with open(os.path.join(ckpt_dir, "generation_config.json"), "w") as f:
            json.dump(gen_config, f, indent=2)

    trainer_state = {"global_step": step}
    if optimizer is not None:
        trainer_state["learning_rate"] = optimizer.get_lr()
    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
        json.dump(trainer_state, f, indent=2)

    scheduler_state = {
        "base_lr": optimizer.get_lr() if optimizer else None,
        "warmup_steps": grpo_config.warmup_steps if grpo_config else 0,
        "last_step": step,
    }
    torch.save(scheduler_state, os.path.join(ckpt_dir, "scheduler.pt"))

    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    torch.save(rng_state, os.path.join(ckpt_dir, "rng_state.pth"))

    if grpo_config is not None:
        torch.save(asdict(grpo_config), os.path.join(ckpt_dir, "training_args.bin"))

    with open(os.path.join(ckpt_dir, "timestamp.txt"), "w") as f:
        f.write(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC\n"))


class GRPOTrainer:
    def __init__(
        self,
        completer: GRPOCompleter,
        dataset: Any,
        config: GRPOConfig,
        reward_func: Callable[..., List[float]],
        optimizer_dict: dict,
        callbacks: Optional[List[Any]] = None,
        model_source: Optional[str] = None,
    ) -> None:
        self.completer = completer
        self.dataset = dataset
        self.config = config
        self.reward_func = reward_func
        self.optimizer_dict = optimizer_dict
        self.callbacks: List[Any] = callbacks or []
        self.model_source = model_source
        self.model: Any = None

    def _compute_grpo_loss(
        self,
        nlog_probs_old: ttml.autograd.Tensor,
        nlog_probs_new: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        adv_ttml: ttml.autograd.Tensor,
        completions_batch_len: int,
        eps: float,
    ) -> ttml.autograd.Tensor:
        """Compute the clipped GRPO surrogate loss."""
        B_local, Tp = nlog_probs_old.shape()
        ratio = ttml.ops.unary.exp(nlog_probs_old - nlog_probs_new)
        clipped_ratio = ttml.ops.unary.clip(ratio, 1.0 - eps, 1.0 + eps)

        surr1 = ratio * adv_ttml
        surr2 = clipped_ratio * adv_ttml
        surr = ttml.ops.binary.min(surr1, surr2)

        # Per-completion normalised weight: w[i,t] = mask[i,t] / max(sum_t(mask[i,t]), 1)
        mask_val = mask.get_value()
        tokens_per_completion = ttnn.maximum(ttnn.sum(mask_val, dim=1, keepdim=True), 1.0)
        weight_tt = ttml.autograd.create_tensor(ttnn.div(mask_val, tokens_per_completion), requires_grad=False)

        weighted_surr = surr * weight_tt
        weighted_surr_4d = ttml.ops.reshape.reshape(weighted_surr, [1, 1, B_local, Tp])
        return ttml.ops.unary.mean(weighted_surr_4d) * (-float(Tp) / completions_batch_len)

    def train(self) -> None:
        grpo_cfg = self.config
        completer = self.completer
        tt_model = completer.model
        tokenizer = completer.tokenizer
        self.model = tt_model

        optimizer = create_optimizer(tt_model, self.optimizer_dict)
        base_lr = optimizer.get_lr()

        # Device-parallelism state. The trainer currently only handles either
        # single-device or DDP; tensor parallelism is not supported here. We
        # gate the multi-device sharding paths on ``ddp_enabled`` rather than
        # ``num_devices > 1`` so this assumption is explicit at the call sites.
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

        num_batches = 0
        num_steps = 0
        accum_count = 0
        grad_accum = grpo_cfg.gradient_accumulation_steps
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
            num_batches += 1
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

            for mini_epoch in range(grpo_cfg.num_iterations):
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

                    step_time_s = time.perf_counter() - step_t0
                    generation_time_s_for_step = accum_generation_time_s

                    num_steps += 1
                    all_rewards = np.concatenate(accum_rewards)
                    mean_reward = float(all_rewards.mean())
                    if accum_completion_lens:
                        mean_completion_len = sum(accum_completion_lens) / len(accum_completion_lens)
                        min_completion_len = min(accum_completion_lens)
                        max_completion_len = max(accum_completion_lens)
                    else:
                        mean_completion_len = 0.0
                        min_completion_len = 0
                        max_completion_len = 0

                    if grpo_cfg.logging_steps > 0 and num_steps % grpo_cfg.logging_steps == 0:
                        step_metrics = {
                            "reward_mean": mean_reward,
                            "reward_std": float(all_rewards.std()),
                            "mean_completion_len": mean_completion_len,
                            "min_completion_len": min_completion_len,
                            "max_completion_len": max_completion_len,
                            "lr": base_lr * warmup_factor,
                            "step_time_s": step_time_s,
                            "generation_time_s": generation_time_s_for_step,
                        }
                        for cb in self.callbacks:
                            cb.on_step_end(self, num_steps, **step_metrics)

                    accum_rewards.clear()
                    accum_completion_lens.clear()
                    accum_generation_time_s = 0.0
                    step_t0 = time.perf_counter()

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

            for nlog_old, mask_old in probs_old_list:
                _deallocate_tensors([nlog_old, mask_old])
            _deallocate_tensors(advantages_tt)

        for cb in self.callbacks:
            cb.on_train_end(self)

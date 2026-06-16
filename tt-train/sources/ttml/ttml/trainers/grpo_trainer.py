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
    # Number of completions resident on a single device within one micro-batch.
    # The across-mesh micro-batch size is per_device_train_batch_size *
    # num_devices, and the per micro-batch prompt count is derived from it (see
    # GRPOTrainer.train).
    per_device_train_batch_size: int
    num_iterations: int
    # Number of micro-batches per generation (effective) batch and per optimizer
    # step. The generation batch generates gradient_accumulation_steps *
    # per_device_train_batch_size * num_devices completions, then the trainer
    # accumulates gradients over micro-batches of size per_device_train_batch_size * num_devices
    # before each optimizer step. Larger values mean a larger effective batch per step.
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
    # Deprecated/unused: the number of prompts per generation batch is now
    # derived at runtime from per_device_train_batch_size, num_devices, and
    # num_generations. Kept only so older configs that still set it construct
    # without error; the trainer ignores any value provided here.
    batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        # Warn (once per construction) when a deprecated field is explicitly set.
        # TODO: remove this field and warning once all configs have migrated.
        if self.batch_size is not None:
            logging.warning(
                "grpo_config: 'batch_size' is deprecated and ignored; the generation batch "
                "size is now derived from per_device_train_batch_size, num_devices, "
                "num_generations, and gradient_accumulation_steps. Remove it from your config."
            )


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

    # Backwards-compatibility shim for the transition period.
    # ``micro_batch_size`` was renamed to ``per_device_train_batch_size``. Accept
    # the old name so existing configs keep working, mapping its value onto the
    # new field. TODO: deprecated — remove this shim (and the warning) once all
    # configs have migrated to ``per_device_train_batch_size``.
    if "micro_batch_size" in fields:
        old_value = fields.pop("micro_batch_size")
        if "per_device_train_batch_size" in fields and fields["per_device_train_batch_size"] != old_value:
            raise ValueError(
                "grpo_config: both 'micro_batch_size' (deprecated) and 'per_device_train_batch_size' are set with different values; "
                "remove 'micro_batch_size' and keep only 'per_device_train_batch_size'."
            )
        logging.warning(
            "grpo_config: 'micro_batch_size' is deprecated and will be removed; "
            "use 'per_device_train_batch_size' instead."
        )
        fields.setdefault("per_device_train_batch_size", old_value)

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


def compute_advantages_host(rewards_np: np.ndarray, group_size: int) -> np.ndarray:
    """Compute group-relative advantages on the host, kept in host order.

    ``rewards_np`` has shape ``[B]`` with ``B = num_groups * group_size`` and
    contiguous groups of length ``group_size`` (all completions of prompt 0,
    then prompt 1, ...). Returns an array of the same shape and order where
    each element has had its group (per-prompt) mean subtracted.

    Doing the group reduction on the host means the advantages never need to be
    co-located by group on a device, so groups are free to straddle devices.
    The advantages are deliberately returned in host order (NOT regrouped per
    device) so that each micro-batch slice can later be sharded along axis 0 in
    the exact same group-agnostic, host-order way that
    :meth:`GRPOCompleter.compute_nlog_probs` shards its token tensors. That
    alignment is what keeps every completion paired with its own advantage on
    every device; see :func:`upload_micro_advantages`.
    """
    B = rewards_np.shape[0]
    assert B % group_size == 0, "rewards length must be divisible by group_size"
    grouped = rewards_np.reshape(-1, group_size).astype(np.float32)
    advantages = grouped - grouped.mean(axis=1, keepdims=True)
    return advantages.reshape(B)


def upload_micro_advantages(adv_np: np.ndarray, mapper: Any, num_devices: int) -> Any:
    """Upload one micro-batch's advantages, sharded to match ``compute_nlog_probs``.

    ``adv_np`` is the host-order advantage slice for a single micro-batch (shape
    ``[mb]``, where ``mb`` is the micro-batch size). It is sharded along axis 0
    across the mesh, so device ``d`` receives host rows
    ``[d * mb_local : (d + 1) * mb_local]`` — the SAME contiguous,
    group-agnostic split that :meth:`GRPOCompleter.compute_nlog_probs` applies
    to its ``[mb, T]`` token tensors for the very same micro-batch. Because both
    tensors are sharded the same way over the same host-order list, device-local
    row ``r`` of the advantages corresponds to device-local row ``r`` of the
    log-probs, i.e. the same completion.

    Returns a ``ttnn.Tensor`` of global shape ``[mb, 1]`` (per device
    ``[mb_local, 1]``), ready to broadcast-multiply the per-completion loss.
    """
    mb = adv_np.shape[0]
    assert mb % num_devices == 0, f"micro-batch size ({mb}) must be divisible by num_devices ({num_devices})"
    mb_local = mb // num_devices
    adv_4d = adv_np.reshape(mb, 1, 1, 1).astype(np.float32)
    adv_ttml = ttml.autograd.Tensor.from_numpy(adv_4d, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    adv_rm = ttnn.to_layout(adv_ttml.get_value(), ttnn.Layout.ROW_MAJOR)
    return ttnn.reshape(adv_rm, [mb_local, 1])


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
        ddp_world_size: int = 1,
    ) -> ttml.autograd.Tensor:
        """Compute the clipped GRPO surrogate loss.

        ``completions_batch_len`` is the *global* number of completions in the
        optimizer step. Under DDP, ``ttml.core.distributed.synchronize_gradients``
        *averages* (not sums) the per-device gradients, so normalising the loss
        by the global count would leave an extra ``1 / ddp_world_size`` factor
        after that averaging. To keep gradients invariant to the device count we
        normalise by the *per-device* completion count instead
        (``completions_batch_len / ddp_world_size``); the gradient averaging then
        restores the intended global-mean gradient. ``ddp_world_size`` is 1 when
        DDP is disabled, leaving the single-device path unchanged.
        """
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
        per_device_batch_len = completions_batch_len / ddp_world_size
        return ttml.ops.unary.mean(weighted_surr_4d) * (-float(B_local) * float(Tp) / per_device_batch_len)

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

        # Derive the across-mesh micro-batch size (in completions), the per
        # micro-batch prompt count, and the generation (effective) batch size up
        # front, and validate the divisibility relationships so misconfigurations
        # fail with a clear message instead of a cryptic shard assert deep in
        # ``compute_nlog_probs`` (or a silently ragged final micro-batch).
        #
        # ``per_device_train_batch_size`` is the number of completions resident
        # on a single device within one micro-batch, so the whole mesh handles
        # ``completions_per_microbatch = per_device_train_batch_size *
        # num_devices`` completions per micro-batch. The per micro-batch prompt
        # count is ``completions_per_microbatch // num_generations``. By
        # construction ``completions_per_microbatch`` is divisible by
        # ``num_devices``, so each micro-batch always shards evenly along axis 0.
        #
        # The generation (effective) batch spans ``gradient_accumulation_steps``
        # micro-batches: each batch generates ``grad_accum`` times the per
        # micro-batch prompt count, then the trainer runs one forward/backward
        # pass per micro-batch accumulating gradients before a single optimizer
        # step. Increasing ``gradient_accumulation_steps`` therefore generates
        # proportionally more completions per batch and trains over that many
        # micro-batches between optimizer steps.
        if grpo_cfg.per_device_train_batch_size <= 0:
            raise ValueError(
                f"per_device_train_batch_size must be positive, got {grpo_cfg.per_device_train_batch_size}"
            )
        grad_accum = grpo_cfg.gradient_accumulation_steps
        if grad_accum <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {grad_accum}")
        if grpo_cfg.num_generations <= 0:
            raise ValueError(f"num_generations must be positive, got {grpo_cfg.num_generations}")
        completions_per_microbatch = grpo_cfg.per_device_train_batch_size * num_devices
        if completions_per_microbatch % grpo_cfg.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size * num_devices ({grpo_cfg.per_device_train_batch_size} * "
                f"{num_devices} = {completions_per_microbatch}) must be divisible by "
                f"num_generations ({grpo_cfg.num_generations}) so the per micro-batch prompt count is an integer"
            )
        prompts_per_microbatch = completions_per_microbatch // grpo_cfg.num_generations
        generation_batch_prompts = prompts_per_microbatch * grad_accum

        total_prompts = min(grpo_cfg.prompts_to_train, len(self.dataset))
        if total_prompts % generation_batch_prompts != 0:
            raise ValueError(
                f"prompts_to_train ({total_prompts}) must be divisible by the generation batch size "
                f"(prompts_per_microbatch * gradient_accumulation_steps = {prompts_per_microbatch} * "
                f"{grad_accum} = {generation_batch_prompts}) to avoid a ragged final batch that can break "
                "micro-batch sharding"
            )
        dataset = self.dataset.select(range(total_prompts))
        prompts = [tokenizer.encode(row["prompt"]) for row in dataset]
        extra_columns = {k: list(dataset[k]) for k in dataset.column_names if k != "prompt"}

        num_batches = 0
        num_steps = 0
        step_t0 = time.perf_counter()

        for cb in self.callbacks:
            cb.on_train_begin(self)

        # Each iteration yields one generation (effective) batch worth of
        # completions, i.e. ``grad_accum`` micro-batches.
        for prompts_batch, completions_batch, dataset_columns_dict, generation_time_s in iter_batched_completions(
            completer, prompts, extra_columns, generation_batch_prompts, grpo_cfg.num_generations
        ):
            num_batches += 1

            completions_strs = [tokenizer.decode(c, skip_special_tokens=True) for c in completions_batch]
            prompts_strs = [tokenizer.decode(p) for p in prompts_batch]
            rewards = dispatch_reward(self.reward_func, completions_strs, prompts_strs, dataset_columns_dict)
            rewards_np = np.array(rewards, dtype=np.float32)

            advantages_np = compute_advantages_host(rewards_np, grpo_cfg.num_generations)
            completion_lens = [len(c) for c in completions_batch]

            # Reference (old) log-probs for every micro-batch in the generation
            # batch, computed once and reused across mini-epochs.
            probs_old_list = []
            tt_model.eval()
            with no_grad():
                for p, c in iter_micro_batch(prompts_batch, completions_batch, completions_per_microbatch):
                    nlog_old, mask = completer.compute_nlog_probs(p, c)
                    nlog_old.set_requires_grad(False)
                    mask.set_requires_grad(False)
                    probs_old_list.append((nlog_old, mask))

            for mini_epoch in range(grpo_cfg.num_iterations):
                tt_model.train()
                optimizer.zero_grad()

                # Accumulate gradients over all ``grad_accum`` micro-batches of
                # the generation batch before taking a single optimizer step.
                for i, (p, c) in enumerate(
                    iter_micro_batch(prompts_batch, completions_batch, completions_per_microbatch),
                ):
                    B = len(c)
                    # The advantages and the micro-batch token tensors are both
                    # sharded along axis 0 over the identical host-order slice
                    # [start : start + B], so each device pairs a completion's
                    # log-probs with that same completion's advantage.
                    start = i * completions_per_microbatch
                    adv_micro_np = advantages_np[start : start + B]
                    adv_slice_val = upload_micro_advantages(adv_micro_np, dp_mapper, num_devices)
                    adv_ttml = ttml.autograd.create_tensor(adv_slice_val, requires_grad=False)

                    nlog_old, mask_old = probs_old_list[i]
                    nlog_probs_new, mask_new = completer.compute_nlog_probs(p, c)

                    # ``len(prompts_batch)`` is the global completion count of the
                    # whole generation batch (all ``grad_accum`` micro-batches), so
                    # accumulating the per-micro-batch losses yields the mean over
                    # the full effective batch.
                    loss = self._compute_grpo_loss(
                        nlog_old,
                        nlog_probs_new,
                        mask_old,
                        adv_ttml,
                        len(prompts_batch),
                        grpo_cfg.epsilon,
                        ddp_world_size=num_devices if ddp_enabled else 1,
                    )

                    loss.backward(retain_graph=False)
                    ttml.autograd.AutoContext.get_instance().reset_graph()

                    _deallocate_tensors([nlog_probs_new, mask_new, adv_ttml, loss])

                warmup_factor = 1.0 if grpo_cfg.warmup_steps == 0 else min(1.0, (num_steps + 1) / grpo_cfg.warmup_steps)
                optimizer.set_lr(base_lr * warmup_factor)

                if ddp_enabled:
                    ttml.core.distributed.synchronize_gradients(tt_model.parameters())

                for cb in self.callbacks:
                    cb.on_before_optimizer_step(self)
                optimizer.step()
                optimizer.zero_grad()

                step_time_s = time.perf_counter() - step_t0
                # Generation runs once per effective batch; attribute its cost to
                # the first mini-epoch's step only.
                generation_time_s_for_step = generation_time_s if mini_epoch == 0 else 0.0

                num_steps += 1
                mean_reward = float(rewards_np.mean())
                if completion_lens:
                    mean_completion_len = sum(completion_lens) / len(completion_lens)
                    min_completion_len = min(completion_lens)
                    max_completion_len = max(completion_lens)
                else:
                    mean_completion_len = 0.0
                    min_completion_len = 0
                    max_completion_len = 0

                if grpo_cfg.logging_steps > 0 and num_steps % grpo_cfg.logging_steps == 0:
                    step_metrics = {
                        "reward_mean": mean_reward,
                        "reward_std": float(rewards_np.std()),
                        "mean_completion_len": mean_completion_len,
                        "min_completion_len": min_completion_len,
                        "max_completion_len": max_completion_len,
                        "lr": base_lr * warmup_factor,
                        "step_time_s": step_time_s,
                        "generation_time_s": generation_time_s_for_step,
                    }
                    for cb in self.callbacks:
                        cb.on_step_end(self, num_steps, **step_metrics)

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

        for cb in self.callbacks:
            cb.on_train_end(self)

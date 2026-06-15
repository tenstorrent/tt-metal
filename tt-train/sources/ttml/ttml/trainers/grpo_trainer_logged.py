# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO trainer with topology logging.

A copy of ``grpo_trainer.py`` that snapshots every parameter's
``tensor_topology().placements()`` (and the gradient's placements when
initialized) at the most interesting points in the training step:

  - on entry to ``train()`` (initial state)
  - after every ``loss.backward()`` (first time grads get a label)
  - before / after ``synchronize_gradients`` (the all-reduce relabel point)
  - before / after ``optimizer.step()``     (parameter writeback)
  - after ``optimizer.zero_grad()``         (sanity check)

Only changes between snapshots are emitted by default, plus a small focus
list of parameters that are dumped unconditionally (``lm_head``,
``embed_tokens``, ``fc``). Set ``TTML_LOG_TOPOLOGY_ALL=1`` to dump every
parameter on every snapshot.

Usage:

    # in your training entry point (e.g. boolq_training_example.py), replace
    #   from ttml.trainers import GRPOTrainer, get_grpo_config
    # with
    from ttml.trainers.grpo_trainer_logged import GRPOTrainer
    from ttml.trainers import get_grpo_config

To also see C++-side transitions (in ``Tensor::set_value`` /
``Tensor::set_grad`` / ``synchronize_tensor``), set ``TTML_LOG_TOPOLOGY=1``
before launching the run.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import ttml
import ttnn

from ttml.common.utils import create_optimizer, no_grad

from ttml.trainers.grpo_trainer import (
    GRPOCompleter,
    GRPOConfig,
    _deallocate_tensors,
    compute_advantages_ttnn,
    dispatch_reward,
    get_grpo_config,
    iter_batched_completions,
    iter_micro_batch,
    save_checkpoint,
)

__all__ = [
    "GRPOCompleter",
    "GRPOConfig",
    "GRPOTrainer",
    "get_grpo_config",
]


_LOG = logging.getLogger("grpo_trainer_logged")
if not _LOG.handlers:
    _LOG.setLevel(logging.INFO)


_LOG_ALL = os.environ.get("TTML_LOG_TOPOLOGY_ALL", "").lower() in ("1", "true", "yes")
_FOCUS_PARAM_KEYWORDS: Tuple[str, ...] = ("lm_head", "embed_tokens", "fc.weight", "fc/weight")


# Canonical, hashable, comparable representation of one placement.
# The raw ttnn.PlacementReplicate / ttnn.PlacementShard objects are nanobind
# bindings whose __eq__ only accepts the same exact type, so comparing a
# Replicate to a Shard raises TypeError, and even comparing two Shard(0)
# instances can return False. We normalize at the boundary in
# _safe_placements so all downstream comparisons use pure Python primitives.
_PlacementKey = Tuple[Any, ...]  # ("replicate",) or ("shard", dim)


def _placement_key(p: Any) -> _PlacementKey:
    if isinstance(p, ttnn.PlacementShard):
        try:
            return ("shard", int(p.dim))
        except Exception:
            return ("shard", "?")
    if isinstance(p, ttnn.PlacementReplicate):
        return ("replicate",)
    return ("unknown", repr(p))


def _placement_repr(p: _PlacementKey) -> str:
    if p[0] == "shard":
        return f"Shard({p[1]})"
    if p[0] == "replicate":
        return "Replicate"
    return f"<{p[0]}:{p[1] if len(p) > 1 else ''}>"


def _placements_repr(placements: Optional[Tuple[_PlacementKey, ...]]) -> str:
    if placements is None:
        return "<none>"
    return "[" + ", ".join(_placement_repr(p) for p in placements) + "]"


def _safe_placements(ttnn_tensor: Any) -> Optional[Tuple[_PlacementKey, ...]]:
    if ttnn_tensor is None:
        return None
    try:
        raw = ttnn_tensor.tensor_topology().placements()
    except Exception:
        return None
    try:
        return tuple(_placement_key(p) for p in raw)
    except Exception:
        return None


def _as_autograd_tensor(param: Any) -> Any:
    return param.tensor if hasattr(param, "tensor") else param


def _value_placements(param: Any) -> Optional[Tuple[_PlacementKey, ...]]:
    try:
        return _safe_placements(_as_autograd_tensor(param).get_value())
    except Exception:
        return None


def _grad_placements(param: Any) -> Optional[Tuple[_PlacementKey, ...]]:
    tensor = _as_autograd_tensor(param)
    try:
        if not tensor.is_grad_initialized():
            return None
        return _safe_placements(tensor.get_grad())
    except Exception:
        return None


def _is_focus(name: str) -> bool:
    return any(k in name for k in _FOCUS_PARAM_KEYWORDS)


def _has_shard(placements: Optional[Tuple[_PlacementKey, ...]]) -> bool:
    if placements is None:
        return False
    return any(p and p[0] == "shard" for p in placements)


class _TopologyTracker:
    """Compare placements of every parameter (value + grad) across snapshots.

    Emits one line per parameter that changed since the previous snapshot, plus
    one line per parameter whose name matches ``_FOCUS_PARAM_KEYWORDS``. If
    ``TTML_LOG_TOPOLOGY_ALL=1`` or ``force_all=True``, also dumps every
    parameter's current topology.
    """

    def __init__(self) -> None:
        self._prev: dict[
            str,
            Tuple[Optional[Tuple[_PlacementKey, ...]], Optional[Tuple[_PlacementKey, ...]]],
        ] = {}

    def snapshot(self, label: str, model: Any, *, force_all: bool = False) -> None:
        try:
            self._snapshot_impl(label, model, force_all=force_all)
        except Exception as exc:
            # A topology logger must never crash training. Log and keep going,
            # and reset the previous-snapshot cache so the next call starts
            # from a clean baseline rather than diffing against a half-built
            # dict.
            _LOG.exception("[topology %-60s] snapshot failed: %s", label, exc)
            self._prev = {}

    def _snapshot_impl(self, label: str, model: Any, *, force_all: bool = False) -> None:
        curr: dict[
            str,
            Tuple[Optional[Tuple[_PlacementKey, ...]], Optional[Tuple[_PlacementKey, ...]]],
        ] = {}
        changes: list[
            Tuple[
                str,
                Tuple[Optional[Tuple[_PlacementKey, ...]], Optional[Tuple[_PlacementKey, ...]]],
                Tuple[Optional[Tuple[_PlacementKey, ...]], Optional[Tuple[_PlacementKey, ...]]],
            ]
        ] = []
        focus: list[Tuple[str, Optional[Tuple[_PlacementKey, ...]], Optional[Tuple[_PlacementKey, ...]]]] = []

        # Deterministic order so diffs across snapshots are easy to grep.
        for name, param in sorted(model.parameters().items(), key=lambda kv: kv[0]):
            v = _value_placements(param)
            g = _grad_placements(param)
            curr[name] = (v, g)

            prev = self._prev.get(name, (None, None))
            value_changed = prev[0] != v
            grad_changed = prev[1] != g
            if value_changed or grad_changed:
                changes.append((name, prev, (v, g)))
            elif _is_focus(name):
                focus.append((name, v, g))

        if not changes and not focus and not (_LOG_ALL or force_all):
            _LOG.info("[topology %-60s] no transitions, no focus matches (%d params)", label, len(curr))
            self._prev = curr
            return

        _LOG.info(
            "[topology %-60s] changes=%d focus=%d total=%d",
            label,
            len(changes),
            len(focus),
            len(curr),
        )

        for name, prev, now in changes:
            v_mark = " VALUE-SHARDED" if _has_shard(now[0]) else ""
            g_mark = " GRAD-SHARDED" if _has_shard(now[1]) else ""
            _LOG.info(
                "  CHANGE %s%s%s",
                name,
                v_mark,
                g_mark,
            )
            _LOG.info(
                "         value: %s -> %s",
                _placements_repr(prev[0]),
                _placements_repr(now[0]),
            )
            _LOG.info(
                "         grad : %s -> %s",
                _placements_repr(prev[1]),
                _placements_repr(now[1]),
            )

        for name, v, g in focus:
            v_mark = " VALUE-SHARDED" if _has_shard(v) else ""
            g_mark = " GRAD-SHARDED" if _has_shard(g) else ""
            _LOG.info(
                "  FOCUS  %s value=%s grad=%s%s%s",
                name,
                _placements_repr(v),
                _placements_repr(g),
                v_mark,
                g_mark,
            )

        if _LOG_ALL or force_all:
            for name, (v, g) in curr.items():
                _LOG.info(
                    "  ALL    %s value=%s grad=%s",
                    name,
                    _placements_repr(v),
                    _placements_repr(g),
                )

        self._prev = curr


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
        self._tracker = _TopologyTracker()

    def _compute_grpo_loss(
        self,
        nlog_probs_old: ttml.autograd.Tensor,
        nlog_probs_new: ttml.autograd.Tensor,
        mask: ttml.autograd.Tensor,
        adv_ttml: ttml.autograd.Tensor,
        completions_batch_len: int,
        eps: float,
    ) -> ttml.autograd.Tensor:
        B_local, Tp = nlog_probs_old.shape()
        ratio = ttml.ops.unary.exp(nlog_probs_old - nlog_probs_new)
        clipped_ratio = ttml.ops.unary.clip(ratio, 1.0 - eps, 1.0 + eps)

        surr1 = ratio * adv_ttml
        surr2 = clipped_ratio * adv_ttml
        surr = ttml.ops.binary.min(surr1, surr2)

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

        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        device = autograd_ctx.get_device()
        num_devices: int = device.get_num_devices()
        ddp_enabled: bool = (
            autograd_ctx.is_parallelism_context_initialized()
            and autograd_ctx.get_parallelism_context().is_ddp_enabled()
        )
        dp_mapper: Any = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0) if ddp_enabled else None
        dp_composer: Any = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0) if ddp_enabled else None

        _LOG.info(
            "GRPOTrainer (logged) starting: ddp_enabled=%s num_devices=%d",
            ddp_enabled,
            num_devices,
        )
        self._tracker.snapshot("00 enter train()", tt_model, force_all=True)

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
        self._tracker.snapshot("01 after first zero_grad()", tt_model)

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

            self._tracker.snapshot(
                f"02 batch={num_batches} after eval pass (no_grad)",
                tt_model,
            )

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
                    self._tracker.snapshot(
                        f"03 batch={num_batches} mini_epoch={mini_epoch} micro={i} after backward()",
                        tt_model,
                    )
                    ttml.autograd.AutoContext.get_instance().reset_graph()

                    _deallocate_tensors([nlog_probs_new, mask_new, adv_ttml, loss])

                accum_count += 1

                if accum_count == grad_accum:
                    warmup_factor = (
                        1.0 if grpo_cfg.warmup_steps == 0 else min(1.0, (num_steps + 1) / grpo_cfg.warmup_steps)
                    )
                    optimizer.set_lr(base_lr * warmup_factor)

                    self._tracker.snapshot(
                        f"04 batch={num_batches} BEFORE synchronize_gradients",
                        tt_model,
                    )
                    if ddp_enabled:
                        ttml.core.distributed.synchronize_gradients(tt_model.parameters())
                    self._tracker.snapshot(
                        f"05 batch={num_batches} AFTER  synchronize_gradients",
                        tt_model,
                    )

                    for cb in self.callbacks:
                        cb.on_before_optimizer_step(self)

                    self._tracker.snapshot(
                        f"06 batch={num_batches} BEFORE optimizer.step()",
                        tt_model,
                    )
                    optimizer.step()
                    self._tracker.snapshot(
                        f"07 batch={num_batches} AFTER  optimizer.step()",
                        tt_model,
                    )

                    optimizer.zero_grad()
                    self._tracker.snapshot(
                        f"08 batch={num_batches} AFTER  optimizer.zero_grad()",
                        tt_model,
                    )
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

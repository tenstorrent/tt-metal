# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Supervised Fine-Tuning (SFT) trainer."""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ttml.modules.lora import LoraConfig

import numpy as np
import ttnn
from tqdm import tqdm

logger = logging.getLogger(__name__)

import ttml
from ttml.common.utils import no_grad
from ttml.datasets import Batch, TTMLDataloader


class TrainerCallback:
    """Base class for SFTTrainer callbacks.

    Override any subset of hooks to customise training behaviour.
    All methods are no-ops by default.
    """

    def on_train_begin(self, trainer: "SFTTrainer") -> None:
        pass

    def on_step_end(self, trainer: "SFTTrainer", step: int, loss: float, lr: float) -> None:
        pass

    def on_eval_end(self, trainer: "SFTTrainer", step: int, eval_loss: float) -> None:
        pass

    def on_before_optimizer_step(self, trainer: "SFTTrainer") -> None:
        pass

    def on_save(self, trainer: "SFTTrainer", step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: "SFTTrainer") -> None:
        pass


@dataclass
class SFTConfig:
    """Configuration for :class:`SFTTrainer`.

    Training-loop cadence and default learning-rate settings live here.
    Optimizer-specific parameters should be passed via the *optimizer*
    argument of :class:`SFTTrainer` instead.

    Attributes:
        max_steps: Total number of optimiser steps.
        gradient_accumulation_steps: Micro-batches accumulated before each
            optimiser step.
        eval_interval: Run evaluation every *N* steps (0 to disable).
        save_interval: Save a checkpoint every *N* steps (0 to disable).
        checkpoint_dir: Directory for checkpoint files.
        seed: Optional RNG seed for reproducibility.
        max_seq_len: Maximum sequence length (currently unused; kept for
            backward compatibility).
        learning_rate: Peak learning rate, also used to initialise the default
            AdamW optimizer when no explicit optimizer is provided.
        warmup_steps: Linear warmup steps for the default schedule.
        max_grad_norm: Maximum gradient norm for clipping (0 to disable).
        log_interval: Update the progress bar / fire ``on_step_end``
            callbacks every *N* steps.
        gradient_checkpointing: Enable activation recomputation to reduce
            memory usage at the cost of ~30 % extra compute.  Sets the
            model's ``runner_type`` to ``MemoryEfficient``.
    """

    # ---- loop ----
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1
    eval_interval: int = 200
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    seed: Optional[int] = None
    max_seq_len: int = 1024
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    max_grad_norm: float = 0.0
    log_interval: int = 1
    gradient_checkpointing: bool = False


class SFTTrainer:
    """Supervised fine-tuning trainer for Tenstorrent devices.

    The trainer owns the training loop, optimizer, and scheduler.  It knows
    nothing about model internals or dataset structure — only the interfaces
    defined by :class:`~ttml.modules.module_base.AbstractModuleBase` and
    :class:`~ttml.datasets.TTMLDataloader`.

    Multi-device training (e.g. DDP) is supported by combining two
    extension points:

    * A **collate function** that shards batch tensors across the mesh
      (via ``shard_tensor_to_mesh_mapper``).
    * The :meth:`~TrainerCallback.on_before_optimizer_step` callback to
      synchronise gradients before the optimiser step.

    Loss aggregation across devices is handled automatically via a default
    ``concat_mesh_to_tensor_composer(device, 0)``.  Pass a custom
    ``loss_composer`` to override.

    Example::

        from ttml.trainers import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            config=SFTConfig(max_steps=5_000, learning_rate=2e-5, max_grad_norm=1.0),
            optimizer={"type": "AdamW", "lr": 2e-5, "weight_decay": 0.01},
        )
        trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataloader: TTMLDataloader,
        eval_dataloader: Optional[TTMLDataloader],
        config: SFTConfig,
        optimizer: Any = None,
        peft_config: Optional[LoraConfig] = None,
        lr_schedule: Optional[Callable[[int], float]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        compute_loss_func: Optional[Callable] = None,
        loss_composer: Any = None,
        attention_mask: Any = None,
    ) -> None:
        """
        Args:
            model: Model to fine-tune.
            train_dataloader: Training data loader.
            eval_dataloader: Evaluation data loader (may be ``None``).
            config: Trainer configuration.
            optimizer: One of:
                - A ``ttml.optimizers.OptimizerBase`` instance (used directly).
                - A ``dict`` with at least a ``"type"`` key, forwarded to
                  ``ttml.optimizers.create_optimizer(dict, model.parameters())``.
                - ``None`` (default) -- creates a default AdamW with
                  ``learning_rate=config.learning_rate``.
            peft_config: Optional PEFT configuration (e.g. :class:`LoraConfig`).
                When provided the model is automatically wrapped with the
                appropriate adapter (currently :class:`LoraModel`).
            lr_schedule: Optional callable ``step -> lr``.
            callbacks: Optional list of :class:`TrainerCallback` instances.
            compute_loss_func: Optional callable ``(model_output, batch) -> loss``
                that replaces the default masked cross-entropy.  Must return a
                scalar :class:`ttml.autograd.Tensor` loss already reduced over
                batch and sequence dimensions.  When this hook is provided the
                caller is fully responsible for applying ``batch.loss_mask``
                (or any other prompt/pad masking strategy); the default masking
                logic is bypassed entirely.
            loss_composer: Optional mesh composer passed to
                ``loss.to_numpy(composer=...)`` when extracting scalar loss
                values for logging. By default a `concat_mesh_to_tensor_composer` with dim=0 is used.
                (Covers both single-device and multi-device DDP configurations.)
            attention_mask: Optional attention mask passed as the second
                argument to ``model(input_ids, mask)``.  ``None`` (default)
                lets the model generate a causal mask on the fly.

        Note:
            When ``config.gradient_checkpointing`` is ``True`` the model's
            ``runner_type`` is switched to ``MemoryEfficient``, which
            recomputes activations during backward instead of caching them.
        """
        if config.seed is not None:
            ttml.autograd.AutoContext.get_instance().set_seed(config.seed)

        if peft_config is not None:
            from ttml.modules.lora import LoraModel

            model = LoraModel(model, peft_config)

        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing(model)

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.step = 0  # 0-based; incremented after each optimizer step

        self._optimizer = self._build_optimizer(optimizer)
        self._lr_schedule = lr_schedule if lr_schedule is not None else self._build_lr_schedule()
        self._attention_mask = attention_mask
        self._loss_fn = ttml.ops.loss.cross_entropy_loss
        self._callbacks = callbacks or []
        self._compute_loss_override = compute_loss_func
        self._loss_composer = self._build_loss_composer(loss_composer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop.

        The dataloader is cycled automatically: when one epoch ends the
        iterator is reset so that ``max_steps`` is always reached regardless
        of dataset size.
        """
        self.model.train()
        for cb in self._callbacks:
            cb.on_train_begin(self)
        data_iter = iter(self.train_dataloader)
        cfg = self.config

        def _next_batch() -> Batch:
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                return next(data_iter)

        bar = tqdm(range(cfg.max_steps), desc="SFTTrainer")
        for _ in bar:
            # self.step is 0-based so external lr_schedule callables (e.g.
            # SpeedrunScheduler.lr_at) receive the expected step index.
            lr = self._lr_schedule(self.step)
            self._optimizer.set_lr(lr)
            self._optimizer.zero_grad()

            micro_losses = []
            for _ in range(cfg.gradient_accumulation_steps):
                batch = _next_batch()
                loss = self._compute_loss(batch)
                micro_losses.append(float(loss.to_numpy(ttnn.DataType.FLOAT32, composer=self._loss_composer).mean()))

                if cfg.gradient_accumulation_steps > 1:
                    loss = ttml.ops.binary.mul(loss, 1.0 / cfg.gradient_accumulation_steps)
                loss.backward(False)
                ttml.autograd.AutoContext.get_instance().reset_graph()

            for cb in self._callbacks:
                cb.on_before_optimizer_step(self)

            if cfg.max_grad_norm > 0:
                ttml.core.clip_grad_norm(self.model.parameters(), cfg.max_grad_norm, 2.0, False)

            self._optimizer.step()
            self.step += 1

            step_loss = float(np.mean(micro_losses))
            if cfg.log_interval > 0 and self.step % cfg.log_interval == 0:
                bar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{lr:.2e}"}, refresh=False)
                for cb in self._callbacks:
                    cb.on_step_end(self, self.step, step_loss, lr)

            if cfg.eval_interval > 0 and self.step % cfg.eval_interval == 0:
                if self.eval_dataloader is not None:
                    val_loss = self._eval()
                    bar.set_postfix(
                        {
                            "loss": f"{step_loss:.4f}",
                            "val_loss": f"{val_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        },
                        refresh=False,
                    )
                    for cb in self._callbacks:
                        cb.on_eval_end(self, self.step, val_loss)

            if cfg.save_interval > 0 and self.step % cfg.save_interval == 0 and self.step > 0:
                self._save_checkpoint()
                for cb in self._callbacks:
                    cb.on_save(
                        self,
                        self.step,
                        os.path.join(cfg.checkpoint_dir, f"step_{self.step}.pkl"),
                    )

        for cb in self._callbacks:
            cb.on_train_end(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: Batch):
        """Forward pass + loss computation.

        If *compute_loss_func* was provided it is called instead of the
        default masked cross-entropy.
        """
        logits = self.model(batch.input_ids, self._attention_mask)  # [B, 1, T, V]
        if self._compute_loss_override is not None:
            return self._compute_loss_override(logits, batch)

        # Validate loss_mask normalisation: for a standard SFT collate the mask
        # should sum to B*T (one entry per token position).  A large deviation
        # usually indicates a custom collate that forgot to normalise.
        if batch.loss_mask is not None:
            mask_np = batch.loss_mask.to_numpy(ttnn.DataType.FLOAT32, composer=self._loss_composer)
            B, _, T, _ = mask_np.shape
            expected = float(B * T)
            actual = float(mask_np.sum())
            # use relative tolerance to avoid BF16 precision issues
            if abs(actual - expected) / expected > 0.01:
                logger.warning(
                    "loss_mask sum (%.2f) differs from expected B*T (%d). "
                    "If you are using a custom collate function, make sure "
                    "the mask is correctly normalised.",
                    actual,
                    int(expected),
                )

        loss = self._loss_fn(logits, batch.labels, ttml.ops.ReduceType.NONE)  # [B, 1, T, 1]
        loss = loss * batch.loss_mask  # zero out prompt + padding
        return ttml.ops.unary.mean(loss)

    def _eval(self) -> float:
        """Run one pass over the eval dataloader and return mean loss."""
        self.model.eval()
        losses = []
        with no_grad():
            for batch in self.eval_dataloader:
                loss = self._compute_loss(batch)
                losses.append(float(loss.to_numpy(ttnn.DataType.FLOAT32, composer=self._loss_composer).mean()))
        self.model.train()
        return float(np.mean(losses))

    def _save_checkpoint(self) -> None:
        """Persist model parameters as a pickle checkpoint.

        .. note::
            When a ``peft_config`` is used and the model is wrapped in
            :class:`LoraModel`, this currently saves *all* parameters (base +
            LoRA).  Ideally only the LoRA adapter weights should be persisted
            to keep checkpoints small and avoid redundant copies of the frozen
            base weights.

        TODO: when ``peft_config`` is set, filter ``self.model.parameters()``
        to save only LoRA adapter parameters (e.g. those whose name contains
        ``lora_``).
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f"step_{self.step}.pkl")

        state = {}
        for name, param in self.model.parameters().items():
            tensor = param.tensor if hasattr(param, "tensor") else param
            state[name] = tensor.to_numpy(ttnn.DataType.FLOAT32)

        with open(path, "wb") as f:
            pickle.dump({"step": self.step, "model_state": state}, f)

    def _build_optimizer(self, optimizer: Any):
        """Resolve the *optimizer* argument into an ``OptimizerBase``.

        * ``OptimizerBase`` instance -- returned as-is.
        * ``dict`` -- forwarded to ``ttml.optimizers.create_optimizer``.
        * ``None`` -- a default AdamW is created with ``learning_rate=config.learning_rate``.
        """
        if isinstance(optimizer, ttml.optimizers.OptimizerBase):
            return optimizer

        if isinstance(optimizer, dict):
            return ttml.optimizers.create_optimizer(optimizer, self.model.parameters())

        # Default: AdamW with lr from config
        return ttml.optimizers.create_optimizer(
            {"type": "AdamW", "lr": self.config.learning_rate},
            self.model.parameters(),
        )

    def _build_lr_schedule(self):
        """Return a callable ``step -> lr`` implementing linear warmup then constant."""
        peak_lr = self.config.learning_rate
        warmup = max(0, self.config.warmup_steps)

        def schedule(step: int) -> float:
            if warmup > 0 and step <= warmup:
                return peak_lr * step / warmup
            return peak_lr

        return schedule

    def _build_loss_composer(self, loss_composer: Optional[Any] = None):
        """Create a mesh composer for aggregating loss across devices."""
        if loss_composer is not None:
            return loss_composer
        device = ttml.autograd.AutoContext.get_instance().get_device()
        return ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    @staticmethod
    def _enable_gradient_checkpointing(model: Any) -> None:
        """Switch the model's runner type to ``MemoryEfficient``.

        Works with both bare models (Llama, NanoGPT) and LoraModel wrappers.
        The model config is a frozen dataclass, so we replace it wholesale.
        """
        target = model.model if hasattr(model, "model") else model
        cfg = getattr(target, "config", None)
        if cfg is None or not hasattr(cfg, "runner_type"):
            return
        target.config = replace(cfg, runner_type=ttml.models.RunnerType.MemoryEfficient)

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Single-device Supervised Fine-Tuning (SFT) trainer."""

import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import ttnn
from tqdm import tqdm

import ttml
from ttml.common.data import build_causal_mask
from ttml.common.utils import no_grad
from ttml.datasets import Batch, TTMLDataloader


@dataclass
class SFTConfig:
    """Configuration for :class:`SFTTrainer`.

    Optimizer params, scheduler params, and loop cadence are all in one place.
    The design intentionally mirrors TRL's ``SFTConfig`` so that users who know
    TRL face minimal friction.
    """

    # ---- loop ----
    max_steps: int = 1000
    gradient_accumulation_steps: int = 1
    eval_interval: int = 200
    save_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    seed: Optional[int] = None

    # ---- sequence ----
    max_seq_len: int = 1024

    # ---- optimizer (AdamW) ----
    lr: float = 2e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # ---- scheduler (linear warmup → constant) ----
    warmup_steps: int = 0


class SFTTrainer:
    """Supervised fine-tuning trainer for a single Tenstorrent device.

    The trainer owns the training loop, optimizer, and scheduler.  It knows
    nothing about model internals or dataset structure — only the interfaces
    defined by :class:`~ttml.modules.module_base.AbstractModuleBase` and
    :class:`~ttml.datasets.TTMLDataloader`.

    The training script becomes a pure wiring exercise::

        from ttml.trainers import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            config=SFTConfig(max_steps=5_000, lr=2e-5),
        )
        trainer.train()
    """

    def __init__(
        self,
        model: Any,
        train_dataloader: TTMLDataloader,
        eval_dataloader: Optional[TTMLDataloader],
        config: SFTConfig,
        lr_schedule: Optional[Callable[[int], float]] = None,
    ) -> None:
        if config.seed is not None:
            ttml.autograd.AutoContext.get_instance().set_seed(config.seed)

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.step = 0  # 0-based; incremented after each optimizer step

        self._optimizer = self._build_optimizer()
        self._lr_schedule = (
            lr_schedule if lr_schedule is not None else self._build_lr_schedule()
        )
        self._causal_mask = self._build_causal_mask()
        self._loss_fn = ttml.ops.loss.cross_entropy_loss

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
                micro_losses.append(float(loss.to_numpy().mean()))

                scaled = ttml.ops.binary.mul(
                    loss, 1.0 / cfg.gradient_accumulation_steps
                )
                scaled.backward(False)
                ttml.autograd.AutoContext.get_instance().reset_graph()

            self._optimizer.step()
            self.step += 1

            step_loss = float(np.mean(micro_losses))
            bar.set_postfix(
                {"loss": f"{step_loss:.4f}", "lr": f"{lr:.2e}"}, refresh=False
            )

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

            if cfg.save_interval > 0 and self.step % cfg.save_interval == 0:
                self._save_checkpoint()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_loss(self, batch: Batch):
        """Forward pass + masked cross-entropy loss.

        ``batch.loss_mask`` carries the per-token weights: 0.0 for prompt and
        padding positions, nonzero for completion positions.  Taking the mean
        of the masked loss gives the per-completion-token loss.
        """
        logits = self.model(batch.input_ids, self._causal_mask)  # [B, 1, T, V]
        loss = self._loss_fn(
            logits, batch.labels, ttml.ops.ReduceType.NONE
        )  # [B, 1, T, 1]
        loss = loss * batch.loss_mask  # zero out prompt + padding
        return ttml.ops.unary.mean(loss)

    def _eval(self) -> float:
        """Run one pass over the eval dataloader and return mean loss."""
        self.model.eval()
        losses = []
        with no_grad():
            for batch in self.eval_dataloader:
                loss = self._compute_loss(batch)
                losses.append(float(loss.to_numpy().mean()))
        self.model.train()
        return float(np.mean(losses))

    def _save_checkpoint(self) -> None:
        """Persist model parameters as a pickle checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f"step_{self.step}.pkl")

        state = {}
        for name, param in self.model.parameters().items():
            tensor = param.tensor if hasattr(param, "tensor") else param
            state[name] = tensor.to_numpy(ttnn.DataType.FLOAT32)

        with open(path, "wb") as f:
            pickle.dump({"step": self.step, "model_state": state}, f)

    def _build_optimizer(self):
        cfg = self.config
        adamw_cfg = ttml.optimizers.AdamWConfig.make(
            cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay
        )
        return ttml.optimizers.AdamW(self.model.parameters(), adamw_cfg)

    def _build_lr_schedule(self):
        """Return a callable ``step -> lr`` implementing linear warmup then constant."""
        peak_lr = self.config.lr
        warmup = max(0, self.config.warmup_steps)

        def schedule(step: int) -> float:
            if warmup > 0 and step <= warmup:
                return peak_lr * step / warmup
            return peak_lr

        return schedule

    def _build_causal_mask(self):
        mask_np = build_causal_mask(self.config.max_seq_len)  # [1, 1, T, T]
        return ttml.autograd.Tensor.from_numpy(
            mask_np, ttnn.Layout.ROW_MAJOR, ttml.autograd.DataType.BFLOAT16
        )

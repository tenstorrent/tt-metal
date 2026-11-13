# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

from .callback import Callback


class CheckpointCallback(Callback):
    """Saves checkpoints as ``config.checkpoint`` says.

    Checkpoints hold what the model trains (``Trainer.save_checkpoint``): the LoRA adapter, or all weights.
    """

    def __init__(self) -> None:
        self.manager = None
        self._last_metrics: dict[str, float] = {}

    def on_train_start(self, trainer: Any) -> None:
        # A callback outlives the run; never carry the previous run's metrics into this one's checkpoints.
        self._last_metrics = {}
        self.manager = trainer.create_checkpoint_manager(trainer.config.checkpoint, trainer.logger)

    def on_validation_end(self, trainer: Any, val_loss: Any) -> None:
        # Runs before that step's `on_step_end`, so the step's checkpoint carries it (for best rotation).
        self._last_metrics["val/loss"] = trainer.loss_to_float(val_loss)

    def on_step_end(self, trainer: Any, global_step: int) -> None:
        if self.manager.should_save_checkpoint(global_step):
            self._save(trainer)

    def on_train_epoch_end(self, trainer: Any) -> None:
        if self.manager.should_save_checkpoint(trainer.global_step, epoch=trainer.epoch):
            self._save(trainer)

    def on_train_end(self, trainer: Any) -> None:
        if trainer.config.checkpoint.save_strategy.type == "none":
            return
        name = trainer.config.checkpoint.final_checkpoint_name
        path = self._save(trainer, name)
        trainer.logger.log_artifact(path, artifact_type="model", name=name)

    def _save(self, trainer: Any, name: Optional[str] = None) -> str:
        return self.manager.save_checkpoint(
            step=trainer.global_step, epoch=trainer.epoch, metrics=dict(self._last_metrics), name=name
        )

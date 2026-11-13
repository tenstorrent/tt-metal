# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

from .callback import Callback


class CheckpointCallback(Callback):
    """Starts the run from a checkpoint and saves checkpoints, as ``config.checkpoint`` says.

    Checkpoints hold what the model trains (``Trainer.save_checkpoint``): the LoRA adapter.
    """

    def __init__(self) -> None:
        self.manager = None
        self._last_metrics: dict[str, float] = {}

    def on_train_start(self, trainer: Any) -> None:
        # A callback outlives the run; never carry the previous run's metrics into this one's checkpoints.
        self._last_metrics = {}
        config = trainer.config.checkpoint
        self.manager = trainer.create_checkpoint_manager(config, trainer.logger)
        init = config.init_strategy
        if init is None:
            return
        if init.type == "resume":
            trainer.resume_from_checkpoint(init.path)
            trainer.logger.info(f"Resuming after step {trainer.global_step} from {init.path}")
        else:
            # Weights only: training still starts at step 0, so the stored step, epoch and metrics are unused.
            trainer.load_checkpoint(init.path)
            trainer.logger.info(f"Initialized the trained weights from {init.path}")

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

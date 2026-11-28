# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any, Optional

from ..managers.checkpoint import CheckpointManager
from .callback import Callback
from .metrics import MetricsCallback


class CheckpointCallback(Callback):
    """Saves checkpoints with ``Trainer.save_checkpoint``: every ``frequency`` steps of a "step"
    ``config.checkpoint.save_strategy``, and on the run's last step, which is also logged as the model artifact. A
    "none" strategy saves nothing.

    Every checkpoint carries the metrics of its own step: the ``values`` of the trainer's ``MetricsCallback``, which
    comes before this one in the callbacks. So the save frequency must be a multiple of the metrics' ``frequency``,
    and the last step logs and validates too. Without a ``MetricsCallback``, checkpoints carry no metrics.
    """

    def __init__(self) -> None:
        self.manager = None
        self.metrics: Optional[MetricsCallback] = None
        self._total_steps = 0

    def on_train_start(self, trainer: Any) -> None:
        # A callback outlives the run; a run that saves nothing has no manager.
        self.manager = None
        self.metrics = None
        if trainer.config.checkpoint.save_strategy.type == "none":
            return
        self.metrics = self._metrics_callback(trainer)
        self._check_config(trainer)
        self._total_steps = trainer.total_steps()
        self.manager = CheckpointManager(trainer.config.checkpoint, save=trainer.save_checkpoint, logger=trainer.logger)

    def _metrics_callback(self, trainer: Any) -> Optional[MetricsCallback]:
        callbacks = trainer.callback_handler.callbacks
        position = next(i for i, callback in enumerate(callbacks) if callback is self)
        for i, callback in enumerate(callbacks):
            if isinstance(callback, MetricsCallback):
                if i > position:
                    raise ValueError(
                        "MetricsCallback must come before CheckpointCallback in the callbacks, so a step's metrics "
                        "are logged before its checkpoint is saved"
                    )
                return callback
        return None

    def _check_config(self, trainer: Any) -> None:
        """Raise, stopping the job before it trains, if its checkpoints couldn't carry the metrics of their step."""
        checkpoint = trainer.config.checkpoint
        rotation = checkpoint.rotation_strategy
        if self.metrics is None:
            if rotation.type == "best":
                raise ValueError(
                    f"checkpoint.rotation_strategy: best ranks checkpoints by {rotation.metric}, and without a "
                    "MetricsCallback checkpoints carry no metrics"
                )
            return

        frequency = checkpoint.save_strategy.frequency
        metrics = trainer.config.metrics
        _check_multiple(frequency, metrics.train.frequency, "metrics.train.frequency")
        if trainer.validation_enabled:
            _check_multiple(frequency, metrics.validation.frequency, "metrics.validation.frequency")

        names = self.metrics.tracked_names(trainer)
        if rotation.type == "best" and rotation.metric not in names:
            reason = "" if trainer.validation_enabled else "; the run does not validate"
            raise ValueError(
                f"checkpoint.rotation_strategy: best ranks checkpoints by {rotation.metric}, and the run tracks only "
                f"{names}{reason}"
            )

    def on_step_end(self, trainer: Any, global_step: int) -> None:
        if self.manager is None:
            return
        if global_step == self._total_steps:
            path = self._save(trainer)
            trainer.logger.log_artifact(
                path, artifact_type="model", name="model", metadata={"checkpoint": os.path.basename(path)}
            )
        elif self.manager.should_save_checkpoint(global_step):
            self._save(trainer)

    def _save(self, trainer: Any) -> str:
        metrics = dict(self.metrics.values) if self.metrics is not None else {}
        return self.manager.save_checkpoint(step=trainer.global_step, epoch=trainer.epoch, metrics=metrics)


def _check_multiple(save_frequency: int, metrics_frequency: int, field: str) -> None:
    if save_frequency % metrics_frequency:
        raise ValueError(
            f"checkpoint.save_strategy.frequency ({save_frequency}) must be a multiple of {field} "
            f"({metrics_frequency}), so every checkpoint carries the metrics of its own step"
        )

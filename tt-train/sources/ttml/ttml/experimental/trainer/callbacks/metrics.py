# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import traceback
from typing import Any

from .callback import Callback


class MetricsCallback(Callback):
    """Logs the model size, and the metrics ``config.metrics`` asks for, through ``trainer.logger``.

    ``values`` keeps the latest value of every metric logged in the run, by its logged name, for other callbacks
    to read, as ``CheckpointCallback`` does. ``train/epoch`` is logged too, but is the run's progress, not a metric,
    so ``values`` leaves it out. A callback that reads ``values`` comes after this one in the callbacks, so a step's
    metrics are in it when its own hooks for that step run.

    TODO: once more callbacks track metrics, move ``values`` to one tracker the trainer owns: every metrics callback
    writes into it, ``CallbackHandler`` runs those callbacks before the others, and the others read from it.
    """

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        # Sum of the step losses since the last log, and their number: `frequency`, or fewer for the run's last
        # log, on its last step.
        self._running_loss: Any = None
        self._running_steps = 0
        self._total_steps = 0

    def on_train_start(self, trainer: Any) -> None:
        # A callback outlives the run; drop what a failed run left behind.
        self.values = {}
        self._running_loss = None
        self._running_steps = 0
        self._total_steps = trainer.total_steps()
        total, trainable = trainer.parameter_counts()
        trainer.logger.log_model_info(
            {
                "model_name": trainer.config.model.name,
                "total_params": total,
                "trainable_params": trainable,
                "train_dataset_size": _dataset_size(trainer.train_dataloader, trainer.config.batch_size),
                "val_dataset_size": _dataset_size(trainer.val_dataloader, trainer.config.batch_size),
            }
        )

    def on_optimizer_step_end(self, trainer: Any, window_loss: Any) -> None:
        self._running_loss = window_loss if self._running_loss is None else self._running_loss + window_loss
        self._running_steps += 1

    def on_step_end(self, trainer: Any, global_step: int) -> None:
        config = trainer.config.metrics.train
        if global_step % config.frequency == 0 or global_step == self._total_steps:
            available = {
                "loss": trainer.loss_to_float(self._running_loss) / self._running_steps,
                "learning_rate": trainer.current_learning_rate(),
            }
            metrics = _select(available, config.metric_names, "train")
            self.values.update(metrics)
            # The trainer's 0-based epoch, as checkpoints record it.
            trainer.logger.log_metrics({**metrics, "train/epoch": trainer.epoch}, step=global_step, commit=False)
            self._running_loss = None
            self._running_steps = 0
        # Commit once per step, so validation metrics logged during the step land on it too.
        trainer.logger.log_metrics({}, step=global_step, commit=True)

    def on_validation_end(self, trainer: Any, val_loss: Any) -> None:
        names = trainer.config.metrics.validation.metric_names
        metrics = _select({"loss": trainer.loss_to_float(val_loss)}, names, "val")
        if metrics:
            trainer.logger.log_metrics(metrics, step=trainer.global_step, commit=False)
            self.values.update(metrics)

    def tracked_names(self, trainer: Any) -> list[str]:
        """The names ``values`` keys the run's metrics by. ``trainer`` is set up."""
        config = trainer.config.metrics
        # Placeholder values: only the names, and how many values a metric has, matter here.
        train = {"loss": 0.0, "learning_rate": trainer.current_learning_rate()}
        names = list(_select(train, config.train.metric_names, "train"))
        if trainer.validation_enabled:
            names += _select({"loss": 0.0}, config.validation.metric_names, "val")
        return names

    def on_setup_error(self, trainer: Any, exception: BaseException) -> None:
        if trainer.logger is not None:
            trainer.logger.error(f"Setup failed: {exception}", traceback.format_exc())

    def on_train_error(self, trainer: Any, exception: BaseException) -> None:
        if trainer.logger is not None:
            trainer.logger.error(f"Training failed: {exception}", traceback.format_exc())


def _select(available: dict[str, Any], requested: list[str], phase: str) -> dict[str, float]:
    """``{phase}/{name}`` per requested metric; a metric of several values logs ``{phase}/{name}/{i}`` for each."""
    metrics = {}
    for name in requested:
        if name not in available:
            continue
        value = available[name]
        if not isinstance(value, list):
            metrics[f"{phase}/{name}"] = value
        elif len(value) == 1:
            metrics[f"{phase}/{name}"] = value[0]
        else:
            metrics.update({f"{phase}/{name}/{i}": v for i, v in enumerate(value)})
    return metrics


def _dataset_size(dataloader: Any, batch_size: int) -> Any:
    if dataloader is None:
        return None
    try:
        return len(dataloader) * batch_size
    except TypeError:
        return None

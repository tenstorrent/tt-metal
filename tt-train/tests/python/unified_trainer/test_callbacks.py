# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MetricsCallback and CheckpointCallback on a framework-free trainer. Device-free."""

import json
import os

from unified_trainer_fakes import FakeTrainer, make_config

from ttml.unified_trainer import CheckpointCallback, MetricsCallback


class LoggedMetrics:
    """Wraps ``trainer.logger.log_metrics`` to record what reaches it."""

    def __init__(self) -> None:
        self.calls = []

    def on_train_start(self, trainer):
        log_metrics = trainer.logger.log_metrics

        def record(metrics, step=None, commit=True):
            self.calls.append((dict(metrics), step, commit))
            log_metrics(metrics, step=step, commit=commit)

        trainer.logger.log_metrics = record

    def logged(self):
        return [(metrics, step) for metrics, step, _ in self.calls if metrics]


def run(trainer, **config):
    trainer.setup(make_config(**config))
    try:
        trainer.train()
    finally:
        trainer.cleanup()


def test_metrics_averages_the_loss_over_steps_freq_and_logs_validation():
    logged = LoggedMetrics()
    trainer = FakeTrainer([logged, MetricsCallback()])
    run(trainer, metrics={"steps_freq": 2, "train_metrics": ["loss", "learning_rate"]}, val_steps_freq=4)
    assert logged.logged() == [
        ({"val/loss": 4.0}, 0),
        ({"train/loss": 3.5, "train/learning_rate": 0.1}, 2),
        ({"val/loss": 4.0}, 4),
        ({"train/loss": 1.5, "train/learning_rate": 0.1}, 4),
    ]
    # Every step commits once.
    assert [step for metrics, step, commit in logged.calls if commit] == [1, 2, 3, 4]


def test_metrics_logs_only_what_is_asked_for():
    logged = LoggedMetrics()
    run(FakeTrainer([logged, MetricsCallback()]), metrics={"steps_freq": 4, "train_metrics": [], "val_metrics": []})
    assert logged.logged() == []


def checkpoint_config(tmp_path, **overrides):
    return {"project_dir": str(tmp_path), **overrides}


def test_checkpoints_every_steps_freq_and_at_the_end(tmp_path):
    trainer = FakeTrainer([CheckpointCallback()])
    run(trainer, checkpoint=checkpoint_config(tmp_path, save_strategy="step", steps_freq=2, keep_all_checkpoints=True))
    names = [os.path.basename(path) for path in trainer.saved]
    assert names[0].startswith("checkpoint_step2_epoch0_") and names[1].startswith("checkpoint_step4_epoch0_")
    assert names[2] == "final"
    assert [metadata["step"] for metadata in trainer.saved.values()] == [2, 4, 4]
    history = json.loads((tmp_path / "checkpoints" / "checkpoint_history.json").read_text())
    assert [entry["name"] for entry in history["checkpoints"]] == names


def test_checkpoints_carry_the_latest_validation_loss(tmp_path):
    trainer = FakeTrainer([CheckpointCallback()])
    run(trainer, val_steps_freq=2, checkpoint=checkpoint_config(tmp_path, save_strategy="step", steps_freq=2))
    assert [metadata["metrics"] for metadata in trainer.saved.values()] == [{"val/loss": 4.0}] * 3


def test_no_checkpoints_with_save_strategy_none(tmp_path):
    trainer = FakeTrainer([CheckpointCallback()])
    run(trainer, checkpoint=checkpoint_config(tmp_path))
    assert trainer.saved == {}


def test_epoch_checkpoints(tmp_path):
    trainer = FakeTrainer([CheckpointCallback()])
    run(trainer, num_epochs=2, checkpoint=checkpoint_config(tmp_path, save_strategy="epoch", keep_all_checkpoints=True))
    assert [(m["step"], m["epoch"]) for m in trainer.saved.values()] == [(4, 0), (8, 1), (8, 1)]


def test_init_from_the_last_checkpoint_of_the_previous_run(tmp_path):
    first = FakeTrainer([CheckpointCallback()])
    run(first, checkpoint=checkpoint_config(tmp_path, save_strategy="step", steps_freq=4))
    final = os.path.join(str(tmp_path), "checkpoints", "final")

    second = FakeTrainer([CheckpointCallback()])
    run(second, checkpoint=checkpoint_config(tmp_path, init_from_checkpoint=True))
    assert second.loaded == [final]

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MetricsCallback and CheckpointCallback on a framework-free trainer. Device-free."""

import json
import os

from unified_trainer_fakes import FakeTrainer, Recorder, make_config

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


def test_metrics_logs_each_learning_rate_of_several_optimizers():
    logged = LoggedMetrics()
    trainer = FakeTrainer([logged, MetricsCallback()])
    trainer.current_learning_rate = lambda: [0.1, 0.01]
    run(trainer, metrics={"steps_freq": 4, "train_metrics": ["learning_rate"], "val_metrics": []})
    assert logged.logged() == [({"train/learning_rate/0": 0.1, "train/learning_rate/1": 0.01}, 4)]


def test_metrics_logs_only_what_is_asked_for():
    logged = LoggedMetrics()
    run(FakeTrainer([logged, MetricsCallback()]), metrics={"steps_freq": 4, "train_metrics": [], "val_metrics": []})
    assert logged.logged() == []


def checkpoint_config(tmp_path, **overrides):
    return {"project_dir": str(tmp_path), **overrides}


EVERY_2_STEPS = {"type": "step", "frequency": 2}
KEEP_ALL = {"type": "none"}


def final_checkpoint(tmp_path):
    return os.path.join(str(tmp_path), "checkpoints", "final")


def test_checkpoints_every_frequency_steps_and_at_the_end(tmp_path):
    trainer = FakeTrainer([CheckpointCallback()])
    run(trainer, checkpoint=checkpoint_config(tmp_path, save_strategy=EVERY_2_STEPS, rotation_strategy=KEEP_ALL))
    names = [os.path.basename(path) for path in trainer.saved]
    assert names[0].startswith("checkpoint_step2_epoch0_") and names[1].startswith("checkpoint_step4_epoch0_")
    assert names[2] == "final"
    assert [metadata["step"] for metadata in trainer.saved.values()] == [2, 4, 4]
    history = json.loads((tmp_path / "checkpoints" / "checkpoint_history.json").read_text())
    assert [entry["name"] for entry in history["checkpoints"]] == names


def test_checkpoints_carry_the_latest_validation_loss(tmp_path):
    trainer = FakeTrainer([CheckpointCallback()])
    run(trainer, val_steps_freq=2, checkpoint=checkpoint_config(tmp_path, save_strategy=EVERY_2_STEPS))
    assert [metadata["metrics"] for metadata in trainer.saved.values()] == [{"val/loss": 4.0}] * 3


def test_no_checkpoints_with_save_strategy_none(tmp_path):
    trainer = FakeTrainer([CheckpointCallback()])
    run(trainer, checkpoint=checkpoint_config(tmp_path))
    assert trainer.saved == {}


def test_epoch_checkpoints(tmp_path):
    trainer = FakeTrainer([CheckpointCallback()])
    run(
        trainer,
        num_epochs=2,
        checkpoint=checkpoint_config(tmp_path, save_strategy={"type": "epoch"}, rotation_strategy=KEEP_ALL),
    )
    assert [(m["step"], m["epoch"]) for m in trainer.saved.values()] == [(4, 0), (8, 1), (8, 1)]


def test_init_loads_the_weights_and_trains_from_step_0(tmp_path):
    first = FakeTrainer([CheckpointCallback()])
    run(first, checkpoint=checkpoint_config(tmp_path, save_strategy=EVERY_2_STEPS, save_optim=True))

    recorder = Recorder()
    second = FakeTrainer([CheckpointCallback(), recorder])
    init = {"type": "init", "path": final_checkpoint(tmp_path)}
    run(second, checkpoint=checkpoint_config(tmp_path / "second", init_strategy=init))
    assert second.loaded == [final_checkpoint(tmp_path)]
    assert recorder.args("on_step_end") == [(1,), (2,), (3,), (4,)]


def test_resume_continues_after_the_checkpoint_s_step(tmp_path):
    first = FakeTrainer([CheckpointCallback()])
    run(first, max_steps=2, checkpoint=checkpoint_config(tmp_path, save_strategy=EVERY_2_STEPS, save_optim=True))

    recorder = Recorder()
    second = FakeTrainer([CheckpointCallback(), recorder])
    resume = {"type": "resume", "path": final_checkpoint(tmp_path)}
    second.setup(make_config(checkpoint=checkpoint_config(tmp_path / "second", init_strategy=resume)))
    try:
        second.train()
        # The two steps of the first run, then its last two batches.
        assert second.optimizer.steps == 4
    finally:
        second.cleanup()
    assert recorder.args("on_step_end") == [(3,), (4,)]
    assert recorder.args("on_optimizer_step_end") == [(2.0,), (1.0,)]


def test_resume_needs_the_optimizer_state(tmp_path, expect_error):
    first = FakeTrainer([CheckpointCallback()])
    run(first, max_steps=2, checkpoint=checkpoint_config(tmp_path, save_strategy=EVERY_2_STEPS))

    second = FakeTrainer([CheckpointCallback()])
    resume = {"type": "resume", "path": final_checkpoint(tmp_path)}
    with expect_error(FileNotFoundError, "has no optimizer state"):
        run(second, checkpoint=checkpoint_config(tmp_path / "second", init_strategy=resume))


def test_after_a_resume_the_train_loss_averages_the_steps_it_trained(tmp_path):
    first = FakeTrainer([CheckpointCallback()])
    run(first, max_steps=1, checkpoint=checkpoint_config(tmp_path, save_strategy=EVERY_2_STEPS, save_optim=True))

    logged = LoggedMetrics()
    second = FakeTrainer([CheckpointCallback(), logged, MetricsCallback()])
    resume = {"type": "resume", "path": final_checkpoint(tmp_path)}
    run(
        second,
        metrics={"steps_freq": 2, "val_metrics": []},
        checkpoint=checkpoint_config(tmp_path / "second", init_strategy=resume),
    )
    # Step 2 is the only step of its window this run trained; steps 3 and 4 make the next one.
    assert logged.logged() == [({"train/loss": 3.0}, 2), ({"train/loss": 1.5}, 4)]

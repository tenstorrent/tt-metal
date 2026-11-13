# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What a long-lived host (tt-inference-server's training runner) relies on: one trainer built at warmup
runs many jobs, a callback may stop a job by raising, and a failed setup reaches the callbacks. Device-free."""

import contextlib

import pytest
from unified_trainer_fakes import FakeTrainer, Recorder, make_config


class StopTraining(Exception):
    """The host's own early-stop exception: the trainer knows nothing about it."""


class JobCallback:
    """Reads only what the host's callbacks read from the trainer, and stops at ``max_steps``."""

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.metrics, self.saved, self.errors = [], [], []
        self._running = None

    def on_train_start(self, trainer):
        self.batches_per_epoch = len(trainer.train_dataloader)
        stopping = trainer.config.stopping_criteria
        self.label = f"Epoch {trainer.epoch + 1}/{stopping.limit}" if stopping.type == "epoch" else "Epoch"

    def on_optimizer_step_end(self, trainer, window_loss):
        self._running = window_loss if self._running is None else self._running + window_loss

    def on_step_end(self, trainer, global_step):
        steps_freq = trainer.config.metrics.train.steps_freq
        if global_step % steps_freq == 0:
            loss = trainer.loss_to_float(self._running) / steps_freq
            self.metrics.append((global_step, trainer.epoch, loss, trainer.current_learning_rate()))
            self._running = None
        checkpoint = trainer.config.checkpoint
        if checkpoint.save_strategy.type == "step" and global_step % checkpoint.save_strategy.frequency == 0:
            path = f"{checkpoint.project_dir}/ckpt-step-{global_step}"
            trainer.save_checkpoint(path)
            self.saved.append(path)

    def on_train_batch_end(self, trainer):
        if trainer.global_step >= self.max_steps:
            raise StopTraining(f"Reached max_steps={self.max_steps}")

    def on_setup_error(self, trainer, exception):
        self.errors.append(("setup", exception))

    def on_train_error(self, trainer, exception):
        self.errors.append(("train", exception))


def job_config(tmp_path, **overrides):
    return make_config(
        metrics={"train": {"steps_freq": 2}},
        checkpoint={"save_strategy": {"type": "step", "frequency": 2}, "project_dir": str(tmp_path)},
        **overrides,
    )


def run_job(trainer, config):
    """The runner's sequence for one job."""
    try:
        trainer.setup(config)
        with contextlib.suppress(StopTraining):
            trainer.train()
    finally:
        trainer.cleanup()


def test_after_warmup_the_trainer_holds_no_run():
    trainer = FakeTrainer(callbacks=[JobCallback(max_steps=10)])
    assert trainer.config is None
    assert not hasattr(trainer, "model")
    assert callable(trainer.callback_handler)


def test_a_job_stopped_by_a_callback_ends_cleanly(tmp_path):
    job = JobCallback(max_steps=3)
    trainer = FakeTrainer(callbacks=[job])
    run_job(trainer, job_config(tmp_path))

    assert [m[:3] for m in job.metrics] == [(2, 0, 3.5)]
    assert job.metrics[0][3] == pytest.approx([0.1])
    assert job.saved == [f"{tmp_path}/ckpt-step-2"]
    assert list(trainer.saved) == job.saved
    assert job.batches_per_epoch == 4
    assert [(phase, type(e)) for phase, e in job.errors] == [("train", StopTraining)]
    assert trainer.config is None and not hasattr(trainer, "model")


def test_the_next_job_starts_from_scratch(tmp_path):
    job, recorder = JobCallback(max_steps=100), Recorder()
    trainer = FakeTrainer(callbacks=[job, recorder])
    run_job(trainer, job_config(tmp_path / "a"))
    run_job(trainer, job_config(tmp_path / "b", optimizer={"lr": 0.5}))

    assert recorder.args("on_step_end") == [(1,), (2,), (3,), (4,)] * 2
    assert [m[3] for m in job.metrics] == [[0.1], [0.1], [0.5], [0.5]]
    assert trainer.events.count("device.open") == trainer.events.count("device.close") == 2


def test_a_failed_setup_reaches_the_callbacks_and_frees_everything(tmp_path, expect_error):
    job = JobCallback(max_steps=10)
    trainer = FakeTrainer(callbacks=[job], fail_in="model")
    with expect_error(RuntimeError, "model failed"):
        run_job(trainer, job_config(tmp_path))
    assert [(phase, str(e)) for phase, e in job.errors] == [("setup", "model failed")]
    assert trainer.events[-1] == "device.close"
    assert trainer.config is None

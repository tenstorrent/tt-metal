# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The training loop of ``ttml.unified_trainer.Trainer``: hook order, gradient accumulation, validation
scheduling, the stopping criteria, failures and cleanup. Device-free."""

import pytest
from unified_trainer_fakes import FakeTrainer, RaiseAt, Recorder, after_epochs, after_steps, make_config

TRAIN_BATCH = ["on_train_batch_start", "on_forward_start", "on_forward_end", "on_backward_start", "on_backward_end"]
STEP = ["on_optimizer_step_start", "on_optimizer_step_end"]
VALIDATION = [
    "on_validation_start",
    "on_validation_batch_start",
    "on_validation_batch_end",
    "on_validation_batch_start",
    "on_validation_batch_end",
    "on_validation_end",
]


def validation_every(steps):
    """The ``metrics`` section of a run that validates every ``steps`` optimizer steps."""
    return {"validation": {"steps_freq": steps}}


def run(trainer, **config):
    trainer.setup(make_config(**config))
    try:
        trainer.train()
    finally:
        trainer.cleanup()


def test_hook_order_with_accumulation_and_validation():
    recorder = Recorder()
    run(FakeTrainer([recorder]), gradient_accumulation_steps=2, metrics=validation_every(1))

    window = TRAIN_BATCH + ["on_train_batch_end"] + TRAIN_BATCH + STEP + VALIDATION
    assert recorder.hooks() == [
        "on_train_start",
        *VALIDATION,
        "on_train_epoch_start",
        *window,
        "on_step_end",
        "on_train_batch_end",
        *window,
        "on_step_end",
        "on_train_batch_end",
        "on_train_epoch_end",
        "on_train_end",
    ]


def test_window_loss_is_the_mean_of_its_micro_batches():
    recorder = Recorder()
    run(FakeTrainer([recorder]), gradient_accumulation_steps=2)
    assert recorder.args("on_optimizer_step_end") == [(3.5,), (1.5,)]
    assert recorder.args("on_step_end") == [(1,), (2,)]


def test_validation_runs_before_training_and_every_steps_freq_steps():
    recorder = Recorder()
    run(FakeTrainer([recorder]), metrics=validation_every(2))
    # Before training, then after steps 2 and 4; each is the mean of the validation losses.
    assert recorder.args("on_validation_end") == [(4.0,), (4.0,), (4.0,)]
    hooks = recorder.hooks()
    ends = [i for i, hook in enumerate(hooks) if hook == "on_validation_end"]
    steps = [i for i, hook in enumerate(hooks) if hook == "on_step_end"]
    assert ends[0] < steps[0] and steps[0] < ends[1] < steps[1] and steps[2] < ends[2] < steps[3]


def test_no_validation_without_data_or_frequency():
    for trainer, config in (
        (FakeTrainer([Recorder()], val_losses=None), {"metrics": validation_every(1)}),
        (FakeTrainer([Recorder()]), {}),
    ):
        recorder = trainer.callback_handler.callbacks[0]
        run(trainer, **config)
        assert "on_validation_start" not in recorder.hooks()


def test_empty_validation_set_reports_zero():
    recorder = Recorder()
    run(FakeTrainer([recorder], val_losses=[]), metrics=validation_every(1))
    assert recorder.args("on_validation_end")[0] == (0.0,)


def test_an_epoch_limit_ends_the_run_after_that_many_epochs():
    recorder = Recorder()
    run(FakeTrainer([recorder]), stopping_criteria=after_epochs(3))
    # 4 steps an epoch.
    assert recorder.args("on_step_end") == [(step,) for step in range(1, 13)]
    assert recorder.hooks().count("on_train_epoch_start") == recorder.hooks().count("on_train_epoch_end") == 3
    assert recorder.hooks()[-1] == "on_train_end"


def test_a_step_limit_ends_the_run_mid_epoch():
    recorder = Recorder()
    run(FakeTrainer([recorder]), stopping_criteria=after_steps(3))
    assert recorder.args("on_step_end") == [(1,), (2,), (3,)]
    assert recorder.hooks().count("on_train_epoch_start") == 1
    # The epoch it ends in still ends, and so does the run.
    assert recorder.hooks()[-2:] == ["on_train_epoch_end", "on_train_end"]


@pytest.mark.parametrize("limit, epochs", [(4, 1), (6, 2), (8, 2), (9, 3)])
def test_a_step_limit_takes_as_many_epochs_as_its_steps(limit, epochs):
    recorder = Recorder()
    run(FakeTrainer([recorder]), stopping_criteria=after_steps(limit))
    # 4 steps an epoch: a limit at the end of an epoch starts no other.
    assert recorder.args("on_step_end") == [(step,) for step in range(1, limit + 1)]
    assert recorder.hooks().count("on_train_epoch_start") == recorder.hooks().count("on_train_epoch_end") == epochs


def test_a_step_limit_counts_optimizer_steps_not_batches():
    recorder = Recorder()
    run(FakeTrainer([recorder]), gradient_accumulation_steps=2, stopping_criteria=after_steps(3))
    # 2 steps an epoch, of 2 batches each: the third step is in the second epoch.
    assert recorder.args("on_step_end") == [(1,), (2,), (3,)]
    assert recorder.hooks().count("on_train_batch_start") == 6
    assert recorder.hooks().count("on_train_epoch_start") == 2


def test_a_step_limit_no_epoch_gets_closer_to_is_rejected(expect_error):
    recorder = Recorder()
    trainer = FakeTrainer([recorder], train_losses=[1.0])
    config = make_config(gradient_accumulation_steps=2, stopping_criteria=after_steps(3))
    with expect_error(ValueError, "stopping_criteria: an epoch of 1 batches makes no optimizer step"):
        trainer.setup(config)
    # Before the device opens.
    assert recorder.hooks() == ["on_setup_error"] and "device.open" not in trainer.events
    trainer.cleanup()

    # An epoch limit ends such a run, without a step.
    run(trainer, gradient_accumulation_steps=2, stopping_criteria=after_epochs(2))
    assert recorder.hooks().count("on_train_epoch_end") == 2 and "on_step_end" not in recorder.hooks()


def test_a_partial_window_never_leaks_into_the_next_epoch():
    recorder = Recorder()
    trainer = FakeTrainer([recorder], train_losses=[1.0, 2.0, 3.0])
    run(trainer, gradient_accumulation_steps=2, stopping_criteria=after_epochs(2))
    # One full window per epoch; the third batch's loss is dropped, not added to the next epoch's window.
    assert recorder.args("on_optimizer_step_end") == [(1.5,), (1.5,)]


def test_gradients_are_zeroed_at_each_epoch_start():
    trainer = FakeTrainer(train_losses=[1.0, 2.0, 3.0])
    trainer.setup(make_config(gradient_accumulation_steps=2, stopping_criteria=after_epochs(2)))
    trainer.train()
    # Per epoch: one zero at the start and one after its single optimizer step.
    assert (trainer.optimizer.steps, trainer.optimizer.zero_grads) == (2, 4)
    trainer.cleanup()


class Steps:
    """Records ``(step, epoch, window loss)`` per optimizer step."""

    def __init__(self) -> None:
        self.steps = []

    def on_optimizer_step_end(self, trainer, window_loss):
        self.loss = window_loss

    def on_step_end(self, trainer, global_step):
        self.steps.append((global_step, trainer.epoch, self.loss))


@pytest.mark.parametrize(
    "train_losses, step, expected",
    [
        # Two steps per epoch.
        ((4.0, 3.0, 2.0, 1.0), 1, [(2, 0, 1.5), (3, 1, 3.5), (4, 1, 1.5)]),
        ((4.0, 3.0, 2.0, 1.0), 2, [(3, 1, 3.5), (4, 1, 1.5)]),
        ((4.0, 3.0, 2.0, 1.0), 3, [(4, 1, 1.5)]),
        # One step per epoch: the third batch is a partial window.
        ((1.0, 2.0, 3.0), 1, [(2, 1, 1.5)]),
    ],
)
def test_a_resumed_run_trains_on_the_batches_after_its_step(tmp_path, train_losses, step, expected):
    steps = Steps()
    trainer = FakeTrainer([steps], train_losses=train_losses)
    trainer.setup(make_config(gradient_accumulation_steps=2, stopping_criteria=after_epochs(2)))
    try:
        trainer.save_checkpoint(str(tmp_path), {"step": step}, optimizer=True)
        trainer.resume_from_checkpoint(str(tmp_path))
        trainer.train()
    finally:
        trainer.cleanup()
    assert steps.steps == expected


@pytest.mark.parametrize(
    "step, expected",
    [
        # Two steps per epoch, and a limit of 3 steps: the steps of the checkpoint count towards it.
        (1, [(2, 0, 1.5), (3, 1, 3.5)]),
        (2, [(3, 1, 3.5)]),
        (3, []),
        (5, []),
    ],
)
def test_a_resumed_run_ends_at_the_step_limit_of_the_run_it_continues(tmp_path, step, expected):
    steps = Steps()
    trainer = FakeTrainer([steps])
    trainer.setup(make_config(gradient_accumulation_steps=2, stopping_criteria=after_steps(3)))
    try:
        trainer.save_checkpoint(str(tmp_path), {"step": step}, optimizer=True)
        trainer.resume_from_checkpoint(str(tmp_path))
        trainer.train()
    finally:
        trainer.cleanup()
    assert steps.steps == expected


def test_an_exception_goes_to_on_train_error_and_out_of_train(expect_error):
    recorder = Recorder()
    error = ValueError("boom")
    trainer = FakeTrainer([RaiseAt("on_step_end", 2, error), recorder])
    trainer.setup(make_config())
    with expect_error(ValueError, "boom"):
        trainer.train()
    assert recorder.args("on_train_error") == [(error,)]
    assert "on_setup_error" not in recorder.hooks()
    assert "on_train_end" not in recorder.hooks()
    assert trainer.logger._finished
    trainer.cleanup()


def test_a_failing_on_train_start_also_reaches_on_train_error(expect_error):
    recorder = Recorder()
    error = RuntimeError("no start")
    trainer = FakeTrainer([RaiseAt("on_train_start", 0, error), recorder])
    trainer.setup(make_config())
    with expect_error(RuntimeError, "no start"):
        trainer.train()
    assert recorder.args("on_train_error") == [(error,)]
    trainer.cleanup()


@pytest.mark.parametrize("step", ["check", "data", "model"])
def test_a_failed_setup_goes_to_on_setup_error_and_out_of_setup(step, expect_error):
    recorder = Recorder()
    trainer = FakeTrainer([recorder], fail_in=step)
    with expect_error(RuntimeError, f"{step} failed"):
        trainer.setup(make_config())
    [(error,)] = recorder.args("on_setup_error")
    assert str(error) == f"{step} failed"
    assert recorder.hooks() == ["on_setup_error"]
    trainer.cleanup()


def test_train_needs_setup(expect_error):
    with expect_error(RuntimeError, "setup"):
        FakeTrainer().train()


def test_setup_runs_checks_and_data_before_the_device():
    trainer = FakeTrainer()
    trainer.setup(make_config())
    assert trainer.events == ["check", "data", "device.open", "model"]
    trainer.cleanup()


@pytest.mark.parametrize("step", ["check", "data"])
def test_a_failed_check_or_data_load_never_opens_the_device(step, expect_error):
    trainer = FakeTrainer(fail_in=step)
    with expect_error(RuntimeError, f"{step} failed"):
        trainer.setup(make_config())
    trainer.cleanup()
    assert "device.open" not in trainer.events


def test_cleanup_after_a_failed_model_load_closes_the_device(expect_error):
    trainer = FakeTrainer(fail_in="model")
    with expect_error(RuntimeError, "model failed"):
        trainer.setup(make_config())
    trainer.cleanup()
    assert trainer.events[-1] == "device.close"


def test_cleanup_drops_the_run_and_keeps_the_host_state():
    recorder = Recorder()
    trainer = FakeTrainer([recorder])
    run(trainer, metrics=validation_every(1))
    assert trainer.config is None and trainer.logger is None
    assert (trainer.global_step, trainer.epoch, trainer.step_loss) == (0, 0, None)
    for attr in ("model", "optimizer", "train_dataloader", "val_dataloader", "device_manager"):
        assert not hasattr(trainer, attr)
    # Callbacks, and what the subclass set up in __init__, outlive the run.
    assert trainer.callback_handler.callbacks == [recorder]
    assert trainer.train_losses == [4.0, 3.0, 2.0, 1.0]
    assert trainer.events[-1] == "device.close"


def test_cleanup_is_idempotent_and_safe_before_setup():
    trainer = FakeTrainer()
    trainer.cleanup()
    trainer.setup(make_config())
    trainer.cleanup()
    trainer.cleanup()
    assert trainer.events.count("device.close") == 1
    assert trainer.train_losses == [4.0, 3.0, 2.0, 1.0]


def test_setup_again_cleans_up_the_previous_run():
    trainer = FakeTrainer()
    trainer.setup(make_config())
    trainer.setup(make_config(optimizer={"lr": 0.5}))
    assert trainer.events.count("device.close") == 1
    assert trainer.current_learning_rate() == [0.5]
    trainer.cleanup()

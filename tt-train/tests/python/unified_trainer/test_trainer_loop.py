# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The training loop of ``ttml.unified_trainer.Trainer``: hook order, gradient accumulation, validation
scheduling, max_steps, failures and cleanup. Device-free."""

import pytest
from unified_trainer_fakes import FakeTrainer, RaiseAt, Recorder, make_config

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


def run(trainer, **config):
    trainer.setup(make_config(**config))
    try:
        trainer.train()
    finally:
        trainer.cleanup()


def test_hook_order_with_accumulation_and_validation():
    recorder = Recorder()
    run(FakeTrainer([recorder]), gradient_accumulation_steps=2, val_steps_freq=1)

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


def test_validation_runs_before_training_and_every_val_steps_freq_steps():
    recorder = Recorder()
    run(FakeTrainer([recorder]), val_steps_freq=2)
    # Before training, then after steps 2 and 4; each is the mean of the validation losses.
    assert recorder.args("on_validation_end") == [(4.0,), (4.0,), (4.0,)]
    hooks = recorder.hooks()
    ends = [i for i, hook in enumerate(hooks) if hook == "on_validation_end"]
    steps = [i for i, hook in enumerate(hooks) if hook == "on_step_end"]
    assert ends[0] < steps[0] and steps[0] < ends[1] < steps[1] and steps[2] < ends[2] < steps[3]


def test_no_validation_without_data_or_frequency():
    for trainer, config in (
        (FakeTrainer([Recorder()], val_losses=None), {"val_steps_freq": 1}),
        (FakeTrainer([Recorder()]), {}),
    ):
        recorder = trainer.callback_handler.callbacks[0]
        run(trainer, **config)
        assert "on_validation_start" not in recorder.hooks()


def test_empty_validation_set_reports_zero():
    recorder = Recorder()
    run(FakeTrainer([recorder], val_losses=[]), val_steps_freq=1)
    assert recorder.args("on_validation_end")[0] == (0.0,)


def test_max_steps_ends_the_run_mid_epoch():
    recorder = Recorder()
    run(FakeTrainer([recorder]), max_steps=3, num_epochs=2)
    assert recorder.args("on_step_end") == [(1,), (2,), (3,)]
    assert recorder.hooks().count("on_train_epoch_start") == 1
    assert recorder.hooks()[-1] == "on_train_end"


def test_a_partial_window_never_leaks_into_the_next_epoch():
    recorder = Recorder()
    trainer = FakeTrainer([recorder], train_losses=[1.0, 2.0, 3.0])
    run(trainer, gradient_accumulation_steps=2, num_epochs=2)
    # One full window per epoch; the third batch's loss is dropped, not added to the next epoch's window.
    assert recorder.args("on_optimizer_step_end") == [(1.5,), (1.5,)]


def test_gradients_are_zeroed_at_each_epoch_start():
    trainer = FakeTrainer(train_losses=[1.0, 2.0, 3.0])
    trainer.setup(make_config(gradient_accumulation_steps=2, num_epochs=2))
    trainer.train()
    # Per epoch: one zero at the start and one after its single optimizer step.
    assert (trainer.optimizer.steps, trainer.optimizer.zero_grads) == (2, 4)
    trainer.cleanup()


def test_an_exception_goes_to_on_error_and_out_of_train(expect_error):
    recorder = Recorder()
    error = ValueError("boom")
    trainer = FakeTrainer([RaiseAt("on_step_end", 2, error), recorder])
    trainer.setup(make_config())
    with expect_error(ValueError, "boom"):
        trainer.train()
    assert recorder.args("on_error") == [(error,)]
    assert "on_train_end" not in recorder.hooks()
    assert trainer.logger._finished
    trainer.cleanup()


def test_a_failing_on_train_start_also_reaches_on_error(expect_error):
    recorder = Recorder()
    error = RuntimeError("no start")
    trainer = FakeTrainer([RaiseAt("on_train_start", 0, error), recorder])
    trainer.setup(make_config())
    with expect_error(RuntimeError, "no start"):
        trainer.train()
    assert recorder.args("on_error") == [(error,)]
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
    run(trainer, val_steps_freq=1)
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
    trainer.setup(make_config(learning_rate=0.5))
    assert trainer.events.count("device.close") == 1
    assert trainer.current_learning_rate() == 0.5
    trainer.cleanup()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dispatching callback hooks: a hook that raises, an error hook too, stops the dispatch and fails the run; a failing
error hook keeps the run's exception as its ``__context__``, and ``cleanup`` still frees the callbacks' state.
Device-free."""

import contextlib
import json
import os

from ttml.experimental.trainer import Callback, CheckpointCallback, MetricsCallback, Trainer, TrainerConfig


class _Device:
    """Records the callbacks' state as it closes: what would still hold device tensors past the device."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.state_at_close = None

    def close(self):
        self.state_at_close = dict(self.trainer.callback_handler.state)


class _StubTrainer(Trainer):
    """Trains on ``batches``, plain floats; ``fail_in`` names the step that raises, ``"train"`` on the last batch."""

    def __init__(self, callbacks, fail_in, batches=(1.0,)):
        super().__init__(callbacks)
        self.fail_in = fail_in
        self.batches = list(batches)

    def _load_dataloaders(self):
        if self.fail_in == "setup":
            raise RuntimeError("setup failed")
        return self.batches, None

    def _create_device_manager(self):
        return _Device(self)

    def _load_model(self):
        return None

    def _load_optimizer(self):
        return None

    def _prepare_batch(self, batch):
        return batch

    def _forward(self, batch):
        if self.fail_in == "train" and batch == self.batches[-1]:
            raise RuntimeError("train failed")
        return batch

    def _backward(self, loss):
        pass

    def _accumulate_loss(self, running, loss):
        return loss if running is None else running + loss

    def _clip_gradients(self, max_norm):
        pass

    def _optimizer_step(self):
        pass

    def _zero_grad(self):
        pass

    def _train_mode(self):
        pass

    def _eval_mode(self):
        pass

    def _no_grad(self):
        return contextlib.nullcontext()

    def loss_to_float(self, loss):
        return float(loss)

    def parameter_counts(self):
        return 0, 0

    def current_learning_rate(self):
        return [0.0]

    def save_checkpoint(self, path, metadata=None):
        pass


class _Recorder(Callback):
    """Records the hooks it gets into ``calls``; the hooks named in ``raises`` raise after recording."""

    def __init__(self, name, calls, raises=()):
        self.name = name
        self.calls = calls
        self.raises = raises

    def _record(self, hook, *args):
        self.calls.append((self.name, hook, *args))
        if hook in self.raises:
            raise ValueError(f"{self.name}.{hook} failed")

    def on_train_start(self, trainer):
        self._record("on_train_start")

    def on_setup_error(self, trainer, exception):
        self._record("on_setup_error", str(exception))

    def on_train_error(self, trainer, exception):
        self._record("on_train_error", str(exception))


def _config(**overrides):
    return TrainerConfig(
        model={"name": "stub"}, dataset={"id": "stub"}, optimizer={"lr": 1e-3}, batch_size=1, **overrides
    )


def _best_rotation_config(project_dir, metric):
    return _config(
        metrics={"train": {"frequency": 1}},
        checkpoint={
            "save_strategy": {"type": "step", "frequency": 1, "project_dir": str(project_dir)},
            "rotation_strategy": {"type": "best", "keep_n": 1, "metric": metric},
        },
    )


def test_failing_setup_error_hook_fails_setup(expect_error):
    calls = []
    first = _Recorder("first", calls, raises={"on_setup_error"})
    trainer = _StubTrainer([first, _Recorder("second", calls)], fail_in="setup")
    try:
        with expect_error(ValueError, "first.on_setup_error failed") as caught:
            trainer.setup(_config())
    finally:
        trainer.cleanup()
    assert calls == [("first", "on_setup_error", "setup failed")]
    assert str(caught.value.__context__) == "setup failed"


def test_failing_train_error_hook_fails_train(expect_error):
    calls = []
    first = _Recorder("first", calls, raises={"on_train_error"})
    trainer = _StubTrainer([first, _Recorder("second", calls)], fail_in="train")
    trainer.setup(_config())
    try:
        with expect_error(ValueError, "first.on_train_error failed") as caught:
            trainer.train()
    finally:
        trainer.cleanup()
    assert calls == [
        ("first", "on_train_start"),
        ("second", "on_train_start"),
        ("first", "on_train_error", "train failed"),
    ]
    assert str(caught.value.__context__) == "train failed"


def test_failing_hook_stops_dispatch_and_ends_run(expect_error):
    calls = []
    first = _Recorder("first", calls, raises={"on_train_start"})
    trainer = _StubTrainer([first, _Recorder("second", calls)], fail_in=None)
    trainer.setup(_config())
    try:
        with expect_error(ValueError, "first.on_train_start failed"):
            trainer.train()
    finally:
        trainer.cleanup()
    assert calls == [
        ("first", "on_train_start"),
        ("first", "on_train_error", "first.on_train_start failed"),
        ("second", "on_train_error", "first.on_train_start failed"),
    ]


def test_metrics_log_the_summed_loss_from_callback_state():
    metrics = MetricsCallback()
    trainer = _StubTrainer([metrics], fail_in=None, batches=(1.0, 3.0))
    trainer.setup(_config())
    try:
        trainer.train()
        # Logged once, on the last step: the mean of both steps' losses.
        assert metrics.values["train/loss"] == 2.0
        assert trainer.callback_handler.state == {}
    finally:
        trainer.cleanup()


def test_best_rotation_by_a_train_metric_needs_no_validation(tmp_path):
    trainer = _StubTrainer([MetricsCallback(), CheckpointCallback()], fail_in=None, batches=(3.0, 1.0, 2.0, 4.0))
    trainer.setup(_best_rotation_config(tmp_path, "train/loss"))
    try:
        trainer.train()
    finally:
        trainer.cleanup()
    with open(os.path.join(tmp_path, "checkpoints", "checkpoint_history.json")) as f:
        history = json.load(f)["checkpoints"]
    # The lowest train/loss, step 2's, and the latest.
    assert [(entry["step"], entry["metrics"]["train/loss"]) for entry in history] == [(2, 1.0), (4, 4.0)]


def test_best_rotation_by_a_validation_metric_needs_validation(tmp_path, expect_error):
    with expect_error(ValueError, "val/loss, a validation metric, so set metrics.validation.frequency"):
        _best_rotation_config(tmp_path, "val/loss")


def test_cleanup_frees_callback_state_an_error_hook_left_behind(expect_error):
    first = _Recorder("first", [], raises={"on_train_error"})
    # Two steps, so the run fails with the first one's loss summed and not yet logged.
    trainer = _StubTrainer([first, MetricsCallback()], fail_in="train", batches=(1.0, 2.0))
    trainer.setup(_config())
    device = trainer.device_manager
    try:
        with expect_error(ValueError, "first.on_train_error failed"):
            trainer.train()
        assert trainer.callback_handler.state == {"metrics/running_loss": 1.0, "metrics/running_steps": 1}
    finally:
        trainer.cleanup()
    assert device.state_at_close == {}

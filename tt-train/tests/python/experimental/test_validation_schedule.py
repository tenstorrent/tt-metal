# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""When ``Trainer.train`` validates: before training, every ``metrics.validation.frequency`` optimizer steps, and on
the last step, after its optimizer step; and how: on every validation batch, in eval mode without gradients, back
in train mode after. Device-free."""

import contextlib

import pytest

from ttml.experimental.trainer import Callback, Trainer, TrainerConfig


class _Device:
    def close(self):
        pass


class _StubTrainer(Trainer):
    """Trains on ``train_batches`` and validates on ``val_batches``, plain floats that are their own loss.

    Records each forward's batch with the mode and gradient state it ran in, in ``forwards``, and each optimizer
    step, in ``events``.
    """

    def __init__(self, train_batches, val_batches, events, callbacks=None):
        super().__init__(callbacks)
        self.train_batches = list(train_batches)
        self.val_batches = None if val_batches is None else list(val_batches)
        self.events = events
        self.forwards = []
        self.mode = None
        self.grad_enabled = True

    def _load_dataloaders(self):
        return self.train_batches, self.val_batches

    def _create_device_manager(self):
        return _Device()

    def _load_model(self):
        return None

    def _load_optimizer(self):
        return None

    def _prepare_batch(self, batch):
        return batch

    def _forward(self, batch):
        self.forwards.append((batch, self.mode, self.grad_enabled))
        return batch

    def _backward(self, loss):
        pass

    def _accumulate_loss(self, running, loss):
        return loss if running is None else running + loss

    def _clip_gradients(self, max_norm):
        pass

    def _optimizer_step(self):
        self.events.append(("step", self.global_step + 1))

    def _zero_grad(self):
        pass

    def _train_mode(self):
        self.mode = "train"

    def _eval_mode(self):
        self.mode = "eval"

    @contextlib.contextmanager
    def _no_grad(self):
        self.grad_enabled = False
        try:
            yield
        finally:
            self.grad_enabled = True

    def loss_to_float(self, loss):
        return float(loss)

    def parameter_counts(self):
        return 0, 0

    def current_learning_rate(self):
        return [0.0]

    def save_checkpoint(self, path, metadata=None):
        pass


class _ValidationRecorder(Callback):
    """Records ``("validate", global_step, val_loss)`` for each validation into ``events``."""

    def __init__(self, events):
        self.events = events

    def on_validation_end(self, trainer, val_loss):
        self.events.append(("validate", trainer.global_step, trainer.loss_to_float(val_loss)))


def _config(frequency, stopping, gradient_accumulation_steps=1):
    return TrainerConfig(
        model={"name": "stub"},
        dataset={"id": "stub"},
        optimizer={"lr": 1e-3},
        batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        stopping_criteria=stopping,
        metrics={"validation": {"frequency": frequency}},
    )


def _train(config, train_batches=(1.0,) * 8, val_batches=(1.0,)):
    """Run ``config``; return the trainer and its events, optimizer steps and validations in the order they ran."""
    events = []
    trainer = _StubTrainer(train_batches, val_batches, events, callbacks=[_ValidationRecorder(events)])
    trainer.setup(config)
    try:
        trainer.train()
    finally:
        trainer.cleanup()
    return trainer, events


def _validated_at(events):
    return [step for kind, step, *_ in events if kind == "validate"]


@pytest.mark.parametrize(
    "steps, frequency, validated_at",
    [
        (4, 2, [0, 2, 4]),
        # The last step isn't a multiple of the frequency, and still validates.
        (5, 2, [0, 2, 4, 5]),
        # Nor is it with a frequency past the run's end.
        (3, 5, [0, 3]),
        (3, 1, [0, 1, 2, 3]),
    ],
    ids=["last_step_a_multiple", "last_step_not_a_multiple", "frequency_past_the_end", "every_step"],
)
def test_validates_before_training_every_frequency_steps_and_on_the_last_step(steps, frequency, validated_at):
    _, events = _train(_config(frequency, {"type": "step", "limit": steps}))
    assert _validated_at(events) == validated_at


def test_validation_counts_optimizer_steps_and_follows_the_steps_update():
    """With gradient accumulation, steps are optimizer steps, not batches; the short last window's step validates
    too. Each validation sees the weights of the step it follows."""
    _, events = _train(_config(2, {"type": "epoch", "limit": 1}, gradient_accumulation_steps=2), (1.0,) * 5)
    assert [event[:2] for event in events] == [
        ("validate", 0),
        ("step", 1),
        ("step", 2),
        ("validate", 2),
        ("step", 3),
        ("validate", 3),
    ]


def test_validation_runs_every_batch_in_eval_mode_without_gradients():
    trainer, events = _train(_config(1, {"type": "step", "limit": 1}), (10.0,), (1.0, 2.0, 6.0))
    # val_loss is the mean of the validation batches' losses.
    assert events == [("validate", 0, 3.0), ("step", 1), ("validate", 1, 3.0)]
    assert trainer.forwards == [
        (1.0, "eval", False),
        (2.0, "eval", False),
        (6.0, "eval", False),
        (10.0, "train", True),
        (1.0, "eval", False),
        (2.0, "eval", False),
        (6.0, "eval", False),
    ]
    assert trainer.mode == "train"


def test_a_frequency_of_0_never_validates():
    trainer, events = _train(_config(0, {"type": "step", "limit": 3}))
    assert _validated_at(events) == []
    assert all(mode == "train" for _, mode, _ in trainer.forwards)


@pytest.mark.parametrize(
    "val_batches, reason",
    [(None, "the dataset has no validation split"), ((), "the validation split has no batch")],
    ids=["no_split", "no_batch"],
)
def test_a_frequency_needs_a_validation_batch(val_batches, reason, expect_error):
    trainer = _StubTrainer((1.0,), val_batches, [])
    try:
        with expect_error(ValueError, f"metrics.validation.frequency: it is 2, but {reason}"):
            trainer.setup(_config(2, {"type": "step", "limit": 1}))
    finally:
        trainer.cleanup()

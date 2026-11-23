# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient accumulation in ``Trainer.train``: each optimizer step gets the mean gradient of its window's
micro-batches, windows run across epoch boundaries, a run's short last window is scaled by its own size, and
clipping sees the whole window's gradient. Device-free."""

import contextlib

import numpy as np
import pytest

from ttml.experimental.trainer import Callback, Trainer, TrainerConfig


class _Loss:
    """A micro-batch loss and its gradient; scaling the loss scales the gradient, as backward would."""

    def __init__(self, value, grad):
        self.value = value
        self.grad = grad

    def __truediv__(self, divisor):
        return _Loss(self.value / divisor, self.grad / divisor)


class _Model:
    def __init__(self, dim):
        self.grad = np.zeros(dim)


class _Device:
    def close(self):
        pass


class _LinearTrainer(Trainer):
    """Trains on ``batches``, vectors ``x`` with the loss ``sum(x)``, so a batch's gradient is ``x`` itself.

    Records the gradient each optimizer step gets, after clipping, in ``step_grads``, and the norm each clip sees
    in ``clip_norms``.
    """

    def __init__(self, batches, callbacks=None):
        super().__init__(callbacks)
        self.batches = [np.asarray(batch, dtype=np.float64) for batch in batches]
        self.step_grads = []
        self.clip_norms = []
        self.zero_grad_calls = 0

    def _load_dataloaders(self):
        return self.batches, None

    def _create_device_manager(self):
        return _Device()

    def _load_model(self):
        return _Model(len(self.batches[0]))

    def _load_optimizer(self):
        return None

    def _prepare_batch(self, batch):
        return batch

    def _forward(self, batch):
        return _Loss(float(batch.sum()), batch.copy())

    def _backward(self, loss):
        self.model.grad += loss.grad

    def _accumulate_loss(self, running, loss):
        return loss.value if running is None else running + loss.value

    def _clip_gradients(self, max_norm):
        norm = float(np.linalg.norm(self.model.grad))
        self.clip_norms.append(norm)
        if norm > max_norm:
            self.model.grad *= max_norm / norm

    def _optimizer_step(self):
        self.step_grads.append(self.model.grad.copy())
        self.model.grad[:] = 0.0

    def _zero_grad(self):
        self.zero_grad_calls += 1
        self.model.grad[:] = 0.0

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


class _StepRecorder(Callback):
    """Records the gradient ``on_optimizer_step_start`` sees and the ``window_loss`` of every step."""

    def __init__(self):
        self.grads_at_step_start = []
        self.window_losses = []
        self.epochs_at_step = []

    def on_optimizer_step_start(self, trainer):
        self.grads_at_step_start.append(trainer.model.grad.copy())

    def on_optimizer_step_end(self, trainer, window_loss):
        self.window_losses.append(trainer.loss_to_float(window_loss))
        self.epochs_at_step.append(trainer.epoch)


# Distinct per batch and per component, so a batch left out of a window, or one counted twice, shows.
_BATCHES = [[1.0, -2.0], [4.0, 8.0], [-16.0, 32.0], [64.0, -128.0]]


def _config(gradient_accumulation_steps, stopping, max_grad_norm=0.0):
    return TrainerConfig(
        model={"name": "stub"},
        dataset={"id": "stub"},
        optimizer={"lr": 1e-3},
        batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        stopping_criteria=stopping,
    )


def _train(batches, config):
    recorder = _StepRecorder()
    trainer = _LinearTrainer(batches, callbacks=[recorder])
    trainer.setup(config)
    try:
        trainer.train()
        total_steps, global_step = trainer.total_steps(), trainer.global_step
    finally:
        trainer.cleanup()
    return trainer, recorder, total_steps, global_step


@pytest.mark.parametrize(
    "batches_per_epoch, accumulation_steps, stopping, windows",
    [
        # Two full windows within one epoch.
        (4, 2, {"type": "epoch", "limit": 1}, [[(0, 0), (0, 1)], [(0, 2), (0, 3)]]),
        # The middle window takes the first epoch's last batch and the second epoch's first.
        (3, 2, {"type": "epoch", "limit": 2}, [[(0, 0), (0, 1)], [(0, 2), (1, 0)], [(1, 1), (1, 2)]]),
        # The run's last window has one batch, scaled by 1 and not by 1 / 2.
        (3, 2, {"type": "epoch", "limit": 1}, [[(0, 0), (0, 1)], [(0, 2)]]),
        # Short last windows only come from an epoch limit: a step limit takes whole windows, across epochs.
        (2, 3, {"type": "step", "limit": 2}, [[(0, 0), (0, 1), (1, 0)], [(1, 1), (2, 0), (2, 1)]]),
    ],
    ids=["within_epoch", "across_epochs", "short_last_window", "step_limit_across_epochs"],
)
def test_each_step_gets_the_mean_gradient_of_its_window(batches_per_epoch, accumulation_steps, stopping, windows):
    batches = _BATCHES[:batches_per_epoch]
    trainer, recorder, total_steps, global_step = _train(batches, _config(accumulation_steps, stopping))

    expected_grads = [np.mean([batches[batch] for _, batch in window], axis=0) for window in windows]
    expected_losses = [float(np.mean([np.sum(batches[batch]) for _, batch in window])) for window in windows]
    assert total_steps == global_step == len(windows)
    assert len(trainer.step_grads) == len(windows)
    for step_grad, expected in zip(trainer.step_grads, expected_grads):
        np.testing.assert_allclose(step_grad, expected)
    np.testing.assert_allclose(recorder.window_losses, expected_losses)
    # A step falls in the epoch of its window's last batch.
    assert recorder.epochs_at_step == [window[-1][0] for window in windows]
    # Only before the first window: gradients are reset by the optimizer step, never at an epoch boundary.
    assert trainer.zero_grad_calls == 1
    assert trainer.clip_norms == []


def test_clipping_sees_the_gradient_of_the_whole_window():
    # Each scaled micro-batch gradient, [1.5, 0] and [0, 2], is within the max norm of 2; their sum, norm 2.5, isn't.
    batches = [[3.0, 0.0], [0.0, 4.0]]
    config = _config(2, {"type": "epoch", "limit": 1}, max_grad_norm=2.0)
    trainer, recorder, _, global_step = _train(batches, config)

    assert global_step == 1
    assert trainer.clip_norms == pytest.approx([2.5])
    np.testing.assert_allclose(trainer.step_grads, [[1.2, 1.6]])
    # `on_optimizer_step_start` runs before clipping.
    np.testing.assert_allclose(recorder.grads_at_step_start, [[1.5, 2.0]])


def test_clipping_sees_the_short_last_window_scaled_by_its_own_size():
    # The last window has one batch, gradient [6, 8]: norm 10, not the 5 a 1 / 2 scale would give.
    batches = [[3.0, 0.0], [0.0, 4.0], [6.0, 8.0]]
    config = _config(2, {"type": "epoch", "limit": 1}, max_grad_norm=5.0)
    trainer, _, _, global_step = _train(batches, config)

    assert global_step == 2
    assert trainer.clip_norms == pytest.approx([2.5, 10.0])
    np.testing.assert_allclose(trainer.step_grads, [[1.5, 2.0], [3.0, 4.0]])

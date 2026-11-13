# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A framework-free ``Trainer`` for the loop, callback and host-contract tests.

Batches and losses are floats, so every hook argument is easy to check; the device, model and optimizer
are recorders.
"""

from __future__ import annotations

import contextlib
import json
import os
from types import SimpleNamespace
from typing import Any, Optional, Sequence

from ttml.unified_trainer import TrainerConfig
from ttml.unified_trainer.implementations.base import Trainer


def make_config(**overrides: Any) -> TrainerConfig:
    fields = {"dataset": {"id": "sst2"}, "model": {"name": "fake"}, "optimizer": {"lr": 0.1}, "batch_size": 1}
    return TrainerConfig.model_validate({**fields, **overrides})


class FakeDevice:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("device.close")


class FakeTrainer(Trainer):
    """Trains on ``train_losses``: each batch is its own loss. ``fail_in`` names a setup step that raises."""

    def __init__(
        self,
        callbacks: Any = None,
        *,
        train_losses: Sequence[float] = (4.0, 3.0, 2.0, 1.0),
        val_losses: Optional[Sequence[float]] = (5.0, 3.0),
        fail_in: Optional[str] = None,
    ) -> None:
        super().__init__(callbacks)
        self.train_losses = list(train_losses)
        self.val_losses = None if val_losses is None else list(val_losses)
        self.fail_in = fail_in
        self.events: list[str] = []
        self.saved: dict[str, Optional[dict]] = {}
        self.loaded: list[str] = []

    def _fail(self, step: str) -> None:
        if self.fail_in == step:
            raise RuntimeError(f"{step} failed")

    def _check_config(self) -> None:
        self.events.append("check")
        self._fail("check")

    def _load_dataloaders(self):
        self.events.append("data")
        self._fail("data")
        return list(self.train_losses), self.val_losses

    def _create_device_manager(self):
        self.events.append("device.open")
        device = FakeDevice(self.events)
        self._fail("device")
        return device

    def _load_model(self):
        self.events.append("model")
        self._fail("model")
        return SimpleNamespace(mode="train")

    def _load_optimizer(self):
        return SimpleNamespace(steps=0, zero_grads=0, lr=self.config.optimizer[0].lr)

    def _prepare_batch(self, batch):
        return batch

    def _forward(self, batch):
        return batch

    def _backward(self, loss) -> None:
        pass

    def _accumulate_loss(self, running, loss):
        return loss if running is None else running + loss

    def _optimizer_step(self) -> None:
        self.optimizer.steps += 1
        self._zero_grad()

    def _zero_grad(self) -> None:
        self.optimizer.zero_grads += 1

    def _train_mode(self) -> None:
        self.model.mode = "train"

    def _eval_mode(self) -> None:
        self.model.mode = "eval"

    def _no_grad(self):
        return contextlib.nullcontext()

    def loss_to_float(self, loss) -> float:
        return float(loss)

    def parameter_counts(self) -> tuple[int, int]:
        return 10, 2

    def current_learning_rate(self) -> list[float]:
        return [self.optimizer.lr]

    def save_checkpoint(self, path: str, metadata: Optional[dict] = None, optimizer: bool = False) -> None:
        """Writes the metadata and, with ``optimizer``, the optimizer's step count, for another trainer to load."""
        os.makedirs(path, exist_ok=True)
        state = {"metadata": metadata or {}}
        if optimizer:
            state["optimizer_steps"] = self.optimizer.steps
        with open(os.path.join(path, "state.json"), "w") as f:
            json.dump(state, f)
        self.saved[path] = metadata

    def load_checkpoint(self, path: str, optimizer: bool = False) -> dict:
        with open(os.path.join(path, "state.json")) as f:
            state = json.load(f)
        if optimizer:
            if "optimizer_steps" not in state:
                raise FileNotFoundError(f"{path} has no optimizer state")
            self.optimizer.steps = state["optimizer_steps"]
        self.loaded.append(path)
        return state["metadata"]


class Recorder:
    """A callback that records every hook call as ``(hook, args)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def __getattr__(self, name: str):
        if not name.startswith("on_"):
            raise AttributeError(name)
        return lambda trainer, *args: self.calls.append((name, args))

    def hooks(self) -> list[str]:
        return [hook for hook, _ in self.calls]

    def args(self, hook: str) -> list[tuple]:
        return [args for name, args in self.calls if name == hook]


class RaiseAt:
    """A callback that raises ``exception`` from ``hook`` once ``trainer.global_step`` reaches ``step``."""

    def __init__(self, hook: str, step: int, exception: BaseException) -> None:
        self.hook, self.step, self.exception = hook, step, exception

    def __getattr__(self, name: str):
        if name != self.hook:
            raise AttributeError(name)

        def hook(trainer, *args):
            if trainer.global_step >= self.step:
                raise self.exception

        return hook

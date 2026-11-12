# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Iterable, Union


class Callback:
    """Hooks into ``Trainer.train`` and ``Trainer.validate``, called in the order listed here, then the hooks
    for a failed ``Trainer.setup`` or ``Trainer.train``.

    Every hook is a no-op, so a subclass overrides only what it needs. ``CallbackHandler`` calls a hook only
    on callbacks that have it, so a callback need not subclass this at all.

    A hook may raise to end training: the exception goes to every callback's ``on_train_error`` and then out
    of ``train()``.
    """

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_epoch_start(self, trainer: Any) -> None:
        pass

    def on_train_batch_start(self, trainer: Any, batch: Any) -> None:
        pass

    def on_forward_start(self, trainer: Any, batch: Any) -> None:
        pass

    def on_forward_end(self, trainer: Any, loss: Any) -> None:
        pass

    def on_backward_start(self, trainer: Any, loss: Any) -> None:
        pass

    def on_backward_end(self, trainer: Any, loss: Any) -> None:
        pass

    def on_optimizer_step_start(self, trainer: Any) -> None:
        pass

    def on_optimizer_step_end(self, trainer: Any, window_loss: Any) -> None:
        """``window_loss`` is the mean loss of the step's micro-batches; read it with ``trainer.loss_to_float``."""

    def on_step_end(self, trainer: Any, global_step: int) -> None:
        pass

    def on_train_batch_end(self, trainer: Any) -> None:
        pass

    def on_train_epoch_end(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Only after the loop completes; a failed or stopped run calls ``on_train_error`` instead."""

    def on_validation_start(self, trainer: Any) -> None:
        pass

    def on_validation_batch_start(self, trainer: Any, batch: Any) -> None:
        pass

    def on_validation_batch_end(self, trainer: Any, batch: Any, loss: Any) -> None:
        pass

    def on_validation_end(self, trainer: Any, val_loss: Any) -> None:
        pass

    def on_setup_error(self, trainer: Any, exception: BaseException) -> None:
        """``setup`` failed and will re-raise ``exception``; ``trainer.logger`` is ``None`` if it failed first."""

    def on_train_error(self, trainer: Any, exception: BaseException) -> None:
        """``train`` failed or a hook stopped it, and ``train`` will re-raise ``exception``."""


class CallbackHandler:
    """Calls a hook on every callback that has it, in order."""

    def __init__(self, trainer: Any, callbacks: Union[Any, Iterable[Any], None] = None) -> None:
        self.trainer = trainer
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.callbacks = list(callbacks)

    def __call__(self, hook: str, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if method is not None:
                method(self.trainer, *args, **kwargs)

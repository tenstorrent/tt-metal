# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Base callback class shared by all ttml trainers."""

from __future__ import annotations

from typing import Any


class TrainerCallback:
    """Base class for trainer callbacks.

    Override any subset of hooks to customise training behaviour.
    All methods are no-ops by default.
    """

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        pass

    def on_eval_end(self, trainer: Any, step: int, eval_loss: float) -> None:
        pass

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

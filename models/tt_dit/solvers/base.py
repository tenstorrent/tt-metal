# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    from collections.abc import Sequence


class Solver(ABC):
    def __init__(self) -> None:
        self._sigmas = None
        self._alphas = None

    def set_schedule(self, sigmas: Sequence[float], alphas: Sequence[float]) -> None:
        """Set the noise and signal schedules.

        Args:
            sigmas: Full noise schedule (length = number of steps + 1).
            alphas: Full signal schedule (length = number of steps + 1).
        """
        self._sigmas = sigmas
        self._alphas = alphas

    @abstractmethod
    def step(self, *, step: int, latent: ttnn.Tensor, velocity_pred: ttnn.Tensor) -> ttnn.Tensor:
        """Advance the latent one step toward the clean data.

        Args:
            step: Current step index into the sigmas/alphas schedule.
            latent: Noisy latent at the current step.
            velocity_pred: Predicted velocity at the current step.

        Returns:
            The predicted latent at the next step.
        """

    def _assert_schedule(self) -> None:
        """Assert that the noise and signal schedules have been set."""
        if self._sigmas is None or self._alphas is None:
            msg = "schedule must be set before stepping"
            raise ValueError(msg)

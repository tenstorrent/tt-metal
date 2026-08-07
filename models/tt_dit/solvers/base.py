# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import ttnn

from .schedule import Schedule


class Solver(ABC):
    """On-device solver for one denoising trajectory.

    A solver owns the scheduler defining its discretization (where it has one) and the
    `Schedule` derived from it, so the schedule has a single owner across a run.
    """

    def __init__(self, *, scheduler: Any = None) -> None:
        self._scheduler = scheduler
        self._schedule = None

    @property
    def scheduler(self) -> Any:
        """The scheduler this solver was built from, or `None` if it has none."""
        return self._scheduler

    @property
    def schedule(self) -> Schedule:
        if self._schedule is None:
            msg = "schedule must be set before stepping"
            raise ValueError(msg)
        return self._schedule

    @property
    def sigmas(self) -> tuple[float, ...]:
        return self.schedule.sigmas

    @property
    def alphas(self) -> tuple[float, ...]:
        return self.schedule.alphas

    @property
    def timesteps(self) -> tuple[float, ...]:
        return self.schedule.timesteps

    @abstractmethod
    def set_schedule(
        self, num_inference_steps: int | None = None, *, shift: float | None = None, **kwargs: Any
    ) -> None:
        """Derive and adopt the schedule for one denoising run.

        Args:
            num_inference_steps: Number of denoising steps.
            shift: Flow shift for this run only; the construction-time value is used when
                omitted. Each solver maps this onto its own scheduler's spelling.
            kwargs: Solver-specific schedule arguments.
        """

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

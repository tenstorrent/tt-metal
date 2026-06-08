# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin

import ttnn


class Solver(ABC):
    def __init__(self, scheduler: SchedulerMixin) -> None:
        if not isinstance(scheduler, SchedulerMixin):
            msg = f"scheduler must be a diffusers SchedulerMixin, got {type(scheduler).__name__}"
            raise ValueError(msg)
        self._scheduler = scheduler
        self._sigmas = None
        self._alphas = None
        self._timesteps = None

    @property
    def scheduler(self) -> SchedulerMixin:
        return self._scheduler

    @property
    def sigmas(self) -> list[float] | None:
        """Returns the active sigma schedule, or None when no schedule has been provided."""
        return self._sigmas

    @property
    def alphas(self) -> list[float] | None:
        """Returns the active alpha schedule, or None when no schedule has been provided."""
        return self._alphas

    @property
    def timesteps(self) -> torch.Tensor | None:
        """Returns the active timesteps, or None when no timesteps have been provided."""
        return self._timesteps

    def set_schedule(self, num_inference_steps: int | None = None, *, device: object = None, **kwargs: object) -> None:
        """Forward to ``scheduler.set_timesteps`` and cache sigmas/alphas for device stepping."""
        self._scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        sigmas = self._scheduler.sigmas
        self._sigmas = sigmas.tolist()
        self._alphas = (1.0 - sigmas).tolist()
        self._timesteps = self._scheduler.timesteps

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
        if self._sigmas is None or self._alphas is None:
            msg = "schedule must be set before stepping"
            raise ValueError(msg)

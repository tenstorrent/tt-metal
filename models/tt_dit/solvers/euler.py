# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ttnn

from .base import Solver
from .schedule import Schedule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class EulerSolver(Solver):
    def __init__(self, *, scheduler: FlowMatchEulerDiscreteScheduler | None = None) -> None:
        super().__init__(scheduler=scheduler)

        # Only meaningful with a scheduler; without one every schedule is taken as given.
        self._default_shift = scheduler.shift if scheduler is not None else None

    def set_schedule(
        self,
        num_inference_steps: int | None = None,
        *,
        sigmas: Sequence[float] | None = None,
        shift: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Derive and adopt the schedule for one denoising run.

        Args:
            num_inference_steps: Number of denoising steps.
            sigmas: Explicit sigma schedule. Required when the solver has no scheduler, and
                then taken exactly as given; otherwise forwarded to `set_timesteps`.
            shift: Flow shift for this run only; the scheduler's construction-time value is
                used when omitted. Only accepted when the solver has a scheduler.
            kwargs: Forwarded to the scheduler's `set_timesteps`.
        """
        if self._scheduler is None:
            if sigmas is None or num_inference_steps is not None or shift is not None or kwargs:
                msg = "a scheduler-less EulerSolver accepts only `sigmas`"
                raise ValueError(msg)
            self._schedule = Schedule.from_sigmas(sigmas)
            return

        # set_shift writes `_shift`, which is what set_timesteps reads. The config copy is
        # currently unused, but keep it in sync so anything re-deriving from the config
        # (from_config, save_config) sees the shift actually in effect.
        value = self._default_shift if shift is None else shift
        self._scheduler.set_shift(value)
        self._scheduler.register_to_config(shift=value)
        self._schedule = Schedule.from_scheduler(self._scheduler, num_inference_steps, sigmas=sigmas, **kwargs)

    def step(self, *, step: int, latent: ttnn.Tensor, velocity_pred: ttnn.Tensor) -> ttnn.Tensor:
        sigma_curr = self.sigmas[step]
        sigma_next = self.sigmas[step + 1]

        return latent + (sigma_next - sigma_curr) * velocity_pred

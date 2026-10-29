# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_NUM_TRAIN_TIMESTEPS = 1000


@dataclass(frozen=True)
class Schedule:
    """The discretization a solver steps along for one denoising run.

    Attributes:
        sigmas: Noise schedule (length = number of steps + 1, terminating at 0).
        alphas: Signal schedule (length = number of steps + 1).
        timesteps: Model conditioning timesteps (length = number of steps).
        num_train_timesteps: Training step count the timesteps are scaled to.
    """

    sigmas: tuple[float, ...]
    alphas: tuple[float, ...]
    timesteps: tuple[float, ...]
    num_train_timesteps: int = _DEFAULT_NUM_TRAIN_TIMESTEPS

    def __len__(self) -> int:
        return len(self.timesteps)

    @classmethod
    def from_sigmas(
        cls,
        sigmas: Sequence[float],
        *,
        num_train_timesteps: int = _DEFAULT_NUM_TRAIN_TIMESTEPS,
    ) -> Schedule:
        """Build a schedule from an explicit sigma schedule, with no scheduler involved.

        Args:
            sigmas: Full noise schedule (length = number of steps + 1), taken as given.
            num_train_timesteps: Training step count the timesteps are scaled to.
        """
        sigmas = tuple(float(s) for s in sigmas)
        return cls(
            sigmas=sigmas,
            alphas=tuple(1.0 - s for s in sigmas),
            timesteps=tuple(s * num_train_timesteps for s in sigmas[:-1]),
            num_train_timesteps=num_train_timesteps,
        )

    @classmethod
    def from_scheduler(
        cls,
        scheduler: Any,
        num_inference_steps: int | None = None,
        **set_timesteps_kwargs: Any,
    ) -> Schedule:
        """Build a schedule by discretizing a diffusers scheduler.

        Args:
            scheduler: Scheduler to discretize.
            num_inference_steps: Number of denoising steps.
            set_timesteps_kwargs: Forwarded verbatim to `scheduler.set_timesteps`.
        """
        scheduler.set_timesteps(num_inference_steps, **set_timesteps_kwargs)

        sigmas = tuple(float(s) for s in scheduler.sigmas.tolist())
        return cls(
            sigmas=sigmas,
            alphas=tuple(1.0 - s for s in sigmas),
            timesteps=tuple(float(t) for t in scheduler.timesteps.tolist()),
            num_train_timesteps=scheduler.config.num_train_timesteps,
        )


def calculate_shift(image_seq_len: int, scheduler: FlowMatchEulerDiscreteScheduler) -> float:
    """Resolution-dependent mu used by FlowMatchEulerDiscreteScheduler's dynamic shifting."""
    base_seq_len = scheduler.config.get("base_image_seq_len", 256)
    max_seq_len = scheduler.config.get("max_image_seq_len", 4096)
    base_shift = scheduler.config.get("base_shift", 0.5)
    max_shift = scheduler.config.get("max_shift", 1.15)

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b

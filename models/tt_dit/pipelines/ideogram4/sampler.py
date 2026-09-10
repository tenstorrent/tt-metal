# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ideogram 4.0 Euler flow-matching sampler.

Faithful port of the reference ``scheduler.py`` + ``sampler_configs.py``. The
logit-normal schedule and step intervals are host-side math (identical to the
reference); only the per-step Euler update ``z <- z + v * (s - t)`` runs on
device. Asymmetric CFG (``v = gw*v_cond + (1-gw)*v_uncond``) is applied on device
by the pipeline; this module owns the schedule, the guidance-weight sequence, and
the device step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn


@dataclass(frozen=True)
class LogitNormalSchedule:
    """Reference LogitNormalSchedule: maps uniform t in [0,1] to a flow-matching time."""

    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(torch.float64)
        z = torch.special.ndtri(t)
        y = self.mean + self.std * z
        t_ = torch.special.expit(y)
        t_ = 1 - t_
        t_min = 1.0 / (1 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * self.logsnr_min))
        return t_.clamp(t_min, t_max).to(torch.float32)


def get_schedule_for_resolution(
    image_resolution: tuple[int, int],
    known_resolution: tuple[int, int] = (512, 512),
    known_mean: float = 1.0,
    std: float = 1.0,
) -> LogitNormalSchedule:
    """Resolution-aware schedule: mean shifts by 0.5*log(num_pixels / known_pixels)."""
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)


@dataclass(frozen=True, kw_only=True)
class SamplerParameters:
    num_steps: int
    guidance_schedule: tuple[float, ...]
    mu: float
    std: float = 1.0

    def __post_init__(self) -> None:
        if len(self.guidance_schedule) != self.num_steps:
            raise ValueError(
                f"guidance_schedule has length {len(self.guidance_schedule)}, expected num_steps={self.num_steps}"
            )


# guidance_schedule is in loop-INDEX order: index 0 is the LAST (polish) step.
PRESETS: dict[str, SamplerParameters] = {
    "V4_QUALITY_48": SamplerParameters(num_steps=48, guidance_schedule=(3.0,) * 3 + (7.0,) * 45, mu=0.0, std=1.5),
    "V4_DEFAULT_20": SamplerParameters(num_steps=20, guidance_schedule=(3.0,) * 2 + (7.0,) * 18, mu=0.0, std=1.75),
    "V4_TURBO_12": SamplerParameters(num_steps=12, guidance_schedule=(3.0,) * 1 + (7.0,) * 11, mu=0.5, std=1.75),
}


class Ideogram4Sampler:
    """Owns the logit-normal schedule, guidance-weight sequence, and the device Euler step.

    The denoising loop runs ``for i in reversed(range(num_steps))``; at step ``i`` the
    flow-matching times are ``t = schedule(intervals[i+1])`` and ``s = schedule(intervals[i])``
    and the Euler update is ``z <- z + v * (s - t)``.
    """

    def __init__(self, params: SamplerParameters, *, height: int, width: int) -> None:
        self.params = params
        self.schedule = get_schedule_for_resolution((height, width), known_mean=params.mu, std=params.std)
        self.step_intervals = make_step_intervals(params.num_steps)

    @classmethod
    def from_preset(cls, name: str, *, height: int, width: int) -> "Ideogram4Sampler":
        return cls(PRESETS[name], height=height, width=width)

    @property
    def num_steps(self) -> int:
        return self.params.num_steps

    def times_for_step(self, i: int) -> tuple[float, float]:
        """Return (t, s) for loop index i — the current and next flow-matching times."""
        t_val = float(self.schedule(self.step_intervals[i + 1].unsqueeze(0)).item())
        s_val = float(self.schedule(self.step_intervals[i].unsqueeze(0)).item())
        return t_val, s_val

    def guidance_weight(self, i: int) -> float:
        return float(self.params.guidance_schedule[i])

    def step(self, z: ttnn.Tensor, velocity: ttnn.Tensor, i: int | None = None, *, scale=None) -> ttnn.Tensor:
        """Device Euler update: z <- z + velocity * (s - t).

        ``scale`` is the step size ``s - t``. When omitted it is computed from ``times_for_step(i)``
        (Python float — used by the standalone sampler test). The traced pipeline passes ``scale``
        as a 1-element DEVICE tensor instead (so the same scalar can be re-patched per step inside
        one reused trace, where a Python float would be baked in) — routing production through here.
        """
        if scale is None:
            t_val, s_val = self.times_for_step(i)
            scale = s_val - t_val
        return z + velocity * scale

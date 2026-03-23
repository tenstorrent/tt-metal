# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import NamedTuple


class Schedule(NamedTuple):
    sigmas: list[float]
    alphas: list[float]


def linear(num_steps: int, *, sigma_small: float | None = None) -> Schedule:
    """Linear flow-matching schedule.

    Produces num_steps sigmas evenly spaced from 1.0 to sigma_small, plus a terminal 0
    (num_steps + 1 total). alpha = 1 - sigma at each point.

    When sigma_small is None, produces uniform spacing from 1.0 to 0.0.
    """
    if sigma_small is None:
        sigma_small = 1 / num_steps

    sigmas = [1.0 - i * (1.0 - sigma_small) / (num_steps - 1) for i in range(num_steps)]
    sigmas.append(0.0)
    alphas = [1 - s for s in sigmas]
    return Schedule(sigmas, alphas)


def shifted_linear(num_steps: int, *, shift: float, sigma_small: float | None = None) -> Schedule:
    """Shifted linear flow-matching schedule."""

    def _shift(t: float) -> float:
        return shift * t / (1 + (shift - 1) * t)

    base = linear(num_steps, sigma_small=sigma_small)
    sigmas = [_shift(t) for t in base.sigmas[:-1]]
    sigmas.append(0.0)
    alphas = [1 - s for s in sigmas]
    return Schedule(sigmas, alphas)


def linear_quadratic(
    num_steps: int,
    *,
    threshold_noise: float = 0.025,
    linear_steps: int | None = None,
) -> Schedule:
    """Piecewise linear-quadratic schedule."""
    if linear_steps is None:
        linear_steps = num_steps // 2

    linear_part = [i * threshold_noise / linear_steps for i in range(linear_steps)]

    quadratic_steps = num_steps - linear_steps
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_part = [quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)]

    # sigma = 1 - schedule_value, so high schedule values -> low noise
    sigmas = [1.0 - x for x in linear_part + quadratic_part]
    # Add the clean endpoint.
    sigmas.append(0.0)
    alphas = [1 - s for s in sigmas]

    return Schedule(sigmas, alphas)

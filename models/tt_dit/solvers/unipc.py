# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import ttnn

from .base import Solver
from .schedule import Schedule

if TYPE_CHECKING:
    from collections.abc import Sequence

    from diffusers.schedulers import UniPCMultistepScheduler


@dataclass(frozen=True)
class _State:
    clean_preds: tuple[ttnn.Tensor, ...]
    corrected: ttnn.Tensor
    oldest_idx: int = 0


class UniPCVariant(Enum):
    BH1 = "bh1"  # B(h) = h
    BH2 = "bh2"  # B(h) = 1 - e^(-h)

    def b(self, h: float) -> float:
        if self is UniPCVariant.BH1:
            return h
        return -math.expm1(-h)


class UniPCSolver(Solver):
    def __init__(self, *, order: int, variant: UniPCVariant, scheduler: UniPCMultistepScheduler) -> None:
        if order not in (1, 2):
            msg = f"only order 1 and 2 are supported, got {order}"
            raise ValueError(msg)

        super().__init__(scheduler=scheduler)

        self.order = order
        self.variant = variant
        self._state = None
        self._default_shift = scheduler.config.flow_shift

    def set_schedule(
        self, num_inference_steps: int | None = None, *, shift: float | None = None, **kwargs: Any
    ) -> None:
        """Derive and adopt the schedule for one denoising run.

        Args:
            num_inference_steps: Number of denoising steps.
            shift: Flow shift for this run only; the scheduler's construction-time value
                is used when omitted.
            kwargs: Forwarded to the scheduler's `set_timesteps`.
        """
        # UniPC reads the shift from its config rather than from set_timesteps.
        self._scheduler.register_to_config(flow_shift=self._default_shift if shift is None else shift)
        self._schedule = Schedule.from_scheduler(self._scheduler, num_inference_steps, **kwargs)

        if self._state is not None:
            self._state = _State(self._state.clean_preds, self._state.corrected, 0)

    def step(self, *, step: int, latent: ttnn.Tensor, velocity_pred: ttnn.Tensor) -> ttnn.Tensor:
        clean_pred = latent - self.sigmas[step] * velocity_pred

        state = self._state or _State(
            tuple(ttnn.empty_like(latent) for _ in range(self.order)),
            ttnn.empty_like(latent),
        )
        clean_preds = _ordered_clean_preds(state.clean_preds, state.oldest_idx)

        if step != 0:
            corrected = self._correct(
                order=_taper(self.order, step - 1, len(self.sigmas) - 1),
                latent=state.corrected,
                step=step - 1,
                clean_preds=(*clean_preds, clean_pred),
            )
        else:
            corrected = latent

        ttnn.copy(corrected, state.corrected)
        del corrected

        ttnn.copy(clean_pred, state.clean_preds[state.oldest_idx])
        oldest_idx = (state.oldest_idx + 1) % self.order
        clean_preds = _ordered_clean_preds(state.clean_preds, oldest_idx)
        del clean_pred

        predicted = self._predict(
            order=_taper(self.order, step, len(self.sigmas) - 1),
            latent=state.corrected,
            step=step,
            clean_preds=clean_preds,
        )

        self._state = _State(state.clean_preds, state.corrected, oldest_idx)
        return predicted

    def _predict(
        self, *, order: int, latent: ttnn.Tensor, step: int, clean_preds: Sequence[ttnn.Tensor]
    ) -> ttnn.Tensor:
        sigma_curr, sigma_next = self.sigmas[step : step + 2]
        alpha_curr, alpha_next = self.alphas[step : step + 2]

        lam_curr = _log(alpha_curr) - _log(sigma_curr)
        lam_next = _log(alpha_next) - _log(sigma_next)
        h = lam_next - lam_curr

        coeff_latent = sigma_next / sigma_curr
        coeff_curr = -alpha_next * math.expm1(-h)

        latent = coeff_latent * latent + coeff_curr * clean_preds[-1]

        if order == 1:
            return latent

        lam_prev = _log(self.alphas[step - 1]) - _log(self.sigmas[step - 1])
        r = (lam_curr - lam_prev) / h
        w = alpha_next * self.variant.b(h) * 0.5 / r

        return latent + w * (clean_preds[-1] - clean_preds[-2])

    def _correct(
        self, *, order: int, latent: ttnn.Tensor, step: int, clean_preds: Sequence[ttnn.Tensor]
    ) -> ttnn.Tensor:
        sigma_curr, sigma_next = self.sigmas[step : step + 2]
        alpha_curr, alpha_next = self.alphas[step : step + 2]

        lam_curr = _log(alpha_curr) - _log(sigma_curr)
        lam_next = _log(alpha_next) - _log(sigma_next)
        h = lam_next - lam_curr

        coeff_latent = sigma_next / sigma_curr
        coeff_clean = -alpha_next * math.expm1(-h)

        latent = coeff_latent * latent + coeff_clean * clean_preds[-2]

        if order == 1:
            # UniC-1: c=0.5, r=1
            w = alpha_next * self.variant.b(h) * 0.5
            return latent + w * (clean_preds[-1] - clean_preds[-2])

        # UniC-2: solve 2x2 system
        exp_neg_h = math.expm1(-h)

        lam_prev = _log(self.alphas[step - 1]) - _log(self.sigmas[step - 1])
        r_1 = (lam_curr - lam_prev) / h

        g1 = (h + exp_neg_h) / h**2
        g2 = (h**2 - 2 * h - 2 * exp_neg_h) / h**2

        if math.isinf(r_1):
            w_prev = 0.0
            w_pred = alpha_next * h * g1
        else:
            det = h * (1 + r_1)
            c_1 = (h * g1 - g2) / det
            c_2 = (g2 + r_1 * h * g1) / det

            w_prev = alpha_next * h * c_1 / r_1
            w_pred = alpha_next * h * c_2

        return latent + w_prev * (clean_preds[-2] - clean_preds[-3]) + w_pred * (clean_preds[-1] - clean_preds[-2])


def _taper(order: int, step: int, num_steps: int) -> int:
    return min(order, step + 1, num_steps - step)


def _ordered_clean_preds(clean_preds: tuple[ttnn.Tensor, ...], oldest_idx: int) -> tuple[ttnn.Tensor, ...]:
    return clean_preds[oldest_idx:] + clean_preds[:oldest_idx]


def _log(x: float, /) -> float:
    return math.log(x) if x != 0 else -math.inf

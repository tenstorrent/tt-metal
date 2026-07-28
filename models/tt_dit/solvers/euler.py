# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ttnn

from .base import Solver


class EulerSolver(Solver):
    def step(self, *, step: int, latent: ttnn.Tensor, velocity_pred: ttnn.Tensor) -> ttnn.Tensor:
        self._assert_schedule()

        sigma_curr = self._sigmas[step]
        sigma_next = self._sigmas[step + 1]

        return latent + (sigma_next - sigma_curr) * velocity_pred

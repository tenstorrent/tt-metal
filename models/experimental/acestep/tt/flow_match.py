# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACE-Step v1.5 flow-matching solver step (TTNN elementwise ops).

Reference: the denoise loop in AceStepConditionGenerationModel.generate_audio:
    xt        = xt - vt * dt          # Euler integration step (dt = t_curr - t_prev)
    x0_pred   = zt - vt * t           # clean-sample estimate (get_x0_from_noise)

vt is the DiT's velocity prediction (validated separately at PCC>=0.999). This module implements
the solver's elementwise device math so the full inference loop's compute runs on-device without
falling back to host. Deterministic (no learnable weights).

CFG guidance (apg_forward/adg_forward) is intentionally excluded: apg_guidance.py is absent from
the HF snapshot (it powers guidance sampling, not the core update) and would only combine two
DiT predictions elementwise — orthogonal to the solver step validated here.
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule


class FlowMatchStep(LightweightModule):
    """Stateless flow-matching Euler solver step. No weights."""

    def __init__(self, mesh_device: ttnn.MeshDevice | None = None):
        self.mesh_device = mesh_device

    def euler_step(self, xt, vt, dt: float):
        """xt <- xt - vt * dt. dt is the (scalar) step size t_curr - t_prev."""
        return ttnn.sub(xt, ttnn.mul(vt, dt))

    def x0_from_noise(self, zt, vt, t: float):
        """Clean-sample estimate: x0 = zt - vt * t (t scalar for the current step)."""
        return ttnn.sub(zt, ttnn.mul(vt, t))

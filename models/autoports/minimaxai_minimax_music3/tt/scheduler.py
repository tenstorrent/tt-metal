# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FlowMatchEuler scheduler for MiniMax-Music3's DiT, as diffusers configures it.

Transcribed from diffusers ``schedulers/scheduling_flow_match_euler_discrete.py`` (Apache-2.0) for
the checkpoint's ``scheduler/scheduler_config.json``: ``num_train_timesteps=1``, ``shift=1.0``,
``invert_sigmas=True``, no dynamic shifting / karras / exponential / beta sigmas, no stochastic
sampling, and the pipeline's ``set_timesteps(sigmas=np.linspace(1, 1/N, N))`` call.

With those settings ``set_timesteps`` reduces to (all float32 arithmetic, as in diffusers)::

    s          = float32(linspace(1, 1/N, N))          # shift == 1 -> unchanged
    sigmas     = 1 - s                                  # invert_sigmas
    timesteps  = sigmas * num_train_timesteps (= 1)     # so timesteps == sigmas[:-1]
    sigmas     = cat(sigmas, [1.0])

and ``step(v, t, x)`` is the Euler update ``x + (sigmas[i + 1] - sigmas[i]) * v`` computed in
float32 and cast back to ``v``'s dtype. The DiT consumes ``t`` directly (0 = noise, 1 = data).

The host-side torch here is deliberately kept on the host: one axpy over ``[1, 128, T]`` per step.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

NUM_TRAIN_TIMESTEPS = 1


class FlowMatchEulerScheduler:
    """``FlowMatchEulerDiscreteScheduler`` restricted to MiniMax-Music3's configuration."""

    def __init__(self, num_inference_steps: int):
        assert num_inference_steps >= 1, num_inference_steps
        self.num_inference_steps = int(num_inference_steps)
        self.set_timesteps()

    # ------------------------------------------------------------------ schedule
    def set_timesteps(self, num_inference_steps: Optional[int] = None) -> None:
        if num_inference_steps is not None:
            self.num_inference_steps = int(num_inference_steps)
        n = self.num_inference_steps
        sigmas = np.linspace(1.0, 1.0 / n, n).astype(np.float32)
        # shift == 1: sigmas = shift * sigmas / (1 + (shift - 1) * sigmas) is the identity.
        sigmas = torch.from_numpy(sigmas).to(torch.float32)
        timesteps = sigmas * NUM_TRAIN_TIMESTEPS
        # invert_sigmas
        sigmas = 1.0 - sigmas
        timesteps = sigmas * NUM_TRAIN_TIMESTEPS
        sigmas = torch.cat([sigmas, torch.ones(1)])
        self.timesteps = timesteps  # [N]
        self.sigmas = sigmas  # [N + 1]
        self._step_index: Optional[int] = None

    @property
    def step_index(self) -> Optional[int]:
        return self._step_index

    def index_for_timestep(self, timestep) -> int:
        t = float(timestep)
        idx = (self.timesteps == t).nonzero()
        if idx.numel() == 0:
            raise ValueError(f"timestep {t} is not in the schedule {self.timesteps.tolist()}")
        # diffusers: the second match if there are several (there never are here), else the first.
        pos = 1 if idx.numel() > 1 else 0
        return int(idx[pos])

    # ------------------------------------------------------------------ update
    def step(self, model_output: torch.Tensor, timestep, sample: torch.Tensor) -> torch.Tensor:
        """One Euler step: ``sample + (sigma_next - sigma) * model_output`` (float32 math, cast to ``model_output.dtype``)."""
        if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor) and not timestep.is_floating_point()):
            raise ValueError("pass one of `scheduler.timesteps`, not an integer index")
        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep)
        sample32 = sample.to(torch.float32)
        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        dt = sigma_next - sigma
        prev_sample = sample32 + dt * model_output.to(torch.float32)
        self._step_index += 1
        return prev_sample.to(model_output.dtype)

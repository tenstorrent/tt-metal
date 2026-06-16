# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN port of HunyuanImage-3.0's FlowMatchDiscreteScheduler (Euler solver).
#
# Mirrors ref/scheduler.py but is torch-free: the sigma/timestep SCHEDULE is a
# handful of scalar values, computed once on host with numpy (linspace + the
# sd3/flux shift) — identical math to the reference, no torch dependency.
#
# The two per-denoising-step operations that touch the (large) latent tensor run
# on device in float32:
#   * Euler update:  prev = sample + (sigma_next - sigma) * model_output
#   * CFG combine:    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
#
# `sigma_next - sigma` and `guidance_scale` are python scalars at call time, so
# the device math is a scalar-scaled add (ttnn.add of a scalar multiply) — no
# host<->device round trips on the latent.

import math

import numpy as np
import ttnn


class HunyuanTtScheduler:
    """Euler flow-matching scheduler with on-device latent updates (torch-free)."""

    order = 1

    def __init__(
        self,
        device,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        reverse: bool = True,
        solver: str = "euler",
        use_flux_shift: bool = False,
        flux_base_shift: float = 0.5,
        flux_max_shift: float = 1.15,
    ):
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.reverse = reverse
        self.solver = solver
        self.use_flux_shift = use_flux_shift
        self.flux_base_shift = flux_base_shift
        self.flux_max_shift = flux_max_shift

        supported_solver = ["euler", "heun-2", "midpoint-2", "kutta-4"]
        if solver not in supported_solver:
            raise ValueError(f"Solver {solver} not supported. Supported solvers: {supported_solver}")

        sigmas = np.linspace(1, 0, num_train_timesteps + 1, dtype=np.float32)
        if not reverse:
            sigmas = sigmas[::-1].copy()
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).astype(np.float32)
        self.timesteps_full = (sigmas * num_train_timesteps).astype(np.float32)

        self._step_index = None
        self._begin_index = None

    # ------------------------------------------------------- schedule (host)
    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def get_timestep_r(self, timestep):
        if self.step_index is None:
            self._init_step_index(timestep)
        return self.timesteps_full[self.step_index + 1]

    def set_timesteps(self, num_inference_steps: int, device=None, n_tokens: int = None):
        self.num_inference_steps = num_inference_steps

        sigmas = np.linspace(1, 0, num_inference_steps + 1, dtype=np.float32)

        if self.use_flux_shift:
            assert isinstance(n_tokens, int), "n_tokens should be provided for flux shift"
            mu = self.get_lin_function(y1=self.flux_base_shift, y2=self.flux_max_shift)(n_tokens)
            sigmas = self.flux_time_shift(mu, 1.0, sigmas)
        elif self.shift != 1.0:
            sigmas = self.sd3_time_shift(sigmas)

        if not self.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas.astype(np.float32)
        self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).astype(np.float32)
        self.timesteps_full = (self.sigmas * self.num_train_timesteps).astype(np.float32)
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = np.nonzero(schedule_timesteps == np.float32(timestep))[0]
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos])

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample, timestep=None):
        return sample

    # --------------------------------------------------------------- shifts
    @staticmethod
    def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @staticmethod
    def flux_time_shift(mu: float, sigma: float, t):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def sd3_time_shift(self, t):
        return (self.shift * t) / (1 + (self.shift - 1) * t)

    # --------------------------------------------------------- step (device)
    def step(self, model_output, timestep, sample):
        """Euler update on device: prev = sample + (sigma_next - sigma) * model_output."""
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        dt = float(sigma_next - sigma)

        # derivative * dt  (Euler: derivative == model_output), then add to sample.
        scaled = ttnn.multiply(model_output, dt)
        prev_sample = ttnn.add(sample, scaled)
        ttnn.deallocate(scaled)

        self._step_index += 1
        return prev_sample


def classifier_free_guidance_tt(pred_cond, pred_uncond, guidance_scale, use_original_formulation=False):
    """On-device CFG combine: pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)."""
    shift = ttnn.subtract(pred_cond, pred_uncond)
    scaled = ttnn.multiply(shift, float(guidance_scale))
    ttnn.deallocate(shift)
    base = pred_cond if use_original_formulation else pred_uncond
    pred = ttnn.add(base, scaled)
    ttnn.deallocate(scaled)
    return pred

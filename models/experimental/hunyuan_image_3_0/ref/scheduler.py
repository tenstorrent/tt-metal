# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone PyTorch reference port of HunyuanImage-3.0's FlowMatchDiscreteScheduler.
#
# This is a faithful extraction of the scheduler used by the upstream
# HunyuanImage-3.0 generation pipeline (hunyuan_image_3_pipeline.py), with the
# diffusers `SchedulerMixin`/`ConfigMixin` machinery stripped out so it can be
# diffed numerically against the TTNN port without pulling in diffusers.
#
# Only the pieces exercised by the default text-to-image path are kept:
#   * sigma/timestep schedule construction (set_timesteps)
#   * sd3 / flux timestep shifting
#   * the Euler (first-order) solver step
#   * the classifier-free-guidance combine
#
# Defaults match generation_config.json: flow_shift=3.0, solver="euler",
# reverse=True, num_inference_steps=50, guidance_scale=5.0.

import math
from typing import Union

import torch


class FlowMatchDiscreteScheduler:
    """Euler flow-matching scheduler (reference)."""

    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        reverse: bool = True,
        solver: str = "euler",
        use_flux_shift: bool = False,
        flux_base_shift: float = 0.5,
        flux_max_shift: float = 1.15,
    ):
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

        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)
        if not reverse:
            sigmas = sigmas.flip(0)
        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)
        self.timesteps_full = (sigmas * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None
        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None

    # ------------------------------------------------------------------ state
    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    @property
    def state_in_first_order(self):
        return self.derivative_1 is None

    # ------------------------------------------------------------- schedule
    def set_timesteps(self, num_inference_steps: int, device=None, n_tokens: int = None):
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        if self.use_flux_shift:
            assert isinstance(n_tokens, int), "n_tokens should be provided for flux shift"
            mu = self.get_lin_function(y1=self.flux_base_shift, y2=self.flux_max_shift)(n_tokens)
            sigmas = self.flux_time_shift(mu, 1.0, sigmas)
        elif self.shift != 1.0:
            sigmas = self.sd3_time_shift(sigmas)

        if not self.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.num_train_timesteps).to(dtype=torch.float32, device=device)
        self.timesteps_full = (sigmas * self.num_train_timesteps).to(dtype=torch.float32, device=device)

        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        return sample

    # --------------------------------------------------------------- shifts
    @staticmethod
    def get_timestep_r(self, timestep: Union[float, torch.FloatTensor]):
        if self.step_index is None:
            self._init_step_index(timestep)
        return self.timesteps_full[self.step_index + 1]

    @staticmethod
    def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @staticmethod
    def flux_time_shift(mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.shift * t) / (1 + (self.shift - 1) * t)

    # ----------------------------------------------------------------- step
    def first_order_method(self, model_output, sigma, sigma_next, sample):
        derivative = model_output
        dt = sigma_next - sigma
        return derivative, dt, sample, True

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ):
        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `step()` is not supported. Make sure to pass one of the `scheduler.timesteps`"
                " as a timestep."
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        if self.solver == "euler":
            derivative, dt, sample, last_inner_step = self.first_order_method(model_output, sigma, sigma_next, sample)
        else:
            raise NotImplementedError(f"Solver {self.solver} not implemented in reference port")

        prev_sample = sample + derivative * dt

        if last_inner_step:
            self._step_index += 1

        if return_dict:
            return {"prev_sample": prev_sample}
        return (prev_sample,)


def classifier_free_guidance(pred_cond, pred_uncond, guidance_scale, use_original_formulation=False):
    """pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)."""
    shift = pred_cond - pred_uncond
    pred = pred_cond if use_original_formulation else pred_uncond
    pred = pred + guidance_scale * shift
    return pred

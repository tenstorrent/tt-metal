# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from loguru import logger


class TtEulerDiscreteScheduler(nn.Module):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        interpolation_type: str = "linear",
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        timestep_spacing: str = "linspace",
        timestep_type: str = "discrete",  # can be "discrete" or "continuous"
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        final_sigmas_type: str = "zero",  # can be "zero" or "sigma_min"
    ):
        # implements the Euler Discrete Scheduler with default params as in
        # https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/scheduler/scheduler_config.json
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        assert beta_schedule == "scaled_linear", "beta_schedule {beta_schedule} is not supported in this version"
        self.beta_schedule = beta_schedule
        assert trained_betas is None, "trained_betas is not supported in this version"
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        assert prediction_type == "epsilon", "prediction_type {prediction_type} is not supported in this version"
        self.prediction_type = prediction_type
        assert (
            interpolation_type == "linear"
        ), "interpolation_type {interpolation_type} is not supported in this version"
        self.interpolation_type = interpolation_type
        assert use_karras_sigmas == False, "karras sigmas are not supported in this version"
        assert use_exponential_sigmas == False, "exponential sigmas are not supported in this version"
        assert use_beta_sigmas == False, "beta sigmas are not supported in this version"
        assert sigma_min is None, "sigma_min is not supported in this version"
        assert sigma_max is None, "sigma_max is not supported in this version"
        assert timestep_spacing == "leading", "timestep_spacing {timestep_spacing} is not supported in this version"
        self.timestep_spacing = timestep_spacing
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
        self.timesteps = timesteps
        assert timestep_type == "discrete", "timestep_type {timestep_type} is not supported in this version"
        self.timestep_type = timestep_type
        self.steps_offset = steps_offset
        assert rescale_betas_zero_snr == False, "rescale_betas_zero_snr is not supported in this version"
        assert final_sigmas_type == "zero", "final_sigmas_type {final_sigmas_type} is not supported in this version"

        self.final_sigmas_type = final_sigmas_type
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.num_inference_steps = None
        sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
        self.sigmas = torch.cat([sigmas, torch.zeros(1)])
        self.is_scale_input_called = False
        self._step_index = None
        self._begin_index = None

    @property
    def init_noise_sigma(self):
        """
        standard deviation of the initial noise distribution.
        """
        max_sigma = max(self.sigmas) if isinstance(self.sigmas, list) else self.sigmas.max()
        if self.timestep_spacing in ["linspace", "trailing"]:
            return max_sigma

        return (max_sigma**2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sample / ((sigma**2 + 1) ** 0.5)

        self.is_scale_input_called = True
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        """
        assert timesteps == None, "timesteps is not supported in this version"
        assert sigmas == None, "sigmas is not supported in this version"
        assert num_inference_steps != None, "num_inference_steps cannot be None in this version"

        self.num_inference_steps = num_inference_steps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        assert self.timestep_spacing == "leading"
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
        timesteps += self.steps_offset

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)
        assert self.interpolation_type == "linear"
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        assert self.final_sigmas_type == "zero"
        sigma_last = 0

        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        assert self.timestep_type == "discrete"
        self.timesteps = torch.from_numpy(timesteps.astype(np.float32)).to(device=device)

        self._step_index = None
        self._begin_index = None
        self.sigmas = sigmas

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
    ) -> Tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).
        """

        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        assert self.step_index is not None, "_init_step_index() should be None before calling step()"
        assert generator is None, "generator is not supported in this version"
        assert return_dict == False, "return_dict==true is not supported in this version"

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
        assert gamma == 0, "gamma > 0 is not supported in this version"
        sigma_hat = sigma

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        assert self.prediction_type == "epsilon"
        pred_original_sample = sample - sigma_hat * model_output

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        return (prev_sample, pred_original_sample)

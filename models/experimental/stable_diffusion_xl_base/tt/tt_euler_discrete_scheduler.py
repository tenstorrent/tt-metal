# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union
import ttnn
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
import ttnn.device


class TtEulerDiscreteScheduler(nn.Module):
    def __init__(
        self,
        device: ttnn.device.Device,
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
        self.step_index = None
        self.begin_index = None
        self.device = device

        # self.update_device_tensor("sigmas")

    def set_step_index(self, step_index: int):
        self.step_index = step_index

    def update_device_tensor(self, tensor_name):
        array = getattr(self, tensor_name)
        setattr(self, "tt_" + tensor_name, [])
        tt_array = getattr(self, "tt_" + tensor_name)

        for val in array:
            tt_array.append(
                ttnn.to_memory_config(
                    ttnn.from_torch(
                        val,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    ).to(device=self.device),
                    ttnn.L1_MEMORY_CONFIG,
                ),
            )

    # pipeline_stable_diffusion_xl.py __call__() step #4
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

        self.begin_index = 0
        self.step_index = self.begin_index

        self.sigmas = sigmas
        self.variance_normalization_factor = (sigmas**2 + 1) ** 0.5

        self.update_device_tensor("sigmas")
        self.update_device_tensor("timesteps")
        self.update_device_tensor("variance_normalization_factor")

    # pipeline_stable_diffusion_xl.py prepare_latents() step # 5
    @property
    def init_noise_sigma(self):
        """
        standard deviation of the initial noise distribution.
        """
        max_sigma = self.sigmas.max()
        assert (
            self.timestep_spacing == "leading"
        ), "timestep_spacing {self.timestep_spacing} is not supported in this version"
        return (max_sigma**2 + 1) ** 0.5

        # no need to do this in ttnn
        # max_sigma = ttnn.max(self.tt_sigmas, dim=0, keepdim=False)
        # return ttnn.sqrt((ttnn.pow(max_sigma,2) + 1))

    # pipeline_stable_diffusion_xl.py __call__() step #9
    def scale_model_input(
        self, sample: ttnn._ttnn.tensor.Tensor, timestep: Union[float, ttnn._ttnn.tensor.Tensor]
    ) -> ttnn._ttnn.tensor.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
        """
        # timestep is not used in this implementation, step_index is already initialized at set_timesteps()
        sigma_normalization_factor = self.variance_normalization_factor[self.step_index]
        sample = sample / sigma_normalization_factor

        self.is_scale_input_called = True
        return sample

    # pipeline_stable_diffusion_xl.py __call__() step #9
    def step(
        self,
        model_output: ttnn._ttnn.tensor.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: ttnn._ttnn.tensor.Tensor,
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

        # this is a potential accuracy pitfall
        # Upcast to avoid precision issues when computing prev_sample
        # sample = sample.to(torch.float32)

        # leaving gamma calculus just in case we hit it
        sigma = self.sigmas[self.step_index]
        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
        assert gamma == 0, "gamma > 0 is not supported in this version"
        tt_sigma = self.tt_sigmas[self.step_index]
        sigma_hat = ttnn.to_layout(tt_sigma, layout=ttnn.TILE_LAYOUT)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        assert self.prediction_type == "epsilon"
        pred_original_sample = sample - model_output * sigma_hat

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) * ttnn.reciprocal(sigma_hat)

        dt = ttnn.to_layout(self.tt_sigmas[self.step_index + 1], layout=ttnn.TILE_LAYOUT) - sigma_hat

        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        # prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self.step_index += 1

        return (prev_sample, pred_original_sample)

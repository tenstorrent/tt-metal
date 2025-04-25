# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
from loguru import logger
import pytest
import torch

from diffusers import DiffusionPipeline
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler


@pytest.mark.parametrize("num_inference_steps", [5])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_euler_discrete_scheduler(device, num_inference_steps):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True
    )

    scheduler = pipe.scheduler

    tt_scheduler = TtEulerDiscreteScheduler(
        scheduler.config.num_train_timesteps,
        scheduler.config.beta_start,
        scheduler.config.beta_end,
        scheduler.config.beta_schedule,
        scheduler.config.trained_betas,
        scheduler.config.prediction_type,
        scheduler.config.interpolation_type,
        scheduler.config.use_karras_sigmas,
        scheduler.config.use_exponential_sigmas,
        scheduler.config.use_beta_sigmas,
        scheduler.config.sigma_min,
        scheduler.config.sigma_max,
        scheduler.config.timestep_spacing,
        scheduler.config.timestep_type,
        scheduler.config.steps_offset,
        scheduler.config.rescale_betas_zero_snr,
        scheduler.config.final_sigmas_type,
    )

    # this is called from pipeline_stable_diffusion_xl.py __call__() step #4
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    tt_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    assert_with_pcc(scheduler.timesteps, tt_scheduler.timesteps, 0.999)
    assert_with_pcc(scheduler.alphas, tt_scheduler.alphas, 0.999)
    assert_with_pcc(scheduler.alphas_cumprod, tt_scheduler.alphas_cumprod, 0.999)
    assert_with_pcc(scheduler.betas, tt_scheduler.betas, 0.999)
    assert_with_pcc(scheduler.sigmas, tt_scheduler.sigmas, 0.999)

    # this is called from pipeline_stable_diffusion_xl.py prepare_latents()
    ref_sigma = scheduler.init_noise_sigma
    tt_sigma = tt_scheduler.init_noise_sigma
    assert_with_pcc(ref_sigma, tt_sigma, 0.999)
    assert ref_sigma == tt_sigma, f"ref_sigma: {ref_sigma}, tt_sigma: {tt_sigma}"

    ref_latent = torch.randn((1, 4, 128, 128), dtype=torch.float32)

    # emulating the pipeline_stable_diffusion_xl.py __call__() step #9
    for i, t in enumerate(scheduler.timesteps):
        ref_scaled_latent = scheduler.scale_model_input(ref_latent, scheduler.timesteps[i])
        tt_scaled_latent = tt_scheduler.scale_model_input(ref_latent, tt_scheduler.timesteps[i])
        passed, msg = assert_with_pcc(ref_scaled_latent, tt_scaled_latent, 0.999)
        logger.debug(f"{i}: scaled_model_input pcc passed: {msg}")

        noise_pred = torch.randn((1, 4, 128, 128), dtype=torch.float32)  # this comes from unet
        ref_prev_sample, ref_pred_original_sample = scheduler.step(
            noise_pred, scheduler.timesteps[i], ref_scaled_latent, return_dict=False
        )
        tt_prev_sample, tt_pred_original_sample = tt_scheduler.step(
            noise_pred, scheduler.timesteps[i], tt_scaled_latent, return_dict=False
        )
        passed, msg = assert_with_pcc(ref_prev_sample, tt_prev_sample, 0.999)
        logger.debug(f"{i}: prev_sample pcc passed: {msg}")
        passed, msg = assert_with_pcc(ref_pred_original_sample, tt_pred_original_sample, 0.999)
        logger.debug(f"{i}: pred_original_sample pcc passed: {msg}")

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from loguru import logger
import pytest
import torch
import ttnn

from diffusers import DiffusionPipeline
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 128 * 128, 4),
    ],
)
@pytest.mark.parametrize("num_inference_steps", [5])
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_euler_discrete_scheduler(device, input_shape, num_inference_steps, is_ci_env):
    try:
        from tracy import signpost
    except ImportError:

        def signpost(*args, **kwargs):
            pass

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )

    scheduler = pipe.scheduler

    tt_scheduler = TtEulerDiscreteScheduler(
        device,
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

    # emulate two runs of the pipeline with different num_inference_steps to ensure that the scheduler is set up correctly
    for _num_inference_steps in [1, num_inference_steps]:
        logger.debug(f"Testing with num_inference_steps: {_num_inference_steps}")
        # this is called from pipeline_stable_diffusion_xl.py __call__() step #4
        scheduler.set_timesteps(num_inference_steps=_num_inference_steps)
        tt_scheduler.set_timesteps(num_inference_steps=_num_inference_steps)

        assert_with_pcc(
            scheduler.timesteps, torch.cat([ttnn.to_torch(t).unsqueeze(0) for t in tt_scheduler.timesteps]), 0.999
        )
        assert_with_pcc(scheduler.alphas, tt_scheduler.alphas, 0.999)
        assert_with_pcc(scheduler.alphas_cumprod, tt_scheduler.alphas_cumprod, 0.999)
        assert_with_pcc(scheduler.betas, tt_scheduler.betas, 0.999)
        assert_with_pcc(scheduler.sigmas, tt_scheduler.sigmas, 0.999)

        # this is called from pipeline_stable_diffusion_xl.py prepare_latents() #5
        ref_sigma = scheduler.init_noise_sigma
        tt_sigma = tt_scheduler.init_noise_sigma
        assert_with_pcc(ref_sigma, tt_sigma, 0.999)

        ref_latent = torch.randn(input_shape, dtype=torch.float32)
        tt_latent = ttnn.from_torch(ref_latent, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device=device)
        tt_latent = ttnn.to_memory_config(tt_latent, ttnn.L1_MEMORY_CONFIG)

        # emulating the pipeline_stable_diffusion_xl.py __call__() step #9
        for i, t in enumerate(scheduler.timesteps):
            signpost(f"euler_discrete_scheduler_step {i=}")
            ref_scaled_latent = scheduler.scale_model_input(ref_latent, scheduler.timesteps[i])
            tt_scaled_latent = tt_scheduler.scale_model_input(tt_latent, None)
            torch_scaled_latent = ttnn.from_device(tt_scaled_latent).to_torch()
            passed, msg = assert_with_pcc(ref_scaled_latent, torch_scaled_latent, 0.999)
            logger.debug(f"{i}: scaled_model_input pcc passed: {msg}")

            noise_pred = torch.randn(input_shape, dtype=torch.float32)  # this comes from unet
            tt_noise_pred = ttnn.from_torch(noise_pred, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device=device)

            ref_prev_sample, _ = scheduler.step(
                noise_pred, scheduler.timesteps[i], ref_scaled_latent, return_dict=False
            )
            tt_prev_sample, _ = tt_scheduler.step(tt_noise_pred, None, tt_scaled_latent, return_dict=False)
            if i < (len(scheduler.timesteps) - 1):
                tt_scheduler.inc_step_index()
            torch_prev_sample = ttnn.from_device(tt_prev_sample).to_torch()
            passed, msg = assert_with_pcc(ref_prev_sample, torch_prev_sample, 0.999)
            logger.debug(f"{i}: prev_sample pcc passed: {msg}")


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 128 * 128, 4),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("num_inference_steps", [20])
def test_euler_discrete_scheduler_add_noise(device, input_shape, num_inference_steps, is_ci_env, reset_seeds):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )

    scheduler = pipe.scheduler
    tt_scheduler = TtEulerDiscreteScheduler(
        device,
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

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    tt_scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # Set begin index to 1 in both cases, to mimic the case in pipeline_stable_diffusion_xl_inpaint.py
    begin_index = 1
    scheduler.set_begin_index(begin_index)
    tt_scheduler.set_begin_index(begin_index)

    torch_original_samples = torch.randn(input_shape, dtype=torch.float32)
    tt_original_samples = ttnn.from_torch(torch_original_samples, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(
        device=device
    )

    torch_noise = torch.randn(input_shape, dtype=torch.float32)
    tt_noise = ttnn.from_torch(torch_noise, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT).to(device=device)

    torch_timesteps = scheduler.timesteps
    ttnn_timesteps = tt_scheduler.timesteps

    latent_timestep = torch_timesteps[:1]
    torch_noisy_sample = scheduler.add_noise(torch_original_samples, torch_noise, latent_timestep)

    tt_latent_timestep = ttnn.unsqueeze(ttnn_timesteps[:1][0], dim=0)
    tt_noisy_sample = tt_scheduler.add_noise(tt_original_samples, tt_noise, tt_latent_timestep, begin_index)

    tt_noisy_sample = ttnn.to_torch(tt_noisy_sample)
    assert_with_pcc(tt_noisy_sample, torch_noisy_sample, 0.999)

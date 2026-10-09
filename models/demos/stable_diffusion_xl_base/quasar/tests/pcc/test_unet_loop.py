# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quasar port of tests/pcc/test_unet_loop.py (1024x1024 only).

The Wormhole test runs the full SDXL pipeline (CLIP text encoders, trace capture, golden
cache). On Quasar the loop is reduced to what the UNet port owns: the classifier-free-guidance
denoising loop of ``TtUNet2DConditionModel`` + ``TtEulerDiscreteScheduler`` against the torch
UNet + diffusers scheduler, with random prompt embeddings in place of the text encoders and
without trace capture. Weights come from the HF cache or are random (see
``sdxl_quasar_test_utils.load_torch_unet``).
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.stable_diffusion_xl_base.quasar.tests.pcc.test_euler_discrete_scheduler import make_tt_scheduler
from models.demos.stable_diffusion_xl_base.quasar.tests.sdxl_quasar_test_utils import (
    load_scheduler,
    load_torch_unet,
    to_device_tile,
)
from models.demos.stable_diffusion_xl_base.quasar.tt import qsr
from models.demos.stable_diffusion_xl_base.quasar.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.quasar.tt.tt_unet import TtUNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc

UNET_LOOP_PCC = {"1024x1024": {"10": 0.93, "50": 0.905}}
UNET_LOOP_SEED = {"1024x1024": {"10": 42, "50": 0}}
GUIDANCE_SCALE = 5.0


def tt_unet_step(tt_unet, tt_scheduler, tt_latents, input_shape, prompt_embeds, time_ids, text_embeds):
    B, C, H, W = input_shape
    latent_model_input = tt_scheduler.scale_model_input(tt_latents, None)
    noise_pred, output_shape = tt_unet.forward(
        latent_model_input,
        [B, C, H, W],
        timestep=tt_scheduler.tt_timestep,
        encoder_hidden_states=prompt_embeds,
        time_ids=time_ids,
        text_embeds=text_embeds,
    )
    return noise_pred, output_shape


@torch.no_grad()
def run_unet_loop(
    device, is_ci_env, is_ci_v2_env, unet_location, pipeline_location, image_resolution, num_steps, debug_mode
):
    height, width = image_resolution
    resolution_key = f"{height}x{width}"
    torch.manual_seed(UNET_LOOP_SEED[resolution_key].get(str(num_steps), 0))

    unet = load_torch_unet(unet_location, is_ci_env, is_ci_v2_env)
    scheduler = load_scheduler(pipeline_location, is_ci_env, is_ci_v2_env)
    tt_scheduler = make_tt_scheduler(device, scheduler)
    tt_unet = TtUNet2DConditionModel(
        device,
        unet.state_dict(),
        "unet",
        model_config=load_model_optimisations(image_resolution),
        debug_mode=debug_mode,
    )

    # Stand-ins for the CLIP encoder outputs: [uncond, cond] prompt embeddings and pooled embeddings.
    prompt_embeds = [torch.randn(1, 77, 2048) * 0.5 for _ in range(2)]
    text_embeds = [torch.randn(1, 1280) * 0.5 for _ in range(2)]
    time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=torch.float32)

    scheduler.set_timesteps(num_inference_steps=num_steps)
    tt_scheduler.set_timesteps(num_inference_steps=num_steps)

    B, C, H, W = 1, 4, height // 8, width // 8
    latents = torch.randn(B, C, H, W) * scheduler.init_noise_sigma
    tt_latents = to_device_tile(torch.permute(latents, (0, 2, 3, 1)).reshape(1, 1, B * H * W, C), device)

    tt_prompt_embeds = [to_device_tile(p, device) for p in prompt_embeds]
    tt_text_embeds = [to_device_tile(p, device, layout=ttnn.ROW_MAJOR_LAYOUT) for p in text_embeds]
    tt_time_ids = to_device_tile(time_ids.reshape(-1), device)

    pcc_per_iter = []
    for i, t in enumerate(scheduler.timesteps):
        # --- torch reference (classifier free guidance)
        latent_model_input = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=torch.cat(prompt_embeds),
            added_cond_kwargs={"text_embeds": torch.cat(text_embeds), "time_ids": torch.cat([time_ids] * 2)},
        ).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + GUIDANCE_SCALE * (noise_text - noise_uncond)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # --- Quasar
        tt_outputs = []
        for s in range(2):
            out, [C, H, W] = tt_unet_step(
                tt_unet, tt_scheduler, tt_latents, [B, C, H, W], tt_prompt_embeds[s], tt_time_ids, tt_text_embeds[s]
            )
            tt_outputs.append(out)
        tt_uncond, tt_text = tt_outputs
        # noise_pred = uncond + scale * (text - uncond)
        tt_text = qsr.add(tt_text, qsr.multiply(tt_uncond, -1.0))
        tt_text = qsr.mul_(tt_text, GUIDANCE_SCALE)
        tt_noise_pred = qsr.add_(tt_uncond, tt_text)
        tt_latents = tt_scheduler.step(tt_noise_pred, None, tt_latents, return_dict=False)[0]
        ttnn.deallocate(tt_text)
        if i < len(scheduler.timesteps) - 1:
            tt_scheduler.inc_step_index()

        torch_tt_latents = ttnn.to_torch(tt_latents).reshape(B, H, W, C).permute(0, 3, 1, 2)
        _, pcc_message = comp_pcc(latents, torch_tt_latents, 0.8)
        logger.info(f"PCC of {i}. iteration is: {pcc_message}")
        pcc_per_iter.append(float(pcc_message))

    pcc_threshold = UNET_LOOP_PCC[resolution_key].get(str(num_steps), 0)
    _, pcc_message = assert_with_pcc(latents, torch_tt_latents, pcc_threshold)
    logger.info(f"PCC of the last iteration is: {pcc_message}")


@pytest.mark.parametrize("image_resolution", [(1024, 1024)], ids=["1024x1024"])
@pytest.mark.timeout(0)
def test_unet_loop(
    device,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    sdxl_base_pipeline_location,
    image_resolution,
    loop_iter_num,
    debug_mode,
):
    run_unet_loop(
        device,
        is_ci_env,
        is_ci_v2_env,
        sdxl_base_unet_location,
        sdxl_base_pipeline_location,
        image_resolution,
        loop_iter_num,
        debug_mode,
    )

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
from loguru import logger
import ttnn
from models.experimental.stable_diffusion_xl_refiner.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_refiner.tests.test_common import (
    SDXL_REFINER_L1_SMALL_SIZE,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc
import matplotlib.pyplot as plt
from models.common.utility_functions import is_wormhole_b0

UNET_LOOP_PCC = {"20": 0.999, "50": 0.997}


def prepare_refiner_tensors(
    device,
    encoder_hidden_states_uncond,
    encoder_hidden_states_cond,
    text_embeds_uncond,
    text_embeds_cond,
    time_ids_uncond,
    time_ids_cond,
):
    # Unconditional (negative) tensors
    ttnn_encoder_hidden_states_uncond = ttnn.from_torch(
        encoder_hidden_states_uncond,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_text_embeds_uncond = ttnn.from_torch(
        text_embeds_uncond,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_time_ids_uncond = ttnn.from_torch(
        time_ids_uncond,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Conditional (positive) tensors
    ttnn_encoder_hidden_states_cond = ttnn.from_torch(
        encoder_hidden_states_cond,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_text_embeds_cond = ttnn.from_torch(
        text_embeds_cond,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_time_ids_cond = ttnn.from_torch(
        time_ids_cond,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return (
        ttnn_encoder_hidden_states_uncond,
        ttnn_encoder_hidden_states_cond,
        ttnn_text_embeds_uncond,
        ttnn_text_embeds_cond,
        ttnn_time_ids_uncond,
        ttnn_time_ids_cond,
    )


def run_tt_refiner_denoising_step(
    ttnn_device,
    tt_unet,
    scaled_latents,
    timestep,
    conditioning_tensors,
    guidance_scale,
):
    # Run single TT refiner denoising step
    (
        ttnn_encoder_hidden_states_uncond,
        ttnn_encoder_hidden_states_cond,
        ttnn_text_embeds_uncond,
        ttnn_text_embeds_cond,
        ttnn_time_ids_uncond,
        ttnn_time_ids_cond,
    ) = conditioning_tensors

    # Create timestep tensor
    ttnn_timestep_tensor = ttnn.from_torch(
        timestep,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Unconditional pass
    ttnn_latents_uncond = ttnn.from_torch(
        scaled_latents,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    B, C, H, W = list(ttnn_latents_uncond.shape)
    ttnn_latents_uncond = ttnn.permute(ttnn_latents_uncond, (0, 2, 3, 1))
    ttnn_latents_uncond = ttnn.reshape(ttnn_latents_uncond, (B, 1, H * W, C))

    noise_uncond, _ = tt_unet.forward(
        ttnn_latents_uncond,
        [B, C, H, W],
        timestep=ttnn_timestep_tensor,
        encoder_hidden_states=ttnn_encoder_hidden_states_uncond,
        time_ids=ttnn_time_ids_uncond,
        text_embeds=ttnn_text_embeds_uncond,
    )

    # Conditional pass
    ttnn_latents_cond = ttnn.from_torch(
        scaled_latents,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_latents_cond = ttnn.permute(ttnn_latents_cond, (0, 2, 3, 1))
    ttnn_latents_cond = ttnn.reshape(ttnn_latents_cond, (B, 1, H * W, C))

    noise_cond, _ = tt_unet.forward(
        ttnn_latents_cond,
        [B, C, H, W],
        timestep=ttnn_timestep_tensor,
        encoder_hidden_states=ttnn_encoder_hidden_states_cond,
        time_ids=ttnn_time_ids_cond,
        text_embeds=ttnn_text_embeds_cond,
    )

    # Clean up tensors
    ttnn.deallocate(ttnn_latents_cond)
    ttnn.deallocate(ttnn_latents_uncond)
    ttnn.deallocate(ttnn_timestep_tensor)

    # Convert noise outputs back to torch tensors for CFG calculation
    noise_uncond = ttnn.to_torch(noise_uncond)
    noise_uncond = noise_uncond.reshape(B, H, W, C)
    noise_uncond = torch.permute(noise_uncond, (0, 3, 1, 2))

    noise_cond = ttnn.to_torch(noise_cond)
    noise_cond = noise_cond.reshape(B, H, W, C)
    noise_cond = torch.permute(noise_cond, (0, 3, 1, 2))

    # CFG
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    return noise_pred


def run_torch_refiner_denoising_step(
    latents,
    scheduler,
    timestep,
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
    guidance_scale,
    unet,
    time_ids_uncond,
    time_ids_cond,
):
    # Run single torch refiner denoising step for comparison
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # Concatenate conditioning
    encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds])
    text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
    time_ids = torch.cat([time_ids_uncond.unsqueeze(0), time_ids_cond.unsqueeze(0)])

    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

    # Predict noise
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    # CFG
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    return noise_pred


@torch.no_grad()
def run_refiner_unet_inference(ttnn_device, is_ci_env, prompts, num_inference_steps):
    torch.manual_seed(0)

    if isinstance(prompts, str):
        prompts = [prompts]

    guidance_scale = 5.0
    denoising_start = 0.8

    height = 1024
    width = 1024

    refiner_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )

    tt_unet = TtUNet2DConditionModel(
        ttnn_device,
        refiner_pipe.unet.state_dict(),
    )

    prompt_embeds = []
    negative_prompt_embeds = []
    pooled_prompt_embeds = []
    negative_pooled_prompt_embeds = []

    # Encode prompts
    for prompt in prompts:
        (
            prompt_embed,
            negative_prompt_embed,
            pooled_prompt_embed,
            negative_pooled_prompt_embed,
        ) = refiner_pipe.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
        )
        prompt_embeds.append(prompt_embed)
        negative_prompt_embeds.append(negative_prompt_embed)
        pooled_prompt_embeds.append(pooled_prompt_embed)
        negative_pooled_prompt_embeds.append(negative_pooled_prompt_embed)

    # Prepare timesteps - using separate schedulers for TT and torch
    tt_scheduler = refiner_pipe.scheduler
    torch_scheduler = refiner_pipe.scheduler.__class__.from_config(refiner_pipe.scheduler.config)

    tt_scheduler.set_timesteps(num_inference_steps)
    torch_scheduler.set_timesteps(num_inference_steps)

    # Calculate timesteps for denoising_start
    discrete_timestep_cutoff = int(
        round(tt_scheduler.config.num_train_timesteps - (denoising_start * tt_scheduler.config.num_train_timesteps))
    )
    num_refiner_steps = (tt_scheduler.timesteps < discrete_timestep_cutoff).sum().item()
    t_start = len(tt_scheduler.timesteps) - num_refiner_steps
    timesteps = tt_scheduler.timesteps[t_start:]

    logger.info(f"Running {num_refiner_steps} refiner denoising steps")

    # Prepare initial latents
    num_channels_latents = refiner_pipe.unet.config.in_channels
    assert num_channels_latents == 4, f"num_channels_latents is {num_channels_latents}, but it should be 4"

    # Create random latents
    latents = torch.randn(
        (1, num_channels_latents, height // 8, width // 8),
        dtype=prompt_embeds[0].dtype,
    )

    # Prepare time_ids
    time_ids_uncond = torch.tensor([height, width, 0.0, 0.0, 2.5], dtype=torch.float32)
    time_ids_cond = torch.tensor([height, width, 0.0, 0.0, 6.0], dtype=torch.float32)

    ttnn.synchronize_device(ttnn_device)
    pcc_per_iter = []

    # Run inference and comparison
    logger.info("Starting refiner inference...")
    for iter in range(len(prompts)):
        prompt = prompts[iter]
        logger.info(f"Running inference for prompt {iter + 1}/{len(prompts)}: {prompt}")

        # Prepare TT conditioning tensors for this prompt
        iter_conditioning_tensors = prepare_refiner_tensors(
            ttnn_device,
            negative_prompt_embeds[iter],
            prompt_embeds[iter],
            negative_pooled_prompt_embeds[iter],
            pooled_prompt_embeds[iter],
            time_ids_uncond,
            time_ids_cond,
        )

        # Reset latents for each prompt
        tt_latents = latents.clone()
        torch_latents = latents.clone()

        # Reset schedulers for each prompt
        tt_scheduler.set_timesteps(num_inference_steps)
        torch_scheduler.set_timesteps(num_inference_steps)

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            scaled_latents = tt_scheduler.scale_model_input(tt_latents, t)

            # Run TT iteration
            noise_pred = run_tt_refiner_denoising_step(
                ttnn_device=ttnn_device,
                tt_unet=tt_unet,
                scaled_latents=scaled_latents,
                timestep=t,
                conditioning_tensors=iter_conditioning_tensors,
                guidance_scale=guidance_scale,
            )
            tt_latents = tt_scheduler.step(noise_pred, t, tt_latents).prev_sample

            # Run torch iteration
            torch_noise_pred = run_torch_refiner_denoising_step(
                latents=torch_latents,
                scheduler=torch_scheduler,
                timestep=t,
                prompt_embeds=prompt_embeds[iter],
                negative_prompt_embeds=negative_prompt_embeds[iter],
                pooled_prompt_embeds=pooled_prompt_embeds[iter],
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[iter],
                guidance_scale=guidance_scale,
                unet=refiner_pipe.unet,
                time_ids_uncond=time_ids_uncond,
                time_ids_cond=time_ids_cond,
            )
            torch_latents = torch_scheduler.step(torch_noise_pred, t, torch_latents).prev_sample

            ttnn.synchronize_device(ttnn_device)

            # Compare outputs
            _, pcc_message = comp_pcc(torch_latents, tt_latents, 0.8)
            logger.info(f"PCC of step {i}: {pcc_message}")
            pcc_per_iter.append(float(pcc_message))

    if not is_ci_env:
        plt.plot(pcc_per_iter, marker="o")
        plt.title("Refiner PCC per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("PCC")
        plt.grid(True)
        plt.savefig("refiner_pcc_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

    _, pcc_message = assert_with_pcc(torch_latents, tt_latents, UNET_LOOP_PCC.get(str(num_inference_steps), 0))
    logger.info(f"Final PCC: {pcc_message}")


@pytest.mark.skipif(not is_wormhole_b0(), reason="SDXL Refiner supported on WH only")
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_REFINER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "prompt",
    (("An astronaut riding a green horse"),),
)
@pytest.mark.timeout(3000)
def test_refiner_unet_loop(
    device,
    is_ci_env,
    prompt,
    loop_iter_num,
):
    return run_refiner_unet_inference(device, is_ci_env, prompt, loop_iter_num)

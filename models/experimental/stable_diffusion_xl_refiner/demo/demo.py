# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger
from models.experimental.stable_diffusion_xl_refiner.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_refiner.tests.test_common import SDXL_REFINER_L1_SMALL_SIZE
from diffusers import (
    DiffusionPipeline,
    UNet2DConditionModel,
)
from PIL import Image
import numpy as np


def setup_pipelines():
    # Base pipeline
    base_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    # Refiner pipeline
    refiner_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    # VAE, tokenizers, text encoders
    vae = base_pipe.vae
    tokenizer_1 = base_pipe.tokenizer
    tokenizer_2 = base_pipe.tokenizer_2
    text_encoder_1 = base_pipe.text_encoder
    text_encoder_2 = base_pipe.text_encoder_2

    return base_pipe, refiner_pipe, vae, tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2


def get_refiner_conditioning(refiner_pipe, prompt: str, negative_prompt: str = ""):
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = refiner_pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
    )

    encoder_hidden_states_uncond = negative_prompt_embeds
    encoder_hidden_states_cond = prompt_embeds
    text_embeds_uncond = negative_pooled_prompt_embeds
    text_embeds_cond = pooled_prompt_embeds

    time_ids_uncond = torch.tensor([1024.0, 1024.0, 0.0, 0.0, 2.5], dtype=torch.float32)
    time_ids_cond = torch.tensor([1024.0, 1024.0, 0.0, 0.0, 6.0], dtype=torch.float32)

    return (
        encoder_hidden_states_uncond,
        encoder_hidden_states_cond,
        text_embeds_uncond,
        text_embeds_cond,
        time_ids_uncond,
        time_ids_cond,
    )


def generate_with_refiner_compare(
    device,
    base_pipe,
    refiner_pipe,
    vae,
    prompt: str,
    negative_prompt: str = "",
    guidance_scale: float = 5.0,
    num_steps: int = 50,
):
    """
    - generate latents using base model
    - run the Hugging Face refiner (for baseline)
    - run your custom refiner on same latents and inputs
    - decode both and return images
    """

    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="unet",
    )
    unet.eval()
    state_dict = unet.state_dict()

    tt_unet = TtUNet2DConditionModel(
        device,
        state_dict,
    )

    logger.info(f"Starting SDXL Refiner demo with prompt: '{prompt}'")
    # Hugging Face base output latents
    logger.info("Running base model (Torch) - denoising to 80%...")
    with torch.no_grad():
        out = base_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="latent",
            return_dict=True,
            denoising_end=0.8,
        )  # only denoise 80% with base
    base_latents = out.images
    logger.info("Base model complete - generated base latents")

    (
        encoder_hidden_states_uncond,
        encoder_hidden_states_cond,
        text_embeds_uncond,
        text_embeds_cond,
        time_ids_uncond,
        time_ids_cond,
    ) = get_refiner_conditioning(refiner_pipe, prompt, negative_prompt)

    # Hugging Face refiner output - baseline for comparison
    logger.info("Running refiner for baseline comparison (Torch)...")
    with torch.no_grad():
        hf_refiner_out = refiner_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            image=base_latents,
            return_dict=True,
            denoising_start=0.8,
        )
    hf_refined_image = hf_refiner_out.images[0]
    logger.info("HuggingFace refiner complete")

    ## TT refiner
    logger.info("Running TT refiner...")
    scheduler = refiner_pipe.scheduler

    # Set timesteps and handle denoising_start
    scheduler.set_timesteps(num_steps)
    denoising_start = 0.8

    # Using HuggingFace's exact timestep calculation logic
    # This is from get_timesteps method in the HF pipeline
    discrete_timestep_cutoff = int(
        round(scheduler.config.num_train_timesteps - (denoising_start * scheduler.config.num_train_timesteps))
    )

    num_refiner_steps = (scheduler.timesteps < discrete_timestep_cutoff).sum().item()

    t_start = len(scheduler.timesteps) - num_refiner_steps
    timesteps = scheduler.timesteps[t_start:]

    logger.info(f"🕐 TT refiner will run {num_refiner_steps} denoising steps (from 80% to 100%)")

    #
    latents = base_latents

    # Prepare TT tensors
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

    # Denoising loop
    for t in timesteps:
        scaled_latents = scheduler.scale_model_input(latents, t)

        ttnn_timestep_tensor = ttnn.from_torch(
            scheduler.timesteps[scheduler.step_index],
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Unconditional pass
        ttnn_latents_uncond = ttnn.from_torch(
            scaled_latents,
            dtype=ttnn.bfloat16,
            device=device,
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
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        B, C, H, W = list(ttnn_latents_cond.shape)
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

        # Convert noise outputs back to torch tensors for CFG calculation
        noise_uncond = ttnn.to_torch(noise_uncond)
        noise_uncond = noise_uncond.reshape(B, H, W, C)
        noise_uncond = torch.permute(noise_uncond, (0, 3, 1, 2))

        noise_cond = ttnn.to_torch(noise_cond)
        noise_cond = noise_cond.reshape(B, H, W, C)
        noise_cond = torch.permute(noise_cond, (0, 3, 1, 2))

        # Combine via CFG
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Step scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    logger.info("TT refiner denoising complete - converting latents to image...")
    with torch.no_grad():
        latents_float32 = latents.to(torch.float32)
        scaled_latents = latents_float32 / vae.config.scaling_factor
        dec = vae.decode(scaled_latents).sample
        dec = (dec / 2 + 0.5).clamp(0, 1)
        image_custom = dec.squeeze().permute(1, 2, 0).cpu().numpy()
    # Format image as PIL
    image_custom_pil = Image.fromarray((image_custom * 255).astype(np.uint8))

    return hf_refined_image, image_custom_pil


def run_demo_inference(
    device,
    prompt,
    is_ci_env,
):
    if is_ci_env:
        pytest.skip("Skipping demo test in CI environment")

    torch.manual_seed(0)

    base_pipe, refiner_pipe, vae, _, _, _, _ = setup_pipelines()

    hf_image, custom_image = generate_with_refiner_compare(
        device=device,
        base_pipe=base_pipe,
        refiner_pipe=refiner_pipe,
        vae=vae,
        prompt=prompt,
        negative_prompt="",
    )

    # Save or display
    logger.info("Saving generated images...")
    hf_image.save("hf_refined.png")
    custom_image.save("custom_refined.png")
    logger.info("Images saved:")
    logger.info("  hf_refined.png - HuggingFace refiner output (baseline)")
    logger.info("  custom_refined.png - TT refiner output")
    logger.info("Demo complete!")


@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_REFINER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "prompt",
    (("An astronaut riding a green horse"),),
)
def test_demo(
    device,
    prompt,
    is_ci_env,
):
    return run_demo_inference(
        device,
        prompt,
        is_ci_env,
    )

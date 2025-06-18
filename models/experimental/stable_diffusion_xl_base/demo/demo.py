# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
from loguru import logger
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    retrieve_timesteps,
    run_tt_iteration,
)
import os
import gc


@torch.no_grad()
def run_demo_inference(
    ttnn_device, is_ci_env, prompts, num_inference_steps, classifier_free_guidance, vae_on_device, evaluation_range
):
    start_from, _ = evaluation_range
    torch.manual_seed(0)

    if isinstance(prompts, str):
        prompts = [prompts]

    # In case of classifier free guidance this is set:
    # - guidance_scale = 5.0
    # - 2 runs of unet per iteration
    # For non classifier free guidance do:
    # - guidance_scale = 1.0
    # - 1 run of unet per iteration
    if classifier_free_guidance == True:
        guidance_scale = 5.0
    else:
        guidance_scale = 1.0

    # 0. Set up default height and width for unet
    height = 1024
    width = 1024

    # 1. Load components
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    # 2. Load tt_unet, tt_vae and tt_scheduler
    tt_model_config = ModelOptimisations(conv_w_dtype=ttnn.bfloat16)
    tt_unet = TtUNet2DConditionModel(
        ttnn_device,
        pipeline.unet.state_dict(),
        "unet",
        model_config=tt_model_config,
        transformer_weights_dtype=ttnn.bfloat16,
    )
    tt_vae = (
        TtAutoencoderKL(ttnn_device, pipeline.vae.state_dict(), model_config=tt_model_config) if vae_on_device else None
    )
    tt_scheduler = TtEulerDiscreteScheduler(
        ttnn_device,
        pipeline.scheduler.config.num_train_timesteps,
        pipeline.scheduler.config.beta_start,
        pipeline.scheduler.config.beta_end,
        pipeline.scheduler.config.beta_schedule,
        pipeline.scheduler.config.trained_betas,
        pipeline.scheduler.config.prediction_type,
        pipeline.scheduler.config.interpolation_type,
        pipeline.scheduler.config.use_karras_sigmas,
        pipeline.scheduler.config.use_exponential_sigmas,
        pipeline.scheduler.config.use_beta_sigmas,
        pipeline.scheduler.config.sigma_min,
        pipeline.scheduler.config.sigma_max,
        pipeline.scheduler.config.timestep_spacing,
        pipeline.scheduler.config.timestep_type,
        pipeline.scheduler.config.steps_offset,
        pipeline.scheduler.config.rescale_betas_zero_snr,
        pipeline.scheduler.config.final_sigmas_type,
    )
    pipeline.scheduler = tt_scheduler

    cpu_device = "cpu"

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
        ) = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=cpu_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=classifier_free_guidance,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        prompt_embeds.append(prompt_embed)
        negative_prompt_embeds.append(negative_prompt_embed)
        pooled_prompt_embeds.append(pooled_prompt_embed)
        negative_pooled_prompt_embeds.append(negative_pooled_prompt_embed)

    # Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, cpu_device, None, None)

    # Convert timesteps to ttnn
    ttnn_timesteps = []
    for t in timesteps:
        scalar_tensor = torch.tensor(t).unsqueeze(0)
        ttnn_timesteps.append(
            ttnn.from_torch(
                scalar_tensor,
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    num_channels_latents = pipeline.unet.config.in_channels
    assert num_channels_latents == 4, f"num_channels_latents is {num_channels_latents}, but it should be 4"

    latents = pipeline.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds[0].dtype,
        cpu_device,
        None,
        None,
    )

    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(None, 0.0)
    add_text_embeds = pooled_prompt_embeds
    text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim
    assert (
        text_encoder_projection_dim == 1280
    ), f"text_encoder_projection_dim is {text_encoder_projection_dim}, but it should be 1280"

    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_time_ids = pipeline._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds[0].dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    negative_add_time_ids = add_time_ids

    if classifier_free_guidance:
        ttnn_prompt_embeds = [
            [
                ttnn.from_torch(
                    negative_prompt_embed,
                    dtype=ttnn.bfloat16,
                    device=ttnn_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.from_torch(
                    prompt_embed,
                    dtype=ttnn.bfloat16,
                    device=ttnn_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            ]
            for negative_prompt_embed, prompt_embed in zip(negative_prompt_embeds, prompt_embeds)
        ]
        ttnn_add_text_embeds = [
            [
                ttnn.from_torch(
                    negative_pooled_prompt_embed,
                    dtype=ttnn.bfloat16,
                    device=ttnn_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.from_torch(
                    add_text_embed,
                    dtype=ttnn.bfloat16,
                    device=ttnn_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            ]
            for negative_pooled_prompt_embed, add_text_embed in zip(negative_pooled_prompt_embeds, add_text_embeds)
        ]
        ttnn_add_time_ids = [
            ttnn.from_torch(
                negative_add_time_ids.squeeze(0),
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.from_torch(
                add_time_ids.squeeze(0),
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        ]
        ttnn_added_cond_kwargs = [
            [
                {
                    "text_embeds": ttnn_add_text_embed[0],
                    "time_ids": ttnn_add_time_ids[0],
                },
                {
                    "text_embeds": ttnn_add_text_embed[1],
                    "time_ids": ttnn_add_time_ids[1],
                },
            ]
            for ttnn_add_text_embed in ttnn_add_text_embeds
        ]
    else:
        ttnn_prompt_embeds = [
            ttnn.from_torch(
                prompt_embeds,
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ]
        ttnn_add_text_embeds = [
            ttnn.from_torch(
                add_text_embeds,
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ]
        ttnn_add_time_ids = [
            ttnn.from_torch(
                add_time_ids.squeeze(0),
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ]
        ttnn_added_cond_kwargs = [
            {
                "text_embeds": ttnn_add_text_embeds[0],
                "time_ids": ttnn_add_time_ids[0],
            }
        ]

    scaling_factor = ttnn.from_torch(
        torch.Tensor([pipeline.vae.config.scaling_factor]),
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info("Performing warmup run, to make use of program caching in actual inference...")
    B, C, H, W = latents.shape

    # All device code will work with channel last tensors
    latents = torch.permute(latents, (0, 2, 3, 1))
    latents = latents.reshape(1, 1, B * H * W, C)

    latents_clone = latents.clone()

    latents = ttnn.from_torch(
        latents,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # UNet will deallocate the input tensor
    latent_model_input = ttnn.clone(latents)

    # Compile run of Scheduler and UNet
    run_tt_iteration(
        ttnn_device,
        tt_unet,
        tt_scheduler,
        latent_model_input,
        [B, C, H, W],
        ttnn_prompt_embeds[0][0],
        ttnn_added_cond_kwargs[0][0],
        ttnn_timesteps[0],
        0,
    )
    if not is_ci_env and not os.path.exists("output"):
        os.mkdir("output")

    images = []
    logger.info("Starting ttnn inference...")
    for iter in range(len(prompts)):
        logger.info(f"Running inference for prompt {iter + 1}/{len(prompts)}: {prompts[iter]}")
        for i, t in tqdm(enumerate(ttnn_timesteps), total=len(ttnn_timesteps)):
            unet_outputs = []
            for unet_slice in range(len(ttnn_prompt_embeds[iter])):
                latent_model_input = latents
                noise_pred, noise_shape = run_tt_iteration(
                    ttnn_device,
                    tt_unet,
                    tt_scheduler,
                    latent_model_input,
                    [B, C, H, W],
                    ttnn_prompt_embeds[iter][unet_slice],
                    ttnn_added_cond_kwargs[iter][unet_slice],
                    t,
                    i,
                )
                C, H, W = noise_shape

                unet_outputs.append(noise_pred)

            # perform guidance
            if classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = unet_outputs
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                ttnn.deallocate(noise_pred_uncond)
                ttnn.deallocate(noise_pred_text)
            else:
                noise_pred = unet_outputs[0]

            latents = tt_scheduler.step(
                noise_pred, tt_scheduler.timesteps[i], latents, **extra_step_kwargs, return_dict=False
            )[0]

            ttnn.deallocate(noise_pred)
            latents = ttnn.move(latents)

        tt_scheduler.set_step_index(0)

        if vae_on_device:
            latents = ttnn.div(latents, scaling_factor)

            # Workaround for #22017
            ttnn_device.disable_and_clear_program_cache()

            logger.info("Running TT VAE")
            image = tt_vae.forward(latents, [B, C, H, W])
            ttnn_device.enable_program_cache()
        else:
            latents = ttnn.from_device(latents).to_torch()
            latents = latents.reshape(B, H, W, C)
            latents = torch.permute(latents, (0, 3, 1, 2))

            latents = latents.to(pipeline.vae.dtype)

            # VAE upcasting to float32 is happening in the reference SDXL demo if VAE dtype is float16. If it's bfloat16, it will not be upcasted.
            latents = latents / pipeline.vae.config.scaling_factor

            image = pipeline.vae.decode(latents, return_dict=False)[0]

        image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
        images.append(image)

        if is_ci_env:
            logger.info(f"Image {iter + 1}/{len(prompts)} generated successfully")
        else:
            image.save(f"output/output{iter + start_from}.png")
            logger.info(f"Image saved to output/output{iter + start_from}.png")

        if vae_on_device:
            ttnn.deallocate(latents)
        else:
            del latents
            gc.collect()

        latents = latents_clone.clone()
        latents = ttnn.from_torch(
            latents,
            dtype=ttnn.bfloat16,
            device=ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return images


@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "prompt",
    (("An astronaut riding a green horse"),),
)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "classifier_free_guidance",
    [
        (True),
        (False),
    ],
    ids=("with_classifier_free_guidance", "no_classifier_free_guidance"),
)
@pytest.mark.parametrize(
    "vae_on_device",
    [
        (True),
        (False),
    ],
    ids=("device_vae", "host_vae"),
)
def test_demo(
    device,
    use_program_cache,
    is_ci_env,
    prompt,
    num_inference_steps,
    classifier_free_guidance,
    vae_on_device,
    evaluation_range,
):
    return run_demo_inference(
        device, is_ci_env, prompt, num_inference_steps, classifier_free_guidance, vae_on_device, evaluation_range
    )

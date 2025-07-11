# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
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
    run_tt_image_gen,
)
import os
from models.utility_functions import profiler


@torch.no_grad()
def run_demo_inference(ttnn_device, is_ci_env, prompts, num_inference_steps, vae_on_device, evaluation_range):
    batch_size = 1

    start_from, _ = evaluation_range
    torch.manual_seed(0)

    if isinstance(prompts, str):
        prompts = [prompts]

    needed_padding = (batch_size - len(prompts) % batch_size) % batch_size
    prompts = prompts + [""] * needed_padding

    guidance_scale = 5.0

    # 0. Set up default height and width for unet
    height = 1024
    width = 1024

    # 1. Load components
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(ttnn_device)):
        # 2. Load tt_unet, tt_vae and tt_scheduler
        tt_model_config = ModelOptimisations()
        tt_unet = TtUNet2DConditionModel(
            ttnn_device,
            pipeline.unet.state_dict(),
            "unet",
            model_config=tt_model_config,
        )
        tt_vae = (
            TtAutoencoderKL(ttnn_device, pipeline.vae.state_dict(), tt_model_config, batch_size)
            if vae_on_device
            else None
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

    all_embeds = [
        pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=cpu_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        for prompt in prompts
    ]

    # Reorder all_embeds to prepare for splitting across devices
    items_per_core = len(all_embeds) // batch_size  # this will always be a multiple of batch_size because of padding

    if batch_size > 1:  # If batch_size is 1, no need to reorder
        reordered = []
        for i in range(batch_size):
            for j in range(items_per_core):
                index = i + j * batch_size
                reordered.append(all_embeds[index])
        all_embeds = reordered

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = zip(*all_embeds)

    prompt_embeds_torch = torch.cat(prompt_embeds, dim=0)
    negative_prompt_embeds_torch = torch.cat(negative_prompt_embeds, dim=0)
    pooled_prompt_embeds_torch = torch.cat(pooled_prompt_embeds, dim=0)
    negative_pooled_prompt_embeds_torch = torch.cat(negative_pooled_prompt_embeds, dim=0)

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
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device),
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

    torch_prompt_embeds = torch.stack([negative_prompt_embeds_torch, prompt_embeds_torch], dim=1)
    torch_add_text_embeds = torch.stack([negative_pooled_prompt_embeds_torch, pooled_prompt_embeds_torch], dim=1)
    ttnn_prompt_embeds = ttnn.from_torch(
        torch_prompt_embeds,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=1),
    )
    ttnn_add_text_embeds = ttnn.from_torch(
        torch_add_text_embeds,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=1),
    )

    torch_add_time_ids = torch.stack([negative_add_time_ids.squeeze(0), add_time_ids.squeeze(0)], dim=0)

    ttnn_time_ids = ttnn.from_torch(
        torch_add_time_ids,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=0),
    )
    ttnn_time_ids = ttnn.squeeze(ttnn_time_ids, dim=0)

    scaling_factor = ttnn.from_torch(
        torch.Tensor([pipeline.vae.config.scaling_factor]),
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device),
    )

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
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device),
    )

    # UNet will deallocate the input tensor
    latent_model_input = ttnn.clone(latents)

    logger.info("Performing warmup run, to make use of program caching in actual inference...")
    run_tt_image_gen(
        ttnn_device,
        tt_unet,
        tt_scheduler,
        latent_model_input,
        ttnn_prompt_embeds,
        ttnn_time_ids,
        ttnn_add_text_embeds,
        [ttnn_timesteps[0]],
        extra_step_kwargs,
        guidance_scale,
        scaling_factor,
        [B, C, H, W],
        tt_vae if vae_on_device else pipeline.vae,
        batch_size,
        0,
    )
    profiler.clear()

    if not is_ci_env and not os.path.exists("output"):
        os.mkdir("output")

    images = []
    logger.info("Starting ttnn inference...")
    for iter in range(len(prompts) // batch_size):
        logger.info(
            f"Running inference for prompts {iter * batch_size + 1}-{iter * batch_size + batch_size}/{len(prompts)}"
        )
        imgs = run_tt_image_gen(
            ttnn_device,
            tt_unet,
            tt_scheduler,
            latents,
            ttnn_prompt_embeds,
            ttnn_time_ids,
            ttnn_add_text_embeds,
            ttnn_timesteps,
            extra_step_kwargs,
            guidance_scale,
            scaling_factor,
            [B, C, H, W],
            tt_vae if vae_on_device else pipeline.vae,
            batch_size,
            iter,
        )

        logger.info(
            f"Denoising loop for {batch_size} promts completed in {profiler.times['denoising_loop'][-1]:.2f} seconds"
        )
        logger.info(
            f"{'On device VAE' if vae_on_device else 'Host VAE'} decoding completed in {profiler.times['vae_decode'][-1]:.2f} seconds"
        )
        for idx, img in enumerate(imgs):
            if iter == len(prompts) // batch_size - 1 and idx >= batch_size - needed_padding:
                break
            img = img.unsqueeze(0)
            img = pipeline.image_processor.postprocess(img, output_type="pil")[0]
            images.append(img)
            if is_ci_env:
                logger.info(f"Image {len(images)}/{len(prompts) // batch_size} generated successfully")
            else:
                img.save(f"output/output{len(images) + start_from}.png")
                logger.info(f"Image saved to output/output{len(images) + start_from}.png")

        latents = latents_clone.clone()
        latents = ttnn.from_torch(
            latents,
            dtype=ttnn.bfloat16,
            device=ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device),
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
    "vae_on_device",
    [
        (True),
        (False),
    ],
    ids=("device_vae", "host_vae"),
)
def test_demo(
    mesh_device,
    is_ci_env,
    prompt,
    num_inference_steps,
    vae_on_device,
    evaluation_range,
):
    return run_demo_inference(mesh_device, is_ci_env, prompt, num_inference_steps, vae_on_device, evaluation_range)

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from diffusers import DiffusionPipeline
from loguru import logger
from conftest import is_galaxy
from models.experimental.stable_diffusion_xl_base.conftest import get_device_name

import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from transformers import CLIPTextModelWithProjection, CLIPTextModel
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    batch_encode_prompt_on_device,
    create_tt_clip_text_encoders,
    retrieve_timesteps,
    run_tt_image_gen,
    prepare_input_tensors,
    allocate_input_tensors,
    create_user_tensors,
    warmup_tt_text_encoders,
)
import os
from models.utility_functions import profiler


@torch.no_grad()
def run_demo_inference(
    ttnn_device,
    is_ci_env,
    prompts,
    num_inference_steps,
    vae_on_device,
    encoders_on_device,
    evaluation_range,
    capture_trace,
):
    print("device name: ", get_device_name())
    print("num_avail_devices: ", ttnn.GetNumAvailableDevices()) 
    print("cluster type: ", ttnn.cluster.get_cluster_type())
    print("is_ci_env!!!!!!!: ", is_ci_env)
    
    batch_size = ttnn_device.get_num_devices()

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
    profiler.start("diffusion_pipeline_from_pretrained")
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    profiler.end("diffusion_pipeline_from_pretrained")

    assert isinstance(pipeline.text_encoder, CLIPTextModel), "pipeline.text_encoder is not a CLIPTextModel"
    assert isinstance(
        pipeline.text_encoder_2, CLIPTextModelWithProjection
    ), "pipeline.text_encoder_2 is not a CLIPTextModelWithProjection"

    # Have to throttle matmuls due to di/dt
    if is_galaxy():
        logger.info("Setting TT_MM_THROTTLE_PERF for Galaxy")
        os.environ["TT_MM_THROTTLE_PERF"] = "5"
    elif is_ci_env and ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.T3K:
        print("aaaaaa")
        logger.info("Setting TT_MM_THROTTLE_PERF for CI on T3K")
        os.environ["TT_MM_THROTTLE_PERF"] = "5"

    profiler.start("load_tt_componenets")
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
    ttnn.synchronize_device(ttnn_device)
    profiler.end("load_tt_componenets")

    cpu_device = "cpu"

    if encoders_on_device:
        logger.info("Encoding prompts on device...")
        # TT text encoder setup
        tt_text_encoder, tt_text_encoder_2 = create_tt_clip_text_encoders(pipeline, ttnn_device)

        # program cache for text encoders
        warmup_tt_text_encoders(
            tt_text_encoder, tt_text_encoder_2, pipeline.tokenizer, pipeline.tokenizer_2, ttnn_device, batch_size
        )

        all_embeds = []
        profiler.start("encode_prompts")
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_embeds = batch_encode_prompt_on_device(
                pipeline,
                tt_text_encoder,
                tt_text_encoder_2,
                ttnn_device,
                prompt=batch_prompts,  # Pass the entire batch
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
            # batch_encode_prompt_on_device returns a single tuple of 4 tensors,
            # but we need individual tuples for each prompt
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = batch_embeds
            # Split the tensors by batch dimension and create individual tuples
            for j in range(len(batch_prompts)):
                all_embeds.append(
                    (
                        prompt_embeds[j : j + 1],  # Keep batch dimension
                        negative_prompt_embeds[j : j + 1] if negative_prompt_embeds is not None else None,
                        pooled_prompt_embeds[j : j + 1],
                        negative_pooled_prompt_embeds[j : j + 1] if negative_pooled_prompt_embeds is not None else None,
                    )
                )
    else:
        logger.info("Encoding prompts on host...")
        # batched impl of host encoding
        profiler.start("encode_prompts")
        all_embeds = pipeline.encode_prompt(
            prompt=prompts,  # Pass the entire list at once
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
        (
            prompt_embeds_batch,
            negative_prompt_embeds_batch,
            pooled_prompt_embeds_batch,
            negative_pooled_prompt_embeds_batch,
        ) = all_embeds
        all_embeds = list(
            zip(
                torch.split(prompt_embeds_batch, 1, dim=0),
                torch.split(negative_prompt_embeds_batch, 1, dim=0),
                torch.split(pooled_prompt_embeds_batch, 1, dim=0),
                torch.split(negative_pooled_prompt_embeds_batch, 1, dim=0),
            )
        )

    profiler.end("encode_prompts")

    profiler.start("prepare_latents")
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

    prompt_embeds_torch = torch.split(torch.cat(prompt_embeds, dim=0), batch_size, dim=0)
    negative_prompt_embeds_torch = torch.split(torch.cat(negative_prompt_embeds, dim=0), batch_size, dim=0)
    pooled_prompt_embeds_torch = torch.split(torch.cat(pooled_prompt_embeds, dim=0), batch_size, dim=0)
    negative_pooled_prompt_embeds_torch = torch.split(
        torch.cat(negative_pooled_prompt_embeds, dim=0), batch_size, dim=0
    )

    # Prepare timesteps
    ttnn_timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler, num_inference_steps, cpu_device, None, None
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
    tt_latents = torch.permute(latents, (0, 2, 3, 1))
    tt_latents = tt_latents.reshape(1, 1, B * H * W, C)
    tt_latents, tt_prompt_embeds, tt_add_text_embeds = create_user_tensors(
        ttnn_device=ttnn_device,
        latents=tt_latents,
        negative_prompt_embeds=negative_prompt_embeds_torch,
        prompt_embeds=prompt_embeds_torch,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_torch,
        add_text_embeds=pooled_prompt_embeds_torch,
    )

    tt_latents_device, tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device = allocate_input_tensors(
        ttnn_device=ttnn_device,
        tt_latents=tt_latents,
        tt_prompt_embeds=tt_prompt_embeds,
        tt_text_embeds=tt_add_text_embeds,
        tt_time_ids=[negative_add_time_ids, add_time_ids],
    )
    ttnn.synchronize_device(ttnn_device)
    profiler.end("prepare_latents")

    profiler.start("warmup_run")
    logger.info("Performing warmup run on denoising, to make use of program caching in actual inference...")
    prepare_input_tensors(
        [
            tt_latents,
            *tt_prompt_embeds[0],
            tt_add_text_embeds[0][0],
            tt_add_text_embeds[0][1],
        ],
        [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
    )
    _, _, _, output_shape, _ = run_tt_image_gen(
        ttnn_device,
        tt_unet,
        tt_scheduler,
        tt_latents_device,
        tt_prompt_embeds_device,
        tt_time_ids_device,
        tt_text_embeds_device,
        [ttnn_timesteps[0]],
        extra_step_kwargs,
        guidance_scale,
        scaling_factor,
        [B, C, H, W],
        tt_vae if vae_on_device else pipeline.vae,
        batch_size,
        capture_trace=False,
    )
    ttnn.synchronize_device(ttnn_device)
    profiler.end("warmup_run")

    tid = None
    output_device = None
    tid_vae = None
    if capture_trace:
        logger.info("Capturing model trace...")
        profiler.start("capture_model_trace")
        prepare_input_tensors(
            [
                tt_latents,
                *tt_prompt_embeds[0],
                tt_add_text_embeds[0][0],
                tt_add_text_embeds[0][1],
            ],
            [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
        )
        _, tid, output_device, output_shape, tid_vae = run_tt_image_gen(
            ttnn_device,
            tt_unet,
            tt_scheduler,
            tt_latents_device,
            tt_prompt_embeds_device,
            tt_time_ids_device,
            tt_text_embeds_device,
            [ttnn_timesteps[0]],
            extra_step_kwargs,
            guidance_scale,
            scaling_factor,
            [B, C, H, W],
            tt_vae if vae_on_device else pipeline.vae,
            batch_size,
            capture_trace=True,
        )
        ttnn.synchronize_device(ttnn_device)
        profiler.end("capture_model_trace")

    logger.info("=" * 80)
    for key, data in profiler.times.items():
        logger.info(f"{key}: {data[-1]:.2f} seconds")
    logger.info("=" * 80)

    profiler.clear()

    if not is_ci_env and not os.path.exists("output"):
        os.mkdir("output")

    images = []
    logger.info("Starting ttnn inference...")
    for iter in range(len(prompts) // batch_size):
        logger.info(
            f"Running inference for prompts {iter * batch_size + 1}-{iter * batch_size + batch_size}/{len(prompts)}"
        )
        prepare_input_tensors(
            [
                tt_latents,
                *tt_prompt_embeds[iter],
                tt_add_text_embeds[iter][0],
                tt_add_text_embeds[iter][1],
            ],
            [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
        )
        imgs, tid, output_device, output_shape, tid_vae = run_tt_image_gen(
            ttnn_device,
            tt_unet,
            tt_scheduler,
            tt_latents_device,
            tt_prompt_embeds_device,
            tt_time_ids_device,
            tt_text_embeds_device,
            ttnn_timesteps,
            extra_step_kwargs,
            guidance_scale,
            scaling_factor,
            [B, C, H, W],
            tt_vae if vae_on_device else pipeline.vae,
            batch_size,
            tid=tid,
            output_device=output_device,
            output_shape=output_shape,
            tid_vae=tid_vae,
        )

        logger.info(
            f"Prepare input tensors for {batch_size} prompts completed in {profiler.times['prepare_input_tensors'][-1]:.2f} seconds"
        )
        logger.info(f"Image gen for {batch_size} prompts completed in {profiler.times['image_gen'][-1]:.2f} seconds")
        logger.info(
            f"Denoising loop for {batch_size} promts completed in {profiler.times['denoising_loop'][-1]:.2f} seconds"
        )
        logger.info(
            f"{'On device VAE' if vae_on_device else 'Host VAE'} decoding completed in {profiler.times['vae_decode'][-1]:.2f} seconds"
        )
        logger.info(f"Output tensor read completed in {profiler.times['read_output_tensor'][-1]:.2f} seconds")

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
    if capture_trace:
        ttnn.release_trace(ttnn_device, tid)
        if vae_on_device:
            ttnn.release_trace(ttnn_device, tid_vae)
    return images


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE, "trace_region_size": SDXL_TRACE_REGION_SIZE}], indirect=True
)
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
@pytest.mark.parametrize(
    "encoders_on_device",
    [
        (True),
        (False),
    ],
    ids=("device_encoders", "host_encoders"),
)
@pytest.mark.parametrize(
    "capture_trace",
    [
        (True),
        (False),
    ],
    ids=("with_trace", "no_trace"),
)
def test_demo(
    mesh_device,
    is_ci_env,
    prompt,
    num_inference_steps,
    vae_on_device,
    encoders_on_device,
    capture_trace,
    evaluation_range,
):
    return run_demo_inference(
        mesh_device,
        is_ci_env,
        prompt,
        num_inference_steps,
        vae_on_device,
        encoders_on_device,
        evaluation_range,
        capture_trace,
    )

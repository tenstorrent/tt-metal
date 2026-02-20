# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
from loguru import logger
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    retrieve_timesteps,
    run_tt_denoising,
    run_torch_denoising,
    prepare_input_tensors,
    allocate_input_tensors,
    create_user_tensors,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc
import matplotlib.pyplot as plt
from models.common.utility_functions import is_wormhole_b0

# TODO: test 20 instead of 10 unet iterations
UNET_LOOP_PCC = {"10": 0.90, "50": 0.92}


@torch.no_grad()
def run_unet_inference(ttnn_device, is_ci_env, prompts, num_inference_steps, debug_mode):
    torch.manual_seed(0)

    if isinstance(prompts, str):
        prompts = [prompts]

    guidance_scale = 5.0

    # 0. Set up default height and width for unet
    height = 1024
    width = 1024

    # 1. Load components
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )

    # 2. Load tt_unet and tt_scheduler
    tt_model_config = ModelOptimisations()
    tt_unet = TtUNet2DConditionModel(
        ttnn_device,
        pipeline.unet.state_dict(),
        "unet",
        model_config=tt_model_config,
        debug_mode=debug_mode,
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
        prompt_embeds.append(prompt_embed)
        negative_prompt_embeds.append(negative_prompt_embed)
        pooled_prompt_embeds.append(pooled_prompt_embed)
        negative_pooled_prompt_embeds.append(negative_pooled_prompt_embed)

    # Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, cpu_device, None, None)
    ttnn_timesteps, tt_num_inference_steps = retrieve_timesteps(
        tt_scheduler, num_inference_steps, cpu_device, None, None
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
    B, C, H, W = latents.shape

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

    tt_latents = torch.permute(latents, (0, 2, 3, 1))
    tt_latents = tt_latents.reshape(1, 1, B * H * W, C)

    tt_latents, tt_prompt_embeds, tt_add_text_embeds = create_user_tensors(
        ttnn_device=ttnn_device,
        latents=tt_latents,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds=prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        add_text_embeds=add_text_embeds,
    )

    tt_latents_device, tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device = allocate_input_tensors(
        ttnn_device=ttnn_device,
        tt_latents=tt_latents,
        tt_prompt_embeds=tt_prompt_embeds,
        tt_text_embeds=tt_add_text_embeds,
        tt_time_ids=[negative_add_time_ids, add_time_ids],
    )

    prompt_embeds = [torch.cat([t1, t2], dim=0) for t1, t2 in zip(negative_prompt_embeds, prompt_embeds)]
    add_text_embeds = [torch.cat([t1, t2], dim=0) for t1, t2 in zip(negative_pooled_prompt_embeds, add_text_embeds)]
    add_time_ids = [torch.cat([t1, t2], dim=0) for t1, t2 in zip(negative_add_time_ids, add_time_ids)]
    added_cond_kwargs = [{"text_embeds": t1, "time_ids": t2} for t1, t2 in zip(add_text_embeds, add_time_ids)]

    logger.info("Performing warmup run, to make use of program caching in actual inference...")

    prepare_input_tensors(
        [
            tt_latents,
            *tt_prompt_embeds[0],
            tt_add_text_embeds[0][0],
            tt_add_text_embeds[0][1],
        ],
        [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
    )
    run_tt_denoising(
        ttnn_device=ttnn_device,
        tt_latents_device=tt_latents_device,
        tt_latents_output=None,
        tt_unet=tt_unet,
        tt_scheduler=tt_scheduler,
        input_shape=[B, C, H, W],
        ttnn_prompt_embeds=tt_prompt_embeds_device,
        ttnn_add_text_embeds=tt_text_embeds_device,
        ttnn_add_time_ids=tt_time_ids_device,
        guidance_scale=guidance_scale,
        extra_step_kwargs=extra_step_kwargs,
        tid=None,
        compile_run=True,
    )

    tid = None
    if not debug_mode:
        prepare_input_tensors(
            [
                tt_latents,
                *tt_prompt_embeds[0],
                tt_add_text_embeds[0][0],
                tt_add_text_embeds[0][1],
            ],
            [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
        )
        tid, _, _, _ = run_tt_denoising(
            ttnn_device=ttnn_device,
            tt_latents_device=tt_latents_device,
            tt_latents_output=None,
            tt_unet=tt_unet,
            tt_scheduler=tt_scheduler,
            input_shape=[B, C, H, W],
            ttnn_prompt_embeds=tt_prompt_embeds_device,
            ttnn_add_text_embeds=tt_text_embeds_device,
            ttnn_add_time_ids=tt_time_ids_device,
            guidance_scale=guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
            tid=None,
        )

    ttnn.synchronize_device(ttnn_device)
    pcc_per_iter = []
    tt_latents_output = None
    logger.info("Starting ttnn inference...")
    for iter in range(len(prompts)):
        prepare_input_tensors(
            [
                tt_latents,
                *tt_prompt_embeds[iter],
                tt_add_text_embeds[iter][0],
                tt_add_text_embeds[iter][1],
            ],
            [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
        )
        logger.info(f"Running inference for prompt {iter + 1}/{len(prompts)}: {prompts[iter]}")
        for i, (t, tt_t) in tqdm(enumerate(zip(timesteps, ttnn_timesteps)), total=len(ttnn_timesteps)):
            tid, tt_latents_device, tt_latents_output, [C, H, W] = run_tt_denoising(
                ttnn_device=ttnn_device,
                tt_latents_device=tt_latents_device,
                tt_latents_output=tt_latents_output,
                tt_unet=tt_unet,
                tt_scheduler=tt_scheduler,
                input_shape=[B, C, H, W],
                ttnn_prompt_embeds=tt_prompt_embeds_device,
                ttnn_add_text_embeds=tt_text_embeds_device,
                ttnn_add_time_ids=tt_time_ids_device,
                guidance_scale=guidance_scale,
                extra_step_kwargs=extra_step_kwargs,
                tid=tid,
                compile_run=debug_mode,
            )
            latents = run_torch_denoising(
                latents=latents,
                iter=iter,
                pipeline=pipeline,
                prompt_embeds=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                t=t,
                guidance_scale=guidance_scale,
                extra_step_kwargs=extra_step_kwargs,
            )

            ttnn.synchronize_device(ttnn_device)
            if i < (len(ttnn_timesteps) - 1):
                tt_scheduler.inc_step_index()

            torch_tt_latents = tt_latents_device.cpu(blocking=False)
            ttnn.synchronize_device(ttnn_device)
            torch_tt_latents = ttnn.to_torch(torch_tt_latents)
            torch_tt_latents = torch.reshape(torch_tt_latents, (B, H, W, C))
            torch_tt_latents = torch.permute(torch_tt_latents, (0, 3, 1, 2))

            _, pcc_message = comp_pcc(latents, torch_tt_latents, 0.8)
            logger.info(f"PCC of {i}. iteration is: {pcc_message}")
            pcc_per_iter.append(float(pcc_message))

        tt_scheduler.set_step_index(0)
    if tid is not None:
        ttnn.release_trace(ttnn_device, tid)
    if not is_ci_env:
        plt.plot(pcc_per_iter, marker="o")
        plt.title("PCC per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("PCC")
        plt.grid(True)
        plt.savefig("pcc_plot.png", dpi=300, bbox_inches="tight")
        plt.close()

    _, pcc_message = assert_with_pcc(latents, torch_tt_latents, UNET_LOOP_PCC.get(str(num_inference_steps), 0))
    logger.info(f"PCC of the last iteration is: {pcc_message}")


@pytest.mark.skipif(not is_wormhole_b0(), reason="SDXL supported on WH only")
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE, "trace_region_size": SDXL_TRACE_REGION_SIZE}], indirect=True
)
@pytest.mark.parametrize(
    "prompt",
    (("An astronaut riding a green horse"),),
)
@pytest.mark.timeout(3000)
def test_unet_loop(
    device,
    is_ci_env,
    prompt,
    loop_iter_num,
    debug_mode,
):
    return run_unet_inference(device, is_ci_env, prompt, loop_iter_num, debug_mode)

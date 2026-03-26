# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

os.environ["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"] = "35000"

import matplotlib.pyplot as plt
import pytest
import torch
import tracy
from diffusers import DiffusionPipeline
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.stable_diffusion_xl_base.tests.test_common import (
    SDXL_BASE_REFINER_TRACE_REGION_SIZE,
    SDXL_FABRIC_CONFIG,
    allocate_input_tensors,
    create_user_tensors,
    metal_command_queue_trace_supported,
    prepare_device,
    prepare_input_tensors,
    retrieve_timesteps,
    run_torch_denoising,
    run_tt_denoising,
    run_tt_iteration,
)
from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.demos.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc

# TODO: test 20 instead of 10 unet iterations
UNET_LOOP_PCC = {
    "1024x1024": {"10": 0.93, "50": 0.913},
    "512x512": {"10": 0.84, "50": 0.917},
}

UNET_LOOP_SEED = {
    "1024x1024": {"10": 42, "50": 0},
    "512x512": {"10": 123, "50": 1024},
}


def _allocate_cfg_parallel_like_pipeline(ttnn_device, tt_latents_host, tt_pe_host, tt_te_host, torch_time_ids):
    """Match TtSDXLPipeline.__allocate_device_tensors (single fused batch row)."""
    is_mesh = isinstance(ttnn_device, ttnn._ttnn.multi_device.MeshDevice)
    tt_latents_device = ttnn.allocate_tensor_on_device(
        tt_latents_host.shape,
        tt_latents_host.dtype,
        tt_latents_host.layout,
        ttnn_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_prompt_embeds_device = ttnn.allocate_tensor_on_device(
        tt_pe_host[0].shape,
        tt_pe_host[0].dtype,
        tt_pe_host[0].layout,
        ttnn_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_text_embeds_device = ttnn.allocate_tensor_on_device(
        tt_te_host[0].shape,
        tt_te_host[0].dtype,
        tt_te_host[0].layout,
        ttnn_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_time_ids_device = ttnn.from_torch(
        torch_time_ids,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=ttnn_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_device, list(ttnn_device.shape), dims=(0, None)) if is_mesh else None,
    )
    tt_time_ids_device = ttnn.squeeze(tt_time_ids_device, dim=0)
    ttnn.synchronize_device(ttnn_device)
    return tt_latents_device, tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device


def _create_user_tensors_cfg_parallel_like_pipeline(
    ttnn_device, latents_torch, all_prompt_embeds_torch, torch_add_text_embeds
):
    """Match TtSDXLPipeline.__create_user_tensors mesh mappers."""
    is_mesh = isinstance(ttnn_device, ttnn._ttnn.multi_device.MeshDevice)
    tt_latents = ttnn.from_torch(
        latents_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_device, list(ttnn_device.shape), dims=(None, 0)) if is_mesh else None,
    )
    tt_prompt_embeds = ttnn.from_torch(
        all_prompt_embeds_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_device, list(ttnn_device.shape), dims=(1, 0)) if is_mesh else None,
    )
    tt_add_text_embeds = ttnn.from_torch(
        torch_add_text_embeds,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_device, list(ttnn_device.shape), dims=(1, 0)) if is_mesh else None,
    )
    ttnn.synchronize_device(ttnn_device)
    return tt_latents, tt_prompt_embeds, tt_add_text_embeds


def _split_cfg_parallel_branches_for_run_tt_denoising(
    tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device
):
    """Slice fused [1,2,...] device tensors into two branches for run_tt_denoising."""
    s = list(tt_prompt_embeds_device.shape)
    pe0 = ttnn.slice(tt_prompt_embeds_device, [0, 0, 0, 0], [1, 1, s[2], s[3]])
    pe1 = ttnn.slice(tt_prompt_embeds_device, [0, 1, 0, 0], [1, 2, s[2], s[3]])
    if int(pe0.shape[1]) == 1:
        pe0 = ttnn.squeeze(pe0, 1)
        pe1 = ttnn.squeeze(pe1, 1)

    st = list(tt_text_embeds_device.shape)
    te0 = ttnn.slice(tt_text_embeds_device, [0, 0, 0], [1, 1, st[2]])
    te1 = ttnn.slice(tt_text_embeds_device, [0, 1, 0], [1, 2, st[2]])
    if int(te0.shape[1]) == 1:
        te0 = ttnn.squeeze(te0, 1)
        te1 = ttnn.squeeze(te1, 1)

    ts = list(tt_time_ids_device.shape)
    td0 = ttnn.slice(tt_time_ids_device, [0, 0], [1, ts[1]])
    td1 = ttnn.slice(tt_time_ids_device, [1, 0], [2, ts[1]])

    return [pe0, pe1], [te0, te1], [td0, td1]


def _run_tt_denoising_cfg_parallel(
    ttnn_device,
    tt_latents_device,
    tt_unet,
    tt_scheduler,
    input_shape,
    ttnn_prompt_embeds,
    ttnn_add_text_embeds,
    ttnn_add_time_ids,
    guidance_scale,
    extra_step_kwargs,
    tid=None,
    compile_run=False,
):
    """CFG-parallel step: fused UNet forward + all_gather + guidance combine."""
    B, C, H, W = input_shape
    do_metal_trace = (not compile_run) and metal_command_queue_trace_supported()
    if tid is None:
        tid = ttnn.begin_trace_capture(ttnn_device, cq_id=0) if do_metal_trace else None
        noise_pred, noise_shape = run_tt_iteration(
            tt_unet,
            tt_scheduler,
            tt_latents_device,
            [B, C, H, W],
            ttnn_prompt_embeds,
            ttnn_add_time_ids,
            ttnn_add_text_embeds,
        )
        C, H, W = noise_shape

        noise_pred_interleaved = ttnn.to_memory_config(noise_pred, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(noise_pred)
        noise_pred = ttnn.all_gather(
            noise_pred_interleaved,
            dim=0,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(noise_pred_interleaved)
        noise_pred = noise_pred[..., :4]
        noise_pred_uncond = ttnn.unsqueeze(noise_pred[0], 0)
        noise_pred_text = ttnn.unsqueeze(noise_pred[1], 0)

        noise_pred_text = ttnn.sub_(noise_pred_text, noise_pred_uncond)
        noise_pred_text = ttnn.mul_(noise_pred_text, guidance_scale)
        noise_pred = ttnn.add_(noise_pred_uncond, noise_pred_text)

        tt_latents_device = tt_scheduler.step(
            noise_pred, None, tt_latents_device, **extra_step_kwargs, return_dict=False
        )[0]
        ttnn.deallocate(noise_pred_uncond)
        ttnn.deallocate(noise_pred_text)

        if do_metal_trace:
            ttnn.end_trace_capture(ttnn_device, tid, cq_id=0)
    else:
        ttnn.execute_trace(ttnn_device, tid, cq_id=0, blocking=True)
    return tid, tt_latents_device, [C, H, W]


@torch.no_grad()
def run_unet_inference(
    ttnn_device,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_pipeline_location,
    image_resolution,
    prompts,
    num_inference_steps,
    debug_mode,
    use_cfg_parallel=False,
):
    prepare_device(ttnn_device, use_cfg_parallel)

    # Get seed from configuration
    height, width = image_resolution
    resolution_key = f"{height}x{width}"
    seed = UNET_LOOP_SEED.get(resolution_key, {}).get(str(num_inference_steps), 0)
    torch.manual_seed(seed)

    if isinstance(prompts, str):
        prompts = [prompts]

    guidance_scale = 5.0

    # 1. Load components - use CIv2 LFC when available
    pipeline = DiffusionPipeline.from_pretrained(
        sdxl_base_pipeline_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
    )

    # 2. Load tt_unet and tt_scheduler (same as TtSDXLPipeline.__load_tt_components: replicate to mesh)
    tt_model_config = load_model_optimisations(image_resolution)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(ttnn_device)):
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

    positive_prompt_embeds = prompt_embeds

    if use_cfg_parallel:
        # Same tensor layout as TtSDXLPipeline.generate_input_tensors / __create_user_tensors / __allocate_device_tensors
        all_prompt_embeds_torch = torch.cat(
            [torch.stack(negative_prompt_embeds, dim=0), torch.stack(positive_prompt_embeds, dim=0)], dim=1
        )
        torch_add_text_embeds_combined = torch.cat(
            [torch.stack(negative_pooled_prompt_embeds, dim=0), torch.stack(pooled_prompt_embeds, dim=0)], dim=1
        )
        torch_add_time_ids = torch.stack([negative_add_time_ids.squeeze(0), add_time_ids.squeeze(0)], dim=0)

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = _create_user_tensors_cfg_parallel_like_pipeline(
            ttnn_device,
            tt_latents,
            all_prompt_embeds_torch,
            torch_add_text_embeds_combined,
        )
        (
            tt_latents_device,
            tt_prompt_embeds_device,
            tt_text_embeds_device,
            tt_time_ids_device,
        ) = _allocate_cfg_parallel_like_pipeline(
            ttnn_device,
            tt_latents,
            tt_prompt_embeds,
            tt_add_text_embeds,
            torch_add_time_ids,
        )
    else:
        tt_latents, tt_prompt_embeds, tt_add_text_embeds = create_user_tensors(
            ttnn_device=ttnn_device,
            latents=tt_latents,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds=positive_prompt_embeds,
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

    prompt_embeds = [torch.cat([t1, t2], dim=0) for t1, t2 in zip(negative_prompt_embeds, positive_prompt_embeds)]
    add_text_embeds = [torch.cat([t1, t2], dim=0) for t1, t2 in zip(negative_pooled_prompt_embeds, add_text_embeds)]
    add_time_ids = [torch.cat([t1, t2], dim=0) for t1, t2 in zip(negative_add_time_ids, add_time_ids)]
    added_cond_kwargs = [{"text_embeds": t1, "time_ids": t2} for t1, t2 in zip(add_text_embeds, add_time_ids)]

    def prepare_iter_and_branch_tensors(iter_idx):
        """Copy host→device for prompt iter; return tensors for current run mode."""
        if use_cfg_parallel:
            # Same host rows as __allocate_device_tensors (tt_pe_host[0].shape); do not rebuild with
            # from_torch per iter — that can diverge logical_shape from the allocated device buffers.
            prepare_input_tensors(
                [tt_latents, tt_prompt_embeds[iter_idx], tt_add_text_embeds[iter_idx]],
                [tt_latents_device, tt_prompt_embeds_device, tt_text_embeds_device],
            )
            return tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device
        prepare_input_tensors(
            [
                tt_latents,
                *tt_prompt_embeds[iter_idx],
                tt_add_text_embeds[iter_idx][0],
                tt_add_text_embeds[iter_idx][1],
            ],
            [tt_latents_device, *tt_prompt_embeds_device, *tt_text_embeds_device],
        )
        return tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device

    logger.info("Performing warmup run, to make use of program caching in actual inference...")

    ped, ted, tidd = prepare_iter_and_branch_tensors(0)
    if use_cfg_parallel:
        _run_tt_denoising_cfg_parallel(
            ttnn_device=ttnn_device,
            tt_latents_device=tt_latents_device,
            tt_unet=tt_unet,
            tt_scheduler=tt_scheduler,
            input_shape=[B, C, H, W],
            ttnn_prompt_embeds=ped,
            ttnn_add_text_embeds=ted,
            ttnn_add_time_ids=tidd,
            guidance_scale=guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
            tid=None,
            compile_run=True,
        )
    else:
        run_tt_denoising(
            ttnn_device=ttnn_device,
            tt_latents_device=tt_latents_device,
            tt_latents_output=None,
            tt_unet=tt_unet,
            tt_scheduler=tt_scheduler,
            input_shape=[B, C, H, W],
            ttnn_prompt_embeds=ped,
            ttnn_add_text_embeds=ted,
            ttnn_add_time_ids=tidd,
            guidance_scale=guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
            tid=None,
            compile_run=True,
        )

    tid = None
    if not debug_mode:
        ped, ted, tidd = prepare_iter_and_branch_tensors(0)
        if use_cfg_parallel:
            tid, _, _ = _run_tt_denoising_cfg_parallel(
                ttnn_device=ttnn_device,
                tt_latents_device=tt_latents_device,
                tt_unet=tt_unet,
                tt_scheduler=tt_scheduler,
                input_shape=[B, C, H, W],
                ttnn_prompt_embeds=ped,
                ttnn_add_text_embeds=ted,
                ttnn_add_time_ids=tidd,
                guidance_scale=guidance_scale,
                extra_step_kwargs=extra_step_kwargs,
                tid=None,
            )
        else:
            tid, _, _, _ = run_tt_denoising(
                ttnn_device=ttnn_device,
                tt_latents_device=tt_latents_device,
                tt_latents_output=None,
                tt_unet=tt_unet,
                tt_scheduler=tt_scheduler,
                input_shape=[B, C, H, W],
                ttnn_prompt_embeds=ped,
                ttnn_add_text_embeds=ted,
                ttnn_add_time_ids=tidd,
                guidance_scale=guidance_scale,
                extra_step_kwargs=extra_step_kwargs,
                tid=None,
            )

    ttnn.synchronize_device(ttnn_device)
    pcc_per_iter = []
    tt_latents_output = None
    logger.info("Starting ttnn inference...")
    tracy.signpost("VALID_START")
    for iter in range(len(prompts)):
        ped, ted, tidd = prepare_iter_and_branch_tensors(iter)
        logger.info(f"Running inference for prompt {iter + 1}/{len(prompts)}: {prompts[iter]}")
        for i, (t, tt_t) in tqdm(enumerate(zip(timesteps, ttnn_timesteps)), total=len(ttnn_timesteps)):
            if use_cfg_parallel:
                tid, tt_latents_device, [C, H, W] = _run_tt_denoising_cfg_parallel(
                    ttnn_device=ttnn_device,
                    tt_latents_device=tt_latents_device,
                    tt_unet=tt_unet,
                    tt_scheduler=tt_scheduler,
                    input_shape=[B, C, H, W],
                    ttnn_prompt_embeds=ped,
                    ttnn_add_text_embeds=ted,
                    ttnn_add_time_ids=tidd,
                    guidance_scale=guidance_scale,
                    extra_step_kwargs=extra_step_kwargs,
                    tid=tid,
                    compile_run=debug_mode,
                )
            else:
                tid, tt_latents_device, tt_latents_output, [C, H, W] = run_tt_denoising(
                    ttnn_device=ttnn_device,
                    tt_latents_device=tt_latents_device,
                    tt_latents_output=tt_latents_output,
                    tt_unet=tt_unet,
                    tt_scheduler=tt_scheduler,
                    input_shape=[B, C, H, W],
                    ttnn_prompt_embeds=ped,
                    ttnn_add_text_embeds=ted,
                    ttnn_add_time_ids=tidd,
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
            mesh_composer = (
                ttnn.ConcatMeshToTensor(ttnn_device, dim=0)
                if isinstance(ttnn_device, ttnn._ttnn.multi_device.MeshDevice)
                else None
            )
            torch_tt_latents = ttnn.to_torch(torch_tt_latents, mesh_composer=mesh_composer)
            # Same as run_tt_image_gen readback: concat on dim=0 stacks device shards; take batch slice only.
            if mesh_composer is not None:
                torch_tt_latents = torch_tt_latents[:B, ...]
            torch_tt_latents = torch.reshape(torch_tt_latents, (B, H, W, C))
            torch_tt_latents = torch.permute(torch_tt_latents, (0, 3, 1, 2))

            _, pcc_message = comp_pcc(latents, torch_tt_latents, 0.8)
            logger.info(f"PCC of {i}. iteration is: {pcc_message}")
            pcc_per_iter.append(float(pcc_message))

        tt_scheduler.set_step_index(0)
    tracy.signpost("VALID_END")
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

    resolution_key = f"{height}x{width}"
    pcc_threshold = UNET_LOOP_PCC.get(resolution_key, {}).get(str(num_inference_steps), 0)
    _, pcc_message = assert_with_pcc(latents, torch_tt_latents, pcc_threshold)
    logger.info(f"PCC of the last iteration is: {pcc_message}")


@pytest.mark.parametrize(
    "image_resolution",
    [
        # 1024x1024 image resolution
        (1024, 1024),
    ],
    ids=["1024x1024"],
)
@pytest.mark.parametrize(
    "device_params, use_cfg_parallel",
    [
        (
            {
                "trace_region_size": SDXL_BASE_REFINER_TRACE_REGION_SIZE,
                "fabric_config": SDXL_FABRIC_CONFIG,
            },
            True,
        ),
        (
            {
                "trace_region_size": SDXL_BASE_REFINER_TRACE_REGION_SIZE,
            },
            False,
        ),
    ],
    indirect=["device_params"],
    ids=["use_cfg_parallel", "no_cfg_parallel"],
)
@pytest.mark.parametrize(
    "prompt",
    (("An astronaut riding a green horse"),),
)
@pytest.mark.timeout(3300)
def test_unet_loop(
    validate_fabric_compatibility,
    mesh_device,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_pipeline_location,
    image_resolution,
    prompt,
    use_cfg_parallel,
    loop_iter_num,
    debug_mode,
):
    if image_resolution == (512, 512) and is_blackhole():
        pytest.skip("512x512 not supported on Blackhole")
    return run_unet_inference(
        mesh_device,
        is_ci_env,
        is_ci_v2_env,
        sdxl_base_pipeline_location,
        image_resolution,
        prompt,
        loop_iter_num,
        debug_mode,
        use_cfg_parallel=use_cfg_parallel,
    )

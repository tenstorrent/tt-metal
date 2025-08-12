# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import ttnn
import torch
import inspect
from typing import List, Optional, Union
from models.utility_functions import profiler

from tqdm import tqdm
import ttnn

from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL

SDXL_L1_SMALL_SIZE = 47000
SDXL_TRACE_REGION_SIZE = 34000000
SDXL_CI_WEIGHTS_PATH = "/mnt/MLPerf/tt_dnn-models/hf_home"


# Copied from sdxl pipeline
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def run_tt_iteration(
    tt_unet,
    tt_scheduler,
    input_tensor,
    input_shape,
    ttnn_prompt_embeds,
    time_ids,
    text_embeds,
):
    B, C, H, W = input_shape

    input_tensor = tt_scheduler.scale_model_input(input_tensor, None)
    ttnn_noise_pred, output_shape = tt_unet.forward(
        input_tensor,
        [B, C, H, W],
        timestep=tt_scheduler.tt_timestep,
        encoder_hidden_states=ttnn_prompt_embeds,
        time_ids=time_ids,
        text_embeds=text_embeds,
    )

    return ttnn_noise_pred, output_shape


# Runs a single iteration of the tt image generation
# This includes the following steps:
# - n denoising loops
# - vae
def run_tt_image_gen(
    ttnn_device,
    tt_unet,
    tt_scheduler,
    tt_latents,
    tt_prompt_embeds,
    tt_time_ids,
    tt_text_embeds,
    tt_timesteps,
    tt_extra_step_kwargs,
    guidance_scale,
    scaling_factor,
    input_shape,
    vae,  # can be host vae or tt vae
    batch_size,
    output_device=None,
    output_shape=None,
    tid=None,
    tid_vae=None,
    capture_trace=False,
):
    assert not (capture_trace and len(tt_timesteps) != 1), "Trace should capture only 1 iteration"
    profiler.start("image_gen")
    profiler.start("denoising_loop")

    for i, t in tqdm(enumerate(tt_timesteps), total=len(tt_timesteps)):
        unet_outputs = []
        if tid is None or capture_trace:
            tid = ttnn.begin_trace_capture(ttnn_device, cq_id=0) if capture_trace else None
            for unet_slice in range(len(tt_time_ids)):
                latent_model_input = tt_latents
                noise_pred, _ = run_tt_iteration(
                    tt_unet,
                    tt_scheduler,
                    latent_model_input,
                    input_shape,
                    tt_prompt_embeds[unet_slice],
                    tt_time_ids[unet_slice],
                    tt_text_embeds[unet_slice],
                )

                unet_outputs.append(noise_pred)

            # perform guidance
            noise_pred_uncond, noise_pred_text = unet_outputs
            noise_pred_text = ttnn.sub_(noise_pred_text, noise_pred_uncond)
            noise_pred_text = ttnn.mul_(noise_pred_text, guidance_scale)
            noise_pred = ttnn.add_(noise_pred_uncond, noise_pred_text)

            tt_latents = tt_scheduler.step(noise_pred, None, tt_latents, **tt_extra_step_kwargs, return_dict=False)[0]

            ttnn.deallocate(noise_pred_uncond)
            ttnn.deallocate(noise_pred_text)

            if capture_trace:
                ttnn.end_trace_capture(ttnn_device, tid, cq_id=0)
        else:
            ttnn.execute_trace(ttnn_device, tid, cq_id=0, blocking=False)

        if i < (len(tt_timesteps) - 1):
            tt_scheduler.inc_step_index()

    ttnn.synchronize_device(ttnn_device)

    # reset scheduler
    tt_scheduler.set_step_index(0)

    profiler.end("denoising_loop")

    vae_on_device = isinstance(vae, TtAutoencoderKL)

    profiler.start("vae_decode")
    if vae_on_device:
        if tid_vae is None or capture_trace:
            tid_vae = ttnn.begin_trace_capture(ttnn_device, cq_id=0) if capture_trace else None
            tt_latents = ttnn.div(tt_latents, scaling_factor)

            logger.info("Running TT VAE")
            output_tensor, [C, H, W] = vae.forward(tt_latents, input_shape)
            ttnn.deallocate(tt_latents)

            if capture_trace:
                ttnn.end_trace_capture(ttnn_device, tid_vae, cq_id=0)
            output_device = output_tensor
            output_shape = [input_shape[0], C, H, W]
        else:
            ttnn.execute_trace(ttnn_device, tid_vae, cq_id=0, blocking=False)

        ttnn.synchronize_device(ttnn_device)
        output_tensor = ttnn.to_torch(output_device, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_device, dim=0)).float()

        B, C, H, W = output_shape
        output_tensor = output_tensor.reshape(batch_size * B, H, W, C)
        imgs = torch.permute(output_tensor, (0, 3, 1, 2))
    else:
        latents = ttnn.to_torch(tt_latents, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_device, dim=0))
        B, C, H, W = input_shape
        latents = latents.reshape(batch_size * B, H, W, C)
        latents = torch.permute(latents, (0, 3, 1, 2))
        latents = latents.to(vae.dtype)

        # VAE upcasting to float32 is happening in the reference SDXL demo if VAE dtype is float16. If it's bfloat16, it will not be upcasted.
        latents = latents / vae.config.scaling_factor
        warmup_run = len(tt_timesteps) == 1
        if warmup_run == False:
            # Do not run host VAE if we are on a warmup run
            imgs = vae.decode(latents, return_dict=False)[0]
        else:
            imgs = None
        del latents
        gc.collect()
    profiler.end("vae_decode")
    profiler.end("image_gen")

    return imgs, tid, output_device, output_shape, tid_vae


def prepare_input_tensors(host_tensors, device_tensors):
    for host_tensor, device_tensor in zip(host_tensors, device_tensors):
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)


def allocate_input_tensors(ttnn_device, tt_latents, tt_prompt_embeds, tt_text_embeds, tt_time_ids):
    is_mesh_device = isinstance(ttnn_device, ttnn._ttnn.multi_device.MeshDevice)
    tt_latents_device = ttnn.allocate_tensor_on_device(
        tt_latents.shape,
        tt_latents.dtype,
        tt_latents.layout,
        ttnn_device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_prompt_embeds_device = [
        ttnn.allocate_tensor_on_device(
            tt_prompt_embeds[0][0].shape,
            tt_prompt_embeds[0][0].dtype,
            tt_prompt_embeds[0][0].layout,
            ttnn_device,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.allocate_tensor_on_device(
            tt_prompt_embeds[0][1].shape,
            tt_prompt_embeds[0][1].dtype,
            tt_prompt_embeds[0][1].layout,
            ttnn_device,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ]

    tt_text_embeds_device = [
        ttnn.allocate_tensor_on_device(
            tt_text_embeds[0][0].shape,
            tt_text_embeds[0][0].dtype,
            tt_text_embeds[0][0].layout,
            ttnn_device,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        ttnn.allocate_tensor_on_device(
            tt_text_embeds[0][1].shape,
            tt_text_embeds[0][1].dtype,
            tt_text_embeds[0][1].layout,
            ttnn_device,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
    ]

    tt_time_ids_device = [
        ttnn.from_torch(
            tt_time_ids[0].squeeze(0),
            dtype=ttnn.bfloat16,
            device=ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device) if is_mesh_device else None,
        ),
        ttnn.from_torch(
            tt_time_ids[1].squeeze(0),
            dtype=ttnn.bfloat16,
            device=ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device) if is_mesh_device else None,
        ),
    ]

    return tt_latents_device, tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device


def create_user_tensors(
    ttnn_device, latents, negative_prompt_embeds, prompt_embeds, negative_pooled_prompt_embeds, add_text_embeds
):
    is_mesh_device = isinstance(ttnn_device, ttnn._ttnn.multi_device.MeshDevice)
    tt_latents = ttnn.from_torch(
        latents,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device) if is_mesh_device else None,
    )

    tt_prompt_embeds = [
        [
            ttnn.from_torch(
                negative_prompt_embed,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=0) if is_mesh_device else None,
            ),
            ttnn.from_torch(
                prompt_embed,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=0) if is_mesh_device else None,
            ),
        ]
        for negative_prompt_embed, prompt_embed in zip(negative_prompt_embeds, prompt_embeds)
    ]

    tt_add_text_embeds = [
        [
            ttnn.from_torch(
                negative_pooled_prompt_embed,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=0) if is_mesh_device else None,
            ),
            ttnn.from_torch(
                add_text_embed,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=0) if is_mesh_device else None,
            ),
        ]
        for negative_pooled_prompt_embed, add_text_embed in zip(negative_pooled_prompt_embeds, add_text_embeds)
    ]

    return tt_latents, tt_prompt_embeds, tt_add_text_embeds

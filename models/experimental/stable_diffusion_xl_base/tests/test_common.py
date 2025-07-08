# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import torch
import inspect
from typing import List, Optional, Union
from models.utility_functions import profiler

from tqdm import tqdm
import ttnn

from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL

SDXL_L1_SMALL_SIZE = 57344


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
    ttnn_device,
    tt_unet,
    tt_scheduler,
    input_tensor,
    input_shape,
    ttnn_prompt_embeds,
    time_ids,
    text_embeds,
    ttnn_timestep,
    i,
):
    B, C, H, W = input_shape

    input_tensor = tt_scheduler.scale_model_input(input_tensor, tt_scheduler.tt_timesteps[i])
    ttnn_noise_pred, output_shape = tt_unet.forward(
        input_tensor,
        [B, C, H, W],
        timestep=ttnn_timestep,
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
    iter,
):
    profiler.start("denoising_loop")
    for i, t in tqdm(enumerate(tt_timesteps), total=len(tt_timesteps)):
        unet_outputs = []

        latent_model_input = tt_latents
        noise_pred, noise_shape = run_tt_iteration(
            ttnn_device,
            tt_unet,
            tt_scheduler,
            latent_model_input,
            input_shape,
            tt_prompt_embeds[iter],  # ovo sharduj
            tt_time_ids,  # ovo sharduj
            ttnn.unsqueeze(tt_text_embeds[iter], dim=0),  # ovo sharduj, ostalo repliciraj
            t,
            i,
        )
        C, H, W = noise_shape

        unet_outputs.append(noise_pred)
        mem_config = ttnn.create_sharded_memory_config(
            shape=(2, 1, H * W, 8),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        noise_pred = ttnn.to_layout(noise_pred, ttnn.ROW_MAJOR_LAYOUT)
        noise_pred = ttnn.pad(noise_pred, [(0, 0), (0, 0), (0, 0), (0, 4)], 0)
        noise_pred = ttnn.sharded_to_interleaved(noise_pred, ttnn.L1_MEMORY_CONFIG)

        noise_pred = ttnn.all_gather(noise_pred, dim=0, memory_config=mem_config)
        noise_pred = ttnn.to_memory_config(noise_pred, ttnn.DRAM_MEMORY_CONFIG)
        noise_pred = ttnn.to_layout(noise_pred, ttnn.TILE_LAYOUT)

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        ttnn.deallocate(noise_pred_uncond)
        ttnn.deallocate(noise_pred_text)

        noise_pred = noise_pred[..., :4]
        noise_pred = ttnn.unsqueeze(noise_pred, dim=0)

        tt_latents = tt_scheduler.step(
            noise_pred, tt_scheduler.timesteps[i], tt_latents, **tt_extra_step_kwargs, return_dict=False
        )[0]

        ttnn.deallocate(noise_pred)
        tt_latents = ttnn.move(tt_latents)

    # reset scheduler
    tt_scheduler.set_step_index(0)

    ttnn.synchronize_device(ttnn_device)
    profiler.end("denoising_loop")

    vae_on_device = isinstance(vae, TtAutoencoderKL)
    profiler.start("vae_decode")
    if vae_on_device:
        tt_latents = ttnn.div(tt_latents, scaling_factor)

        logger.info("Running TT VAE")
        imgs = vae.forward(tt_latents, input_shape)
        ttnn.deallocate(tt_latents)
        ttnn.synchronize_device(ttnn_device)
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

    return imgs

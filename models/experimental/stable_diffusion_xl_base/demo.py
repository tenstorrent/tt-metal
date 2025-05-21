# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import List, Optional, Union
import pytest
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
from loguru import logger
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL


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


def run_tt_unet(
    ttnn_device, tt_unet, torch_latent_model_input, ttnn_prompt_embeds, ttnn_added_cond_kwargs, ttnn_timestep
):
    B, C, H, W = torch_latent_model_input.shape
    ttnn_latent_model_input = ttnn.from_torch(
        torch_latent_model_input,
        dtype=ttnn.bfloat16,
        device=ttnn_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_latent_model_input = ttnn.permute(ttnn_latent_model_input, (0, 2, 3, 1))
    ttnn_latent_model_input = ttnn.reshape(ttnn_latent_model_input, (1, 1, B * H * W, C))

    ttnn_noise_pred, output_shape = tt_unet.forward(
        ttnn_latent_model_input,
        [B, C, H, W],
        timestep=ttnn_timestep,
        encoder_hidden_states=ttnn_prompt_embeds,
        added_cond_kwargs=ttnn_added_cond_kwargs,
    )

    noise_pred = ttnn.to_torch(ttnn_noise_pred)
    noise_pred = noise_pred.reshape(B, output_shape[1], output_shape[2], output_shape[0])
    noise_pred = torch.permute(noise_pred, (0, 3, 1, 2))
    return noise_pred


@torch.no_grad()
def run_demo_inference(
    ttnn_device,
    prompt,
    num_inference_steps,
    classifier_free_guidance,
    vae_on_device,
):
    torch.manual_seed(0)

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

    # 2. Load tt_unet
    tt_unet = TtUNet2DConditionModel(
        ttnn_device,
        pipeline.unet.state_dict(),
        "unet",
        conv_weights_dtype=ttnn.bfloat16,
        transformer_weights_dtype=ttnn.bfloat16,
    )
    tt_vae = TtAutoencoderKL(ttnn_device, pipeline.vae.state_dict()) if vae_on_device else None

    cpu_device = "cpu"

    # Encode prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
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
        prompt_embeds.dtype,
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
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    negative_add_time_ids = add_time_ids

    if classifier_free_guidance:
        ttnn_prompt_embeds = [
            ttnn.from_torch(
                negative_prompt_embeds,
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.from_torch(
                prompt_embeds,
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        ]
        ttnn_add_text_embeds = [
            ttnn.from_torch(
                negative_pooled_prompt_embeds,
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            ttnn.from_torch(
                add_text_embeds,
                dtype=ttnn.bfloat16,
                device=ttnn_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
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
            {
                "text_embeds": ttnn_add_text_embeds[0],
                "time_ids": ttnn_add_time_ids[0],
            },
            {
                "text_embeds": ttnn_add_text_embeds[1],
                "time_ids": ttnn_add_time_ids[1],
            },
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
                layout=ttnn.TILE_LAYOUT,
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

    logger.info("Performing warmup run, to make use of program caching in actual inference...")
    latent_model_input = latents
    latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timesteps[0])
    run_tt_unet(
        ttnn_device,
        tt_unet,
        latent_model_input,
        ttnn_prompt_embeds[0],
        ttnn_added_cond_kwargs[0],
        ttnn_timesteps[0],
    )

    logger.info("Starting ttnn inference...")
    for i, t in tqdm(enumerate(ttnn_timesteps), total=len(ttnn_timesteps)):
        unet_outputs = []
        for unet_slice in range(len(ttnn_prompt_embeds)):
            latent_model_input = latents
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, timesteps[i])

            noise_pred = run_tt_unet(
                ttnn_device,
                tt_unet,
                latent_model_input,
                ttnn_prompt_embeds[unet_slice],
                ttnn_added_cond_kwargs[unet_slice],
                t,
            )

            unet_outputs.append(noise_pred)

        if len(unet_outputs) > 1:
            noise_pred = torch.cat(unet_outputs, dim=0)
        else:
            noise_pred = unet_outputs[0]
        # perform guidance
        if classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # guidance rescale is 0, skip next step
        latents = pipeline.scheduler.step(noise_pred, timesteps[i], latents, **extra_step_kwargs, return_dict=False)[0]

    latents = latents.to(pipeline.vae.dtype)
    # VAE upcasting to float32 is happening in the reference SDXL demo if VAE dtype is float16. If it's bfloat16, it will not be upcasted.
    latents = latents / pipeline.vae.config.scaling_factor

    if vae_on_device:
        # Workaround for #22017
        ttnn_device.disable_and_clear_program_cache()

        B, C, H, W = list(latents.shape)
        latents = torch.permute(latents, (0, 2, 3, 1))
        latents = latents.reshape(1, 1, B * H * W, C)
        ttnn_latents = ttnn.from_torch(
            latents,
            dtype=ttnn.bfloat16,
            device=ttnn_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        logger.info("Running TT VAE")
        image = tt_vae.forward(ttnn_latents, [B, C, H, W])
    else:
        image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]

    image.save("output.png")
    logger.info(f"Image saved to output.png")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 6 * 16384}], indirect=True)
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
    prompt,
    num_inference_steps,
    classifier_free_guidance,
    vae_on_device,
):
    return run_demo_inference(
        device,
        prompt,
        num_inference_steps,
        classifier_free_guidance,
        vae_on_device=vae_on_device,
    )

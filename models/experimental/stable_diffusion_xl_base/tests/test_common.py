# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import ttnn
import torch
import inspect
from typing import List, Optional, Union

from ttnn.distributed.distributed import ConcatMeshToTensor
from models.experimental.tt_dit.encoders.clip.model_clip import CLIPEncoder, CLIPConfig
from models.experimental.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.common.utility_functions import profiler

import ttnn

from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL

# For basic SDXL demo, L1 small size of 23000 is enough,
# but for inpainting/img2img, we need larger L1 small due
# to having an extra VAE encode call, which increases it.
# For simplicity, increase both to 29000 as there's enough
# space left in base variant as well.
SDXL_L1_SMALL_SIZE = 30000
SDXL_TRACE_REGION_SIZE = 34000000
SDXL_CI_WEIGHTS_PATH = "/mnt/MLPerf/tt_dnn-models/hf_home"
SDXL_FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_1D


def create_tt_clip_text_encoders(pipeline, ttnn_device):
    text_encoder_1 = pipeline.text_encoder
    config_1 = CLIPConfig(
        vocab_size=text_encoder_1.config.vocab_size,
        embed_dim=text_encoder_1.config.hidden_size,
        ff_dim=text_encoder_1.config.intermediate_size,
        num_heads=text_encoder_1.config.num_attention_heads,
        num_hidden_layers=text_encoder_1.config.num_hidden_layers,
        max_prompt_length=77,
        layer_norm_eps=text_encoder_1.config.layer_norm_eps,
        attention_dropout=text_encoder_1.config.attention_dropout,
        hidden_act=text_encoder_1.config.hidden_act,
    )
    ccl_manager = None

    # Note: Factor for SDXL should always be 1; since we don't support TP
    parallel_config_1 = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
    )

    tt_text_encoder = CLIPEncoder(
        config_1, ttnn_device, ccl_manager, parallel_config_1, text_encoder_1.config.eos_token_id
    )
    tt_text_encoder.load_state_dict(text_encoder_1.state_dict())

    text_encoder_2 = pipeline.text_encoder_2
    config_2 = CLIPConfig(
        vocab_size=text_encoder_2.config.vocab_size,
        embed_dim=text_encoder_2.config.hidden_size,
        ff_dim=text_encoder_2.config.intermediate_size,
        num_heads=text_encoder_2.config.num_attention_heads,
        num_hidden_layers=text_encoder_2.config.num_hidden_layers,
        max_prompt_length=77,
        layer_norm_eps=text_encoder_2.config.layer_norm_eps,
        attention_dropout=text_encoder_2.config.attention_dropout,
        hidden_act=text_encoder_2.config.hidden_act,
    )

    # Note: Factor for SDXL should always be 1; since we don't support TP
    parallel_config_2 = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
    )

    tt_text_encoder_2 = CLIPEncoder(
        config_2, ttnn_device, ccl_manager, parallel_config_2, text_encoder_2.config.eos_token_id
    )
    tt_text_encoder_2.load_state_dict(text_encoder_2.state_dict())

    return tt_text_encoder, tt_text_encoder_2


def warmup_tt_text_encoders(tt_text_encoder, tt_text_encoder_2, tokenizer, tokenizer_2, ttnn_device, batch_size):
    logger.info("Performing warmup run on encoding, to make use of program caching in actual inference...")
    batch_size = ttnn_device.get_num_devices()
    dummy_prompt = ["abc"] * batch_size
    dummy_ids = tokenizer(
        dummy_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    dummy_ids_2 = tokenizer(
        dummy_prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    tt_tokens_1 = ttnn.from_torch(
        dummy_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=0),
    )
    tt_tokens_2 = ttnn.from_torch(
        dummy_ids_2,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=0),
    )

    _, _ = tt_text_encoder(tt_tokens_1, ttnn_device, with_projection=False)
    _, _ = tt_text_encoder_2(tt_tokens_2, ttnn_device, with_projection=True)
    ttnn.synchronize_device(ttnn_device)


# encode_prompt function, adapted from sdxl pipeline to work with on device tt text encoders
# batch size (lenght of prompts) must be equal to number of devices
def batch_encode_prompt_on_device(
    pipeline,
    tt_text_encoder,
    tt_text_encoder_2,
    ttnn_device,
    prompt: str,
    prompt_2: Optional[str] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    negative_prompt: Optional[str] = None,
    negative_prompt_2: Optional[str] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    lora_scale: Optional[float] = None,
    clip_skip: Optional[int] = None,
    use_cfg_parallel: bool = False,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            used in both text-encoders
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        lora_scale (`float`, *optional*):
            A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    num_devices = ttnn_device.get_num_devices()
    num_prompts = len(prompt)
    if use_cfg_parallel and num_prompts < num_devices:
        # Pad prompts by appending empty strings to match num_devices
        prompt = prompt + [""] * (num_devices - len(prompt))
        if prompt_2 is not None:
            prompt_2 = prompt_2 + [""] * (num_devices - len(prompt_2))

    assert len(prompt) == num_devices, "Prompt length must be equal to number of devices"
    assert lora_scale is None, "Lora scale is not supported currently with on device text encoders"
    assert clip_skip is None, "Clip skip is not supported currently with on device text encoders"
    assert prompt_embeds is None, "Prompt embeds is not supported currently with on device text encoders"
    assert (
        negative_prompt_embeds is None
    ), "Negative prompt embeds is not supported currently with on device text encoders"
    assert (
        do_classifier_free_guidance is True
    ), "Non - Classifier free guidance is not supported currently with on device text encoders"

    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Define tokenizers and text encoders
    tokenizers = (
        [pipeline.tokenizer, pipeline.tokenizer_2] if pipeline.tokenizer is not None else [pipeline.tokenizer_2]
    )
    text_encoders = [tt_text_encoder, tt_text_encoder_2] if tt_text_encoder is not None else [tt_text_encoder_2]

    if prompt_embeds is None:
        prompt_2 = prompt_2 or prompt

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]

        for ind, (prompt, tokenizer, text_encoder) in enumerate(zip(prompts, tokenizers, text_encoders)):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            tt_tokens = ttnn.from_torch(
                text_input_ids,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=0),
            )

            tt_sequence_output, tt_pooled_output = text_encoder(tt_tokens, ttnn_device, with_projection=(ind > 0))

            tt_sequence_output_torch = ttnn.to_torch(
                tt_sequence_output[-2],
                mesh_composer=ConcatMeshToTensor(ttnn_device, dim=0),
            ).to(torch.float32)
            tt_pooled_output_torch = ttnn.to_torch(
                tt_pooled_output, mesh_composer=ConcatMeshToTensor(ttnn_device, dim=0)
            ).to(torch.float32)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            # ----------- WARNING: ----------
            # The comment above is from the reference implementation of SDXL pipeline encode prompts function.
            # It clearly states that we are only interested in the pooled output of the final text encoder, but is in fact taking the last hidden state of the first text encoder.
            # I think this may be a bug in the reference implementation, but at the moment, we'll do the same (take the last hidden state of the first text encoder)
            if ind == 0:
                tt_pooled_prompt_embeds = ttnn.to_torch(
                    tt_sequence_output[-1],
                    mesh_composer=ConcatMeshToTensor(ttnn_device, dim=0),
                ).to(torch.float32)
                pooled_prompt_embeds = tt_pooled_prompt_embeds.to(torch.float32)
            else:
                pooled_prompt_embeds = tt_pooled_output_torch

            if clip_skip is None:
                prompt_embeds = tt_sequence_output_torch
            else:
                assert False, "Clip skip not none path not tested, use at your own risk!"
                prompt_embeds = ttnn.to_torch(
                    tt_sequence_output[-(clip_skip + 2)],
                    mesh_composer=ConcatMeshToTensor(ttnn_device, dim=0),
                ).to(torch.float32)

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    zero_out_negative_prompt = negative_prompt is None and pipeline.config.force_zeros_for_empty_prompt

    if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    elif do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt_2 = negative_prompt_2 or negative_prompt

        # normalize str to list
        negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        negative_prompt_2 = (
            batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
        )

        uncond_tokens: List[str]
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = [negative_prompt, negative_prompt_2]

        negative_prompt_embeds_list = []
        for ind, (negative_prompt, tokenizer, text_encoder) in enumerate(zip(uncond_tokens, tokenizers, text_encoders)):
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            tt_tokens = ttnn.from_torch(
                uncond_input.input_ids,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                mesh_mapper=ttnn.ShardTensorToMesh(ttnn_device, dim=0),
            )
            tt_sequence_output_neg, tt_pooled_output_neg = text_encoder(
                tt_tokens, ttnn_device, with_projection=(ind > 0)
            )
            tt_sequence_output_neg_torch = ttnn.to_torch(
                tt_sequence_output_neg[-2],
                mesh_composer=ConcatMeshToTensor(ttnn_device, dim=0),
            ).to(torch.float32)
            tt_pooled_output_neg_torch = ttnn.to_torch(
                tt_pooled_output_neg, mesh_composer=ConcatMeshToTensor(ttnn_device, dim=0)
            ).to(torch.float32)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            # ----------- WARNING: ----------
            # The comment above is from the reference implementation of SDXL pipeline encode prompts function.
            # It clearly states that we are only interested in the pooled output of the final text encoder, but is in fact taking the last hidden state of the first text encoder.
            # I think this may be a bug in the reference implementation, but at the moment, we'll do the same (take the last hidden state of the first text encoder)            # We are only ALWAYS interested in the pooled output of the final text encoder
            if ind == 0:
                tt_pooled_prompt_embeds = (
                    ttnn.to_torch(
                        tt_sequence_output_neg[-1],
                        mesh_composer=ConcatMeshToTensor(ttnn_device, dim=0),
                    )
                ).to(torch.float32)
                negative_pooled_prompt_embeds = tt_pooled_prompt_embeds
            else:
                negative_pooled_prompt_embeds = tt_pooled_output_neg_torch

            negative_prompt_embeds = tt_sequence_output_neg_torch

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    if pipeline.text_encoder_2 is not None:
        prompt_embeds = prompt_embeds.to(dtype=pipeline.text_encoder_2.dtype, device=device)
    else:
        prompt_embeds = prompt_embeds.to(dtype=pipeline.unet.dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        if pipeline.text_encoder_2 is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=pipeline.text_encoder_2.dtype, device=device)
        else:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=pipeline.unet.dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
        bs_embed * num_images_per_prompt, -1
    )
    if do_classifier_free_guidance:
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

    slice_to = num_prompts if use_cfg_parallel else None
    return (
        prompt_embeds[:slice_to],
        negative_prompt_embeds[:slice_to],
        pooled_prompt_embeds[:slice_to],
        negative_pooled_prompt_embeds[:slice_to],
    )


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


def prepare_image_latents(
    torch_pipeline,
    tt_pipeline,
    batch_size,
    num_channels_latents,
    height,
    width,
    cpu_device,
    dtype,
    image=None,
    is_strength_max=True,
    add_noise=True,
    latents=None,  # passed in latents
):
    # 4, 5, 8
    assert image is not None, "Image is not provided"
    assert image.shape[1] == 3, "Image is not 3 channels"
    assert add_noise is True, "Add noise should be True"
    assert torch_pipeline.vae_scale_factor == 8, "Vae scale factor should be 8"
    assert latents is None, "Latents are not supported for inpainting pipeline atm"

    shape = (
        1,
        num_channels_latents,
        int(height) // torch_pipeline.vae_scale_factor,
        int(width) // torch_pipeline.vae_scale_factor,
    )

    cpu_device = torch.device("cpu")
    image = image.to(device=cpu_device, dtype=dtype)

    if tt_pipeline.pipeline_config.vae_on_device:
        image_latents = [latent.sample() for latent in tt_pipeline.tt_vae.encode(image).latent_dist]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = [
            torch_pipeline.vae.encode(img).latent_dist.sample() for img in torch.chunk(image, chunks=batch_size, dim=0)
        ]
        image_latents = torch.cat(image_latents, dim=0)
    image_latents = tt_pipeline.torch_pipeline.vae.config.scaling_factor * image_latents
    image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

    torch_noise = torch.randn(shape, generator=None, device=cpu_device, dtype=dtype)
    torch_noise = torch_noise.repeat(batch_size // torch_noise.shape[0], 1, 1, 1)
    if is_strength_max:
        return torch_noise * tt_pipeline.tt_scheduler.init_noise_sigma

    tt_noise = ttnn.from_torch(
        torch_noise,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=tt_pipeline.ttnn_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            tt_pipeline.ttnn_device, list(tt_pipeline.ttnn_device.shape), dims=(None, 0)
        ),
    )
    tt_image_latents = ttnn.from_torch(
        image_latents,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=tt_pipeline.ttnn_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            tt_pipeline.ttnn_device, list(tt_pipeline.ttnn_device.shape), dims=(None, 0)
        ),
    )
    latents = tt_pipeline.tt_scheduler.add_noise(tt_image_latents, tt_noise)

    return ttnn.to_torch(
        latents,
        mesh_composer=ttnn.ConcatMeshToTensor(tt_pipeline.ttnn_device, dim=0),
    )[:batch_size, ...]


# adapted from sdxl inpaint pipeline: diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py
def prepare_mask_latents_inpainting(
    tt_inpainting_pipeline,
    mask,
    masked_image,
    batch_size,
    height,
    width,
    dtype,
    cpu_device,
    masked_image_latents=None,
):
    assert masked_image is not None, "Masked image must be provided at the moment"
    assert masked_image_latents is None, "Masked image latents are not supported for inpainting pipeline at the moment"

    # resize the mask to latents shape as we concatenate the mask to the latents
    # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
    # and half precision
    mask = torch.nn.functional.interpolate(
        mask,
        size=(
            height // tt_inpainting_pipeline.torch_pipeline.vae_scale_factor,
            width // tt_inpainting_pipeline.torch_pipeline.vae_scale_factor,
        ),
    )
    mask = mask.to(device=cpu_device, dtype=dtype)

    if masked_image is not None:
        if masked_image_latents is None:
            masked_image = masked_image.to(device=cpu_device, dtype=dtype)
            if tt_inpainting_pipeline.pipeline_config.vae_on_device == False:
                masked_image_latents = tt_inpainting_pipeline.torch_pipeline._encode_vae_image(
                    masked_image, generator=None
                )
            else:
                masked_image_latents = [
                    mask.sample() for mask in tt_inpainting_pipeline.tt_vae.encode(masked_image).latent_dist
                ]
                masked_image_latents = torch.cat(masked_image_latents, dim=0)
                masked_image_latents = (
                    tt_inpainting_pipeline.torch_pipeline.vae.config.scaling_factor * masked_image_latents
                )

        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=cpu_device, dtype=dtype)

    return mask, masked_image_latents


# Adapted from sdxl inpaint/img2img pipelines: diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_inpaint.py
def get_timesteps(tt_scheduler, num_inference_steps, strength, denoising_start=None):
    assert denoising_start is None, "denoising_start is not supported in this version"
    # This code path is only working if denoising_start is None, else more logic is needed
    # Denoising start is used in conjuction with SDXL Refiner pipeline.

    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)

    timesteps = tt_scheduler.timesteps[t_start * tt_scheduler.order :]

    # set_begin_index will update the step index as well to avoid doing so during trace capture
    tt_scheduler.set_begin_index(t_start * tt_scheduler.order)
    return timesteps, num_inference_steps - t_start


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


def run_tt_iteration_inpainting(
    tt_unet,
    tt_scheduler,
    tt_image_latents,
    tt_masked_image_latents,
    tt_mask,
    image_latents_shape,
    ttnn_prompt_embeds,
    time_ids,
    text_embeds,
):
    B, C, H, W = image_latents_shape

    input_tensor = tt_scheduler.scale_model_input(tt_image_latents, None)
    input_tensor = ttnn.concat([input_tensor, tt_mask, tt_masked_image_latents], dim=-1)

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
    num_steps,
    tt_extra_step_kwargs,
    guidance_scale,
    scaling_factor,
    input_shape,
    vae,  # can be host vae or tt vae
    batch_size,
    persistent_buffer,
    semaphores,
    output_device=None,
    output_shape=None,
    tid=None,
    tid_vae=None,
    capture_trace=False,
    use_cfg_parallel=False,
    guidance_rescale=0.0,
    one_minus_guidance_rescale=1.0,
):
    assert not (capture_trace and num_steps != 1), "Trace should capture only 1 iteration"
    profiler.start("image_gen")
    profiler.start("denoising_loop")

    for i in range(num_steps):
        unet_outputs = []
        if tid is None or capture_trace:
            tid = ttnn.begin_trace_capture(ttnn_device, cq_id=0) if capture_trace else None
            for unet_slice in range(tt_prompt_embeds.shape[0]):
                latent_model_input = tt_latents
                noise_pred, _ = run_tt_iteration(
                    tt_unet,
                    tt_scheduler,
                    latent_model_input,
                    input_shape,
                    tt_prompt_embeds[unet_slice] if not use_cfg_parallel else tt_prompt_embeds,
                    tt_time_ids if use_cfg_parallel else tt_time_ids[unet_slice],
                    ttnn.unsqueeze(tt_text_embeds[unet_slice], dim=0) if not use_cfg_parallel else tt_text_embeds,
                )

                unet_outputs.append(noise_pred)

            if use_cfg_parallel:
                noise_pred_interleaved = ttnn.to_memory_config(noise_pred, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(noise_pred)
                noise_pred = noise_pred_interleaved
                noise_pred_out = ttnn.experimental.all_gather_async(
                    noise_pred,
                    dim=0,
                    persistent_output_tensor=persistent_buffer,
                    multi_device_global_semaphore=semaphores,
                    num_links=1,
                    cluster_axis=0,
                    mesh_device=ttnn_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                )
                ttnn.deallocate(noise_pred)
                noise_pred = noise_pred_out
                noise_pred = noise_pred[..., :4]
                noise_pred_uncond, noise_pred_text = ttnn.unsqueeze(noise_pred[0], 0), ttnn.unsqueeze(noise_pred[1], 0)
            else:
                noise_pred_uncond, noise_pred_text = unet_outputs

            # ttnn.clone doesn't work with L1 sharded tensors
            noise_pred_text_new = ttnn.to_memory_config(noise_pred_text, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(noise_pred_text)
            noise_pred_text = noise_pred_text_new
            noise_pred_text_orig = ttnn.clone(noise_pred_text)

            # perform guidance
            noise_pred_text = ttnn.sub_(noise_pred_text, noise_pred_uncond)
            noise_pred_text = ttnn.mul_(noise_pred_text, guidance_scale)
            noise_pred = ttnn.add(noise_pred_uncond, noise_pred_text)

            ttnn.deallocate(noise_pred_uncond)
            ttnn.deallocate(noise_pred_text)

            noise_pred_new = ttnn.to_memory_config(noise_pred, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(noise_pred)
            noise_pred = noise_pred_new

            # perform guidance rescale
            std_text = ttnn.std(noise_pred_text_orig, dim=[1, 2, 3], keepdim=True)
            std_cfg = ttnn.std(noise_pred, dim=[1, 2, 3], keepdim=True)

            std_ratio = ttnn.div(std_text, std_cfg)

            noise_pred_rescaled = ttnn.mul(noise_pred, std_ratio)

            rescaled_term = ttnn.mul(noise_pred_rescaled, guidance_rescale)
            original_term = ttnn.mul(noise_pred, one_minus_guidance_rescale)
            ttnn.deallocate(noise_pred)
            noise_pred = ttnn.add(rescaled_term, original_term)
            ttnn.deallocate(std_text)
            ttnn.deallocate(std_cfg)
            ttnn.deallocate(std_ratio)
            ttnn.deallocate(noise_pred_rescaled)
            ttnn.deallocate(rescaled_term)
            ttnn.deallocate(original_term)
            ttnn.deallocate(noise_pred_text_orig)
            noise_pred = ttnn.move(noise_pred)

            tt_latents = tt_scheduler.step(noise_pred, None, tt_latents, **tt_extra_step_kwargs, return_dict=False)[0]

            if capture_trace:
                ttnn.end_trace_capture(ttnn_device, tid, cq_id=0)
        else:
            ttnn.execute_trace(ttnn_device, tid, cq_id=0, blocking=False)

        if i < (num_steps - 1):
            tt_scheduler.inc_step_index()

    ttnn.synchronize_device(ttnn_device)

    # set_begin_index resets both begin and step index of the scheduler
    tt_scheduler.set_begin_index(0)

    profiler.end("denoising_loop")

    vae_on_device = isinstance(vae, TtAutoencoderKL)

    if vae_on_device:
        profiler.start("vae_decode")
        if tid_vae is None or capture_trace:
            tid_vae = ttnn.begin_trace_capture(ttnn_device, cq_id=0) if capture_trace else None
            tt_latents = ttnn.div(tt_latents, scaling_factor)

            logger.info("Running TT VAE")
            output_tensor, [C, H, W] = vae.decode(tt_latents, input_shape)
            ttnn.deallocate(tt_latents)

            if capture_trace:
                ttnn.end_trace_capture(ttnn_device, tid_vae, cq_id=0)
            output_device = output_tensor
            output_shape = [input_shape[0], C, H, W]
        else:
            ttnn.execute_trace(ttnn_device, tid_vae, cq_id=0, blocking=False)

        ttnn.synchronize_device(ttnn_device)
        profiler.end("vae_decode")

        profiler.start("read_output_tensor")
        output_tensor = ttnn.to_torch(output_device, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_device, dim=0)).float()[
            :batch_size, ...
        ]
        ttnn.synchronize_device(ttnn_device)
        profiler.end("read_output_tensor")

        B, C, H, W = output_shape
        output_tensor = output_tensor.reshape(batch_size * B, H, W, C)
        imgs = torch.permute(output_tensor, (0, 3, 1, 2))
    else:
        profiler.start("read_output_tensor")
        latents = ttnn.to_torch(tt_latents, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_device, dim=0))[:batch_size, ...]
        ttnn.synchronize_device(ttnn_device)
        profiler.end("read_output_tensor")
        profiler.start("vae_decode")
        B, C, H, W = input_shape
        latents = latents.reshape(batch_size * B, H, W, C)
        latents = torch.permute(latents, (0, 3, 1, 2))
        latents = latents.to(vae.dtype)

        # VAE upcasting to float32 is happening in the reference SDXL demo if VAE dtype is float16. If it's bfloat16, it will not be upcasted.
        latents = latents / vae.config.scaling_factor
        warmup_run = num_steps == 1
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


# Runs a single iteration of the tt image generation
# This includes the following steps:
# - n denoising loops
# - vae
def run_tt_image_gen_inpainting(
    ttnn_device,
    tt_unet,
    tt_scheduler,
    tt_latents,
    tt_masked_image_latents,
    tt_mask,
    tt_prompt_embeds,
    tt_time_ids,
    tt_text_embeds,
    num_steps,
    tt_extra_step_kwargs,
    guidance_scale,
    scaling_factor,
    combined_latents_shape,  # 9 channels
    image_latents_shape,  # 4 channels
    vae,  # can be host vae or tt vae
    batch_size,
    persistent_buffer,
    semaphores,
    output_device=None,
    output_shape=None,
    tid=None,
    tid_vae=None,
    capture_trace=False,
    use_cfg_parallel=False,
    guidance_rescale=0.0,
    one_minus_guidance_rescale=1.0,
):
    assert not (capture_trace and num_steps != 1), "Trace should capture only 1 iteration"
    profiler.start("image_gen")
    profiler.start("denoising_loop")

    for i in range(num_steps):  # tqdm(enumerate(tt_timesteps), total=len(tt_timesteps)):
        unet_outputs = []
        if tid is None or capture_trace:
            tid = ttnn.begin_trace_capture(ttnn_device, cq_id=0) if capture_trace else None
            for unet_slice in range(tt_prompt_embeds.shape[0]):
                latent_model_input = tt_latents
                noise_pred, _ = run_tt_iteration_inpainting(
                    tt_unet,
                    tt_scheduler,
                    latent_model_input,
                    tt_masked_image_latents,
                    tt_mask,
                    combined_latents_shape,
                    tt_prompt_embeds[unet_slice] if not use_cfg_parallel else tt_prompt_embeds,
                    tt_time_ids if use_cfg_parallel else tt_time_ids[unet_slice],
                    ttnn.unsqueeze(tt_text_embeds[unet_slice], dim=0) if not use_cfg_parallel else tt_text_embeds,
                )

                unet_outputs.append(noise_pred)

            if use_cfg_parallel:
                noise_pred_interleaved = ttnn.to_memory_config(noise_pred, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(noise_pred)
                noise_pred = noise_pred_interleaved
                noise_pred_out = ttnn.experimental.all_gather_async(
                    noise_pred,
                    dim=0,
                    persistent_output_tensor=persistent_buffer,
                    multi_device_global_semaphore=semaphores,
                    num_links=1,
                    cluster_axis=0,
                    mesh_device=ttnn_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Linear,
                )
                ttnn.deallocate(noise_pred)
                noise_pred = noise_pred_out
                noise_pred = noise_pred[..., :4]
                noise_pred_uncond, noise_pred_text = ttnn.unsqueeze(noise_pred[0], 0), ttnn.unsqueeze(noise_pred[1], 0)
            else:
                noise_pred_uncond, noise_pred_text = unet_outputs

            # ttnn.clone doesn't work with L1 sharded tensors
            noise_pred_text_new = ttnn.to_memory_config(noise_pred_text, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(noise_pred_text)
            noise_pred_text = noise_pred_text_new
            noise_pred_text_orig = ttnn.clone(noise_pred_text)

            # perform guidance
            noise_pred_text = ttnn.sub_(noise_pred_text, noise_pred_uncond)
            noise_pred_text = ttnn.mul_(noise_pred_text, guidance_scale)
            noise_pred = ttnn.add(noise_pred_uncond, noise_pred_text)

            ttnn.deallocate(noise_pred_uncond)
            ttnn.deallocate(noise_pred_text)

            noise_pred_new = ttnn.to_memory_config(noise_pred, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(noise_pred)
            noise_pred = noise_pred_new

            # perform guidance rescale
            std_text = ttnn.std(noise_pred_text_orig, dim=[1, 2, 3], keepdim=True)
            std_cfg = ttnn.std(noise_pred, dim=[1, 2, 3], keepdim=True)

            std_ratio = ttnn.div(std_text, std_cfg)

            noise_pred_rescaled = ttnn.mul(noise_pred, std_ratio)

            rescaled_term = ttnn.mul(noise_pred_rescaled, guidance_rescale)
            original_term = ttnn.mul(noise_pred, one_minus_guidance_rescale)
            ttnn.deallocate(noise_pred)
            noise_pred = ttnn.add(rescaled_term, original_term)
            ttnn.deallocate(std_text)
            ttnn.deallocate(std_cfg)
            ttnn.deallocate(std_ratio)
            ttnn.deallocate(noise_pred_rescaled)
            ttnn.deallocate(rescaled_term)
            ttnn.deallocate(original_term)
            ttnn.deallocate(noise_pred_text_orig)
            noise_pred = ttnn.move(noise_pred)

            tt_latents = tt_scheduler.step(noise_pred, None, tt_latents, **tt_extra_step_kwargs, return_dict=False)[0]

            if capture_trace:
                ttnn.end_trace_capture(ttnn_device, tid, cq_id=0)
        else:
            ttnn.execute_trace(ttnn_device, tid, cq_id=0, blocking=False)

        if i < (num_steps - 1):
            tt_scheduler.inc_step_index()

    ttnn.synchronize_device(ttnn_device)

    # set_begin_index resets both begin and step index of the scheduler
    tt_scheduler.set_begin_index(0)

    profiler.end("denoising_loop")

    vae_on_device = isinstance(vae, TtAutoencoderKL)

    if vae_on_device:
        profiler.start("vae_decode")
        if tid_vae is None or capture_trace:
            tid_vae = ttnn.begin_trace_capture(ttnn_device, cq_id=0) if capture_trace else None
            tt_latents = ttnn.div(tt_latents, scaling_factor)

            logger.info("Running TT VAE")
            output_tensor, [C, H, W] = vae.decode(tt_latents, image_latents_shape)
            ttnn.deallocate(tt_latents)

            if capture_trace:
                ttnn.end_trace_capture(ttnn_device, tid_vae, cq_id=0)
            output_device = output_tensor
            output_shape = [image_latents_shape[0], C, H, W]
        else:
            ttnn.execute_trace(ttnn_device, tid_vae, cq_id=0, blocking=False)

        ttnn.synchronize_device(ttnn_device)
        profiler.end("vae_decode")

        profiler.start("read_output_tensor")
        output_tensor = ttnn.to_torch(output_device, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_device, dim=0)).float()[
            :batch_size, ...
        ]
        ttnn.synchronize_device(ttnn_device)
        profiler.end("read_output_tensor")

        B, C, H, W = output_shape
        output_tensor = output_tensor.reshape(batch_size * B, H, W, C)
        imgs = torch.permute(output_tensor, (0, 3, 1, 2))
    else:
        profiler.start("read_output_tensor")
        latents = ttnn.to_torch(tt_latents, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_device, dim=0))[:batch_size, ...]
        ttnn.synchronize_device(ttnn_device)
        profiler.end("read_output_tensor")
        profiler.start("vae_decode")
        B, C, H, W = image_latents_shape
        latents = latents.reshape(batch_size * B, H, W, C)
        latents = torch.permute(latents, (0, 3, 1, 2))
        latents = latents.to(vae.dtype)

        # VAE upcasting to float32 is happening in the reference SDXL demo if VAE dtype is float16. If it's bfloat16, it will not be upcasted.
        latents = latents / vae.config.scaling_factor
        warmup_run = num_steps == 1
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
    profiler.start("prepare_input_tensors")
    for host_tensor, device_tensor in zip(host_tensors, device_tensors):
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
    if device_tensors:
        ttnn.synchronize_device(device_tensors[0].device())
    profiler.end("prepare_input_tensors")


def allocate_input_tensors(ttnn_device, tt_latents, tt_prompt_embeds, tt_text_embeds, tt_time_ids):
    profiler.start("allocate_input_tensors")
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
    ttnn.synchronize_device(ttnn_device)
    profiler.end("prepare_input_tensors")

    return tt_latents_device, tt_prompt_embeds_device, tt_text_embeds_device, tt_time_ids_device


def create_user_tensors(
    ttnn_device, latents, negative_prompt_embeds, prompt_embeds, negative_pooled_prompt_embeds, add_text_embeds
):
    profiler.start("create_user_tensors")
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
    ttnn.synchronize_device(ttnn_device)
    profiler.end("create_user_tensors")
    return tt_latents, tt_prompt_embeds, tt_add_text_embeds

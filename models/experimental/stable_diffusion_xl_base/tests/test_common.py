# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import ttnn
import torch
import inspect
from typing import List, Optional, Union

from ttnn.distributed.distributed import ConcatMeshToTensor
from models.experimental.stable_diffusion_35_large.tt.clip_encoder import (
    TtCLIPConfig,
    TtCLIPTextTransformer,
    TtCLIPTextTransformerParameters,
)
from models.common.utility_functions import profiler

from tqdm import tqdm
import ttnn

from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL

SDXL_L1_SMALL_SIZE = 27000
SDXL_TRACE_REGION_SIZE = 34000000
SDXL_CI_WEIGHTS_PATH = "/mnt/MLPerf/tt_dnn-models/hf_home"


def create_tt_clip_text_encoders(pipeline, ttnn_device):
    tt_parameters_text_encoder = TtCLIPTextTransformerParameters.from_torch(
        pipeline.text_encoder.state_dict(),
        device=ttnn_device,
        dtype=ttnn.bfloat16,
        parallel_manager=None,
        has_text_projection=False,  # Text encoder 1 does not have text projection
    )
    tt_config_text_encoder = TtCLIPConfig(
        vocab_size=pipeline.text_encoder.config.vocab_size,
        d_model=pipeline.text_encoder.config.hidden_size,
        d_ff=pipeline.text_encoder.config.intermediate_size,
        num_heads=pipeline.text_encoder.config.num_attention_heads,
        num_layers=pipeline.text_encoder.config.num_hidden_layers,
        max_position_embeddings=77,
        layer_norm_eps=pipeline.text_encoder.config.layer_norm_eps,
        attention_dropout=pipeline.text_encoder.config.attention_dropout,
        hidden_act=pipeline.text_encoder.config.hidden_act,
    )
    tt_text_encoder = TtCLIPTextTransformer(tt_parameters_text_encoder, tt_config_text_encoder)

    # TT text encoder 2 setup
    tt_parameters_text_encoder_2 = TtCLIPTextTransformerParameters.from_torch(
        pipeline.text_encoder_2.state_dict(),
        device=ttnn_device,
        dtype=ttnn.bfloat16,
        parallel_manager=None,
        has_text_projection=True,  # Text encoder 2 has text projection
    )

    tt_config_text_encoder_2 = TtCLIPConfig(
        vocab_size=pipeline.text_encoder_2.config.vocab_size,
        d_model=pipeline.text_encoder_2.config.hidden_size,
        d_ff=pipeline.text_encoder_2.config.intermediate_size,
        num_heads=pipeline.text_encoder_2.config.num_attention_heads,
        num_layers=pipeline.text_encoder_2.config.num_hidden_layers,
        max_position_embeddings=77,
        layer_norm_eps=pipeline.text_encoder_2.config.layer_norm_eps,
        attention_dropout=pipeline.text_encoder_2.config.attention_dropout,
        hidden_act=pipeline.text_encoder_2.config.hidden_act,
    )
    tt_text_encoder_2 = TtCLIPTextTransformer(tt_parameters_text_encoder_2, tt_config_text_encoder_2)

    return tt_text_encoder, tt_text_encoder_2


def warmup_tt_text_encoders(tt_text_encoder, tt_text_encoder_2, tokenizer, tokenizer_2, ttnn_device, batch_size):
    logger.info("Performing warmup run on encoding, to make use of program caching in actual inference...")
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

    _, _ = tt_text_encoder(tt_tokens_1, ttnn_device, parallel_manager=None)
    _, _ = tt_text_encoder_2(tt_tokens_2, ttnn_device, parallel_manager=None)
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

    num_devices = ttnn_device.get_num_devices()
    assert len(prompt) == num_devices, "Prompt length must be equal to number of devices"
    assert prompt_2 is None, "Prompt 2 is not supported currently"
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
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

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

            tt_sequence_output, tt_pooled_output = text_encoder(tt_tokens, ttnn_device, parallel_manager=None)

            tt_sequence_output_torch = ttnn.to_torch(
                tt_sequence_output.hidden_states[-2],
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
                    tt_sequence_output.hidden_states[-1],
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
                    tt_sequence_output.hidden_states[-(clip_skip + 2)],
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
            tt_sequence_output_neg, tt_pooled_output_neg = text_encoder(tt_tokens, ttnn_device, parallel_manager=None)
            tt_sequence_output_neg_torch = ttnn.to_torch(
                tt_sequence_output_neg.hidden_states[-2],
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
                        tt_sequence_output_neg.hidden_states[-1],
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

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


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
        output_tensor = ttnn.to_torch(output_device, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_device, dim=0)).float()
        ttnn.synchronize_device(ttnn_device)
        profiler.end("read_output_tensor")

        B, C, H, W = output_shape
        output_tensor = output_tensor.reshape(batch_size * B, H, W, C)
        imgs = torch.permute(output_tensor, (0, 3, 1, 2))
    else:
        profiler.start("read_output_tensor")
        latents = ttnn.to_torch(tt_latents, mesh_composer=ttnn.ConcatMeshToTensor(ttnn_device, dim=0))
        ttnn.synchronize_device(ttnn_device)
        profiler.end("read_output_tensor")
        profiler.start("vae_decode")
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

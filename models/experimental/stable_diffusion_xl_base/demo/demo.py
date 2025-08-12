# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from ast import List
import time
from typing import Optional
from models.experimental.stable_diffusion_35_large.tt.clip_encoder import (
    TtCLIPConfig,
    TtCLIPTextTransformer,
    TtCLIPTextTransformerParameters,
)
import pytest
import torch
from diffusers import DiffusionPipeline
from loguru import logger
from conftest import is_galaxy
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from transformers import CLIPTextModelWithProjection, CLIPTextModel
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    retrieve_timesteps,
    run_tt_image_gen,
    prepare_input_tensors,
    allocate_input_tensors,
    create_user_tensors,
)
import os
from models.utility_functions import profiler


# encode prompt from pipeline called!
# Prompt tmbeds path
# In clip text encoder
# Clip skip none path
# In clip text encoder
# Clip skip none path
# ABC path
# Text encoder 2 path
# CFG path


# encode prompt function from diffusers, need to modify it to work with ttnn and encoders
def my_encode_prompt(
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
    print("encode prompt from pipeline called!")
    device = device or pipeline._execution_device
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt is not None:
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # Define tokenizers and text encoders
    tokenizers = (
        [pipeline.tokenizer, pipeline.tokenizer_2] if pipeline.tokenizer is not None else [pipeline.tokenizer_2]
    )
    text_encoders = (
        [pipeline.text_encoder, pipeline.text_encoder_2]
        if pipeline.text_encoder is not None
        else [pipeline.text_encoder_2]
    )

    total_tokenize_and_encode_time = 0
    if prompt_embeds is None:
        print("Prompt tmbeds path")
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        i = 0
        for ind, (prompt, tokenizer, text_encoder) in enumerate(zip(prompts, tokenizers, text_encoders)):
            tokenizer_start_time = time.time()
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokenizer_end_time = time.time()
            # print(f"tokenizer_{i} time = ", tokenizer_end_time - tokenizer_start_time)
            total_tokenize_and_encode_time += tokenizer_end_time - tokenizer_start_time

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                print("Truncated ids path")  # ovo treba
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            text_encoder_start_time = time.time()
            print("Text input ids shape = ", text_input_ids.shape)

            tt_tokens = ttnn.from_torch(
                text_input_ids,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device),  # fix this
            )

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
            print("Running TT text encoder 1 inference")
            tt_sequence_output, tt_pooled_output = tt_text_encoder(tt_tokens, ttnn_device, parallel_manager=None)
            ttnn.synchronize_device(ttnn_device)

            tt_sequence_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_sequence_output.hidden_states[-2])[0])
            tt_pooled_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled_output)[0])

            print("TT text encoder 1 inference done")

            # print("Prompt embeds ", prompt_embeds)
            text_encoder_end_time = time.time()
            # print(f"text_encoder_{i} time = ", text_encoder_end_time - text_encoder_start_time)
            total_tokenize_and_encode_time += text_encoder_end_time - text_encoder_start_time
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            print("pooled prompt embeds data format is ", pooled_prompt_embeds.dtype)
            # print("Full prompt embeds is: ", prompt_embeds)
            if ind == 0:
                # the reference code says that pooled prompt embeds is actually the pooled prompt embeds, but is in fact last hidden state
                tt_pooled_prompt_embeds = ttnn.to_torch(
                    ttnn.get_device_tensors(tt_sequence_output.hidden_states[-1])[0]
                )
                print("tt_pooled prompt embeds data format is ", tt_pooled_prompt_embeds.dtype)
                pooled_prompt_embeds = tt_pooled_prompt_embeds.to(torch.float32)
                print("pooled prompt embeds shape = ", pooled_prompt_embeds.shape)
                print("tt pooled prompt embeds shape = ", tt_pooled_prompt_embeds.shape)
                print("Using tt path")

            if clip_skip is None:
                print("Clip skip none path")
                prompt_embeds = prompt_embeds.hidden_states[-2]
                if ind == 0:
                    prompt_embeds = tt_sequence_output_torch.to(torch.float32)
                    print("Prompt embeds shape = ", prompt_embeds.shape)
                    print("TT prompt embeds shape = ", tt_sequence_output_torch.shape)
                    print("Using tt path 2")
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                print("Clip skip not none path")
                assert False, "Clip skip not none path not implemented"
                # prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
            i += 1

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

    # get unconditional embeddings for classifier free guidance
    zero_out_negative_prompt = negative_prompt is None and pipeline.config.force_zeros_for_empty_prompt
    print("Zero out negative prompt = ", zero_out_negative_prompt)
    i = 0

    if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
        print("ABC path")
        # ovo vrv ne treba
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    elif do_classifier_free_guidance and negative_prompt_embeds is None:
        print("DEF path")
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
        print("Pre negative prompt loop")
        for ind, (negative_prompt, tokenizer, text_encoder) in enumerate(zip(uncond_tokens, tokenizers, text_encoders)):
            print("Negative prompt path")
            max_length = prompt_embeds.shape[1]
            tokenizer_start_time = time.time()
            uncond_input = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            tokenizer_end_time = time.time()
            # print(f"negative tokenizer_{i} time = ", tokenizer_end_time - tokenizer_start_time)
            total_tokenize_and_encode_time += tokenizer_end_time - tokenizer_start_time

            text_encoder_start_time = time.time()
            tt_tokens = ttnn.from_torch(
                uncond_input.input_ids,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=ttnn_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device),  # fix this
            )
            tt_sequence_output_neg, tt_pooled_output_neg = tt_text_encoder(
                tt_tokens, ttnn_device, parallel_manager=None
            )

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                output_hidden_states=True,
            )
            text_encoder_end_time = time.time()
            # print(f"negative text_encoder_{i} time = ", text_encoder_end_time - text_encoder_start_time)
            total_tokenize_and_encode_time += text_encoder_end_time - text_encoder_start_time
            # We are only ALWAYS interested in the pooled output of the final text encoder
            negative_pooled_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]
            if ind == 0:
                if ind == 0:
                    tt_pooled_prompt_embeds = (
                        ttnn.to_torch(ttnn.get_device_tensors(tt_sequence_output_neg.hidden_states[-1])[0])
                    ).to(torch.float32)
                    negative_pooled_prompt_embeds = tt_pooled_prompt_embeds
                    negative_prompt_embeds = tt_sequence_output_neg.to(torch.float32)
                    print("Using tt path for encoder 1 in negative prompt")

            negative_prompt_embeds_list.append(negative_prompt_embeds)

        negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

    if pipeline.text_encoder_2 is not None:
        print("Text encoder 2 path")
        prompt_embeds = prompt_embeds.to(dtype=pipeline.text_encoder_2.dtype, device=device)
    else:
        print("Text encoder 2 not none path")
        prompt_embeds = prompt_embeds.to(dtype=pipeline.unet.dtype, device=device)

    # print("Total tokenize and encode time = ", total_tokenize_and_encode_time)
    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        print("CFG path")
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


@torch.no_grad()
def run_demo_inference(
    ttnn_device, is_ci_env, prompts, num_inference_steps, vae_on_device, evaluation_range, capture_trace
):
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
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )

    assert isinstance(pipeline.text_encoder, CLIPTextModel), "pipeline.text_encoder is not a CLIPTextModel"
    assert isinstance(
        pipeline.text_encoder_2, CLIPTextModelWithProjection
    ), "pipeline.text_encoder_2 is not a CLIPTextModelWithProjection"

    # TT text encoder setup
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

    # Have to throttle matmuls due to di/dt
    if is_galaxy():
        logger.info("Setting TT_MM_THROTTLE_PERF for Galaxy")
        os.environ["TT_MM_THROTTLE_PERF"] = "5"

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

    # assumes lora scale is None
    # assumes prompt_embeds is None

    embed_start_time = time.time()
    all_embeds = [
        # pipeline.encode_prompt(
        #     prompt=prompt,
        #     prompt_2=None,
        #     device=cpu_device,
        #     num_images_per_prompt=1,
        #     do_classifier_free_guidance=True,
        #     negative_prompt=None,
        #     negative_prompt_2=None,
        #     prompt_embeds=None,
        #     negative_prompt_embeds=None,
        #     pooled_prompt_embeds=None,
        #     negative_pooled_prompt_embeds=None,
        #     lora_scale=None,
        #     clip_skip=None,
        # )
        my_encode_prompt(
            pipeline,
            tt_text_encoder,
            tt_text_encoder_2,
            ttnn_device,
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

    # all_embeds = pipeline.encode_prompt(
    #     prompt=prompts,  # Pass the entire list at once
    #     prompt_2=None,
    #     device=cpu_device,
    #     num_images_per_prompt=1,
    #     do_classifier_free_guidance=True,
    #     negative_prompt=None,
    #     negative_prompt_2=None,
    #     prompt_embeds=None,
    #     negative_prompt_embeds=None,
    #     pooled_prompt_embeds=None,
    #     negative_pooled_prompt_embeds=None,
    #     lora_scale=None,
    #     clip_skip=None,
    # )
    # (
    #     prompt_embeds_batch,
    #     negative_prompt_embeds_batch,
    #     pooled_prompt_embeds_batch,
    #     negative_pooled_prompt_embeds_batch,
    # ) = all_embeds
    # all_embeds = list(
    #     zip(
    #         torch.split(prompt_embeds_batch, 1, dim=0),
    #         torch.split(negative_prompt_embeds_batch, 1, dim=0),
    #         torch.split(pooled_prompt_embeds_batch, 1, dim=0),
    #         torch.split(negative_pooled_prompt_embeds_batch, 1, dim=0),
    #     )
    # )
    embed_end_time = time.time()
    print("embed time = ", embed_end_time - embed_start_time)

    print("All embeds length = ", len(all_embeds))
    # print("all_embeds = ", all_embeds)

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

    tid = None
    output_device = None
    tid_vae = None
    if capture_trace:
        logger.info("Capturing model trace...")
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

        logger.info(f"Image gen for {batch_size} prompts completed in {profiler.times['image_gen'][-1]:.2f} seconds")
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
    "capture_trace",
    [
        (True),
        (False),
    ],
    ids=("with_trace", "no_trace"),
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_demo(
    mesh_device,
    is_ci_env,
    prompt,
    num_inference_steps,
    vae_on_device,
    capture_trace,
    evaluation_range,
):
    return run_demo_inference(
        mesh_device, is_ci_env, prompt, num_inference_steps, vae_on_device, evaluation_range, capture_trace
    )

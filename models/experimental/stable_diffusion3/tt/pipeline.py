# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-FileCopyrightText: Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import tqdm
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from .t5_encoder import TtT5Encoder, TtT5EncoderParameters
from .transformer import TtSD3Transformer2DModel, TtSD3Transformer2DModelParameters


class TtStableDiffusion3Pipeline:
    def __init__(self, *, checkpoint: str, device: ttnn.Device, enable_t5_text_encoder: bool = True) -> None:
        self._device = device

        logger.info("loading models...")
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer")
        self._tokenizer_2 = CLIPTokenizer.from_pretrained(checkpoint, subfolder="tokenizer_2")
        self._tokenizer_3 = T5TokenizerFast.from_pretrained(checkpoint, subfolder="tokenizer_3")
        self._text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(checkpoint, subfolder="text_encoder")
        self._text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(checkpoint, subfolder="text_encoder_2")
        if enable_t5_text_encoder:
            torch_text_encoder_3 = T5EncoderModel.from_pretrained(checkpoint, subfolder="text_encoder_3")
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint, subfolder="scheduler")
        self._vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
        torch_transformer = SD3Transformer2DModel.from_pretrained(
            checkpoint, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        assert isinstance(self._tokenizer_1, CLIPTokenizer)
        assert isinstance(self._tokenizer_2, CLIPTokenizer)
        assert isinstance(self._tokenizer_3, T5TokenizerFast)
        assert isinstance(self._text_encoder_1, CLIPTextModelWithProjection)
        assert isinstance(self._text_encoder_2, CLIPTextModelWithProjection)
        assert isinstance(self._scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(self._vae, AutoencoderKL)
        assert isinstance(torch_transformer, SD3Transformer2DModel)

        logger.info("creating TT-NN transformer...")

        parameters = TtSD3Transformer2DModelParameters.from_torch(
            torch_transformer.state_dict(), device=device, dtype=ttnn.bfloat8_b
        )
        self._tt_transformer = TtSD3Transformer2DModel(
            parameters, num_attention_heads=torch_transformer.config.num_attention_heads
        )
        self._num_channels_latents = torch_transformer.config.in_channels
        self._joint_attention_dim = torch_transformer.config.joint_attention_dim

        self._block_out_channels = self._vae.config.block_out_channels
        self._vae_scaling_factor = self._vae.config.scaling_factor
        self._vae_shift_factor = self._vae.config.shift_factor

        self._vae_scale_factor = 2 ** (len(self._block_out_channels) - 1)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)

        if enable_t5_text_encoder:
            logger.info("creating TT-NN text encoder...")

            parameters = TtT5EncoderParameters.from_torch(
                torch_text_encoder_3.state_dict(), device=device, dtype=ttnn.bfloat16
            )
            self._text_encoder_3 = TtT5Encoder(
                parameters,
                num_heads=torch_text_encoder_3.config.num_heads,
                relative_attention_num_buckets=torch_text_encoder_3.config.relative_attention_num_buckets,
                relative_attention_max_distance=torch_text_encoder_3.config.relative_attention_max_distance,
                layer_norm_epsilon=torch_text_encoder_3.config.layer_norm_epsilon,
            )
        else:
            self._text_encoder_3 = None

    def prepare(
        self,
        *,
        batch_size: int,
        num_images_per_prompt: int = 1,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 4.5,
        max_t5_sequence_length: int = 256,
    ) -> None:
        self._prepared_batch_size = batch_size
        self._prepared_num_images_per_prompt = num_images_per_prompt
        self._prepared_width = width
        self._prepared_height = height
        self._prepared_guidance_scale = guidance_scale
        self._prepared_max_t5_sequence_length = max_t5_sequence_length

        do_classifier_free_guidance = guidance_scale > 1

        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(
            prompt_1=[""],
            prompt_2=[""],
            prompt_3=[""],
            negative_prompt_1=[""],
            negative_prompt_2=[""],
            negative_prompt_3=[""],
            num_images_per_prompt=num_images_per_prompt,
            max_t5_sequence_length=max_t5_sequence_length,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        latents_shape = (
            batch_size * num_images_per_prompt,
            height // self._vae_scale_factor,
            width // self._vae_scale_factor,
            self._num_channels_latents,
        )

        tt_prompt_embeds = ttnn.from_torch(
            prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b
        )
        tt_pooled_prompt_embeds = ttnn.from_torch(
            pooled_prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_timestep = ttnn.allocate_tensor_on_device([1, 1], ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self._device)
        tt_sigma_difference = ttnn.allocate_tensor_on_device([1, 1], ttnn.bfloat16, ttnn.TILE_LAYOUT, self._device)
        tt_latents = ttnn.allocate_tensor_on_device(latents_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self._device)

        # cache
        self._step(
            timestep=tt_timestep,
            latents=tt_latents,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=tt_prompt_embeds,
            pooled_prompt_embeds=tt_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            sigma_difference=tt_sigma_difference,
        )

        # trace
        tid = ttnn.begin_trace_capture(self._device)
        self._step(
            timestep=tt_timestep,
            latents=tt_latents,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=tt_prompt_embeds,
            pooled_prompt_embeds=tt_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            sigma_difference=tt_sigma_difference,
        )
        ttnn.end_trace_capture(self._device, tid)

        self._trace = PipelineTrace(
            tid=tid,
            spatial_input_output=tt_latents,
            prompt_input=tt_prompt_embeds,
            pooled_projection_input=tt_pooled_prompt_embeds,
            timestep_input=tt_timestep,
            sigma_difference_input=tt_sigma_difference,
        )

    def __call__(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        prompt_3: list[str],
        negative_prompt_1: list[str],
        negative_prompt_2: list[str],
        negative_prompt_3: list[str],
        num_inference_steps: int = 40,
        seed: int | None = None,
    ) -> None:
        start_time = time.time()

        batch_size = self._prepared_batch_size
        num_images_per_prompt = self._prepared_num_images_per_prompt
        width = self._prepared_width
        height = self._prepared_height
        guidance_scale = self._prepared_guidance_scale
        max_t5_sequence_length = self._prepared_max_t5_sequence_length

        assert height % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        assert width % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        assert max_t5_sequence_length <= 512  # noqa: PLR2004
        assert batch_size == len(prompt_1)

        do_classifier_free_guidance = guidance_scale > 1
        latents_shape = (
            batch_size * num_images_per_prompt,
            self._num_channels_latents,
            height // self._vae_scale_factor,
            width // self._vae_scale_factor,
        )

        logger.info("encoding prompts...")

        prompt_encoding_start_time = time.time()
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(
            prompt_1=prompt_1,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt_1=negative_prompt_1,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            num_images_per_prompt=num_images_per_prompt,
            max_t5_sequence_length=max_t5_sequence_length,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        prompt_encoding_end_time = time.time()

        logger.info("preparing timesteps...")

        self._scheduler.set_timesteps(num_inference_steps)
        timesteps = self._scheduler.timesteps

        logger.info("preparing latents...")

        if seed is not None:
            torch.manual_seed(seed)
        latents = torch.randn(latents_shape, dtype=prompt_embeds.dtype).permute([0, 2, 3, 1])

        tt_prompt_embeds = ttnn.from_torch(prompt_embeds, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        tt_pooled_prompt_embeds = ttnn.from_torch(pooled_prompt_embeds, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_initial_latents = ttnn.from_torch(latents, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        logger.info("denoising...")
        denoising_start_time = time.time()

        ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._trace.prompt_input)
        ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds, self._trace.pooled_projection_input)
        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._trace.spatial_input_output)

        for i, t in enumerate(tqdm.tqdm(timesteps)):
            tt_timestep = ttnn.full([1, 1], fill_value=t, dtype=ttnn.float32)

            sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]
            tt_sigma_difference = ttnn.full(
                [1, 1], fill_value=sigma_difference, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )

            ttnn.copy_host_to_device_tensor(tt_timestep, self._trace.timestep_input)
            ttnn.copy_host_to_device_tensor(tt_sigma_difference, self._trace.sigma_difference_input)

            self._trace.execute()

        denoising_end_time = time.time()

        logger.info("decoding image...")

        image_decoding_start_time = time.time()

        latents = ttnn.to_torch(self._trace.spatial_input_output).to(torch.float32)
        latents = (latents.permute([0, 3, 1, 2]) / self._vae_scaling_factor) + self._vae_shift_factor

        with torch.no_grad():
            image = self._vae.decoder(latents)
            image = self._image_processor.postprocess(image, output_type="pt")
            assert isinstance(image, torch.Tensor)

        image_decoding_end_time = time.time()

        output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        end_time = time.time()

        logger.info(f"prompt encoding duration: {prompt_encoding_end_time - prompt_encoding_start_time}")
        logger.info(f"denoising duration: {denoising_end_time - denoising_start_time}")
        logger.info(f"image decoding duration: {image_decoding_end_time - image_decoding_start_time}")
        logger.info(f"total runtime: {end_time - start_time}")

        return output

    def _step(
        self,
        *,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        latents: ttnn.Tensor,
        timestep: ttnn.Tensor,
        pooled_prompt_embeds: ttnn.Tensor,
        prompt_embeds: ttnn.Tensor,
        sigma_difference: ttnn.Tensor,
    ) -> None:
        latent_model_input = ttnn.concat([latents, latents]) if do_classifier_free_guidance else latents
        latent_model_batch_size = latent_model_input.shape[0]

        timestep = ttnn.repeat(timestep, ttnn.Shape([latent_model_batch_size, 1]))
        timestep = ttnn.to_layout(timestep, ttnn.TILE_LAYOUT)

        noise_pred = self._tt_transformer(
            spatial=latent_model_input,
            prompt=prompt_embeds,
            pooled_projection=pooled_prompt_embeds,
            timestep=timestep,
        )

        noise_pred = _reshape_noise_pred(
            noise_pred,
            height=latent_model_input.shape[-3],
            width=latent_model_input.shape[-2],
            patch_size=self._tt_transformer.patch_size,
        )

        if do_classifier_free_guidance:
            split_pos = noise_pred.shape[0] // 2
            uncond = noise_pred[0:split_pos]
            cond = noise_pred[split_pos:]
            noise_pred = uncond + guidance_scale * (cond - uncond)

        ttnn.add_(latents, sigma_difference * noise_pred)

    def _encode_prompts(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        prompt_3: list[str],
        negative_prompt_1: list[str],
        negative_prompt_2: list[str],
        negative_prompt_3: list[str],
        num_images_per_prompt: int,
        max_t5_sequence_length: int,
        do_classifier_free_guidance: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer_max_length = self._tokenizer_1.model_max_length

        prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
            prompt=prompt_1,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_1,
            text_encoder=self._text_encoder_1,
            tokenizer_max_length=tokenizer_max_length,
        )

        prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
            prompt=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_2,
            text_encoder=self._text_encoder_2,
            tokenizer_max_length=tokenizer_max_length,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        t5_prompt_embed = _get_t5_prompt_embeds(
            device=self._device,
            prompt=prompt_3,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_t5_sequence_length,
            tokenizer=self._tokenizer_3,
            text_encoder=self._text_encoder_3,
            tokenizer_max_length=tokenizer_max_length,
            joint_attention_dim=self._joint_attention_dim,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if not do_classifier_free_guidance:
            return prompt_embeds, pooled_prompt_embeds

        negative_prompt_embed, negative_pooled_prompt_embed = _get_clip_prompt_embeds(
            prompt=negative_prompt_1,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_1,
            text_encoder=self._text_encoder_1,
            tokenizer_max_length=tokenizer_max_length,
        )
        negative_prompt_2_embed, negative_pooled_prompt_2_embed = _get_clip_prompt_embeds(
            prompt=negative_prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_2,
            text_encoder=self._text_encoder_2,
            tokenizer_max_length=tokenizer_max_length,
        )
        negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

        t5_negative_prompt_embed = _get_t5_prompt_embeds(
            device=self._device,
            prompt=negative_prompt_3,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_t5_sequence_length,
            tokenizer=self._tokenizer_3,
            text_encoder=self._text_encoder_3,
            tokenizer_max_length=tokenizer_max_length,
            joint_attention_dim=self._joint_attention_dim,
        )

        negative_clip_prompt_embeds = torch.nn.functional.pad(
            negative_clip_prompt_embeds,
            (
                0,
                t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1],
            ),
        )

        negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
        negative_pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        return prompt_embeds, pooled_prompt_embeds


@dataclass
class PipelineTrace:
    spatial_input_output: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_projection_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    tid: int

    def execute(self) -> None:
        ttnn.execute_trace(self.spatial_input_output.device(), self.tid)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    *,
    clip_skip: int | None = None,
    device: torch.device | None = None,
    num_images_per_prompt: int,
    prompt: list[str],
    text_encoder: CLIPTextModelWithProjection,
    tokenizer_max_length: int,
    tokenizer: CLIPTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {tokenizer_max_length} tokens: {removed_text}"
        )
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]

    if clip_skip is None:
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_t5_prompt_embeds(
    prompt: list[str],
    *,
    torch_device: torch.device | None = None,
    device: ttnn.Device,
    joint_attention_dim: int,
    max_sequence_length: int,
    num_images_per_prompt: int,
    text_encoder: TtT5Encoder | None,
    tokenizer_max_length: int,
    tokenizer: T5TokenizerFast,
) -> torch.Tensor:
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if text_encoder is None:
        return torch.zeros(
            (
                batch_size * num_images_per_prompt,
                tokenizer_max_length,
                joint_attention_dim,
            ),
            device=torch_device,
            dtype=torch.bfloat16,
        )

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    tt_text_input_ids = ttnn.from_torch(text_input_ids, device=device, layout=ttnn.TILE_LAYOUT)
    tt_prompt_embeds = text_encoder(tt_text_input_ids)
    prompt_embeds = ttnn.to_torch(tt_prompt_embeds)

    prompt_embeds = prompt_embeds.to(device=torch_device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    return prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)


def _reshape_noise_pred(
    noise_pred: ttnn.Tensor,
    *,
    height: int,
    width: int,
    patch_size: int,
) -> ttnn.Tensor:
    patch_count_y = height // patch_size
    patch_count_x = width // patch_size

    shape1 = (
        noise_pred.shape[0] * patch_count_y,
        patch_count_x,
        patch_size,
        -1,
    )

    shape2 = (
        noise_pred.shape[0],
        patch_count_y * patch_size,
        patch_count_x * patch_size,
        -1,
    )

    noise_pred = noise_pred.reshape(shape1)
    noise_pred = ttnn.transpose(noise_pred, 1, 2)
    return noise_pred.reshape(shape2)

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-FileCopyrightText: Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import tqdm
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from ..reference.transformer import FluxTransformer as FluxTransformeReference
from .t5_encoder import T5Encoder, T5EncoderParameters
from .transformer import FluxTransformer, FluxTransformerParameters


class FluxPipeline:
    def __init__(
        self, *, checkpoint: str, device: ttnn.MeshDevice, use_torch_encoder: bool = True, model_location_generator
    ) -> None:
        self._device = device

        logger.info("loading transformer...")

        model_name_checkpoint = model_location_generator(checkpoint, model_subdir="Flux1_Schnell")

        print("model_name_checkpoint", model_name_checkpoint)

        torch_transformer = FluxTransformeReference.from_pretrained(
            model_name_checkpoint, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        assert isinstance(torch_transformer, FluxTransformeReference)

        logger.info("creating TT-NN transformer...")

        parameters = FluxTransformerParameters.from_torch(
            torch_transformer.state_dict(),
            device=device,
            dtype=ttnn.bfloat8_b,
        )
        self._tt_transformer = FluxTransformer(
            parameters, num_attention_heads=torch_transformer.config.num_attention_heads
        )

        self._num_channels_latents = torch_transformer.in_channels // 4
        self._pos_embed = torch_transformer.pos_embed
        del torch_transformer

        logger.info("loading other models...")
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(model_name_checkpoint, subfolder="tokenizer")
        self._tokenizer_2 = T5TokenizerFast.from_pretrained(model_name_checkpoint, subfolder="tokenizer_2")
        self._torch_text_encoder_1 = CLIPTextModel.from_pretrained(model_name_checkpoint, subfolder="text_encoder")
        torch_text_encoder_2 = T5EncoderModel.from_pretrained(
            model_name_checkpoint,
            subfolder="text_encoder_2",
            torch_dtype=torch.float32 if use_torch_encoder else torch.bfloat16,
        )
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name_checkpoint, subfolder="scheduler")
        self._vae = AutoencoderKL.from_pretrained(model_name_checkpoint, subfolder="vae")

        assert isinstance(self._tokenizer_1, CLIPTokenizer)
        assert isinstance(self._tokenizer_2, T5TokenizerFast)
        assert isinstance(self._torch_text_encoder_1, CLIPTextModel)
        assert isinstance(torch_text_encoder_2, T5EncoderModel)
        assert isinstance(self._scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(self._vae, AutoencoderKL)

        self._torch_text_encoder_1.eval()
        self._vae.eval()

        self._vae_scaling_factor = self._vae.config.scaling_factor
        self._vae_shift_factor = self._vae.config.shift_factor

        self._vae_scale_factor = 2 ** len(self._vae.config.block_out_channels)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)

        if use_torch_encoder:
            self._text_encoder_2 = torch_text_encoder_2
        else:
            logger.info("creating TT-NN text encoder...")
            parameters = T5EncoderParameters.from_torch(
                torch_text_encoder_2.state_dict(),
                device=device,
                dtype=ttnn.bfloat8_b,
            )
            self._text_encoder_2 = T5Encoder(
                parameters,
                num_heads=torch_text_encoder_2.config.num_heads,
                relative_attention_num_buckets=torch_text_encoder_2.config.relative_attention_num_buckets,
                relative_attention_max_distance=torch_text_encoder_2.config.relative_attention_max_distance,
                layer_norm_epsilon=torch_text_encoder_2.config.layer_norm_epsilon,
            )

    def prepare(
        self,
        *,
        prompt_count: int,
        num_images_per_prompt: int = 1,
        width: int = 1024,
        height: int = 1024,
        max_t5_sequence_length: int = 512,
    ) -> None:
        self._prepared_prompt_count = prompt_count
        self._prepared_num_images_per_prompt = num_images_per_prompt
        self._prepared_width = width
        self._prepared_height = height
        self._prepared_max_t5_sequence_length = max_t5_sequence_length

        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(
            prompt_1=[""] * prompt_count,
            prompt_2=[""] * prompt_count,
            num_images_per_prompt=num_images_per_prompt,
            max_t5_sequence_length=max_t5_sequence_length,
        )

        latents_height = height // self._vae_scale_factor
        latents_width = width // self._vae_scale_factor
        spatial_sequence_length = latents_width * latents_height

        mesh_height, mesh_width = self._device.shape

        tt_latents = ttnn.allocate_tensor_on_device(
            [
                prompt_count * num_images_per_prompt // mesh_height,
                spatial_sequence_length // mesh_width,
                self._num_channels_latents * 4,
            ],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self._device,
        )
        tt_prompt_embeds = ttnn.from_torch(
            prompt_embeds,
            device=self._device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=ttnn.ShardTensor2dMesh(self._device, tuple(self._device.shape), (0, -1)),
        )
        tt_pooled_prompt_embeds = ttnn.from_torch(
            pooled_prompt_embeds,
            device=self._device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(self._device, tuple(self._device.shape), (0, None)),
        )
        tt_timestep = ttnn.allocate_tensor_on_device([1, 1], ttnn.float32, ttnn.TILE_LAYOUT, self._device)
        tt_sigma_difference = ttnn.allocate_tensor_on_device([1, 1], ttnn.bfloat16, ttnn.TILE_LAYOUT, self._device)
        tt_imagerot1 = ttnn.allocate_tensor_on_device(
            [spatial_sequence_length + max_t5_sequence_length, 128], ttnn.float32, ttnn.TILE_LAYOUT, self._device
        )
        tt_imagerot2 = ttnn.allocate_tensor_on_device(
            [spatial_sequence_length + max_t5_sequence_length, 128], ttnn.float32, ttnn.TILE_LAYOUT, self._device
        )

        self._step(
            latents=tt_latents,
            timestep=tt_timestep,
            pooled_prompt_embeds=tt_pooled_prompt_embeds,
            prompt_embeds=tt_prompt_embeds,
            sigma_difference=tt_sigma_difference,
            image_rotary_emb=(tt_imagerot1, tt_imagerot2),
        )

        # # trace
        # tid = ttnn.begin_trace_capture(self._device)
        # self._step(
        #     timestep=tt_timestep,
        #     latents=tt_latents,
        #     prompt_embeds=tt_prompt_embeds,
        #     pooled_prompt_embeds=tt_pooled_prompt_embeds,
        #     sigma_difference=tt_sigma_difference,
        # )
        # ttnn.end_trace_capture(self._device, tid)
        tid = 0

        self._trace = PipelineTrace(
            tid=tid,
            spatial_input_output=tt_latents,
            prompt_input=tt_prompt_embeds,
            pooled_projection_input=tt_pooled_prompt_embeds,
            timestep_input=tt_timestep,
            sigma_difference_input=tt_sigma_difference,
            imagerot1_input=tt_imagerot1,
            imagerot2_input=tt_imagerot2,
        )

    def __call__(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        num_inference_steps: int = 40,
        seed: int | None = None,
    ) -> None:
        start_time = time.time()

        prompt_count = self._prepared_prompt_count
        num_images_per_prompt = self._prepared_num_images_per_prompt
        width = self._prepared_width
        height = self._prepared_height
        max_t5_sequence_length = self._prepared_max_t5_sequence_length

        # assert height % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        # assert width % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        assert max_t5_sequence_length <= 512
        assert prompt_count == len(prompt_1)
        assert prompt_count == len(prompt_2)

        logger.info("encoding prompts...")

        prompt_encoding_start_time = time.time()
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(
            prompt_1=prompt_1,
            prompt_2=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            max_t5_sequence_length=max_t5_sequence_length,
        )
        prompt_encoding_end_time = time.time()

        logger.info("preparing timesteps...")

        sigmas = np.linspace(
            1.0,
            1 / num_inference_steps,
            num_inference_steps,
        )

        self._scheduler.set_timesteps(sigmas=sigmas)  # type: ignore  # noqa: PGH003
        timesteps = self._scheduler.timesteps

        logger.info("preparing latents...")

        latents_height = height // self._vae_scale_factor
        latents_width = width // self._vae_scale_factor

        unpacked_latents_shape = [
            prompt_count * num_images_per_prompt,
            self._num_channels_latents,
            latents_height * 2,
            latents_width * 2,
        ]

        if seed is not None:
            torch.manual_seed(seed)
        latents = torch.randn(unpacked_latents_shape, dtype=prompt_embeds.dtype)

        latents = _pack_latents(
            latents,
            prompt_count * num_images_per_prompt,
            self._num_channels_latents,
            latents_height,
            latents_width,
        )

        logger.info("preparing embeddings...")
        text_ids = torch.zeros([prompt_embeds.shape[1], 3], dtype=self._torch_text_encoder_1.dtype)

        image_ids = _latent_image_ids(height=latents_height, width=latents_width).to(dtype=prompt_embeds.dtype)

        ids = torch.cat((text_ids, image_ids), dim=0)
        image_rotary_emb = self._pos_embed(ids)

        batch_sharded = ttnn.ShardTensor2dMesh(self._device, tuple(self._device.shape), (0, None))
        unsharded = ttnn.ReplicateTensorToMesh(self._device)

        tt_prompt_embeds = ttnn.from_torch(
            prompt_embeds,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=ttnn.ShardTensor2dMesh(self._device, tuple(self._device.shape), (0, -1)),
        )
        tt_initial_latents = ttnn.from_torch(
            latents,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(self._device, tuple(self._device.shape), (0, -2)),
        )
        tt_pooled_prompt_embeds = ttnn.from_torch(
            pooled_prompt_embeds, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=batch_sharded
        )
        tt_imagerot1 = ttnn.from_torch(
            image_rotary_emb[0], layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded
        )
        tt_imagerot2 = ttnn.from_torch(
            image_rotary_emb[1], layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=unsharded
        )

        logger.info("denoising...")
        ttnn.synchronize_device(self._device)
        denoising_start_time = time.time()

        ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._trace.prompt_input)
        ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds, self._trace.pooled_projection_input)
        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._trace.spatial_input_output)
        ttnn.copy_host_to_device_tensor(tt_imagerot1, self._trace.imagerot1_input)
        ttnn.copy_host_to_device_tensor(tt_imagerot2, self._trace.imagerot2_input)

        for i, t in enumerate(tqdm.tqdm(timesteps)):
            sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]

            # ttnn.full does not support replication for use with mesh device
            # tt_timestep = ttnn.full([1, 1], fill_value=t, dtype=ttnn.float32)
            # tt_sigma_difference = ttnn.full(
            #     [1, 1], fill_value=sigma_difference, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            # )
            tt_timestep = ttnn.from_torch(
                torch.full([1, 1], fill_value=t),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32,
                mesh_mapper=unsharded,
            )
            tt_sigma_difference = ttnn.from_torch(
                torch.full([1, 1], fill_value=sigma_difference),
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=unsharded,
            )

            ttnn.copy_host_to_device_tensor(tt_timestep, self._trace.timestep_input)
            ttnn.copy_host_to_device_tensor(tt_sigma_difference, self._trace.sigma_difference_input)

            # self._trace.execute()
            self._step(
                latents=self._trace.spatial_input_output,
                timestep=self._trace.timestep_input,
                pooled_prompt_embeds=self._trace.pooled_projection_input,
                prompt_embeds=self._trace.prompt_input,
                sigma_difference=sigma_difference,
                image_rotary_emb=(self._trace.imagerot1_input, self._trace.imagerot2_input),
            )

        ttnn.synchronize_device(self._device)
        denoising_end_time = time.time()

        logger.info("decoding images...")

        image_decoding_start_time = time.time()

        composer = ttnn.ConcatMesh2dToTensor(self._device, tuple(self._device.shape), (0, -2))
        latents = ttnn.to_torch(self._trace.spatial_input_output, mesh_composer=composer).to(torch.float32)
        latents = _unpack_latents(latents, height, width, self._vae_scale_factor)
        latents = latents / self._vae_scaling_factor + self._vae_shift_factor

        with torch.no_grad():
            images = self._vae.decoder(latents)
            images = self._image_processor.postprocess(images, output_type="pt")
            assert isinstance(images, torch.Tensor)

        image_decoding_end_time = time.time()

        output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(images))

        end_time = time.time()

        logger.info(f"prompt encoding duration: {prompt_encoding_end_time - prompt_encoding_start_time}")
        logger.info(f"denoising duration: {denoising_end_time - denoising_start_time}")
        logger.info(f"image decoding duration: {image_decoding_end_time - image_decoding_start_time}")
        logger.info(f"total runtime: {end_time - start_time}")

        return output

    def _step(
        self,
        *,
        latents: ttnn.Tensor,
        timestep: ttnn.Tensor,
        pooled_prompt_embeds: ttnn.Tensor,
        prompt_embeds: ttnn.Tensor,
        sigma_difference: ttnn.Tensor,
        image_rotary_emb: tuple[ttnn.Tensor, ttnn.Tensor],
    ) -> None:
        noise_pred = self._tt_transformer.forward(
            spatial=latents,
            prompt=prompt_embeds,
            pooled_projection=pooled_prompt_embeds,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
        )

        ttnn.add_(latents, sigma_difference * noise_pred)

    def _encode_prompts(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        num_images_per_prompt: int,
        max_t5_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenizer_max_length = self._tokenizer_1.model_max_length

        pooled_prompt_embeds = _get_clip_prompt_embeds(
            prompt=prompt_1,
            num_images_per_prompt=num_images_per_prompt,
            tokenizer=self._tokenizer_1,
            text_encoder=self._torch_text_encoder_1,
            tokenizer_max_length=tokenizer_max_length,
        )

        prompt_embeds = _get_t5_prompt_embeds(
            prompt=prompt_2,
            device=self._device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_t5_sequence_length,
            tokenizer=self._tokenizer_2,
            text_encoder=self._text_encoder_2,
        )

        return prompt_embeds, pooled_prompt_embeds


@dataclass
class PipelineTrace:
    spatial_input_output: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_projection_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    imagerot1_input: ttnn.Tensor
    imagerot2_input: ttnn.Tensor
    tid: int

    def execute(self) -> None:
        ttnn.execute_trace(self.spatial_input_output.device(), self.tid)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _get_clip_prompt_embeds(
    prompt: list[str],
    *,
    num_images_per_prompt: int,
    tokenizer_max_length: int,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
) -> torch.Tensor:
    prompt_count = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1]:
        logger.warning("CLIP input text was truncated")

    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False)
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype)

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    return prompt_embeds.view(prompt_count * num_images_per_prompt, -1)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _get_t5_prompt_embeds(
    prompt: list[str],
    *,
    device: ttnn.MeshDevice,
    num_images_per_prompt: int,
    max_sequence_length: int,
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel | T5Encoder,
) -> torch.Tensor:
    prompt_count = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1]:
        logger.warning("T5 input text was truncated")

    if isinstance(text_encoder, T5Encoder):
        tt_text_input_ids = ttnn.from_torch(
            text_input_ids.repeat(num_images_per_prompt, 1),
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, tuple(device.shape), (0, None)),
        )
        tt_prompt_embeds = text_encoder.forward(tt_text_input_ids)
        prompt_embeds = ttnn.to_torch(
            tt_prompt_embeds,
            mesh_composer=ttnn.ConcatMesh2dToTensor(device, tuple(device.shape), (0, -1)),
        )
    else:
        prompt_embeds = text_encoder.forward(text_input_ids)[0]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)

    _, seq_len = text_input_ids.shape
    return prompt_embeds.view(prompt_count * num_images_per_prompt, seq_len, -1)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    # B, C, H * P, W * Q -> B, H * W, C * P * Q
    latents = latents.view(batch_size, num_channels_latents, height, 2, width, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height) * (width), num_channels_latents * 4)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    # B, H * W, C * P * Q -> B, C, H * P, W * Q
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    return latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/flux/pipeline_flux.py
def _latent_image_ids(*, height: int, width: int) -> torch.Tensor:
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    return latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
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

from ..tt.utils import from_torch_fast, to_torch
from .t5_encoder import TtT5Encoder, TtT5EncoderParameters
from .transformer import TtSD3Transformer2DModel, TtSD3Transformer2DModelParameters
from .vae_decoder import TtVaeDecoder, TtVaeDecoderParameters

TILE_SIZE = 32


class TtStableDiffusion3Pipeline:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        device: ttnn.MeshDevice,
        enable_t5_text_encoder: bool = True,
        vae_cpu_fallback: bool = False,
        guidance_cond: int,
        model_location_generator,
    ) -> None:
        self._device = device

        logger.info("loading models...")
        model_name_checkpoint = model_location_generator(checkpoint_name, model_subdir="StableDiffusion_35_Large")

        self._tokenizer_1 = CLIPTokenizer.from_pretrained(model_name_checkpoint, subfolder="tokenizer")
        self._tokenizer_2 = CLIPTokenizer.from_pretrained(model_name_checkpoint, subfolder="tokenizer_2")
        self._tokenizer_3 = T5TokenizerFast.from_pretrained(model_name_checkpoint, subfolder="tokenizer_3")
        self._text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
            model_name_checkpoint, subfolder="text_encoder"
        )
        self._text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_name_checkpoint, subfolder="text_encoder_2"
        )
        if enable_t5_text_encoder:
            torch_text_encoder_3 = T5EncoderModel.from_pretrained(
                model_name_checkpoint,
                subfolder="text_encoder_3",
                torch_dtype=torch.bfloat16,
            )
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name_checkpoint, subfolder="scheduler")
        vae = AutoencoderKL.from_pretrained(
            model_name_checkpoint,
            subfolder="vae",
            torch_dtype=None if vae_cpu_fallback else torch.bfloat16,
        )
        torch_transformer = SD3Transformer2DModel.from_pretrained(
            model_name_checkpoint,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_transformer.eval()

        assert isinstance(self._tokenizer_1, CLIPTokenizer)
        assert isinstance(self._tokenizer_2, CLIPTokenizer)
        assert isinstance(self._tokenizer_3, T5TokenizerFast)
        assert isinstance(self._text_encoder_1, CLIPTextModelWithProjection)
        assert isinstance(self._text_encoder_2, CLIPTextModelWithProjection)
        assert isinstance(self._scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(vae, AutoencoderKL)
        assert isinstance(torch_transformer, SD3Transformer2DModel)

        logger.info("creating TT-NN transformer...")

        if checkpoint_name == "stabilityai/stable-diffusion-3.5-medium":
            embedding_dim = 1536
        else:
            embedding_dim = 2432

        num_devices = device.get_num_devices()
        ## heads padding for T3K TP
        pad_embedding_dim = False
        if os.environ["MESH_DEVICE"] == "T3K" and embedding_dim == 2432:
            pad_embedding_dim = True
            hidden_dim_padding = (
                ((embedding_dim // num_devices // TILE_SIZE) + 1) * TILE_SIZE
            ) * num_devices - embedding_dim
            num_heads = 40
        else:
            hidden_dim_padding = 0
            num_heads = torch_transformer.config.num_attention_heads

        parameters = TtSD3Transformer2DModelParameters.from_torch(
            torch_transformer.state_dict(),
            num_heads=num_heads,
            unpadded_num_heads=torch_transformer.config.num_attention_heads,
            embedding_dim=embedding_dim,
            hidden_dim_padding=hidden_dim_padding,
            device=self._device,
            dtype=ttnn.bfloat8_b if device.get_num_devices() == 1 else ttnn.bfloat16,
        )

        self._tt_transformer = TtSD3Transformer2DModel(
            parameters, guidance_cond=guidance_cond, num_heads=num_heads, device=self._device
        )
        self._num_channels_latents = torch_transformer.config.in_channels
        self._joint_attention_dim = torch_transformer.config.joint_attention_dim

        self._block_out_channels = vae.config.block_out_channels
        self._vae_scaling_factor = vae.config.scaling_factor
        self._vae_shift_factor = vae.config.shift_factor

        self._vae_scale_factor = 2 ** (len(self._block_out_channels) - 1)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)

        if enable_t5_text_encoder:
            logger.info("creating TT-NN text encoder...")

            parameters = TtT5EncoderParameters.from_torch(
                torch_text_encoder_3.state_dict(),
                device=self._device,
                dtype=ttnn.bfloat16,
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

        if not vae_cpu_fallback:
            logger.info("creating VAE decoder...")

            parameters = TtVaeDecoderParameters.from_torch(
                vae.decoder.state_dict(),
                device=self._device.get_devices()[0],
                dtype=ttnn.bfloat16,
            )
            self._vae = TtVaeDecoder(parameters, norm_num_groups=vae.config.norm_num_groups)
        else:
            self._vae = vae

    def prepare(
        self,
        *,
        num_images_per_prompt: int = 1,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 4.5,
        max_t5_sequence_length: int = 256,
        prompt_sequence_length: int = 333,
        spatial_sequence_length: int = 4096,
    ) -> None:
        self._prepared_num_images_per_prompt = num_images_per_prompt
        self._prepared_width = width
        self._prepared_height = height
        self._prepared_guidance_scale = guidance_scale
        self._prepared_max_t5_sequence_length = max_t5_sequence_length
        self._prepared_prompt_sequence_length = prompt_sequence_length

        """
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

        # TODO: pass the patch_size value
        patch_size = 2
        latents_shape = (
            batch_size * num_images_per_prompt,
            height // self._vae_scale_factor,
            (width // self._vae_scale_factor) // patch_size,
            self._num_channels_latents * patch_size,
        )

        tt_prompt_embeds = ttnn.from_torch(
            prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
        )
        tt_pooled_prompt_embeds = ttnn.from_torch(
            pooled_prompt_embeds, device=self._device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),

        )

        tt_timestep = ttnn.allocate_tensor_on_device([batch_size * num_images_per_prompt * (1+do_classifier_free_guidance), 1], ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, self._device)
        tt_sigma_difference = ttnn.allocate_tensor_on_device([1, 1], ttnn.bfloat16, ttnn.TILE_LAYOUT, self._device)
        tt_latents = ttnn.allocate_tensor_on_device(latents_shape, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self._device)

        self._device.disable_and_clear_program_cache()

        # cache
        self._step(
            timestep=tt_timestep,
            latents=tt_latents,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=tt_prompt_embeds,
            pooled_prompt_embeds=tt_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            sigma_difference=tt_sigma_difference,
            prompt_sequence_length=prompt_sequence_length,
            spatial_sequence_length=spatial_sequence_length,
        )
        self._step(
            timestep=tt_timestep,
            latents=tt_latents,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=tt_prompt_embeds,
            pooled_prompt_embeds=tt_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            sigma_difference=tt_sigma_difference,
            prompt_sequence_length=prompt_sequence_length,
            spatial_sequence_length=spatial_sequence_length,
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
            prompt_sequence_length=prompt_sequence_length,
            spatial_sequence_length=spatial_sequence_length,
        )
        ttnn.end_trace_capture(self._device, tid)

        self._trace = PipelineTrace(
            tid=tid,
            spatial_input_output=tt_latents,
            prompt_input=tt_prompt_embeds,
            pooled_projection_input=tt_pooled_prompt_embeds,
            prompt_sequence_length=prompt_sequence_length,
            spatial_sequence_length=spatial_sequence_length,
        )
        """

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

        batch_size = len(prompt_1)
        num_images_per_prompt = self._prepared_num_images_per_prompt

        assert (
            batch_size * num_images_per_prompt == 1
        ), "generating multiple images is not yet supported as it requires another mesh sharding stategy"

        width = self._prepared_width
        height = self._prepared_height
        guidance_scale = self._prepared_guidance_scale
        max_t5_sequence_length = self._prepared_max_t5_sequence_length

        assert height % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        assert width % (self._vae_scale_factor * self._tt_transformer.patch_size) == 0
        assert max_t5_sequence_length <= 512  # noqa: PLR2004

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
        # We let randn generate a permuted latent tensor, so that the generated noise matches the
        # reference implementation.
        latents = torch.randn(latents_shape, dtype=prompt_embeds.dtype).permute(0, 2, 3, 1)

        tt_prompt_embeds = ttnn.from_torch(
            prompt_embeds,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self._device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
        )
        tt_pooled_prompt_embeds = ttnn.from_torch(
            pooled_prompt_embeds,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self._device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
        )
        tt_initial_latents = ttnn.from_torch(
            latents,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self._device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
        )

        logger.info("denoising...")
        denoising_start_time = time.time()

        # ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._trace.prompt_input)
        # ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds, self._trace.pooled_projection_input)
        # ttnn.copy_host_to_device_tensor(tt_initial_latents, self._trace.spatial_input_output)

        latents_step = tt_initial_latents

        transformer_batch_size = batch_size * num_images_per_prompt
        if do_classifier_free_guidance:
            transformer_batch_size *= 2

        for i, t in enumerate(tqdm.tqdm(timesteps)):
            tt_timestep = ttnn.full(
                [transformer_batch_size, 1],
                fill_value=t,
                dtype=ttnn.float32,
                device=self._device,
                layout=ttnn.TILE_LAYOUT,
            )

            sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]
            tt_sigma_difference = ttnn.full(
                [1, 1],
                fill_value=sigma_difference,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                device=self._device,
            )

            # ttnn.copy_host_to_device_tensor(tt_timestep, self._trace.timestep_input)
            # ttnn.copy_host_to_device_tensor(tt_sigma_difference, self._trace.sigma_difference_input)
            # self._trace.execute()

            self._step(
                timestep=tt_timestep,
                latents=latents_step,  # tt_latents,
                do_classifier_free_guidance=do_classifier_free_guidance,
                prompt_embeds=tt_prompt_embeds,
                pooled_prompt_embeds=tt_pooled_prompt_embeds,
                guidance_scale=guidance_scale,
                sigma_difference=tt_sigma_difference,
                prompt_sequence_length=333,
                spatial_sequence_length=4096,
            )

        latents_output = ttnn.get_device_tensors(latents_step)[0]

        denoising_end_time = time.time()

        logger.info("decoding image...")

        image_decoding_start_time = time.time()

        # scaled_latents = 1 / self._vae_scaling_factor * self._trace.spatial_input_output + self._vae_shift_factor
        scaled_latents = 1 / self._vae_scaling_factor * latents_step + self._vae_shift_factor

        if isinstance(self._vae, TtVaeDecoder):
            images = self._vae(scaled_latents)
            torch_images = to_torch(images, dtype=torch.float32).permute(0, 3, 1, 2)
        else:
            torch_latents = to_torch(scaled_latents, dtype=torch.float32).permute([0, 3, 1, 2])
            with torch.no_grad():
                torch_images = self._vae.decoder(torch_latents)

        with torch.no_grad():
            torch_images = self._image_processor.postprocess(torch_images, output_type="pt")
            assert isinstance(torch_images, torch.Tensor)

        image_decoding_end_time = time.time()

        output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(torch_images))

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
        prompt_sequence_length: int,
        spatial_sequence_length: int,
    ) -> None:
        latent_model_input = ttnn.concat([latents, latents]) if do_classifier_free_guidance else latents

        noise_pred = self._tt_transformer(
            spatial=latent_model_input,
            prompt=prompt_embeds,
            pooled_projection=pooled_prompt_embeds,
            timestep=timestep,
            N=spatial_sequence_length,
            L=prompt_sequence_length,
        )

        noise_pred = _reshape_noise_pred(
            noise_pred,
            height=latents.shape[-3],
            width=latents.shape[-2],
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

    tt_text_input_ids = from_torch_fast(text_input_ids, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    tt_prompt_embeds = text_encoder(tt_text_input_ids, device)
    tt_prompt_embeds = ttnn.get_device_tensors(tt_prompt_embeds)[0]
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
    # B, H * W, P * Q * C -> B, H * P, W * Q, C

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

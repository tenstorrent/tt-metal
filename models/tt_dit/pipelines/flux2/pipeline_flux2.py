# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING

import diffusers
import numpy as np
import torch
import tqdm
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger
from PIL import Image

import ttnn

from ...models.transformers.transformer_flux2 import Flux2Transformer
from ...models.vae.vae_flux2 import Flux2VaeDecoder, Flux2VaeEncoder
from ...parallel.config import DiTGParallelConfigNoCFG, EncoderParallelConfig, Flux2VaeParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.padding import PaddingConfig
from ...utils.tensor import fast_device_to_host, float_to_uint8
from ...utils.tracing import StateTensor, traced_function
from .prompt_encoder import PromptEncoder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

# Defaults aligned with diffusers FluxImg2ImgPipeline for image-to-image generation.
DEFAULT_NUM_INFERENCE_STEPS = 28
DEFAULT_IMG2IMG_STRENGTH = 0.6
DEFAULT_GUIDANCE_SCALE = 4.0


class Flux2TransformerState:
    def __init__(self) -> None:
        self._tt_prompt_embeds = StateTensor()
        self._tt_latents_step = StateTensor()
        self._tt_guidance = StateTensor()
        self._tt_spatial_rope_cos = StateTensor()
        self._tt_spatial_rope_sin = StateTensor()
        self._tt_prompt_rope_cos = StateTensor()
        self._tt_prompt_rope_sin = StateTensor()
        self._tt_timestep = StateTensor()
        self._tt_sigma_difference = StateTensor()

    def __getattr__(self, name: str) -> ttnn.Tensor | None:
        return object.__getattribute__(self, f"_{name}").data


class Flux2Pipeline:
    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_name: str = "black-forest-labs/FLUX.2-dev",
        parallel_config: DiTGParallelConfigNoCFG,
        encoder_parallel_config: EncoderParallelConfig,
        vae_parallel: Flux2VaeParallelConfig,
        topology: ttnn.Topology,
        num_links: int,
        height: int = 1024,
        width: int = 1024,
        is_fsdp: bool = False,
        dynamic_load: bool = False,
        trace_warmup: bool = False,
        vae_use_conv3d: bool = False,
        shard_prompt: bool = False,
        warmup_image: Image.Image | None = None,
    ) -> None:
        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self._encoder_parallel_config = encoder_parallel_config
        self._shard_prompt = shard_prompt

        self._vae_parallel = vae_parallel
        self._vae_use_conv3d = vae_use_conv3d

        self._height = height
        self._width = width
        self.is_fsdp = is_fsdp
        self.dynamic_load = dynamic_load

        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Encoder parallel config: {encoder_parallel_config}")
        logger.info(f"VAE parallel: {vae_parallel}")

        self._ccl_manager = CCLManager(self._mesh_device, num_links=num_links, topology=topology)

        logger.info("loading models...")

        self.checkpoint_name = checkpoint_name

        self._torch_transformer = diffusers.Flux2Transformer2DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,  # bfloat16 is the native datatype of the model
        )
        self._torch_transformer.eval()

        self._torch_vae = AutoencoderKLFlux2.from_pretrained(checkpoint_name, subfolder="vae")
        assert isinstance(self._torch_vae, AutoencoderKLFlux2)

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")

        logger.info("creating TT-NN transformer...")

        head_dim = 128
        num_heads = 48
        self._num_channels_latents = 32
        self._patch_size = 2
        self._vae_scale_factor = 8

        if num_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                num_heads,
                head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self.transformer = Flux2Transformer(
            in_channels=128,
            num_layers=8,
            num_single_layers=48,
            attention_head_dim=head_dim,
            num_attention_heads=num_heads,
            joint_attention_dim=15360,
            out_channels=128,
            device=mesh_device,
            ccl_manager=self._ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            is_fsdp=self.is_fsdp,
            shard_prompt=self._shard_prompt,
        )

        self._pos_embed = self._torch_transformer.pos_embed

        self._image_processor = Flux2ImageProcessor(vae_scale_factor=self._vae_scale_factor * self._patch_size)

        logger.info("creating TT-NN text encoder...")
        self._encoder_ccl_manager = CCLManager(self._mesh_device, num_links=num_links, topology=topology)
        self._prompt_encoder = PromptEncoder(
            checkpoint_name=checkpoint_name,
            use_torch_encoder=False,
            device=self._mesh_device,
            parallel_config=self._encoder_parallel_config,
            ccl_manager=self._encoder_ccl_manager,
        )

        logger.info("creating TT-NN VAE decoder...")
        self._vae_ccl_manager = CCLManager(self._mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
        self._vae_decoder = Flux2VaeDecoder(
            out_channels=3,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            z_channels=32,
            device=self._mesh_device,
            parallel_config=self._vae_parallel,
            ccl_manager=self._vae_ccl_manager,
            use_conv3d=self._vae_use_conv3d,
        )

        logger.info("creating TT-NN VAE encoder...")
        self._vae_encoder = Flux2VaeEncoder(
            in_channels=3,
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            z_channels=32,
            device=self._mesh_device,
            parallel_config=self._vae_parallel,
            ccl_manager=self._vae_ccl_manager,
            use_conv3d=self._vae_use_conv3d,
        )

        if self.dynamic_load:
            self.transformer.register_coresident_exclusions(
                self._prompt_encoder._encoder, self._vae_decoder, self._vae_encoder
            )
            self._prompt_encoder._encoder.register_coresident_exclusions(self.transformer)
            self._vae_decoder.register_coresident_exclusions(self.transformer)
            self._vae_encoder.register_coresident_exclusions(self.transformer)

        # Load in reverse order of use so the first-used model (encoder) stays loaded before __call__.
        self._prepare_vae()
        self._prepare_transformer()
        self._prepare_prompt_encoder()

        self.ts = Flux2TransformerState()

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        text_ids = _prepare_ids(text_sequence_length=512)  # matches sequence_length=512 in encode
        image_ids = _prepare_ids(height=self._height // 16, width=self._width // 16)
        prompt_rope_cos, prompt_rope_sin = self._pos_embed.forward(text_ids)
        spatial_rope_cos, spatial_rope_sin = self._pos_embed.forward(image_ids)
        # Store rope tensors pre-shaped as (1, 1, seq, head_dim) so attention blocks never
        # need to reshape per-call (2800 ops per inference). SP sharding is on dim 2 (seq).
        self.ts._tt_spatial_rope_cos.update(
            spatial_rope_cos.unsqueeze(0).unsqueeze(0),
            False,
            mesh_axes=[None, None, sp_axis, None],
            device=self._mesh_device,
        )
        self.ts._tt_spatial_rope_sin.update(
            spatial_rope_sin.unsqueeze(0).unsqueeze(0),
            False,
            mesh_axes=[None, None, sp_axis, None],
            device=self._mesh_device,
        )
        # When sharding the prompt across SP, the prompt rope must be sharded the same way (on the
        # sequence dim) so per-rank tokens get their matching rope positions before the attention
        # gathers q/k/v back to full length.
        prompt_rope_mesh_axes = [None, None, sp_axis, None] if self._shard_prompt else None
        self.ts._tt_prompt_rope_cos.update(
            prompt_rope_cos.unsqueeze(0).unsqueeze(0), False, mesh_axes=prompt_rope_mesh_axes, device=self._mesh_device
        )
        self.ts._tt_prompt_rope_sin.update(
            prompt_rope_sin.unsqueeze(0).unsqueeze(0), False, mesh_axes=prompt_rope_mesh_axes, device=self._mesh_device
        )

        # Warm up with a condition image when the pipeline will be used for img2img, so the
        # persistent StateTensor buffers (latents, spatial rope) get preallocated at the combined
        # noise+image sequence length. Otherwise the text2img e2e warmup sizes them for noise-only
        # latents and the first img2img (traced) call fails the copy-into-existing-buffer.
        self.warmup(warmup_info="e2e warmup", image=warmup_image)  # E2E warmup. Preallocate buffers
        if trace_warmup:  # warmup for trace capture
            self.warmup(traced=True, warmup_info="trace warmup", image=warmup_image)  # warmup for trace capture
            self.warmup(
                traced=True, warmup_info="steady state trace warmup for VAE", image=warmup_image
            )  # TODO: INVESTIGATE. For some reason, the VAE time still incures some overhead without this extra trace

    def warmup(self, traced: bool = False, warmup_info="warmup", image: Image.Image | None = None) -> None:
        logger.info(f"Warming up pipeline___ {warmup_info}...")
        self.__call__(
            prompts=["warmup"],
            num_inference_steps=2,
            seed=0,
            traced=traced,
            image=image,
        )

    def _prepare_transformer(self) -> None:
        cache.load_model(
            tt_model=self.transformer,
            get_torch_state_dict=self._torch_transformer.state_dict,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="transformer",
            parallel_config=self._parallel_config,
            mesh_shape=tuple(self._mesh_device.shape),
            is_fsdp=self.is_fsdp,
        )
        ttnn.synchronize_device(self._mesh_device)

    def _prepare_prompt_encoder(self) -> None:
        self._prompt_encoder.load_weights()
        ttnn.synchronize_device(self._mesh_device)

    def _prepare_vae(self) -> None:
        cache.load_model(
            tt_model=self._vae_decoder,
            get_torch_state_dict=self._torch_vae.state_dict,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="vae",
            parallel_config=self._vae_parallel,
            mesh_shape=tuple(self._mesh_device.shape),
        )
        ttnn.synchronize_device(self._mesh_device)

    def _prepare_vae_encoder(self) -> None:
        # distinct subfolder from the decoder so the two VAE halves don't collide in the weight cache
        cache.load_model(
            tt_model=self._vae_encoder,
            get_torch_state_dict=self._torch_vae.state_dict,
            model_name=os.path.basename(self.checkpoint_name),
            subfolder="vae_encoder",
            parallel_config=self._vae_parallel,
            mesh_shape=tuple(self._mesh_device.shape),
        )
        ttnn.synchronize_device(self._mesh_device)

    @staticmethod
    def create_pipeline(
        *,
        mesh_device: ttnn.MeshDevice,
        sp_axis: int,
        tp_axis: int,
        encoder_tp_axis: int,
        vae_tp_axis: int | None = None,
        vae_h_axis: int | None = None,
        vae_w_axis: int | None = None,
        num_links: int,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        width: int = 1024,
        height: int = 1024,
        is_fsdp: bool = False,
        dynamic_load: bool = False,
        trace_warmup: bool = False,
        checkpoint_name: str = "black-forest-labs/FLUX.2-dev",
        vae_use_conv3d: bool = False,
        shard_prompt: bool = False,
        warmup_image: Image.Image | None = None,
    ) -> Flux2Pipeline:
        dit_parallel_config = DiTGParallelConfigNoCFG(
            tensor_parallel=ParallelFactor(factor=int(mesh_device.shape[tp_axis]), mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=int(mesh_device.shape[sp_axis]), mesh_axis=sp_axis),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=int(mesh_device.shape[encoder_tp_axis]), mesh_axis=encoder_tp_axis)
        )
        vae_parallel = Flux2VaeParallelConfig.from_axes(
            mesh_device, tp_axis=vae_tp_axis, h_axis=vae_h_axis, w_axis=vae_w_axis
        )

        return Flux2Pipeline(
            mesh_device=mesh_device,
            checkpoint_name=checkpoint_name,
            parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel=vae_parallel,
            topology=topology,
            num_links=num_links,
            width=width,
            height=height,
            is_fsdp=is_fsdp,
            dynamic_load=dynamic_load,
            trace_warmup=trace_warmup,
            vae_use_conv3d=vae_use_conv3d,
            shard_prompt=shard_prompt,
            warmup_image=warmup_image,
        )

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        prompts: Sequence[str],
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        image: Image.Image | None = None,
        strength: float = DEFAULT_IMG2IMG_STRENGTH,
        prompt_upsample_temperature: float | None = None,  # prompt upsampling is currently very slow
        seed: int | None = None,
        traced: bool = False,
        profiler=None,
        profiler_iteration: int = 0,
    ) -> list[Image.Image]:
        prompt_count = len(prompts)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        if image is not None:
            if strength < 0 or strength > 1:
                msg = f"The value of strength should be in [0.0, 1.0] but is {strength}"
                raise ValueError(msg)

        latents_height = self._height // self._vae_scale_factor
        latents_width = self._width // self._vae_scale_factor
        transformer_batch_size = prompt_count * num_images_per_prompt
        noise_sequence_length = (latents_height // self._patch_size) * (latents_width // self._patch_size)

        logger.info("encoding prompts...")

        with profiler("encoder", profiler_iteration) if profiler else nullcontext():
            self._prepare_prompt_encoder()
            if prompt_upsample_temperature is not None:
                prompts = self._prompt_encoder.upsample(
                    prompts,
                    max_length=224,  # TODO: this should be higher
                    temperature=prompt_upsample_temperature,
                    images=[image] if image is not None else None,
                    traced=traced,
                )

            prompt_embeds, _mask = self._prompt_encoder.encode(
                prompts,
                num_images_per_prompt=num_images_per_prompt,
                sequence_length=512,
                traced=traced,
            )
            _, prompt_sequence_length, _ = prompt_embeds.shape

        image_latents_torch: torch.Tensor | None = None
        image_latents_for_rope: list[torch.Tensor] | None = None
        if image is not None:
            logger.info("encoding condition image...")
            with profiler("vae_encode", profiler_iteration) if profiler else nullcontext():
                condition_image = self._preprocess_condition_image(image)
                patchified = self._encode_vae_image(condition_image, traced=traced)
            image_latents_torch = _pack_latents(patchified).repeat(transformer_batch_size, 1, 1)
            image_latents_for_rope = [patchified[0]]

        spatial_sequence_length = noise_sequence_length
        if image_latents_torch is not None:
            spatial_sequence_length += image_latents_torch.shape[1]
            self._update_spatial_rope(
                noise_height=latents_height // self._patch_size,
                noise_width=latents_width // self._patch_size,
                image_latents=image_latents_for_rope or [],
                traced=traced,
            )

        logger.info("preparing timesteps...")
        timesteps, sigmas = _schedule(
            self._scheduler,
            step_count=num_inference_steps,
            spatial_sequence_length=noise_sequence_length,
        )
        if image is not None:
            timesteps, sigmas, num_inference_steps = _truncate_timesteps_for_strength(
                self._scheduler,
                timesteps=timesteps,
                sigmas=sigmas,
                num_inference_steps=num_inference_steps,
                strength=strength,
            )
            if num_inference_steps < 1:
                msg = (
                    f"After adjusting num_inference_steps by strength={strength}, the number of pipeline steps is "
                    f"{num_inference_steps}, which is < 1 and not appropriate for this pipeline."
                )
                raise ValueError(msg)

        guidance = torch.full([transformer_batch_size], fill_value=guidance_scale)

        logger.info("preparing latents...")

        if seed is not None:
            torch.manual_seed(seed)

        shape = [transformer_batch_size, self._num_channels_latents, latents_height, latents_width]
        latents = self._patchify(torch.randn(shape).permute(0, 2, 3, 1))

        is_img2img = image_latents_torch is not None
        latents_to_upload = torch.cat([latents, image_latents_torch], dim=1) if is_img2img else latents

        # Shard the prompt sequence across SP (like spatial) when enabled, so per-block matmuls
        # run on 1/sp of the prompt tokens; attention gathers prompt q/k/v for the joint SDPA.
        prompt_embeds_mesh_axes = [None, sp_axis, None] if self._shard_prompt else None
        self.ts._tt_prompt_embeds.update(
            prompt_embeds, traced, mesh_axes=prompt_embeds_mesh_axes, device=self._mesh_device
        )
        self.ts._tt_latents_step.update(
            latents_to_upload, traced, mesh_axes=[None, sp_axis, None], device=self._mesh_device
        )
        self.ts._tt_guidance.update(guidance.unsqueeze(1), traced, device=self._mesh_device)

        logger.info("denoising...")
        self._prepare_transformer()

        with profiler("denoising", profiler_iteration) if profiler else nullcontext():
            for i, t in enumerate(tqdm.tqdm(timesteps)):
                with profiler(f"denoising_step_{i}", profiler_iteration) if profiler else nullcontext():
                    sigma_difference = (sigmas[i + 1] - sigmas[i]).item()

                    self.ts._tt_timestep.update(
                        torch.full([1, 1], fill_value=float(t)),
                        device=self._mesh_device,
                        dtype=ttnn.float32,
                        traced=traced,
                    )

                    self.ts._tt_sigma_difference.update(
                        torch.full([1, 1], fill_value=sigma_difference),
                        device=self._mesh_device,
                        dtype=ttnn.float32,
                        traced=traced,
                    )

                    self._step(
                        spatial=self.ts.tt_latents_step,
                        prompt=self.ts.tt_prompt_embeds,
                        timestep=self.ts.tt_timestep,
                        guidance=self.ts.tt_guidance,
                        sigma_difference=self.ts.tt_sigma_difference,
                        spatial_rope=(self.ts.tt_spatial_rope_cos, self.ts.tt_spatial_rope_sin),
                        prompt_rope=(self.ts.tt_prompt_rope_cos, self.ts.tt_prompt_rope_sin),
                        noise_sequence_length=noise_sequence_length,
                        spatial_sequence_length=spatial_sequence_length,
                        prompt_sequence_length=prompt_sequence_length,
                        traced=traced,
                    )

        logger.info("decoding image...")
        pcv = profiler("vae", profiler_iteration) if profiler else nullcontext()
        with pcv:
            self._prepare_vae()

            tt_latents = self.ts.tt_latents_step
            if is_img2img:
                tt_latents = _extract_noise_latents_from_combined(
                    self._ccl_manager,
                    self._parallel_config,
                    tt_latents,
                    noise_sequence_length=noise_sequence_length,
                )

            # Latents arrive SP-sharded on dim=1 (patchified token dim). Redistribute to
            # whatever spatial sharding the VAE conv pyramid expects before unpatchifying.
            #
            # H (token dim=1): gather SP, then re-shard to H axis if on a different axis.
            # W (spatial dim=2): can only be applied after unpatchify since H/W are interleaved
            #   in patchified tokens; mesh_partition is applied right after the reshape below.
            if self._vae_parallel.h_parallel is not None and self._vae_parallel.h_parallel.mesh_axis == sp_axis:
                pass
            else:
                tt_latents = self._ccl_manager.all_gather(
                    tt_latents,
                    dim=1,
                    mesh_axis=sp_axis,
                    use_hyperparams=True,
                )
                if self._vae_parallel.h_parallel is not None:
                    tt_latents = ttnn.mesh_partition(
                        tt_latents,
                        dim=1,
                        cluster_axis=self._vae_parallel.h_parallel.mesh_axis,
                        memory_config=tt_latents.memory_config(),
                    )

            tt_latents = self._vae_decoder.preprocess_and_unpatchify(
                tt_latents,
                height=self._height // self._vae_scale_factor,
                width=self._width // self._vae_scale_factor,
            )

            if self._vae_parallel.w_parallel is not None:
                tt_latents = ttnn.mesh_partition(
                    tt_latents,
                    dim=2,
                    cluster_axis=self._vae_parallel.w_parallel.mesh_axis,
                    memory_config=tt_latents.memory_config(),
                )

            tt_decoded_output = self._vae_decoder.forward(tt_latents, traced=traced)
            decoded_output = fast_device_to_host(
                tt_decoded_output,
                self._mesh_device,
                [None, None],
                ccl_manager=self._vae_ccl_manager,
                pre_transfer_fn=float_to_uint8,
            )
            output = [Image.fromarray(image.numpy()) for image in decoded_output]

        return output

    @traced_function(device=lambda self: self._mesh_device, clone_prep_inputs=False)
    def _step(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor,
        sigma_difference: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        noise_sequence_length: int,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> None:
        is_img2img = spatial_sequence_length > noise_sequence_length

        noise_pred = self.transformer.forward(
            spatial=spatial,
            prompt=prompt,
            timestep=timestep,
            guidance=guidance,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )

        if is_img2img:
            _apply_img2img_scheduler_step(
                self._ccl_manager,
                self._parallel_config,
                spatial=spatial,
                noise_pred=noise_pred,
                noise_sequence_length=noise_sequence_length,
                sigma_difference=sigma_difference,
            )
            return

        ttnn.multiply_(noise_pred, sigma_difference)
        ttnn.add_(spatial, noise_pred)

    def _preprocess_condition_image(self, image: Image.Image) -> torch.Tensor:
        self._image_processor.check_image_input(image)

        image_width, image_height = image.size
        if image_width * image_height > 1024 * 1024:
            image = self._image_processor._resize_to_target_area(image, 1024 * 1024)
            image_width, image_height = image.size

        multiple_of = self._vae_scale_factor * self._patch_size
        image_width = (image_width // multiple_of) * multiple_of
        image_height = (image_height // multiple_of) * multiple_of
        return self._image_processor.preprocess(image, height=image_height, width=image_width, resize_mode="crop")

    def _encode_vae_image(self, image: torch.Tensor, *, traced: bool = False) -> torch.Tensor:
        # Runs the VAE encoder, .mode(), space-to-depth patchify and batch-norm normalize entirely
        # on device. Returns the normalized patchified latent as host [B, C*p^2, H/16, W/16]
        # (channel-first), matching the previous torch fallback's contract so the packing/rope/cat
        # logic in __call__ is unchanged.
        self._prepare_vae_encoder()

        _, _, img_height, img_width = image.shape
        h_axis = self._vae_parallel.h_parallel.mesh_axis if self._vae_parallel.h_parallel is not None else None
        mesh_axes = [None, h_axis, None, None] if h_axis is not None else None

        image_nhwc = image.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)
        tt_image = tensor.from_torch(
            image_nhwc,
            device=self._mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_axes=mesh_axes,
        )

        enc_height = img_height // self._vae_scale_factor
        enc_width = img_width // self._vae_scale_factor
        patchified = self._vae_encoder.encode_and_patchify(tt_image, height=enc_height, width=enc_width, traced=traced)

        patchified_host = tensor.to_torch(
            patchified,
            mesh_axes=mesh_axes,
            composer_device=self._mesh_device,
        )
        # [B, H/16, W/16, C*p^2] -> [B, C*p^2, H/16, W/16]
        return patchified_host.permute(0, 3, 1, 2).contiguous().to(torch.float32)

    def _update_spatial_rope(
        self,
        *,
        noise_height: int,
        noise_width: int,
        image_latents: list[torch.Tensor],
        traced: bool,
    ) -> None:
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        noise_ids = _prepare_ids(height=noise_height, width=noise_width, text_sequence_length=1)
        if image_latents:
            image_ids = _prepare_image_ids(image_latents).squeeze(0)
            spatial_ids = torch.cat([noise_ids, image_ids], dim=0)
        else:
            spatial_ids = noise_ids

        spatial_rope_cos, spatial_rope_sin = self._pos_embed.forward(spatial_ids)
        self.ts._tt_spatial_rope_cos.update(
            spatial_rope_cos.unsqueeze(0).unsqueeze(0),
            traced,
            mesh_axes=[None, None, sp_axis, None],
            device=self._mesh_device,
        )
        self.ts._tt_spatial_rope_sin.update(
            spatial_rope_sin.unsqueeze(0).unsqueeze(0),
            traced,
            mesh_axes=[None, None, sp_axis, None],
            device=self._mesh_device,
        )

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        # N, H, W, C -> N, (H / P) * (W / P), C * P * P
        batch_size, height, width, channels = latents.shape
        p = self._patch_size

        if height % p != 0 or width % p != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({p})"
            raise ValueError(msg)

        latents = latents.reshape([batch_size, height // p, p, width // p, p, channels])
        return latents.permute(0, 1, 3, 5, 2, 4).flatten(3, 5).flatten(1, 2)


def _truncate_timesteps_for_strength(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    num_inference_steps: int,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
    init_timestep = min(num_inference_steps * strength, num_inference_steps)
    t_start = int(max(num_inference_steps - init_timestep, 0))
    order = scheduler.order
    timesteps = timesteps[t_start * order :]
    sigmas = sigmas[t_start * order : t_start * order + len(timesteps) + 1]
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(t_start * order)

    return timesteps, sigmas, num_inference_steps - t_start


def _schedule(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    step_count: int,
    spatial_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scheduler.set_timesteps(
        sigmas=np.linspace(1.0, 1 / step_count, step_count),
        mu=compute_empirical_mu(image_seq_len=spatial_sequence_length, num_steps=step_count),
    )

    timesteps = scheduler.timesteps
    sigmas = scheduler.sigmas

    assert isinstance(timesteps, torch.Tensor)
    assert isinstance(sigmas, torch.Tensor)

    return timesteps, sigmas


# From https://github.com/black-forest-labs/flux2/blob/ab7cca68018ad3ceadcace9d6ecb1bc1f6f46b4e/src/flux2/sampling.py#L251C1-L266C21
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def _prepare_ids(*, height: int = 1, width: int = 1, text_sequence_length: int = 1) -> torch.Tensor:
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    s = torch.arange(text_sequence_length)

    return torch.cartesian_prod(t, h, w, s)


def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)


def _extract_noise_latents_from_combined(
    ccl_manager: CCLManager,
    parallel_config: DiTGParallelConfigNoCFG,
    combined: ttnn.Tensor,
    *,
    noise_sequence_length: int,
) -> ttnn.Tensor:
    sp_axis = parallel_config.sequence_parallel.mesh_axis
    full_combined = ccl_manager.all_gather(combined, dim=1, mesh_axis=sp_axis, use_hyperparams=True)
    noise_latents = ttnn.slice(
        full_combined,
        [0, 0, 0],
        [full_combined.shape[0], noise_sequence_length, full_combined.shape[2]],
    )
    return ttnn.mesh_partition(
        noise_latents,
        dim=1,
        cluster_axis=sp_axis,
        memory_config=noise_latents.memory_config(),
    )


def _apply_img2img_scheduler_step(
    ccl_manager: CCLManager,
    parallel_config: DiTGParallelConfigNoCFG,
    *,
    spatial: ttnn.Tensor,
    noise_pred: ttnn.Tensor,
    noise_sequence_length: int,
    sigma_difference: ttnn.Tensor,
) -> None:
    sp_axis = parallel_config.sequence_parallel.mesh_axis

    full_pred = ccl_manager.all_gather(noise_pred, dim=1, mesh_axis=sp_axis, use_hyperparams=True)
    noise_pred_only = ttnn.slice(
        full_pred,
        [0, 0, 0],
        [full_pred.shape[0], noise_sequence_length, full_pred.shape[2]],
    )
    ttnn.multiply_(noise_pred_only, sigma_difference)

    full_spatial = ccl_manager.all_gather(spatial, dim=1, mesh_axis=sp_axis, use_hyperparams=True)
    noise_spatial = ttnn.slice(
        full_spatial,
        [0, 0, 0],
        [full_spatial.shape[0], noise_sequence_length, full_spatial.shape[2]],
    )
    image_spatial = ttnn.slice(
        full_spatial,
        [0, noise_sequence_length, 0],
        [full_spatial.shape[0], full_spatial.shape[1], full_spatial.shape[2]],
    )
    updated_noise = ttnn.add(noise_spatial, noise_pred_only)
    full_spatial = ttnn.concat([updated_noise, image_spatial], dim=1)

    updated_combined = ttnn.mesh_partition(
        full_spatial,
        dim=1,
        cluster_axis=sp_axis,
        memory_config=full_spatial.memory_config(),
    )
    ttnn.copy(updated_combined, spatial)


def _prepare_image_ids(image_latents: list[torch.Tensor], *, scale: int = 10) -> torch.Tensor:
    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for latent, t in zip(image_latents, t_coords):
        latent = latent.squeeze(0)
        _, height, width = latent.shape
        x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
        image_latent_ids.append(x_ids)

    return torch.cat(image_latent_ids, dim=0).unsqueeze(0)

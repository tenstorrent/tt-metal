# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import huggingface_hub
import torch
import tqdm
import ttnn
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from loguru import logger

from ...encoders.clip.encoder_pair import CLIPTokenizerEncoderPair
from ...encoders.t5.encoder_pair import T5TokenizerEncoderPair
from ...models.transformers.transformer_motif import MotifTransformer, convert_motif_transformer_state
from ...models.vae.vae_sd35 import VAEDecoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.padding import PaddingConfig
from ...utils.substate import substate

if TYPE_CHECKING:
    from collections.abc import Iterable

    from PIL import Image


@dataclass
class PipelineTrace:
    tid: int
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    latents_output: ttnn.Tensor


class MotifPipeline:
    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        enable_t5_text_encoder: bool = True,
        use_torch_t5_text_encoder: bool = False,
        use_torch_clip_text_encoder: bool = False,
        parallel_config: DiTParallelConfig,
        topology: ttnn.Topology,
        num_links: int,
        height: int = 1024,
        width: int = 1024,
        use_cache: bool,
    ) -> None:
        self.timing_collector = None

        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self._height = height
        self._width = width

        submesh_shape = list(mesh_device.shape)
        submesh_shape[parallel_config.cfg_parallel.mesh_axis] //= parallel_config.cfg_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Original mesh shape: {mesh_device.shape}")
        logger.info(f"Creating submeshes with shape {submesh_shape}")
        self._submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))

        self._ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=topology)
            for submesh_device in self._submesh_devices
        ]

        # Hacky submesh reshapes and assignment to parallelize encoders and VAE
        encoder_device = self._submesh_devices[0]
        self.original_submesh_shape = tuple(encoder_device.shape)
        self.desired_encoder_submesh_shape = tuple(encoder_device.shape)

        if encoder_device.shape[1] != 4:
            # If reshaping, vae_device must be on submesh 0. That means T5 can't fit, so disable it.
            vae_submesh_idx = 0
            if enable_t5_text_encoder and not use_torch_t5_text_encoder:
                logger.warning(
                    "If VAE submesh must be reshaped, VAE must be on submesh 0, and T5 cannot fit. Disabling T5."
                )
                enable_t5_text_encoder = False

            cfg_shape = tuple(encoder_device.shape)
            assert cfg_shape[0] * cfg_shape[1] == 4, f"Cannot reshape {cfg_shape} to a 1x4 mesh"
            logger.info(f"Reshaping submesh device 0 from {cfg_shape} to (1, 4) for CLIP")
            self.desired_encoder_submesh_shape = (1, 4)
        else:
            # vae_device can only be on submesh 1 if submesh is not getting reshaped.
            vae_submesh_idx = 1 if len(self._submesh_devices) > 1 else 0
        vae_device = self._submesh_devices[vae_submesh_idx]

        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=encoder_device.shape[1], mesh_axis=1),
        )
        self.encoder_parallel_config = encoder_parallel_config
        self.encoder_device = encoder_device

        vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))
        self.vae_parallel_config = vae_parallel_config
        self.vae_device = vae_device
        self.vae_submesh_idx = vae_submesh_idx

        logger.info("loading models...")
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id="Motif-Technologies/Motif-Image-6B-Preview",
            filename="motif_image_preview.bin",
            subfolder="checkpoints",
            revision="update_new_ckpt",
        )

        vae_checkpoint = "stabilityai/stable-diffusion-3-medium-diffusers"
        self._torch_vae = AutoencoderKL.from_pretrained(vae_checkpoint, subfolder="vae")
        assert isinstance(self._torch_vae, AutoencoderKL)

        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), mmap=True)

        logger.info("creating TT-NN transformer...")

        num_heads = 30
        head_dim = 64
        num_layers = 30
        self._num_channels_latents = 16
        self._prompt_embedding_dim = MotifTransformer.ENCODED_TEXT_DIM
        self._patch_size = 2
        self._vae_scale_factor = 8

        transformer_state_dict = substate(state_dict, "dit")
        convert_motif_transformer_state(transformer_state_dict, num_layers=num_layers)

        if num_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                num_heads,
                head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self.transformers = []
        for i, submesh_device in enumerate(self._submesh_devices):
            tt_transformer = MotifTransformer(
                patch_size=self._patch_size,
                num_layers=num_layers,
                attention_head_dim=head_dim,
                num_attention_heads=num_heads,
                pooled_projection_dim=2048,
                pos_embed_max_size=64,
                modulation_dim=4096,
                time_embed_dim=4096,
                register_token_num=4,
                latents_height=height // self._vae_scale_factor,
                latents_width=width // self._vae_scale_factor,
                mesh_device=submesh_device,
                ccl_manager=self._ccl_managers[i],
                parallel_config=parallel_config,
                padding_config=padding_config,
            )

            if use_cache:
                cache_path = cache.get_and_create_cache_path(
                    model_name="motif-image-6b",
                    subfolder="transformer",
                    parallel_config=self._parallel_config,
                    mesh_shape=submesh_device.shape,
                    dtype="bf16",
                )
                # create cache if it doesn't exist
                if not cache.cache_dict_exists(cache_path):
                    logger.info(
                        f"Cache does not exist. Creating cache: {cache_path} and loading transformer weights from PyTorch state dict"
                    )
                    tt_transformer.load_torch_state_dict(transformer_state_dict)
                    cache.save_cache_dict(tt_transformer.to_cached_state_dict(cache_path), cache_path)
                else:
                    logger.info(f"Loading transformer weights from cache: {cache_path}")
                    tt_transformer.from_cached_state_dict(cache.load_cache_dict(cache_path))
            else:
                logger.info("Loading transformer weights from PyTorch state dict")
                tt_transformer.load_torch_state_dict(transformer_state_dict)

            self.transformers.append(tt_transformer)
            ttnn.synchronize_device(submesh_device)

        self._latents_scaling = self._torch_vae.config.scaling_factor
        self._latents_shift = self._torch_vae.config.shift_factor

        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)

        if self.desired_encoder_submesh_shape != self.original_submesh_shape:
            # HACK: reshape submesh device 0 to 1D
            self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))

        self._text_encoder = TextEncoder(
            device=encoder_device,
            ccl_manager=self._ccl_managers[0],
            parallel_config=encoder_parallel_config,
            enable_t5=enable_t5_text_encoder,
            use_torch_clip_encoder=use_torch_clip_text_encoder,
            use_torch_t5_encoder=use_torch_t5_text_encoder,
        )

        self._traces = None

        ttnn.synchronize_device(self.encoder_device)

        self._vae_decoder = VAEDecoder.from_torch(
            torch_ref=self._torch_vae.decoder,
            mesh_device=self.vae_device,
            parallel_config=self.vae_parallel_config,
            ccl_manager=self._ccl_managers[vae_submesh_idx],
        )

        if self.desired_encoder_submesh_shape != self.original_submesh_shape:
            # HACK: reshape submesh device 0 to 1D
            # If reshaping, vae device is same as encoder device
            self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        cfg_scale: float,
        prompt_1: list[str],
        prompt_2: list[str],
        prompt_3: list[str],
        negative_prompt_1: list[str | None],
        negative_prompt_2: list[str | None],
        negative_prompt_3: list[str | None],
        linear_quadratic_emulating_steps: int = 100,
        negative_strategy_switch_time: float = 0.85,
        num_inference_steps: int,
        seed: int | None = None,
        traced: bool = False,
    ) -> list[Image.Image]:
        timer = self.timing_collector
        prompt_count = len(prompt_1)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        cfg_factor = self._parallel_config.cfg_parallel.factor

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        with timer.time_section("total") if timer else nullcontext():
            cfg_enabled = cfg_scale > 1
            logger.info("encoding prompts...")

            with timer.time_section("total_encoding") if timer else nullcontext():
                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # HACK: reshape submesh device 0 from 2D to 1D
                    self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))
                prompt_embeds1, pooled_prompt_embeds1, prompt_embeds2, pooled_prompt_embeds2 = self._encode_prompts(
                    prompt_1=prompt_1,
                    prompt_2=prompt_2,
                    prompt_3=prompt_3,
                    negative_prompt_1=negative_prompt_1,
                    negative_prompt_2=negative_prompt_2,
                    negative_prompt_3=negative_prompt_3,
                    num_images_per_prompt=num_images_per_prompt,
                    cfg_enabled=cfg_enabled,
                )
                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # HACK: reshape submesh device 0 from 1D to 2D
                    self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))

            logger.info("preparing timesteps...")
            timesteps, sigmas = _schedule(
                step_count=num_inference_steps,
                linear_quadratic_emulating_steps=linear_quadratic_emulating_steps,
            )

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)

            shape = [
                prompt_count * num_images_per_prompt,
                self._num_channels_latents,
                self._height // self._vae_scale_factor,
                self._width // self._vae_scale_factor,
            ]
            # We let randn generate a permuted latent tensor in float32, so that the generated noise
            # matches the reference implementation.
            latents = self.transformers[0].patchify(
                torch.randn(shape, dtype=torch.float32).to(dtype=torch.bfloat16).permute(0, 2, 3, 1)
            )

            tt_prompt_embeds_device_list = []
            tt_prompt_embeds1_list = []
            tt_prompt_embeds2_list = []
            tt_pooled_prompt_embeds_device_list = []
            tt_pooled_prompt_embeds1_list = []
            tt_pooled_prompt_embeds2_list = []
            tt_latents_step_list = []
            for i, submesh_device in enumerate(self._submesh_devices):
                tt_prompt_embeds_device = tensor.from_torch(
                    prompt_embeds1[i : i + 1] if cfg_factor == 2 else prompt_embeds1,
                    device=submesh_device,
                    on_host=traced,
                )
                tt_prompt_embeds1 = tensor.from_torch(
                    prompt_embeds1[i : i + 1] if cfg_factor == 2 else prompt_embeds1,
                    device=submesh_device,
                    on_host=True,
                )
                tt_prompt_embeds2 = tensor.from_torch(
                    prompt_embeds2[i : i + 1] if cfg_factor == 2 else prompt_embeds2,
                    device=submesh_device,
                    on_host=True,
                )

                tt_pooled_prompt_embeds_device = tensor.from_torch(
                    pooled_prompt_embeds1[i : i + 1] if cfg_factor == 2 else pooled_prompt_embeds1,
                    device=submesh_device,
                    on_host=traced,
                )
                tt_pooled_prompt_embeds1 = tensor.from_torch(
                    pooled_prompt_embeds1[i : i + 1] if cfg_factor == 2 else pooled_prompt_embeds1,
                    device=submesh_device,
                    on_host=True,
                )
                tt_pooled_prompt_embeds2 = tensor.from_torch(
                    pooled_prompt_embeds2[i : i + 1] if cfg_factor == 2 else pooled_prompt_embeds2,
                    device=submesh_device,
                    on_host=True,
                )

                tt_initial_latents = tensor.from_torch(
                    latents, device=submesh_device, on_host=traced, mesh_axes=[None, sp_axis, None]
                )

                if traced:
                    if self._traces is None:
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds_device = tt_prompt_embeds_device.to(submesh_device)
                        tt_pooled_prompt_embeds_device = tt_pooled_prompt_embeds_device.to(submesh_device)
                    else:
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._traces[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds_device, self._traces[i].prompt_input)
                        ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds_device, self._traces[i].pooled_input)

                        tt_initial_latents = self._traces[i].spatial_input
                        tt_prompt_embeds_device = self._traces[i].prompt_input
                        tt_pooled_prompt_embeds_device = self._traces[i].pooled_input

                tt_prompt_embeds_device_list.append(tt_prompt_embeds_device)
                tt_prompt_embeds1_list.append(tt_prompt_embeds1)
                tt_prompt_embeds2_list.append(tt_prompt_embeds2)
                tt_pooled_prompt_embeds_device_list.append(tt_pooled_prompt_embeds_device)
                tt_pooled_prompt_embeds1_list.append(tt_pooled_prompt_embeds1)
                tt_pooled_prompt_embeds2_list.append(tt_pooled_prompt_embeds2)
                tt_latents_step_list.append(tt_initial_latents)

            logger.info("denoising...")

            for i, t in enumerate(tqdm.tqdm(timesteps)):
                with timer.time_step("denoising_step") if timer else nullcontext():
                    sigma_difference = sigmas[i + 1] - sigmas[i]

                    tt_timestep_list = []
                    tt_sigma_difference_list = []
                    for submesh_nr, submesh_device in enumerate(self._submesh_devices):
                        tt_timestep = ttnn.full(
                            [1, 1],
                            fill_value=t,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.float32,
                            device=submesh_device if not traced else None,
                        )
                        tt_timestep_list.append(tt_timestep)

                        tt_sigma_difference = ttnn.full(
                            [1, 1],
                            fill_value=sigma_difference,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            device=submesh_device if not traced else None,
                        )
                        tt_sigma_difference_list.append(tt_sigma_difference)

                        if t >= 1000 * negative_strategy_switch_time:
                            ttnn.copy_host_to_device_tensor(
                                tt_prompt_embeds1_list[submesh_nr],
                                tt_prompt_embeds_device_list[submesh_nr],
                            )
                            ttnn.copy_host_to_device_tensor(
                                tt_pooled_prompt_embeds1_list[submesh_nr],
                                tt_pooled_prompt_embeds_device_list[submesh_nr],
                            )
                        else:
                            ttnn.copy_host_to_device_tensor(
                                tt_prompt_embeds2_list[submesh_nr],
                                tt_prompt_embeds_device_list[submesh_nr],
                            )
                            ttnn.copy_host_to_device_tensor(
                                tt_pooled_prompt_embeds2_list[submesh_nr],
                                tt_pooled_prompt_embeds_device_list[submesh_nr],
                            )

                    tt_latents_step_list = self._step(
                        timestep=tt_timestep_list,
                        latents=tt_latents_step_list,
                        cfg_enabled=cfg_enabled,
                        prompt_embeds=tt_prompt_embeds_device_list,
                        pooled_prompt_embeds=tt_pooled_prompt_embeds_device_list,
                        cfg_scale=cfg_scale,
                        sigma_difference=tt_sigma_difference_list,
                        traced=traced,
                    )

            logger.info("decoding image...")

            with timer.time_section("vae_decoding") if timer else nullcontext():
                # Sync because we don't pass a persistent buffer or a barrier semaphore.
                ttnn.synchronize_device(self.vae_device)

                tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
                    tt_latents_step_list[self.vae_submesh_idx],
                    dim=1,
                    mesh_axis=sp_axis,
                    use_hyperparams=True,
                )

                torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
                # Motif does not apply VAE shift. TODO: Check if this is done on purpose.
                torch_latents = torch_latents / self._latents_scaling  # + self._latents_shift

                torch_latents = self.transformers[0].unpatchify(
                    torch_latents,
                    height=self._height // self._vae_scale_factor,
                    width=self._width // self._vae_scale_factor,
                )

                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # HACK: reshape submesh device 0 from 2D to 1D
                    # If reshaping, vae device is same as encoder device
                    self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))

                tt_latents = tensor.from_torch(torch_latents, device=self.vae_device)
                tt_decoded_output = self._vae_decoder(tt_latents)
                decoded_output = ttnn.to_torch(ttnn.get_device_tensors(tt_decoded_output)[0]).permute(0, 3, 1, 2)

                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # HACK: reshape submesh device 0 from 1D to 2D
                    # If reshaping, vae device is same as encoder device
                    self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))

                image = self._image_processor.postprocess(decoded_output, output_type="pt")
                assert isinstance(image, torch.Tensor)

                output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        return output

    def _step_inner(
        self,
        *,
        cfg_enabled: bool,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled: ttnn.Tensor,
        timestep: ttnn.Tensor,
        submesh_index: int,
    ) -> ttnn.Tensor:
        if cfg_enabled and self._parallel_config.cfg_parallel.factor == 1:
            latent = ttnn.concat([latent, latent])

        return self.transformers[submesh_index].forward(
            spatial=latent,
            prompt=prompt,
            pooled=pooled,
            timestep=timestep,
        )

    def _step(
        self,
        *,
        cfg_enabled: bool,
        cfg_scale: float,
        latents: list[ttnn.Tensor],  # device tensor
        timestep: list[ttnn.Tensor],  # host tensor
        pooled_prompt_embeds: list[ttnn.Tensor],  # device tensor
        prompt_embeds: list[ttnn.Tensor],  # device tensor
        sigma_difference: list[ttnn.Tensor],  # device tensor
        traced: bool,
    ) -> list[ttnn.Tensor]:
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        if traced and self._traces is None:
            self._traces = []
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                timestep_device = timestep[submesh_id].to(submesh_device)
                sigma_difference_device = sigma_difference[submesh_id].to(submesh_device)

                pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    pooled=pooled_prompt_embeds[submesh_id],
                    timestep=timestep_device,
                    submesh_index=submesh_id,
                )

                for device in self._submesh_devices:
                    ttnn.synchronize_device(device)

                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    pooled=pooled_prompt_embeds[submesh_id],
                    timestep=timestep_device,
                    submesh_index=submesh_id,
                )
                ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)

                for device in self._submesh_devices:
                    ttnn.synchronize_device(device)

                self._traces.append(
                    PipelineTrace(
                        spatial_input=latents[submesh_id],
                        prompt_input=prompt_embeds[submesh_id],
                        pooled_input=pooled_prompt_embeds[submesh_id],
                        timestep_input=timestep_device,
                        latents_output=pred,
                        sigma_difference_input=sigma_difference_device,
                        tid=trace_id,
                    )
                )

        noise_pred_list = []
        if traced:
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._traces[submesh_id].timestep_input)
                ttnn.copy_host_to_device_tensor(
                    sigma_difference[submesh_id], self._traces[submesh_id].sigma_difference_input
                )
                ttnn.execute_trace(submesh_device, self._traces[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._traces[submesh_id].latents_output)

            # TODO: If we don't do this, we get noise when tracing is enabled. But why, since sigma
            # difference is only used outside of tracing region?
            sigma_difference_device = [trace.sigma_difference_input for trace in self._traces]
        else:
            for submesh_id in range(len(self._submesh_devices)):
                noise_pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    pooled=pooled_prompt_embeds[submesh_id],
                    timestep=timestep[submesh_id],
                    submesh_index=submesh_id,
                )
                noise_pred_list.append(noise_pred)

            sigma_difference_device = sigma_difference

        if cfg_enabled:
            if self._parallel_config.cfg_parallel.factor == 1:
                split_pos = noise_pred_list[0].shape[0] // 2
                uncond = noise_pred_list[0][0:split_pos]
                cond = noise_pred_list[0][split_pos:]
                noise_pred_list[0] = uncond + cfg_scale * (cond - uncond)
            else:
                # uncond and cond are replicated, so it is fine to get a single tensor from each
                uncond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[0])[0].cpu(blocking=True)).to(
                    torch.float32
                )
                cond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[1])[0].cpu(blocking=True)).to(
                    torch.float32
                )

                torch_noise_pred = uncond + cfg_scale * (cond - uncond)

                noise_pred_list[0] = tensor.from_torch(
                    torch_noise_pred, device=self._submesh_devices[0], mesh_axes=[None, sp_axis, None]
                )

                noise_pred_list[1] = tensor.from_torch(
                    torch_noise_pred, device=self._submesh_devices[1], mesh_axes=[None, sp_axis, None]
                )

        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            ttnn.synchronize_device(submesh_device)  # Helps with accurate time profiling.
            ttnn.multiply_(noise_pred_list[submesh_id], sigma_difference_device[submesh_id])
            ttnn.add_(latents[submesh_id], noise_pred_list[submesh_id])

        return latents

    def _encode_prompts(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        prompt_3: list[str],
        negative_prompt_1: list[str | None],
        negative_prompt_2: list[str | None],
        negative_prompt_3: list[str | None],
        num_images_per_prompt: int,
        cfg_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        timer = self.timing_collector

        no_negative_prompt = [x is None for x in negative_prompt_1]
        negative_prompt_1 = [x if x is not None else "" for x in negative_prompt_1]
        negative_prompt_2 = [x if x is not None else "" for x in negative_prompt_2]
        negative_prompt_3 = [x if x is not None else "" for x in negative_prompt_3]

        with timer.time_section("text_encoding") if timer else nullcontext():
            pos_prompt_embeds, pos_pooled_prompt_embeds = self._text_encoder.encode(
                prompt_1, prompt_2, prompt_3, num_images_per_prompt=num_images_per_prompt
            )

            neg_prompt_embeds, neg_pooled_prompt_embeds = self._text_encoder.encode(
                negative_prompt_1, negative_prompt_2, negative_prompt_3, num_images_per_prompt=num_images_per_prompt
            )

        if not cfg_enabled:
            return pos_prompt_embeds, pos_pooled_prompt_embeds, pos_prompt_embeds, pos_pooled_prompt_embeds

        zeroed_prompt_embeds = neg_prompt_embeds.clone()
        zeroed_pooled_prompt_embeds = neg_pooled_prompt_embeds.clone()

        for i, no_neg in enumerate(no_negative_prompt):
            if no_neg:
                zeroed_prompt_embeds[i] = 0
                zeroed_pooled_prompt_embeds[i] = 0

        prompt_embeds = torch.cat([neg_prompt_embeds, pos_prompt_embeds], dim=0)
        prompt_embeds_alt = torch.cat([zeroed_prompt_embeds, pos_prompt_embeds], dim=0)

        pooled_prompt_embeds = torch.cat([neg_pooled_prompt_embeds, pos_pooled_prompt_embeds], dim=0)
        pooled_prompt_embeds_alt = torch.cat([zeroed_pooled_prompt_embeds, pos_pooled_prompt_embeds], dim=0)

        return prompt_embeds, pooled_prompt_embeds, prompt_embeds_alt, pooled_prompt_embeds_alt


class TextEncoder:
    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: EncoderParallelConfig,
        enable_t5: bool,
        use_torch_clip_encoder: bool,
        use_torch_t5_encoder: bool,
    ) -> None:
        self._device = device
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config

        self._clip_l = CLIPTokenizerEncoderPair(
            "openai/clip-vit-large-patch14",
            skip_norm=False,
            true_clip_skip=0,
            zero_masking=True,
            sequence_length=None,
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            use_torch=use_torch_clip_encoder,
        )

        self._clip_g = CLIPTokenizerEncoderPair(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            skip_norm=False,
            true_clip_skip=0,
            zero_masking=True,
            sequence_length=None,
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            use_torch=use_torch_clip_encoder,
        )

        self._t5 = T5TokenizerEncoderPair(
            "google/flan-t5-xxl",
            zero_masking=True,
            sequence_length=256,
            empty_sequence_length=None,
            embedding_dim=4096,
            device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            use_torch=use_torch_t5_encoder,
            enabled=enable_t5,
            use_attention_mask=True,
        )

    def encode(
        self,
        prompts_1: Iterable[str],
        prompts_2: Iterable[str],
        prompts_3: Iterable[str],
        *,
        num_images_per_prompt: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_l, pooled_clip_l = self._clip_l.encode(prompts=prompts_1, num_images_per_prompt=num_images_per_prompt)
        clip_g, pooled_clip_g = self._clip_g.encode(prompts=prompts_2, num_images_per_prompt=num_images_per_prompt)
        t5 = self._t5.encode(prompts=prompts_3, num_images_per_prompt=num_images_per_prompt)

        clip = torch.cat([clip_l, clip_g], dim=-1)
        clip = torch.nn.functional.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))

        embeds = torch.cat([clip, t5], dim=-2)
        pooled_embeds = torch.cat([pooled_clip_l, pooled_clip_g], dim=-1)

        return embeds, pooled_embeds


def _schedule(*, step_count: int, linear_quadratic_emulating_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert step_count % 2 == 0

    s = step_count
    n = linear_quadratic_emulating_steps
    a = s // 2 / n - 1

    sigmas1 = torch.linspace(1, 0, n + 1)[: s // 2]
    sigmas2 = torch.linspace(0, 1, s // 2 + 1).pow(2) * a - a

    sigmas = torch.concat([sigmas1, sigmas2])
    timesteps = sigmas[:-1] * 1000

    return timesteps, sigmas

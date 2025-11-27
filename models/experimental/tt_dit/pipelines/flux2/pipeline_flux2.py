# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import diffusers
import numpy as np
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from loguru import logger

import ttnn

from ...models.transformers.transformer_flux2 import Flux2Transformer
from ...models.vae.vae_flux2 import Flux2VaeDecoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache, tensor
from ...utils.padding import PaddingConfig
from .prompt_encoder import PromptEncoder

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from PIL import Image


@dataclass
class PipelineTrace:
    tid: int
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    guidance_input: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    latents_output: ttnn.Tensor
    spatial_rope_cos: ttnn.Tensor
    spatial_rope_sin: ttnn.Tensor
    prompt_rope_cos: ttnn.Tensor
    prompt_rope_sin: ttnn.Tensor


class Flux2Pipeline:
    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        use_torch_prompt_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        parallel_config: DiTParallelConfig,
        encoder_parallel_config: EncoderParallelConfig | None = None,
        vae_parallel_config: VAEParallelConfig | None = None,
        topology: ttnn.Topology,
        num_links: int,
        height: int = 1024,
        width: int = 1024,
    ) -> None:
        self.timing_collector = None

        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self._height = height
        self._width = width

        # setup encoder and vae parallel configs.
        if encoder_parallel_config is None:
            encoder_parallel_config = EncoderParallelConfig(
                tensor_parallel=(
                    parallel_config.tensor_parallel
                    if parallel_config.tensor_parallel.mesh_axis == 4
                    else parallel_config.sequence_parallel
                )
            )

        if vae_parallel_config is None:
            vae_parallel_config = VAEParallelConfig(
                tensor_parallel=(
                    parallel_config.tensor_parallel
                    if parallel_config.tensor_parallel.mesh_axis == 4
                    else parallel_config.sequence_parallel
                )
            )

        self._encoder_parallel_config = encoder_parallel_config
        self._vae_parallel_config = vae_parallel_config

        # Create submeshes based on CFG parallel configuration
        submesh_shape = list(mesh_device.shape)
        submesh_shape[parallel_config.sequence_parallel.mesh_axis] = parallel_config.sequence_parallel.factor
        submesh_shape[parallel_config.tensor_parallel.mesh_axis] = parallel_config.tensor_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        # logger.info(f"Original mesh shape: {mesh_device.shape}")
        # logger.info(f"Creating submeshes with shape {submesh_shape}")
        self._submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))[
            0 : parallel_config.cfg_parallel.factor
        ]
        self._ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=topology)
            for submesh_device in self._submesh_devices
        ]

        self.encoder_device = self._submesh_devices[0] if not use_torch_prompt_encoder else None
        self.encoder_mesh_shape = ttnn.MeshShape(1, self._encoder_parallel_config.tensor_parallel.factor)
        self.vae_device = self._submesh_devices[0]
        self.encoder_submesh_idx = 0  # Use submesh 0 for encoder
        self.vae_submesh_idx = 0  # Use submesh 0 for VAE

        logger.info("loading models...")

        checkpoint_name = "black-forest-labs/FLUX.2-dev"

        torch_transformer = diffusers.Flux2Transformer2DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,  # bfloat16 is the native datatype of the model
        )
        torch_transformer.eval()

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

        self.transformers = []
        for i, submesh_device in enumerate(self._submesh_devices):
            tt_transformer = Flux2Transformer(
                in_channels=128,
                num_layers=8,
                num_single_layers=48,
                attention_head_dim=head_dim,
                num_attention_heads=num_heads,
                joint_attention_dim=15360,
                out_channels=128,
                device=submesh_device,
                ccl_manager=self._ccl_managers[i],
                parallel_config=parallel_config,
                padding_config=padding_config,
            )

            if not cache.initialize_from_cache(
                tt_model=tt_transformer,
                torch_state_dict=torch_transformer.state_dict(),
                model_name="flux2-dev",
                subfolder="transformer",
                parallel_config=self._parallel_config,
                mesh_shape=tuple(submesh_device.shape),
                dtype="bf16",
            ):
                logger.info("Loading transformer weights from PyTorch state dict")
                tt_transformer.load_torch_state_dict(torch_transformer.state_dict())

            self.transformers.append(tt_transformer)
            ttnn.synchronize_device(submesh_device)

        self._pos_embed = torch_transformer.pos_embed

        self._image_processor = VaeImageProcessor()

        with self.encoder_reshape(self.encoder_device):
            logger.info("creating TT-NN text encoder...")
            self._prompt_encoder = PromptEncoder(
                checkpoint_name=checkpoint_name,
                use_torch_encoder=use_torch_prompt_encoder,
                device=self.encoder_device,
                parallel_config=self._encoder_parallel_config,
                ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
            )

            if self.encoder_device is not None:
                ttnn.synchronize_device(self.encoder_device)

            if not use_torch_vae_decoder:
                logger.info("creating TT-NN VAE decoder...")
                self._vae_decoder = Flux2VaeDecoder(
                    out_channels=3,
                    block_out_channels=[128, 256, 512, 512],
                    layers_per_block=2,
                    z_channels=32,
                    device=self.vae_device,
                    parallel_config=self._vae_parallel_config,
                    ccl_manager=self._ccl_managers[self.vae_submesh_idx],
                )
                self._vae_decoder.load_torch_state_dict(self._torch_vae.state_dict())
            else:
                self._vae_decoder = None

            if self.encoder_device is not None:
                ttnn.synchronize_device(self.encoder_device)

        self._traces = None

    @contextmanager
    def encoder_reshape(self, device: ttnn.MeshDevice | None) -> Generator[None]:
        if device is None:
            yield
            return

        original_mesh_shape = ttnn.MeshShape(tuple(device.shape))
        assert (
            original_mesh_shape.mesh_size() == self.encoder_mesh_shape.mesh_size()
        ), f"Device cannot be reshaped device shape: {device.shape} encoder mesh shape: {self.encoder_mesh_shape}"
        if original_mesh_shape != self.encoder_mesh_shape:
            device.reshape(self.encoder_mesh_shape)
        yield
        if original_mesh_shape != device.shape:
            device.reshape(original_mesh_shape)

    @staticmethod
    def create_pipeline(
        *,
        mesh_device: ttnn.MeshDevice,
        dit_cfg: tuple[int, int],
        dit_sp: tuple[int, int],
        dit_tp: tuple[int, int],
        encoder_tp: tuple[int, int],
        vae_tp: tuple[int, int],
        use_torch_prompt_encoder: bool = False,
        use_torch_vae_decoder: bool = False,
        num_links: int,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        width: int = 1024,
        height: int = 1024,
    ) -> Flux2Pipeline:
        cfg_factor, cfg_axis = dit_cfg
        sp_factor, sp_axis = dit_sp
        tp_factor, tp_axis = dit_tp
        encoder_tp_factor, encoder_tp_axis = encoder_tp
        vae_tp_factor, vae_tp_axis = vae_tp

        dit_parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=encoder_tp_factor, mesh_axis=encoder_tp_axis)
        )
        vae_parallel_config = VAEParallelConfig(
            tensor_parallel=ParallelFactor(factor=vae_tp_factor, mesh_axis=vae_tp_axis)
        )

        # logger.info(f"Mesh device shape: {mesh_device.shape}")
        logger.info(f"Parallel config: {dit_parallel_config}")
        logger.info(f"Encoder parallel config: {encoder_parallel_config}")
        logger.info(f"VAE parallel config: {vae_parallel_config}")

        return Flux2Pipeline(
            mesh_device=mesh_device,
            use_torch_prompt_encoder=use_torch_prompt_encoder,
            use_torch_vae_decoder=use_torch_vae_decoder,
            parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            topology=topology,
            num_links=num_links,
            width=width,
            height=height,
        )

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 4.0,
        prompts: Sequence[str],
        num_inference_steps: int,
        prompt_upsample_temperature: float | None = None,  # prompt upsampling is currently very slow
        seed: int | None = None,
        traced: bool = False,
    ) -> list[Image.Image]:
        timer = self.timing_collector.reset() if self.timing_collector else None
        prompt_count = len(prompts)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis
        cfg_factor = self._parallel_config.cfg_parallel.factor

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        latents_height = self._height // self._vae_scale_factor
        latents_width = self._width // self._vae_scale_factor
        transformer_batch_size = prompt_count * num_images_per_prompt
        spatial_sequence_length = (latents_height // self._patch_size) * (latents_width // self._patch_size)

        with timer.time_section("total") if timer else nullcontext():
            logger.info("encoding prompts...")

            if prompt_upsample_temperature is not None:
                with timer.time_section("prompt_upsampling") if timer else nullcontext():
                    prompts = self._prompt_encoder.upsample(
                        prompts,
                        max_length=224,  # TODO: this should be higher
                        temperature=prompt_upsample_temperature,
                    )

            with timer.time_section("total_encoding") if timer else nullcontext():
                with self.encoder_reshape(self.encoder_device):
                    prompt_embeds, _mask = self._prompt_encoder.encode(
                        prompts, num_images_per_prompt=num_images_per_prompt, sequence_length=512
                    )
            _, prompt_sequence_length, _ = prompt_embeds.shape

            logger.info("preparing timesteps...")
            timesteps, sigmas = _schedule(
                self._scheduler,
                step_count=num_inference_steps,
                spatial_sequence_length=spatial_sequence_length,
            )

            guidance = torch.full([transformer_batch_size], fill_value=guidance_scale)

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)

            shape = [transformer_batch_size, self._num_channels_latents, latents_height, latents_width]
            latents = self._patchify(torch.randn(shape).permute(0, 2, 3, 1))

            text_ids = _prepare_ids(text_sequence_length=prompt_sequence_length)
            image_ids = _prepare_ids(height=self._height // 16, width=self._width // 16)
            prompt_rope_cos, prompt_rope_sin = self._pos_embed.forward(text_ids)
            spatial_rope_cos, spatial_rope_sin = self._pos_embed.forward(image_ids)

            tt_prompt_embeds_list = []
            tt_latents_step_list = []
            tt_guidance_list = []
            tt_spatial_rope_cos_list = []
            tt_spatial_rope_sin_list = []
            tt_prompt_rope_cos_list = []
            tt_prompt_rope_sin_list = []
            for i, submesh_device in enumerate(self._submesh_devices):
                tt_prompt_embeds = tensor.from_torch(
                    prompt_embeds[i : i + 1] if cfg_factor == 2 else prompt_embeds,
                    device=submesh_device,
                    on_host=traced,
                )

                tt_initial_latents = tensor.from_torch(
                    latents, device=submesh_device, on_host=traced, mesh_axes=[None, sp_axis, None]
                )

                tt_guidance = tensor.from_torch(guidance.unsqueeze(1), device=submesh_device, on_host=traced)

                tt_spatial_rope_cos = tensor.from_torch(
                    spatial_rope_cos, device=submesh_device, on_host=traced, mesh_axes=[sp_axis, None]
                )
                tt_spatial_rope_sin = tensor.from_torch(
                    spatial_rope_sin, device=submesh_device, on_host=traced, mesh_axes=[sp_axis, None]
                )
                tt_prompt_rope_cos = tensor.from_torch(prompt_rope_cos, device=submesh_device, on_host=traced)
                tt_prompt_rope_sin = tensor.from_torch(prompt_rope_sin, device=submesh_device, on_host=traced)

                if traced:
                    if self._traces is None:
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds = tt_prompt_embeds.to(submesh_device)
                        tt_guidance = tt_guidance.to(submesh_device)
                        tt_spatial_rope_cos = tt_spatial_rope_cos.to(submesh_device)
                        tt_spatial_rope_sin = tt_spatial_rope_sin.to(submesh_device)
                        tt_prompt_rope_cos = tt_prompt_rope_cos.to(submesh_device)
                        tt_prompt_rope_sin = tt_prompt_rope_sin.to(submesh_device)
                    else:
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._traces[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._traces[i].prompt_input)
                        ttnn.copy_host_to_device_tensor(tt_guidance, self._traces[i].guidance_input)
                        ttnn.copy_host_to_device_tensor(tt_spatial_rope_cos, self._traces[i].spatial_rope_cos)
                        ttnn.copy_host_to_device_tensor(tt_spatial_rope_sin, self._traces[i].spatial_rope_sin)
                        ttnn.copy_host_to_device_tensor(tt_prompt_rope_cos, self._traces[i].prompt_rope_cos)
                        ttnn.copy_host_to_device_tensor(tt_prompt_rope_sin, self._traces[i].prompt_rope_sin)

                        tt_initial_latents = self._traces[i].spatial_input
                        tt_prompt_embeds = self._traces[i].prompt_input
                        tt_guidance = self._traces[i].guidance_input
                        tt_spatial_rope_cos = self._traces[i].spatial_rope_cos
                        tt_spatial_rope_sin = self._traces[i].spatial_rope_sin
                        tt_prompt_rope_cos = self._traces[i].prompt_rope_cos
                        tt_prompt_rope_sin = self._traces[i].prompt_rope_sin

                tt_prompt_embeds_list.append(tt_prompt_embeds)
                tt_latents_step_list.append(tt_initial_latents)
                tt_guidance_list.append(tt_guidance)
                tt_spatial_rope_cos_list.append(tt_spatial_rope_cos)
                tt_spatial_rope_sin_list.append(tt_spatial_rope_sin)
                tt_prompt_rope_cos_list.append(tt_prompt_rope_cos)
                tt_prompt_rope_sin_list.append(tt_prompt_rope_sin)

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

                    tt_latents_step_list = self._step(
                        timestep=tt_timestep_list,
                        guidance=tt_guidance_list,
                        latents=tt_latents_step_list,
                        cfg_enabled=False,
                        prompt_embeds=tt_prompt_embeds_list,
                        cfg_scale=1.0,
                        sigma_difference=tt_sigma_difference_list,
                        spatial_rope_cos=tt_spatial_rope_cos_list,
                        spatial_rope_sin=tt_spatial_rope_sin_list,
                        prompt_rope_cos=tt_prompt_rope_cos_list,
                        prompt_rope_sin=tt_prompt_rope_sin_list,
                        spatial_sequence_length=spatial_sequence_length,
                        prompt_sequence_length=prompt_sequence_length,
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

                if self._vae_decoder is None:
                    vae = self._torch_vae

                    torch_latents = torch_latents.reshape(
                        [
                            transformer_batch_size,
                            latents_height // self._patch_size,
                            latents_width // self._patch_size,
                            self._num_channels_latents * self._patch_size**2,
                        ]
                    )
                    torch_latents = torch_latents.permute(0, 3, 1, 2).to(torch.float32)

                    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
                    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
                    torch_latents = torch_latents * latents_bn_std + latents_bn_mean

                    batch_size, num_channels_latents, height, width = torch_latents.shape
                    torch_latents = torch_latents.reshape(
                        batch_size, num_channels_latents // (2 * 2), 2, 2, height, width
                    )
                    torch_latents = torch_latents.permute(0, 1, 4, 2, 5, 3)
                    torch_latents = torch_latents.reshape(
                        batch_size, num_channels_latents // (2 * 2), height * 2, width * 2
                    )

                    with torch.no_grad():
                        decoded_output = vae.decode(torch_latents).sample
                else:
                    with self.encoder_reshape(self.encoder_device):
                        tt_latents = tensor.from_torch(torch_latents, device=self.vae_device)
                        tt_latents = self._vae_decoder.preprocess_and_unpatchify(
                            tt_latents,
                            height=self._height // self._vae_scale_factor,
                            width=self._width // self._vae_scale_factor,
                        )
                        tt_decoded_output = self._vae_decoder.forward(tt_latents)
                        decoded_output = ttnn.to_torch(ttnn.get_device_tensors(tt_decoded_output)[0]).permute(
                            0, 3, 1, 2
                        )

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
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor,
        submesh_index: int,
        spatial_rope_cos: ttnn.Tensor,
        spatial_rope_sin: ttnn.Tensor,
        prompt_rope_cos: ttnn.Tensor,
        prompt_rope_sin: ttnn.Tensor,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> ttnn.Tensor:
        if cfg_enabled and self._parallel_config.cfg_parallel.factor == 1:
            latent = ttnn.concat([latent, latent])

        return self.transformers[submesh_index].forward(
            spatial=latent,
            prompt=prompt,
            timestep=timestep,
            guidance=guidance,
            spatial_rope=(spatial_rope_cos, spatial_rope_sin),
            prompt_rope=(prompt_rope_cos, prompt_rope_sin),
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )

    def _step(
        self,
        *,
        cfg_enabled: bool,
        cfg_scale: float,
        latents: list[ttnn.Tensor],  # device tensor
        timestep: list[ttnn.Tensor],  # host tensor
        guidance: list[ttnn.Tensor],  # device tensor
        prompt_embeds: list[ttnn.Tensor],  # device tensor
        sigma_difference: list[ttnn.Tensor],  # device tensor
        spatial_rope_cos: list[ttnn.Tensor],
        spatial_rope_sin: list[ttnn.Tensor],
        prompt_rope_cos: list[ttnn.Tensor],
        prompt_rope_sin: list[ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
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
                    timestep=timestep_device,
                    guidance=guidance[submesh_id],
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )

                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    timestep=timestep_device,
                    guidance=guidance[submesh_id],
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
                    submesh_index=submesh_id,
                )
                ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)

                for device in self._submesh_devices:
                    ttnn.synchronize_device(device)

                self._traces.append(
                    PipelineTrace(
                        spatial_input=latents[submesh_id],
                        prompt_input=prompt_embeds[submesh_id],
                        timestep_input=timestep_device,
                        guidance_input=guidance[submesh_id],
                        spatial_rope_cos=spatial_rope_cos[submesh_id],
                        spatial_rope_sin=spatial_rope_sin[submesh_id],
                        prompt_rope_cos=prompt_rope_cos[submesh_id],
                        prompt_rope_sin=prompt_rope_sin[submesh_id],
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
                    timestep=timestep[submesh_id],
                    guidance=guidance[submesh_id],
                    spatial_rope_cos=spatial_rope_cos[submesh_id],
                    spatial_rope_sin=spatial_rope_sin[submesh_id],
                    prompt_rope_cos=prompt_rope_cos[submesh_id],
                    prompt_rope_sin=prompt_rope_sin[submesh_id],
                    spatial_sequence_length=spatial_sequence_length,
                    prompt_sequence_length=prompt_sequence_length,
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

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        # N, H, W, C -> N, (H / P) * (W / P), C * P * P
        batch_size, height, width, channels = latents.shape
        p = self._patch_size

        if height % p != 0 or width % p != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({p})"
            raise ValueError(msg)

        latents = latents.reshape([batch_size, height // p, p, width // p, p, channels])
        return latents.permute(0, 1, 3, 5, 2, 4).flatten(3, 5).flatten(1, 2)


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

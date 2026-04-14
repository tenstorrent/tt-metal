# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import tqdm
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkProfiler

from ...encoders.mistral.encoder_pair import MistralTokenizerEncoderPair
from ...models.transformers.transformer_flux2 import Flux2Transformer
from ...models.vae.vae_sd35 import VAEDecoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache
from ...utils.padding import PaddingConfig


@dataclass
class PipelineTrace:
    tid: int
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    guidance_input: ttnn.Tensor
    spatial_rope_cos: ttnn.Tensor
    spatial_rope_sin: ttnn.Tensor
    prompt_rope_cos: ttnn.Tensor
    prompt_rope_sin: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    latents_output: ttnn.Tensor


class Flux2Pipeline:
    MISTRAL_SEQUENCE_LENGTH = 512

    def __init__(
        self,
        *,
        checkpoint_name: str,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
        encoder_parallel_config: EncoderParallelConfig = None,
        vae_parallel_config: VAEParallelConfig = None,
        topology: ttnn.Topology,
        num_links: int,
        is_fsdp: bool = False,
        use_torch_text_encoder: bool = False,
        dummy_weights: bool = False,
    ) -> None:
        self._mesh_device = mesh_device
        self._parallel_config = parallel_config
        self._is_fsdp = is_fsdp

        self._encoder_parallel_config = encoder_parallel_config
        if self._encoder_parallel_config is None:
            self._encoder_parallel_config = EncoderParallelConfig(
                tensor_parallel=(
                    parallel_config.tensor_parallel
                    if parallel_config.tensor_parallel.mesh_axis == 4
                    else parallel_config.sequence_parallel
                )
            )

        self._vae_parallel_config = vae_parallel_config
        if self._vae_parallel_config is None:
            self._vae_parallel_config = VAEParallelConfig(
                tensor_parallel=(
                    parallel_config.tensor_parallel
                    if parallel_config.tensor_parallel.mesh_axis == 4
                    else parallel_config.sequence_parallel
                )
            )

        submesh_shape = list(mesh_device.shape)
        submesh_shape[parallel_config.sequence_parallel.mesh_axis] = parallel_config.sequence_parallel.factor
        submesh_shape[parallel_config.tensor_parallel.mesh_axis] = parallel_config.tensor_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Original mesh shape: {mesh_device.shape}")
        logger.info(f"Creating submeshes with shape {submesh_shape}")
        self._submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))[0:1]
        self._ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=topology)
            for submesh_device in self._submesh_devices
        ]

        self.vae_device = self._submesh_devices[0]
        self.vae_submesh_idx = 0
        self.encoder_device = self._submesh_devices[0]
        self.encoder_submesh_idx = 0

        logger.info("loading models...")

        torch_transformer = Flux2Transformer2DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_transformer.eval()

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")
        self._torch_vae = AutoencoderKLFlux2.from_pretrained(checkpoint_name, subfolder="vae")

        logger.info("creating TT-NN Mistral3 text encoder...")
        self._mistral_encoder = MistralTokenizerEncoderPair(
            checkpoint_name,
            device=self.encoder_device,
            ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
            parallel_config=self._encoder_parallel_config,
            use_torch=use_torch_text_encoder,
            is_fsdp=is_fsdp,
        )
        self._mistral_encoder.deallocate_encoder_weights()

        logger.info("creating TT-NN transformer...")

        if torch_transformer.config.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                torch_transformer.config.num_attention_heads,
                torch_transformer.config.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self.transformers = []
        for i, submesh_device in enumerate(self._submesh_devices):
            tt_transformer = Flux2Transformer(
                patch_size=torch_transformer.config.patch_size,
                in_channels=torch_transformer.config.in_channels,
                num_layers=torch_transformer.config.num_layers,
                num_single_layers=torch_transformer.config.num_single_layers,
                attention_head_dim=torch_transformer.config.attention_head_dim,
                num_attention_heads=torch_transformer.config.num_attention_heads,
                joint_attention_dim=torch_transformer.config.joint_attention_dim,
                timestep_guidance_channels=torch_transformer.config.timestep_guidance_channels,
                out_channels=torch_transformer.out_channels,
                mlp_ratio=torch_transformer.config.mlp_ratio,
                guidance_embeds=torch_transformer.config.guidance_embeds,
                mesh_device=submesh_device,
                ccl_manager=self._ccl_managers[i],
                parallel_config=parallel_config,
                padding_config=padding_config,
                is_fsdp=is_fsdp,
                weights_dtype=ttnn.bfloat8_b,
                use_fused_ag_matmul=False,
            )

            if dummy_weights:
                tt_transformer.load_torch_state_dict(
                    _dummy_state_dict(torch_transformer),
                    strict=False,
                )
            else:
                model_name = os.path.basename(checkpoint_name)
                cache.load_model(
                    tt_transformer,
                    get_torch_state_dict=torch_transformer.state_dict,
                    model_name=model_name,
                    subfolder="transformer",
                    parallel_config=parallel_config,
                    mesh_shape=tuple(submesh_device.shape),
                    is_fsdp=is_fsdp,
                    dtype="bf8",
                )

            self.transformers.append(tt_transformer)
            ttnn.synchronize_device(submesh_device)

        self._axes_dim = list(torch_transformer.config.axes_dims_rope)
        self._tt_rope_freqs = []
        for dim in self._axes_dim:
            if dim == 0:
                self._tt_rope_freqs.append(None)
                continue
            freqs = (
                (1.0 / (torch_transformer.config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)))
                .float()
                .repeat_interleave(2)
                .unsqueeze(0)
            )
            self._tt_rope_freqs.append(
                ttnn.from_torch(
                    freqs,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.float32,
                    device=self._submesh_devices[0],
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self._submesh_devices[0]),
                )
            )

        self._in_channels = torch_transformer.config.in_channels
        self._num_channels_latents = torch_transformer.config.in_channels // 4
        self._joint_attention_dim = torch_transformer.config.joint_attention_dim
        self._patch_size = torch_transformer.config.patch_size
        self._with_guidance_embeds = torch_transformer.config.guidance_embeds

        self._block_out_channels = self._torch_vae.config.block_out_channels
        self._vae_scale_factor = 2 ** (len(self._block_out_channels) - 1)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor * 2)

        logger.info("creating TT-NN VAE decoder...")
        torch_vae_decoder = self._torch_vae.decoder
        self._vae_decoder = VAEDecoder(
            block_out_channels=[block.resnets[0].conv2.out_channels for block in torch_vae_decoder.up_blocks][::-1],
            in_channels=torch_vae_decoder.conv_in.in_channels,
            out_channels=torch_vae_decoder.conv_out.out_channels,
            layers_per_block=torch_vae_decoder.layers_per_block,
            norm_num_groups=torch_vae_decoder.mid_block.resnets[0].norm1.num_groups,
            mesh_device=self.vae_device,
            parallel_config=self._vae_parallel_config,
            ccl_manager=self._ccl_managers[self.vae_submesh_idx],
        )
        model_name = os.path.basename(checkpoint_name)
        cache.load_model(
            self._vae_decoder,
            get_torch_state_dict=torch_vae_decoder.state_dict,
            model_name=model_name,
            subfolder="vae_decoder",
            parallel_config=self._vae_parallel_config,
            mesh_shape=tuple(self.vae_device.shape),
        )

        self._traces = None

        bn_mean = self._torch_vae.bn.running_mean.view(1, 1, -1).to(torch.bfloat16)
        bn_std = torch.sqrt(self._torch_vae.bn.running_var.view(1, 1, -1) + self._torch_vae.config.batch_norm_eps).to(
            torch.bfloat16
        )
        self._tt_bn_mean = ttnn.from_torch(
            bn_mean,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.vae_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.vae_device),
        )
        self._tt_bn_std = ttnn.from_torch(
            bn_std,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self.vae_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.vae_device),
        )

        logger.info("warming up for tracing...")
        self.run_single_prompt(prompt="", num_inference_steps=1, seed=0, traced=False)
        self.synchronize_devices()

    @staticmethod
    def create_pipeline(
        checkpoint_name,
        mesh_device,
        dit_sp=None,
        dit_tp=None,
        encoder_tp=None,
        vae_tp=None,
        num_links=None,
        topology=ttnn.Topology.Linear,
        is_fsdp=None,
        use_torch_text_encoder=True,
        dummy_weights=False,
    ):
        wh_config = {
            (1, 2): {"sp": (1, 0), "tp": (2, 1), "vae_tp": (2, 1), "num_links": 1, "is_fsdp": True},
            (2, 4): {"sp": (2, 0), "tp": (4, 1), "vae_tp": (4, 1), "num_links": 1, "is_fsdp": True},
        }
        bh_config = {
            (2, 2): {"sp": (2, 0), "tp": (2, 1), "vae_tp": (2, 1), "num_links": 2, "is_fsdp": True},
            (2, 4): {"sp": (2, 0), "tp": (4, 1), "vae_tp": (4, 1), "num_links": 2, "is_fsdp": True},
        }

        default_config = bh_config if is_blackhole() else wh_config
        sp_factor, sp_axis = dit_sp or default_config[tuple(mesh_device.shape)]["sp"]
        tp_factor, tp_axis = dit_tp or default_config[tuple(mesh_device.shape)]["tp"]
        vae_tp_factor, vae_tp_axis = vae_tp or default_config[tuple(mesh_device.shape)]["vae_tp"]
        encoder_tp_factor, encoder_tp_axis = encoder_tp or (tp_factor, tp_axis)
        num_links = num_links or default_config[tuple(mesh_device.shape)]["num_links"]
        is_fsdp = is_fsdp if is_fsdp is not None else default_config[tuple(mesh_device.shape)]["is_fsdp"]

        dit_parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        )
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=encoder_tp_factor, mesh_axis=encoder_tp_axis)
        )
        vae_parallel_config = VAEParallelConfig(
            tensor_parallel=ParallelFactor(factor=vae_tp_factor, mesh_axis=vae_tp_axis)
        )

        pipeline = Flux2Pipeline(
            checkpoint_name=checkpoint_name,
            mesh_device=mesh_device,
            parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            topology=topology,
            num_links=num_links,
            is_fsdp=is_fsdp,
            use_torch_text_encoder=use_torch_text_encoder,
            dummy_weights=dummy_weights,
        )

        return pipeline

    def run_single_prompt(
        self,
        *,
        width: int = 1024,
        height: int = 1024,
        prompt: str,
        num_inference_steps: int,
        seed: int,
        guidance_scale: float = 4.0,
        traced: bool = True,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ):
        return self(
            width=width,
            height=height,
            prompt=[prompt],
            num_inference_steps=num_inference_steps,
            seed=seed,
            guidance_scale=guidance_scale,
            traced=traced,
            profiler=profiler,
            profiler_iteration=profiler_iteration,
        )

    def __call__(
        self,
        *,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 4.0,
        prompt: list[str],
        num_inference_steps: int,
        seed: int | None = None,
        traced: bool = False,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ):
        prompt_count = len(prompt)
        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        assert (
            height % (self._vae_scale_factor * self._patch_size) == 0
        ), f"height {height} must be divisible by {self._vae_scale_factor * self._patch_size}"
        assert (
            width % (self._vae_scale_factor * self._patch_size) == 0
        ), f"width {width} must be divisible by {self._vae_scale_factor * self._patch_size}"
        assert prompt_count == 1, "generating multiple images is not supported"

        with profiler("total", profiler_iteration) if profiler else nullcontext():
            latents_height_full = 2 * (int(height) // (self._vae_scale_factor * 2))
            latents_width_full = 2 * (int(width) // (self._vae_scale_factor * 2))
            latents_height = latents_height_full // 2
            latents_width = latents_width_full // 2
            spatial_sequence_length = latents_height * latents_width

            logger.info("encoding prompts...")

            with profiler("encoder", profiler_iteration) if profiler else nullcontext():
                prompt_embeds = self._encode_prompt(prompt)
                _, prompt_sequence_length, _ = prompt_embeds.shape

            logger.info("preparing timesteps...")

            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = spatial_sequence_length
            mu = _compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
            self._scheduler.set_timesteps(sigmas=sigmas, mu=mu)

            guidance = (
                torch.full([prompt_count], fill_value=guidance_scale, dtype=torch.float32)
                if self._with_guidance_embeds
                else None
            )

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)

            noise_shape = (prompt_count, self._num_channels_latents * 4, latents_height, latents_width)
            latents = torch.randn(noise_shape, dtype=torch.bfloat16)
            latents = _pack_latents(latents)

            latent_ids = _prepare_latent_ids(prompt_count, latents_height, latents_width)
            text_ids = _prepare_text_ids(prompt_embeds)

            tt_prompt_embeds_list = []
            tt_latents_step_list = []
            tt_guidance_list = []
            tt_spatial_rope_cos_list = []
            tt_spatial_rope_sin_list = []
            tt_prompt_rope_cos_list = []
            tt_prompt_rope_sin_list = []

            for i, submesh_device in enumerate(self._submesh_devices):
                tt_prompt_embeds = ttnn.from_torch(
                    prompt_embeds,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=(None, None),
                    ),
                )

                shard_latents_dims = [None, None]
                shard_latents_dims[sp_axis] = 1
                tt_initial_latents = ttnn.from_torch(
                    latents,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=tuple(shard_latents_dims),
                    ),
                )

                tt_guidance = (
                    ttnn.from_torch(
                        guidance.unsqueeze(-1),
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16,
                        device=submesh_device if not traced else None,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
                    )
                    if guidance is not None
                    else None
                )

                shard_rope_dims = [None, None]
                shard_rope_dims[sp_axis] = 0
                rope_mesh_mapper = ttnn.ShardTensor2dMesh(
                    submesh_device,
                    tuple(submesh_device.shape),
                    dims=tuple(shard_rope_dims),
                )

                tt_spatial_rope_cos, tt_spatial_rope_sin = self._compute_rope_on_device(
                    latent_ids[0],
                    submesh_device,
                    rope_mesh_mapper,
                )
                tt_prompt_rope_cos, tt_prompt_rope_sin = self._compute_rope_on_device(
                    text_ids[0],
                    submesh_device,
                    ttnn.ReplicateTensorToMesh(submesh_device),
                )

                if traced:
                    if self._traces is None:
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds = tt_prompt_embeds.to(submesh_device)
                        if tt_guidance is not None:
                            tt_guidance = tt_guidance.to(submesh_device)
                    else:
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._traces[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._traces[i].prompt_input)
                        ttnn.copy(tt_spatial_rope_cos, self._traces[i].spatial_rope_cos)
                        ttnn.copy(tt_spatial_rope_sin, self._traces[i].spatial_rope_sin)
                        ttnn.copy(tt_prompt_rope_cos, self._traces[i].prompt_rope_cos)
                        ttnn.copy(tt_prompt_rope_sin, self._traces[i].prompt_rope_sin)

                        tt_initial_latents = self._traces[i].spatial_input
                        tt_prompt_embeds = self._traces[i].prompt_input
                        tt_spatial_rope_cos = self._traces[i].spatial_rope_cos
                        tt_spatial_rope_sin = self._traces[i].spatial_rope_sin
                        tt_prompt_rope_cos = self._traces[i].prompt_rope_cos
                        tt_prompt_rope_sin = self._traces[i].prompt_rope_sin

                        if tt_guidance is not None:
                            ttnn.copy_host_to_device_tensor(tt_guidance, self._traces[i].guidance_input)
                            tt_guidance = self._traces[i].guidance_input

                tt_prompt_embeds_list.append(tt_prompt_embeds)
                tt_latents_step_list.append(tt_initial_latents)
                tt_guidance_list.append(tt_guidance)
                tt_spatial_rope_cos_list.append(tt_spatial_rope_cos)
                tt_spatial_rope_sin_list.append(tt_spatial_rope_sin)
                tt_prompt_rope_cos_list.append(tt_prompt_rope_cos)
                tt_prompt_rope_sin_list.append(tt_prompt_rope_sin)

            logger.info("denoising...")

            with profiler("denoising", profiler_iteration) if profiler else nullcontext():
                for step_i, t in enumerate(tqdm.tqdm(self._scheduler.timesteps)):
                    with profiler(f"denoising_step_{step_i}", profiler_iteration) if profiler else nullcontext():
                        sigma_difference = self._scheduler.sigmas[step_i + 1] - self._scheduler.sigmas[step_i]

                        tt_timestep_list = []
                        tt_sigma_difference_list = []
                        for submesh_device in self._submesh_devices:
                            tt_timestep = ttnn.full(
                                [1, 1],
                                fill_value=t,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.float32,
                                device=submesh_device if not traced else None,
                            )
                            tt_timestep_list.append(tt_timestep)

                            tt_sigma_difference = ttnn.full(
                                tt_initial_latents.shape,
                                fill_value=sigma_difference,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16,
                                device=submesh_device if not traced else None,
                            )
                            tt_sigma_difference_list.append(tt_sigma_difference)

                        tt_latents_step_list = self._step(
                            latents=tt_latents_step_list,
                            timestep=tt_timestep_list,
                            prompt_embeds=tt_prompt_embeds_list,
                            sigma_difference=tt_sigma_difference_list,
                            guidance=tt_guidance_list,
                            spatial_rope_cos=tt_spatial_rope_cos_list,
                            spatial_rope_sin=tt_spatial_rope_sin_list,
                            prompt_rope_cos=tt_prompt_rope_cos_list,
                            prompt_rope_sin=tt_prompt_rope_sin_list,
                            spatial_sequence_length=spatial_sequence_length,
                            prompt_sequence_length=prompt_sequence_length,
                            traced=traced,
                        )

            logger.info("decoding image...")

            with profiler("vae", profiler_iteration) if profiler else nullcontext():
                ttnn.synchronize_device(self.vae_device)

                tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
                    tt_latents_step_list[self.vae_submesh_idx],
                    dim=1,
                    mesh_axis=sp_axis,
                    use_hyperparams=True,
                )

                tt_latents = ttnn.multiply(tt_latents, self._tt_bn_std)
                tt_latents = ttnn.add(tt_latents, self._tt_bn_mean)

                tt_latents = ttnn.reshape(
                    tt_latents,
                    [
                        1,
                        latents_height,
                        latents_width,
                        self._num_channels_latents,
                        2,
                        2,
                    ],
                )
                tt_latents = ttnn.permute(tt_latents, (0, 1, 4, 2, 5, 3))
                tt_latents = ttnn.reshape(
                    tt_latents,
                    [
                        1,
                        latents_height * 2,
                        latents_width * 2,
                        self._num_channels_latents,
                    ],
                )
                tt_latents = ttnn.to_layout(tt_latents, ttnn.TILE_LAYOUT)

                tt_decoded_output = self._vae_decoder(tt_latents)
                ttnn.synchronize_device(self.vae_device)

                decoded_output = ttnn.to_torch(ttnn.get_device_tensors(tt_decoded_output)[0]).permute(0, 3, 1, 2)

                image = self._image_processor.postprocess(decoded_output, output_type="pt")
                assert isinstance(image, torch.Tensor)

                output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        return output

    def synchronize_devices(self):
        for device in self._submesh_devices:
            ttnn.synchronize_device(device)

    def _compute_rope_on_device(self, positions, submesh_device, mesh_mapper):
        cos_parts = []
        sin_parts = []
        for axis_i, dim in enumerate(self._axes_dim):
            if dim == 0 or self._tt_rope_freqs[axis_i] is None:
                continue
            pos_col = positions[:, axis_i : axis_i + 1].float()
            tt_pos = ttnn.from_torch(
                pos_col,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.float32,
                device=submesh_device,
                mesh_mapper=mesh_mapper,
            )
            angles = ttnn.multiply(tt_pos, self._tt_rope_freqs[axis_i])
            ttnn.deallocate(tt_pos)
            cos_parts.append(ttnn.cos(angles))
            sin_parts.append(ttnn.sin(angles))
            ttnn.deallocate(angles)
        freqs_cos = ttnn.concat(cos_parts, dim=-1)
        freqs_sin = ttnn.concat(sin_parts, dim=-1)
        for t in cos_parts + sin_parts:
            ttnn.deallocate(t)
        return ttnn.typecast(freqs_cos, ttnn.bfloat16), ttnn.typecast(freqs_sin, ttnn.bfloat16)

    def _encode_prompt(self, prompts: list[str]) -> torch.Tensor:
        self._mistral_encoder.reload_encoder_weights()
        result = self._mistral_encoder.encode(prompts)
        self._mistral_encoder.deallocate_encoder_weights()
        return result

    def _step_inner(
        self,
        *,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor | None,
        spatial_rope_cos: ttnn.Tensor,
        spatial_rope_sin: ttnn.Tensor,
        prompt_rope_cos: ttnn.Tensor,
        prompt_rope_sin: ttnn.Tensor,
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        submesh_index: int,
    ) -> ttnn.Tensor:
        noise_pred = self.transformers[submesh_index].forward(
            spatial=latent,
            prompt=prompt,
            timestep=timestep,
            guidance=guidance,
            spatial_rope=(spatial_rope_cos, spatial_rope_sin),
            prompt_rope=(prompt_rope_cos, prompt_rope_sin),
            spatial_sequence_length=spatial_sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )
        return noise_pred

    def _step(
        self,
        *,
        latents: list[ttnn.Tensor],
        timestep: list[ttnn.Tensor],
        prompt_embeds: list[ttnn.Tensor],
        sigma_difference: list[ttnn.Tensor],
        guidance: list[ttnn.Tensor | None],
        spatial_rope_cos: list[ttnn.Tensor],
        spatial_rope_sin: list[ttnn.Tensor],
        prompt_rope_cos: list[ttnn.Tensor],
        prompt_rope_sin: list[ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        traced: bool,
    ) -> list[ttnn.Tensor]:
        if traced and self._traces is None:
            self._traces = []
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                timestep_device = timestep[submesh_id].to(submesh_device)
                sigma_difference_device = sigma_difference[submesh_id].to(submesh_device)

                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = self._step_inner(
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
                        latents_output=pred,
                        spatial_rope_cos=spatial_rope_cos[submesh_id],
                        spatial_rope_sin=spatial_rope_sin[submesh_id],
                        prompt_rope_cos=prompt_rope_cos[submesh_id],
                        prompt_rope_sin=prompt_rope_sin[submesh_id],
                        sigma_difference_input=sigma_difference_device,
                        tid=trace_id,
                    )
                )

        noise_pred_list = []
        if traced:
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._traces[submesh_id].timestep_input)
                ttnn.copy_host_to_device_tensor(
                    sigma_difference[submesh_id],
                    self._traces[submesh_id].sigma_difference_input,
                )
                sigma_difference_device = self._traces[submesh_id].sigma_difference_input
                ttnn.execute_trace(submesh_device, self._traces[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._traces[submesh_id].latents_output)
        else:
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                noise_pred = self._step_inner(
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
                sigma_difference_device = sigma_difference[submesh_id]

        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            ttnn.synchronize_device(submesh_device)
            ttnn.multiply_(sigma_difference_device, noise_pred_list[submesh_id])
            ttnn.add_(latents[submesh_id], sigma_difference_device)

        return latents


def _dummy_state_dict(torch_model) -> dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in torch_model.state_dict().items()}


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents


def _prepare_latent_ids(
    batch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    t = torch.arange(1)
    h = torch.arange(height)
    w = torch.arange(width)
    l_dim = torch.arange(1)
    latent_ids = torch.cartesian_prod(t, h, w, l_dim)
    return latent_ids.unsqueeze(0).expand(batch_size, -1, -1)


def _prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = prompt_embeds.shape
    out_ids = []
    for _ in range(batch_size):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l_dim = torch.arange(seq_len)
        coords = torch.cartesian_prod(t, h, w, l_dim)
        out_ids.append(coords)
    return torch.stack(out_ids)


def _compute_empirical_mu(
    image_seq_len: int,
    num_steps: int,
    m_10: float = 0.30,
    m_200: float = 3.50,
) -> float:
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b
    return float(mu)

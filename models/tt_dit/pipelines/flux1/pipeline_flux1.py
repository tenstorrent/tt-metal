# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import tqdm
import ttnn
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from loguru import logger
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from models.perf.benchmarking_utils import BenchmarkProfiler

from ...encoders.clip.model_clip import CLIPConfig, CLIPEncoder
from ...encoders.t5.model_t5 import T5Config, T5Encoder
from ...models.transformers.transformer_flux1 import Flux1Transformer
from ...models.vae.vae_sd35 import VAEDecoder
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, ParallelFactor, VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils.padding import PaddingConfig
from ...utils import cache
from models.common.utility_functions import is_blackhole
import os


@dataclass
class PipelineTrace:
    tid: int
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    guidance_input: ttnn.Tensor
    spatial_rope_cos: ttnn.Tensor
    spatial_rope_sin: ttnn.Tensor
    prompt_rope_cos: ttnn.Tensor
    prompt_rope_sin: ttnn.Tensor
    sigma_difference_input: ttnn.Tensor
    latents_output: ttnn.Tensor


class Flux1Pipeline:
    T5_SEQUENCE_LENGTH = 512

    def __init__(
        self,
        *,
        checkpoint_name: str,
        mesh_device: ttnn.MeshDevice,
        enable_t5_text_encoder: bool = True,
        use_torch_t5_text_encoder: bool = False,
        use_torch_clip_text_encoder: bool = False,
        parallel_config: DiTParallelConfig,
        encoder_parallel_config: EncoderParallelConfig = None,
        vae_parallel_config: VAEParallelConfig = None,
        topology: ttnn.Topology,
        num_links: int,
    ) -> None:
        self._mesh_device = mesh_device
        self._parallel_config = parallel_config

        # setup encoder and vae parallel configs.
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

        # No CFG. Create submeshes based on SP and TP
        submesh_shape = list(mesh_device.shape)
        submesh_shape[parallel_config.sequence_parallel.mesh_axis] = parallel_config.sequence_parallel.factor
        submesh_shape[parallel_config.tensor_parallel.mesh_axis] = parallel_config.tensor_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Original mesh shape: {mesh_device.shape}")
        logger.info(f"Creating submeshes with shape {submesh_shape}")
        self._submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))[
            0:1
        ]  # Only create one submesh for now. This can be used to support batching in the future.
        self._ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=topology)
            for submesh_device in self._submesh_devices
        ]

        self.encoder_device = self._submesh_devices[0]
        self.original_submesh_shape = tuple(self.encoder_device.shape)
        self.vae_device = self._submesh_devices[0]
        self.encoder_submesh_idx = 0  # Use submesh 0 for encoder
        self.vae_submesh_idx = 0  # Use submesh 0 for VAE

        logger.info("loading models...")
        self._tokenizer_1 = CLIPTokenizer.from_pretrained(checkpoint_name, subfolder="tokenizer")
        self._t5_tokenizer = T5TokenizerFast.from_pretrained(checkpoint_name, subfolder="tokenizer_2")
        torch_text_encoder_1 = CLIPTextModel.from_pretrained(checkpoint_name, subfolder="text_encoder")
        torch_text_encoder_1.eval()
        if enable_t5_text_encoder:
            torch_t5_text_encoder = T5EncoderModel.from_pretrained(checkpoint_name, subfolder="text_encoder_2")
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_name, subfolder="scheduler")
        self._torch_vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae")

        torch_transformer = FluxTransformer2DModel.from_pretrained(
            checkpoint_name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,  # bfloat16 is the native datatype of the model
        )
        torch_transformer.eval()

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
            tt_transformer = Flux1Transformer(
                patch_size=torch_transformer.config.patch_size,
                in_channels=torch_transformer.config.in_channels,
                num_layers=torch_transformer.config.num_layers,
                num_single_layers=torch_transformer.config.num_single_layers,
                attention_head_dim=torch_transformer.config.attention_head_dim,
                num_attention_heads=torch_transformer.config.num_attention_heads,
                joint_attention_dim=torch_transformer.config.joint_attention_dim,
                pooled_projection_dim=torch_transformer.config.pooled_projection_dim,
                out_channels=torch_transformer.out_channels,
                axes_dims_rope=torch_transformer.config.axes_dims_rope,
                with_guidance_embeds=torch_transformer.config.guidance_embeds,
                mesh_device=submesh_device,
                ccl_manager=self._ccl_managers[i],
                parallel_config=parallel_config,
                padding_config=padding_config,
            )

            model_name = os.path.basename(checkpoint_name)
            if not cache.initialize_from_cache(
                tt_transformer,
                torch_transformer.state_dict(),
                model_name,
                "transformer",
                parallel_config,
                tuple(submesh_device.shape),
            ):
                logger.info(f"Loading transformer weights from PyTorch state dict")
                tt_transformer.load_torch_state_dict(torch_transformer.state_dict())

            self.transformers.append(tt_transformer)
            ttnn.synchronize_device(submesh_device)

        self._pos_embed = torch_transformer.pos_embed

        self._num_channels_latents = torch_transformer.config.in_channels // 4
        self._joint_attention_dim = torch_transformer.config.joint_attention_dim
        self._patch_size = torch_transformer.config.patch_size
        self._with_guidance_embeds = torch_transformer.config.guidance_embeds

        self._block_out_channels = self._torch_vae.config.block_out_channels
        self._latents_scaling = self._torch_vae.config.scaling_factor
        self._latents_shift = self._torch_vae.config.shift_factor

        self._vae_scale_factor = 2 ** len(self._block_out_channels)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._vae_scale_factor)

        logger.info("creating TT-NN CLIP text encoder...")

        if use_torch_clip_text_encoder:
            self._text_encoder_1 = torch_text_encoder_1
        else:
            clip_config_1 = CLIPConfig(
                vocab_size=torch_text_encoder_1.config.vocab_size,
                embed_dim=torch_text_encoder_1.config.hidden_size,
                ff_dim=torch_text_encoder_1.config.intermediate_size,
                num_heads=torch_text_encoder_1.config.num_attention_heads,
                num_hidden_layers=torch_text_encoder_1.config.num_hidden_layers,
                max_prompt_length=77,
                layer_norm_eps=torch_text_encoder_1.config.layer_norm_eps,
                attention_dropout=torch_text_encoder_1.config.attention_dropout,
                hidden_act=torch_text_encoder_1.config.hidden_act,
            )

            self._text_encoder_1 = CLIPEncoder(
                config=clip_config_1,
                mesh_device=self.encoder_device,
                ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
                parallel_config=encoder_parallel_config,
                eos_token_id=2,  # default EOS token ID for CLIP
            )

            self._text_encoder_1.load_torch_state_dict(torch_text_encoder_1.state_dict())

        if enable_t5_text_encoder:
            if use_torch_t5_text_encoder:
                self._t5_text_encoder = torch_t5_text_encoder
            else:
                logger.info("creating TT-NN text encoder...")

                t5_config = T5Config(
                    vocab_size=torch_t5_text_encoder.config.vocab_size,
                    embed_dim=torch_t5_text_encoder.config.d_model,
                    ff_dim=torch_t5_text_encoder.config.d_ff,
                    kv_dim=torch_t5_text_encoder.config.d_kv,
                    num_heads=torch_t5_text_encoder.config.num_heads,
                    num_hidden_layers=torch_t5_text_encoder.config.num_layers,
                    max_prompt_length=self.T5_SEQUENCE_LENGTH,
                    layer_norm_eps=torch_t5_text_encoder.config.layer_norm_epsilon,
                    relative_attention_num_buckets=torch_t5_text_encoder.config.relative_attention_num_buckets,
                    relative_attention_max_distance=torch_t5_text_encoder.config.relative_attention_max_distance,
                )

                self._t5_text_encoder = T5Encoder(
                    config=t5_config,
                    mesh_device=self.encoder_device,
                    ccl_manager=self._ccl_managers[self.encoder_submesh_idx],
                    parallel_config=encoder_parallel_config,
                )

                if not cache.initialize_from_cache(
                    self._t5_text_encoder,
                    torch_t5_text_encoder.state_dict(),
                    model_name,
                    "t5_text_encoder",
                    encoder_parallel_config,
                    tuple(self.encoder_device.shape),
                ):
                    logger.info(f"Loading T5 text encoder weights from PyTorch state dict")
                    self._t5_text_encoder.load_torch_state_dict(torch_t5_text_encoder.state_dict())
        else:
            self._t5_text_encoder = None

        self._traces = None

        ttnn.synchronize_device(self.encoder_device)

        self._vae_decoder = VAEDecoder.from_torch(
            torch_ref=self._torch_vae.decoder,
            mesh_device=self.vae_device,
            parallel_config=self._vae_parallel_config,
            ccl_manager=self._ccl_managers[self.vae_submesh_idx],
        )

        # warmup for safe tracing.
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
        enable_t5_text_encoder=True,
        use_torch_t5_text_encoder=False,
        use_torch_clip_text_encoder=False,
        num_links=None,
        topology=ttnn.Topology.Linear,
    ):
        wh_config = {
            (1, 4): {"sp": (1, 0), "tp": (4, 1), "encoder_tp": (4, 1), "vae_tp": (4, 1), "num_links": 1},
            (2, 4): {"sp": (2, 0), "tp": (4, 1), "encoder_tp": (4, 1), "vae_tp": (4, 1), "num_links": 1},
            (4, 4): {"sp": (4, 0), "tp": (4, 1), "encoder_tp": (4, 1), "vae_tp": (4, 1), "num_links": 1},
            (4, 8): {"sp": (4, 0), "tp": (8, 1), "encoder_tp": (4, 0), "vae_tp": (4, 0), "num_links": 4},
        }
        bh_config = {
            (1, 2): {"sp": (1, 0), "tp": (2, 1), "encoder_tp": (2, 1), "vae_tp": (2, 1), "num_links": 2},
            (2, 2): {"sp": (2, 0), "tp": (2, 1), "encoder_tp": (2, 1), "vae_tp": (2, 1), "num_links": 2},
            (2, 4): {"sp": (2, 0), "tp": (4, 1), "encoder_tp": (4, 1), "vae_tp": (4, 1), "num_links": 2},
        }

        default_config = bh_config if is_blackhole() else wh_config
        sp_factor, sp_axis = dit_sp or default_config[tuple(mesh_device.shape)]["sp"]
        tp_factor, tp_axis = dit_tp or default_config[tuple(mesh_device.shape)]["tp"]
        encoder_tp_factor, encoder_tp_axis = encoder_tp or default_config[tuple(mesh_device.shape)]["encoder_tp"]
        vae_tp_factor, vae_tp_axis = vae_tp or default_config[tuple(mesh_device.shape)]["vae_tp"]
        num_links = num_links or default_config[tuple(mesh_device.shape)]["num_links"]

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

        pipeline = Flux1Pipeline(
            checkpoint_name=checkpoint_name,
            mesh_device=mesh_device,
            enable_t5_text_encoder=enable_t5_text_encoder,
            use_torch_t5_text_encoder=use_torch_t5_text_encoder,
            use_torch_clip_text_encoder=use_torch_clip_text_encoder,
            parallel_config=dit_parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            topology=topology,
            num_links=num_links,
        )

        return pipeline

    def run_single_prompt(
        self,
        *,
        width: int = 1024,
        height: int = 1024,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int,
        seed: int,
        traced: bool = True,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ):
        return self(
            width=width,
            height=height,
            prompt_1=[prompt],
            prompt_2=[prompt],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            num_inference_steps=num_inference_steps,
            seed=seed,
            traced=traced,
            profiler=profiler,
            profiler_iteration=profiler_iteration,
        )

    def __call__(
        self,
        *,
        num_images_per_prompt: int = 1,
        width: int = 1024,
        height: int = 1024,
        cfg_scale: float = 1,  # Flux.1 is not indented to be used with CFG
        guidance_scale: float = 3.5,
        prompt_1: list[str],
        prompt_2: list[str],
        negative_prompt_1: list[str] | None = None,
        negative_prompt_2: list[str] | None = None,
        num_inference_steps: int,
        seed: int | None = None,
        traced: bool = False,
        clip_skip: int = 0,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> list[Image.Image]:
        prompt_count = len(prompt_1)

        sp_axis = self._parallel_config.sequence_parallel.mesh_axis

        assert num_images_per_prompt == 1, "generating multiple images is not supported"
        assert prompt_count == 1, "generating multiple images is not supported"

        with profiler("total", profiler_iteration) if profiler else nullcontext():
            assert height % (self._vae_scale_factor * self._patch_size) == 0
            assert width % (self._vae_scale_factor * self._patch_size) == 0

            cfg_enabled = cfg_scale > 1
            assert not cfg_enabled, "CFG is not supported"

            latents_height = height // self._vae_scale_factor
            latents_width = width // self._vae_scale_factor
            spatial_sequence_length = latents_height * latents_width

            logger.info("encoding prompts...")

            with profiler("encoder", profiler_iteration) if profiler else nullcontext():
                prompt_embeds, pooled_prompt_embeds = self._encode_prompts(
                    prompt_1=prompt_1,
                    prompt_2=prompt_2,
                    negative_prompt_1=negative_prompt_1,
                    negative_prompt_2=negative_prompt_2,
                    num_images_per_prompt=num_images_per_prompt,
                    cfg_enabled=cfg_enabled,
                    clip_skip=clip_skip,
                    profiler=profiler,
                    profiler_iteration=profiler_iteration,
                )
                _, prompt_sequence_length, _ = prompt_embeds.shape

            logger.info("preparing timesteps...")

            self._scheduler.set_timesteps(
                sigmas=np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
                mu=_calculate_shift(
                    spatial_sequence_length,
                    self._scheduler.config.get("base_image_seq_len", 256),
                    self._scheduler.config.get("max_image_seq_len", 4096),
                    self._scheduler.config.get("base_shift", 0.5),
                    self._scheduler.config.get("max_shift", 1.15),
                ),
            )

            guidance = (
                torch.full([prompt_count * num_images_per_prompt], fill_value=guidance_scale)
                if self._with_guidance_embeds
                else None
            )

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)

            # We let randn generate a permuted latent tensor, so that the generated noise matches the
            # reference implementation.
            shape = [
                prompt_count * num_images_per_prompt,
                self._num_channels_latents,
                latents_height * 2,
                latents_width * 2,
            ]
            latents = _pack_latents(
                torch.randn(shape, dtype=torch.bfloat16),
                prompt_count * num_images_per_prompt,
                self._num_channels_latents,
                latents_height,
                latents_width,
            )

            text_ids = torch.zeros([prompt_sequence_length, 3])
            image_ids = _latent_image_ids(height=latents_height, width=latents_width)
            ids = torch.cat((text_ids, image_ids), dim=0)
            rope_cos, rope_sin = self._pos_embed.forward(ids)

            tt_prompt_embeds_list = []
            tt_pooled_prompt_embeds_list = []
            tt_latents_step_list = []
            tt_guidance_list = []
            tt_spatial_rope_cos_list = []
            tt_spatial_rope_sin_list = []
            tt_prompt_rope_cos_list = []
            tt_prompt_rope_sin_list = []
            for i, submesh_device in enumerate(self._submesh_devices):
                tt_prompt_embeds = ttnn.from_torch(
                    prompt_embeds[i : i + 1] if self._parallel_config.cfg_parallel.factor == 2 else prompt_embeds,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=(None, None),
                    ),
                )

                tt_pooled_prompt_embeds = ttnn.from_torch(
                    pooled_prompt_embeds[i : i + 1]
                    if self._parallel_config.cfg_parallel.factor == 2
                    else pooled_prompt_embeds,
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
                shard_latents_dims[sp_axis] = 1  # height of latents
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

                tt_spatial_rope_cos = ttnn.from_torch(
                    rope_cos[prompt_sequence_length:],
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=rope_mesh_mapper,
                )
                tt_spatial_rope_sin = ttnn.from_torch(
                    rope_sin[prompt_sequence_length:],
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=rope_mesh_mapper,
                )
                tt_prompt_rope_cos = ttnn.from_torch(
                    rope_cos[:prompt_sequence_length],
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
                )
                tt_prompt_rope_sin = ttnn.from_torch(
                    rope_sin[:prompt_sequence_length],
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
                )

                if traced:
                    if self._traces is None:
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds = tt_prompt_embeds.to(submesh_device)
                        tt_pooled_prompt_embeds = tt_pooled_prompt_embeds.to(submesh_device)
                        tt_spatial_rope_cos = tt_spatial_rope_cos.to(submesh_device)
                        tt_spatial_rope_sin = tt_spatial_rope_sin.to(submesh_device)
                        tt_prompt_rope_cos = tt_prompt_rope_cos.to(submesh_device)
                        tt_prompt_rope_sin = tt_prompt_rope_sin.to(submesh_device)

                        if tt_guidance is not None:
                            tt_guidance = tt_guidance.to(submesh_device)
                    else:
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._traces[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._traces[i].prompt_input)
                        ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds, self._traces[i].pooled_input)
                        ttnn.copy_host_to_device_tensor(tt_spatial_rope_cos, self._traces[i].spatial_rope_cos)
                        ttnn.copy_host_to_device_tensor(tt_spatial_rope_sin, self._traces[i].spatial_rope_sin)
                        ttnn.copy_host_to_device_tensor(tt_prompt_rope_cos, self._traces[i].prompt_rope_cos)
                        ttnn.copy_host_to_device_tensor(tt_prompt_rope_sin, self._traces[i].prompt_rope_sin)

                        tt_initial_latents = self._traces[i].spatial_input
                        tt_prompt_embeds = self._traces[i].prompt_input
                        tt_pooled_prompt_embeds = self._traces[i].pooled_input
                        tt_spatial_rope_cos = self._traces[i].spatial_rope_cos
                        tt_spatial_rope_sin = self._traces[i].spatial_rope_sin
                        tt_prompt_rope_cos = self._traces[i].prompt_rope_cos
                        tt_prompt_rope_sin = self._traces[i].prompt_rope_sin

                        if tt_guidance is not None:
                            ttnn.copy_host_to_device_tensor(tt_guidance, self._traces[i].guidance_input)
                            tt_guidance = self._traces[i].guidance_input

                tt_prompt_embeds_list.append(tt_prompt_embeds)
                tt_pooled_prompt_embeds_list.append(tt_pooled_prompt_embeds)
                tt_latents_step_list.append(tt_initial_latents)
                tt_guidance_list.append(tt_guidance)
                tt_spatial_rope_cos_list.append(tt_spatial_rope_cos)
                tt_spatial_rope_sin_list.append(tt_spatial_rope_sin)
                tt_prompt_rope_cos_list.append(tt_prompt_rope_cos)
                tt_prompt_rope_sin_list.append(tt_prompt_rope_sin)

            logger.info("denoising...")

            with profiler("denoising", profiler_iteration) if profiler else nullcontext():
                for i, t in enumerate(tqdm.tqdm(self._scheduler.timesteps)):
                    with profiler(f"denoising_step_{i}", profiler_iteration) if profiler else nullcontext():
                        sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]

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
                                # [1, 1],
                                tt_initial_latents.shape,
                                fill_value=sigma_difference,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16,
                                device=submesh_device
                                if not traced
                                else None,  # Not used in trace region, can be on device always.
                            )
                            tt_sigma_difference_list.append(tt_sigma_difference)

                        tt_latents_step_list = self._step(
                            timestep=tt_timestep_list,
                            latents=tt_latents_step_list,
                            cfg_enabled=cfg_enabled,
                            prompt_embeds=tt_prompt_embeds_list,
                            pooled_prompt_embeds=tt_pooled_prompt_embeds_list,
                            cfg_scale=cfg_scale,
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
                # Sync because we don't pass a persistent buffer or a barrier semaphore.
                ttnn.synchronize_device(self.vae_device)

                tt_latents = self._ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
                    tt_latents_step_list[self.vae_submesh_idx],
                    dim=1,
                    mesh_axis=sp_axis,
                    use_hyperparams=True,
                )

                torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
                torch_latents = (torch_latents / self._latents_scaling) + self._latents_shift
                torch_latents = _unpack_latents(torch_latents, height, width, self._vae_scale_factor)

                tt_latents = ttnn.from_torch(
                    torch_latents.permute(0, 2, 3, 1),
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=self.vae_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.vae_device),
                )
                tt_decoded_output = self._vae_decoder(tt_latents)
                decoded_output = ttnn.to_torch(ttnn.get_device_tensors(tt_decoded_output)[0]).permute(0, 3, 1, 2)

                image = self._image_processor.postprocess(decoded_output, output_type="pt")
                assert isinstance(image, torch.Tensor)

                output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        return output

    def synchronize_devices(self):
        for device in self._submesh_devices:
            ttnn.synchronize_device(device)

    def _step_inner(
        self,
        *,
        cfg_enabled: bool,
        latent: ttnn.Tensor,
        prompt: ttnn.Tensor,
        pooled: ttnn.Tensor,
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
        if cfg_enabled and not self._parallel_config.cfg_parallel.factor > 1:
            latent = ttnn.concat([latent, latent])

        noise_pred = self.transformers[submesh_index].forward(
            spatial=latent,
            prompt=prompt,
            pooled=pooled,
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
        cfg_enabled: bool,
        cfg_scale: float,
        latents: list[ttnn.Tensor],  # device tensor
        timestep: list[ttnn.Tensor],  # host tensor
        pooled_prompt_embeds: list[ttnn.Tensor],  # device tensor
        prompt_embeds: list[ttnn.Tensor],  # device tensor
        sigma_difference: list[ttnn.Tensor],  # device tensor
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
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    pooled=pooled_prompt_embeds[submesh_id],
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
                        pooled_input=pooled_prompt_embeds[submesh_id],
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
                    sigma_difference[submesh_id], self._traces[submesh_id].sigma_difference_input
                )
                sigma_difference_device = self._traces[submesh_id].sigma_difference_input
                ttnn.execute_trace(submesh_device, self._traces[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._traces[submesh_id].latents_output)
        else:
            for submesh_id, submesh_device in enumerate(self._submesh_devices):
                noise_pred = self._step_inner(
                    cfg_enabled=cfg_enabled,
                    latent=latents[submesh_id],
                    prompt=prompt_embeds[submesh_id],
                    pooled=pooled_prompt_embeds[submesh_id],
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

        if cfg_enabled:
            if not self._parallel_config.cfg_parallel.factor > 1:
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

                shard_latents_dims = [None, None]
                shard_latents_dims[self._parallel_config.sequence_parallel.mesh_axis] = 1  # height of latents
                noise_pred_list[0] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=self._submesh_devices[0],
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self._submesh_devices[0],
                        tuple(self._submesh_devices[0].shape),
                        dims=tuple(shard_latents_dims),
                    ),
                )

                noise_pred_list[1] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=self._submesh_devices[1],
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self._submesh_devices[1],
                        tuple(self._submesh_devices[1].shape),
                        dims=tuple(shard_latents_dims),
                    ),
                )

        for submesh_id, submesh_device in enumerate(self._submesh_devices):
            ttnn.synchronize_device(submesh_device)  # Helps with accurate time profiling.
            ttnn.multiply_(sigma_difference_device, noise_pred_list[submesh_id])
            ttnn.add_(latents[submesh_id], sigma_difference_device)

        return latents

    def _encode_prompts_partial(
        self,
        prompt_1: list[str],
        prompt_2: list[str],
        num_images_per_prompt: int,
        clip_skip: int = 0,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer_max_length = self._tokenizer_1.model_max_length

        with profiler("clip_encoding", profiler_iteration) if profiler else nullcontext():
            prompt_1_embeds, pooled_prompt_1_embeds = _get_clip_prompt_embeds(
                prompts=prompt_1,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_1,
                text_encoder=self._text_encoder_1,
                sequence_length=tokenizer_max_length,
                mesh_device=self.encoder_device,
                clip_skip=clip_skip,
            )

        with profiler("t5_encoding", profiler_iteration) if profiler else nullcontext():
            t5_prompt_embeds = _get_t5_prompt_embeds(
                prompts=prompt_2,
                text_encoder=self._t5_text_encoder,
                tokenizer=self._t5_tokenizer,
                sequence_length=self.T5_SEQUENCE_LENGTH,
                empty_sequence_length=self.T5_SEQUENCE_LENGTH,
                num_images_per_prompt=num_images_per_prompt,
                mesh_device=self.encoder_device,
                embedding_dim=self._joint_attention_dim,
            )

        prompt_embeds = t5_prompt_embeds
        pooled_prompt_embeds = pooled_prompt_1_embeds

        return prompt_embeds, pooled_prompt_embeds

    def _encode_prompts(
        self,
        *,
        prompt_1: list[str],
        prompt_2: list[str],
        negative_prompt_1: list[str] | None,
        negative_prompt_2: list[str] | None,
        num_images_per_prompt: int,
        cfg_enabled: bool,
        clip_skip: int = 0,
        profiler: BenchmarkProfiler = None,
        profiler_iteration: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts_partial(
            prompt_1=prompt_1,
            prompt_2=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            profiler=profiler,
            profiler_iteration=profiler_iteration,
        )

        if not cfg_enabled:
            return prompt_embeds, pooled_prompt_embeds

        negative_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompts_partial(
            prompt_1=negative_prompt_1,
            prompt_2=negative_prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
            profiler=profiler,
            profiler_iteration=profiler_iteration,
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    *,
    prompts: list[str],
    text_encoder: CLIPEncoder | CLIPTextModel,
    tokenizer: CLIPTokenizer,
    sequence_length: int,
    num_images_per_prompt: int,
    clip_skip: int = 0,
    mesh_device: ttnn.MeshDevice | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
    ).input_ids

    untruncated_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
    ).input_ids

    if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
        logger.warning("CLIP input text was truncated")

    if isinstance(text_encoder, CLIPEncoder):
        assert mesh_device is not None

        tt_tokens = ttnn.from_torch(
            tokens,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )

        tt_prompt_embeds, tt_pooled_prompt_embeds = text_encoder(
            prompt_tokenized=tt_tokens,
            mesh_device=mesh_device,
        )
        tt_prompt_embeds = tt_prompt_embeds[-(clip_skip + 2)]

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
        pooled_prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_pooled_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)
        with torch.no_grad():
            output = text_encoder.forward(tokens, output_hidden_states=True)
        prompt_embeds = output.hidden_states[-(clip_skip + 2)].to("cpu")
        pooled_prompt_embeds = output.pooler_output.to("cpu")

    # In diffusers v0.35.1 `pooled_prompt_embeds` is repeated along the wrong dimension in
    # `StableDiffusion3Pipeline`, effectively mixing up the prompts.
    pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)

    return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_t5_prompt_embeds(
    *,
    prompts: list[str],
    text_encoder: T5Encoder | T5EncoderModel | None,
    tokenizer: T5TokenizerFast,
    sequence_length: int,
    empty_sequence_length: int,
    num_images_per_prompt: int,
    mesh_device: ttnn.MeshDevice | None = None,
    embedding_dim: int,
) -> torch.Tensor:
    if text_encoder is None:
        return torch.zeros([len(prompts) * num_images_per_prompt, empty_sequence_length, embedding_dim])

    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
    ).input_ids

    untruncated_tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding="longest",
    ).input_ids

    if untruncated_tokens.shape[-1] >= tokens.shape[-1] and not torch.equal(tokens, untruncated_tokens):
        logger.warning("T5 input text was truncated")

    if isinstance(text_encoder, T5Encoder):
        assert mesh_device is not None

        tt_tokens = ttnn.from_torch(
            tokens,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.uint32,
            device=mesh_device,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )
        tt_hidden_states = text_encoder(prompt=tt_tokens, device=mesh_device)
        tt_prompt_embeds = tt_hidden_states[-1]

        prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(tt_prompt_embeds)[0])
    else:
        tokens = tokens.to(device=text_encoder.device)
        with torch.no_grad():
            output = text_encoder.forward(tokens)
        prompt_embeds = output.last_hidden_state.to("cpu")

    return prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)


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
    return latents.reshape(batch_size, height * width, num_channels_latents * 4)


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


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b

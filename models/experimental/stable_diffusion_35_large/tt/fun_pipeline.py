# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import tqdm
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from contextlib import contextmanager, nullcontext

from .clip_encoder import TtCLIPTextTransformer, TtCLIPTextTransformerParameters, TtCLIPConfig
from .t5_encoder import TtT5Encoder, TtT5EncoderParameters
from .fun_transformer import sd_transformer, TtSD3Transformer2DModelParameters
from .parallel_config import StableDiffusionParallelManager, EncoderParallelManager, VAEParallelConfig
from .fun_vae_decoder.fun_vae_decoder import sd_vae_decode, TtVaeDecoderParameters

TILE_SIZE = 32


@dataclass
class TimingData:
    clip_encoding_time: float = 0.0
    t5_encoding_time: float = 0.0
    total_encoding_time: float = 0.0
    denoising_step_times: List[float] = field(default_factory=list)
    vae_decoding_time: float = 0.0
    total_time: float = 0.0


class TimingCollector:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.step_timings: Dict[str, List[float]] = {}

    @contextmanager
    def time_section(self, name: str):
        start = time.time()
        yield
        end = time.time()
        self.timings[name] = end - start

    @contextmanager
    def time_step(self, name: str):
        start = time.time()
        yield
        end = time.time()
        if name not in self.step_timings:
            self.step_timings[name] = []
        self.step_timings[name].append(end - start)

    def get_timing_data(self) -> TimingData:
        return TimingData(
            clip_encoding_time=self.timings.get("clip_encoding", 0.0),
            t5_encoding_time=self.timings.get("t5_encoding", 0.0),
            total_encoding_time=self.timings.get("total_encoding", 0.0),
            denoising_step_times=self.step_timings.get("denoising_step", []),
            vae_decoding_time=self.timings.get("vae_decoding", 0.0),
            total_time=self.timings.get("total", 0.0),
        )


@dataclass
class PipelineTrace:
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    pooled_projection_input: ttnn.Tensor
    timestep_input: ttnn.Tensor
    latents_output: ttnn.Tensor
    tid: int


class TtStableDiffusion3Pipeline:
    def __init__(
        self,
        *,
        checkpoint_name: str,
        mesh_device: ttnn.MeshDevice,
        enable_t5_text_encoder: bool = True,
        guidance_cond: int,
        parallel_manager: StableDiffusionParallelManager,
        encoder_parallel_manager: EncoderParallelManager,
        vae_parallel_manager: VAEParallelConfig,
        height: int,
        width: int,
        model_location_generator,
        quiet: bool = False,
    ) -> None:
        self._mesh_device = mesh_device
        self.encoder_parallel_manager = encoder_parallel_manager
        self.vae_parallel_manager = vae_parallel_manager
        self.quiet = quiet
        model_name_checkpoint = model_location_generator(checkpoint_name, model_subdir="StableDiffusion_35_Large")

        if not quiet:
            logger.info("loading models...")
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
            torch_text_encoder_3 = T5EncoderModel.from_pretrained(model_name_checkpoint, subfolder="text_encoder_3")
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_name_checkpoint, subfolder="scheduler")
        self._torch_vae = AutoencoderKL.from_pretrained(model_name_checkpoint, subfolder="vae")

        torch_transformer = SD3Transformer2DModel.from_pretrained(
            model_name_checkpoint,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,  # bfloat16 is the native datatype of the model
        )
        torch_transformer.eval()

        assert isinstance(self._tokenizer_1, CLIPTokenizer)
        assert isinstance(self._tokenizer_2, CLIPTokenizer)
        assert isinstance(self._tokenizer_3, T5TokenizerFast)
        assert isinstance(self._text_encoder_1, CLIPTextModelWithProjection)
        assert isinstance(self._text_encoder_2, CLIPTextModelWithProjection)
        assert isinstance(self._scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(self._torch_vae, AutoencoderKL)
        assert isinstance(torch_transformer, SD3Transformer2DModel)
        if not quiet:
            logger.info("creating TT-NN transformer...")

        if checkpoint_name == "stabilityai/stable-diffusion-3.5-medium":
            embedding_dim = 1536
        else:
            embedding_dim = 2432

        parallel_config = parallel_manager.dit_parallel_config
        ## heads padding
        assert not embedding_dim % torch_transformer.config.num_attention_heads, "Embedding_dim % num_heads != 0"
        pad_embedding_dim = (bool)(
            torch_transformer.config.num_attention_heads
        ) % parallel_config.tensor_parallel.factor
        if pad_embedding_dim:
            head_size = embedding_dim // torch_transformer.config.num_attention_heads
            num_heads = (
                math.ceil(torch_transformer.config.num_attention_heads / parallel_config.tensor_parallel.factor)
                * parallel_config.tensor_parallel.factor
            )
            hidden_dim_padding = (num_heads * head_size) - embedding_dim
        else:
            num_heads = torch_transformer.config.num_attention_heads

        parameters_list = []
        for i, submesh_device in enumerate(parallel_manager.submesh_devices):
            parameters_list.append(
                TtSD3Transformer2DModelParameters.from_torch(
                    torch_transformer.state_dict(),
                    num_heads=num_heads,
                    unpadded_num_heads=torch_transformer.config.num_attention_heads,
                    embedding_dim=embedding_dim,
                    hidden_dim_padding=hidden_dim_padding,
                    device=submesh_device,
                    dtype=ttnn.bfloat8_b if submesh_device.get_num_devices() == 1 else ttnn.bfloat16,
                    guidance_cond=guidance_cond,
                    parallel_config=parallel_config,
                    height=height // 2 ** (len(self._torch_vae.config.block_out_channels) - 1),
                    width=width // 2 ** (len(self._torch_vae.config.block_out_channels) - 1),
                )
            )
            ttnn.synchronize_device(submesh_device)

        self.parallel_manager = parallel_manager
        self.num_heads = num_heads
        self.patch_size = parameters_list[0].pos_embed.patch_size
        # DEBUG
        # self.patch_size = 2
        self.tt_transformer_parameters = parameters_list

        # self._tt_transformer = TtSD3Transformer2DModel(
        #     parameters, guidance_cond=guidance_cond, num_heads=num_heads, device=self._device
        # )
        self._num_channels_latents = torch_transformer.config.in_channels
        self._joint_attention_dim = torch_transformer.config.joint_attention_dim

        self._block_out_channels = self._torch_vae.config.block_out_channels
        self._torch_vae_scaling_factor = self._torch_vae.config.scaling_factor
        self._torch_vae_shift_factor = self._torch_vae.config.shift_factor

        self._torch_vae_scale_factor = 2 ** (len(self._block_out_channels) - 1)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._torch_vae_scale_factor)

        # HACK: reshape submesh device 0 to 1D
        encoder_parallel_manager.mesh_device.reshape(
            ttnn.MeshShape(*encoder_parallel_manager.tensor_parallel.mesh_shape)
        )
        if not quiet:
            logger.info("creating TT-NN CLIP text encoder...")
        parameters_1 = TtCLIPTextTransformerParameters.from_torch(
            self._text_encoder_1.state_dict(),
            device=encoder_parallel_manager.mesh_device,
            dtype=ttnn.bfloat16,
            parallel_manager=encoder_parallel_manager,
        )
        parameters_2 = TtCLIPTextTransformerParameters.from_torch(
            self._text_encoder_2.state_dict(),  # Different weights!
            device=encoder_parallel_manager.mesh_device,
            dtype=ttnn.bfloat16,
            parallel_manager=encoder_parallel_manager,
        )
        self._text_encoder_1 = TtCLIPTextTransformer(
            parameters_1,
            TtCLIPConfig(
                vocab_size=self._text_encoder_1.config.vocab_size,
                d_model=self._text_encoder_1.config.hidden_size,
                d_ff=self._text_encoder_1.config.intermediate_size,
                num_heads=self._text_encoder_1.config.num_attention_heads,
                num_layers=self._text_encoder_1.config.num_hidden_layers,
                max_position_embeddings=77,
                layer_norm_eps=self._text_encoder_1.config.layer_norm_eps,
                attention_dropout=self._text_encoder_1.config.attention_dropout,
                hidden_act=self._text_encoder_1.config.hidden_act,
            ),
        )
        self._text_encoder_2 = TtCLIPTextTransformer(
            parameters_2,
            TtCLIPConfig(
                vocab_size=self._text_encoder_2.config.vocab_size,
                d_model=self._text_encoder_2.config.hidden_size,
                d_ff=self._text_encoder_2.config.intermediate_size,
                num_heads=self._text_encoder_2.config.num_attention_heads,
                num_layers=self._text_encoder_2.config.num_hidden_layers,
                max_position_embeddings=77,
                layer_norm_eps=self._text_encoder_2.config.layer_norm_eps,
                attention_dropout=self._text_encoder_2.config.attention_dropout,
                hidden_act=self._text_encoder_2.config.hidden_act,
            ),
        )

        if enable_t5_text_encoder:
            if not quiet:
                logger.info("creating TT-NN text encoder...")

            parameters = TtT5EncoderParameters.from_torch(
                torch_text_encoder_3.state_dict(),
                device=encoder_parallel_manager.mesh_device,
                dtype=ttnn.bfloat16,
                parallel_manager=encoder_parallel_manager,
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

        # HACK: reshape submesh device 0 from 1D to 2D
        self.encoder_parallel_manager.mesh_device.reshape(
            ttnn.MeshShape(*self.parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape)
        )

        self.timing_collector = None  # Set externally when timing is needed

        self._trace = None

        ttnn.synchronize_device(self.encoder_parallel_manager.mesh_device)

        # HACK: reshape submesh device 0 to 1D
        original_vae_device_shape = tuple(vae_parallel_manager.device.shape)
        if original_vae_device_shape != tuple(encoder_parallel_manager.tensor_parallel.mesh_shape):
            vae_parallel_manager.device.reshape(ttnn.MeshShape(*encoder_parallel_manager.tensor_parallel.mesh_shape))
        self._vae_parameters = TtVaeDecoderParameters.from_torch(
            torch_vae_decoder=self._torch_vae.decoder, dtype=ttnn.bfloat16, parallel_config=vae_parallel_manager
        )

        # HACK: reshape submesh device 0 from 1D to 2D
        if original_vae_device_shape != tuple(encoder_parallel_manager.tensor_parallel.mesh_shape):
            vae_parallel_manager.device.reshape(
                ttnn.MeshShape(*self.parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape)
            )

        self.latents_gather_semaphore = ttnn.create_global_semaphore(
            vae_parallel_manager.device, parallel_manager.ccl_cores, 0
        )
        self.latents_gather_buffer = ttnn.from_torch(
            torch.zeros((1, 128, 128, 16)),
            device=vae_parallel_manager.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(vae_parallel_manager.device),
        )

    def prepare(
        self,
        *,
        batch_size: int,
        num_images_per_prompt: int = 1,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 4.5,
        max_t5_sequence_length: int = 256,
        prompt_sequence_length: int = 333,
        spatial_sequence_length: int = 4096,
    ) -> None:
        self._prepared_batch_size = batch_size
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
            height // self._torch_vae_scale_factor,
            (width // self._torch_vae_scale_factor) // patch_size,
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
        traced: bool = False,
        clip_skip: int | None = None,
    ) -> None:
        timer = self.timing_collector

        with timer.time_section("total") if timer else nullcontext():
            batch_size = self._prepared_batch_size
            num_images_per_prompt = self._prepared_num_images_per_prompt
            width = self._prepared_width
            height = self._prepared_height
            guidance_scale = self._prepared_guidance_scale
            max_t5_sequence_length = self._prepared_max_t5_sequence_length

            assert height % (self._torch_vae_scale_factor * self.patch_size) == 0
            assert width % (self._torch_vae_scale_factor * self.patch_size) == 0
            assert max_t5_sequence_length <= 512  # noqa: PLR2004
            assert batch_size == len(prompt_1)

            do_classifier_free_guidance = guidance_scale > 1
            # TODO: pass the patch_size value
            patch_size = 2
            latents_shape = (
                batch_size * num_images_per_prompt,
                height // self._torch_vae_scale_factor,
                width // self._torch_vae_scale_factor,
                self._num_channels_latents,
            )

            if not self.quiet:
                logger.info(f"Latents shape: {latents_shape}")
                logger.info("encoding prompts...")

            with timer.time_section("total_encoding") if timer else nullcontext():
                # HACK: reshape submesh device 0 from 2D to 1D
                self.encoder_parallel_manager.mesh_device.reshape(
                    ttnn.MeshShape(*self.encoder_parallel_manager.tensor_parallel.mesh_shape)
                )
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
                    clip_skip=clip_skip,
                )
                # HACK: reshape submesh device 0 from 1D to 2D
                self.encoder_parallel_manager.mesh_device.reshape(
                    ttnn.MeshShape(*self.parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape)
                )

            self._scheduler.set_timesteps(num_inference_steps)
            timesteps = self._scheduler.timesteps

            if seed is not None:
                torch.manual_seed(seed)
            latents = torch.randn(latents_shape, dtype=prompt_embeds.dtype)  # .permute([0, 2, 3, 1])

            tt_prompt_embeds_list = []
            tt_pooled_prompt_embeds_list = []
            tt_latents_step_list = []
            for i, submesh_device in enumerate(self.parallel_manager.submesh_devices):
                tt_prompt_embeds = ttnn.from_torch(
                    prompt_embeds[i].unsqueeze(0)
                    if self.parallel_manager.dit_parallel_config.cfg_parallel.factor == 2
                    else prompt_embeds,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        self.parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
                        dims=[None, None],
                    ),
                )

                tt_pooled_prompt_embeds = ttnn.from_torch(
                    pooled_prompt_embeds[i].unsqueeze(0)
                    if self.parallel_manager.dit_parallel_config.cfg_parallel.factor == 2
                    else pooled_prompt_embeds,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        self.parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
                        dims=[None, None],
                    ),
                )

                shard_latents_dims = [None, None]
                shard_latents_dims[
                    self.parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis
                ] = 1  # height of latents
                tt_initial_latents = ttnn.from_torch(
                    latents,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        self.parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
                        dims=shard_latents_dims,
                    ),
                )
                if traced:
                    if self._trace is None:
                        # Push inputs to device
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds = tt_prompt_embeds.to(submesh_device)
                        tt_pooled_prompt_embeds = tt_pooled_prompt_embeds.to(submesh_device)
                    else:
                        # Copy inputs to trace
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._trace[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._trace[i].prompt_input)
                        ttnn.copy_host_to_device_tensor(tt_pooled_prompt_embeds, self._trace[i].pooled_projection_input)
                        # Ensure trace inputs are passed to function
                        tt_initial_latents = self._trace[i].spatial_input
                        tt_prompt_embeds = self._trace[i].prompt_input
                        tt_pooled_prompt_embeds = self._trace[i].pooled_projection_input

                tt_prompt_embeds_list.append(tt_prompt_embeds)
                tt_pooled_prompt_embeds_list.append(tt_pooled_prompt_embeds)
                tt_latents_step_list.append(tt_initial_latents)

            if not self.quiet:
                logger.info("denoising...")

            for i, t in enumerate(tqdm.tqdm(timesteps, disable=self.quiet)):
                with timer.time_step("denoising_step") if timer else nullcontext():
                    sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]

                    tt_timestep_list = []
                    tt_sigma_difference_list = []
                    for submesh_device in self.parallel_manager.submesh_devices:
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
                            device=submesh_device,  # Not used in trace region, can be on device always.
                        )
                        tt_sigma_difference_list.append(tt_sigma_difference)

                    tt_latents_step_list = self._step(
                        timestep=tt_timestep_list,
                        latents=tt_latents_step_list,  # tt_latents,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        prompt_embeds=tt_prompt_embeds_list,
                        pooled_prompt_embeds=tt_pooled_prompt_embeds_list,
                        guidance_scale=guidance_scale,
                        sigma_difference=tt_sigma_difference_list,
                        prompt_sequence_length=333,
                        spatial_sequence_length=4096,
                        traced=traced,
                    )
            if not self.quiet:
                logger.info("decoding image...")

            with timer.time_section("vae_decoding") if timer else nullcontext():
                # All gather replacement
                tt_latents = ttnn.experimental.all_gather_async(
                    input_tensor=tt_latents_step_list[0],
                    dim=1,
                    multi_device_global_semaphore=self.latents_gather_semaphore,
                    topology=ttnn.Topology.Linear,
                    mesh_device=self.vae_parallel_manager.device,
                    cluster_axis=self.parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis,
                    num_links=1,
                )

                torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
                torch_latents = (torch_latents / self._torch_vae_scaling_factor) + self._torch_vae_shift_factor

                # HACK: reshape submesh device 0 from 2D to 1D
                original_vae_device_shape = tuple(self.vae_parallel_manager.device.shape)
                if original_vae_device_shape != tuple(self.encoder_parallel_manager.tensor_parallel.mesh_shape):
                    self.vae_parallel_manager.device.reshape(
                        ttnn.MeshShape(*self.encoder_parallel_manager.tensor_parallel.mesh_shape)
                    )

                tt_latents = ttnn.from_torch(
                    torch_latents,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=self.vae_parallel_manager.device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.vae_parallel_manager.device),
                )
                decoded_output = sd_vae_decode(tt_latents, self._vae_parameters)
                decoded_output = ttnn.to_torch(ttnn.get_device_tensors(decoded_output)[0]).permute(0, 3, 1, 2)
                # HACK: reshape submesh device 0 from 1D to 2D
                if original_vae_device_shape != tuple(self.encoder_parallel_manager.tensor_parallel.mesh_shape):
                    self.vae_parallel_manager.device.reshape(
                        ttnn.MeshShape(*self.parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape)
                    )
                # image = self._torch_vae.decoder(tt_latents)
                image = self._image_processor.postprocess(decoded_output, output_type="pt")
                if not self.quiet:
                    logger.info(f"postprocessed image shape: {image.shape}")
                assert isinstance(image, torch.Tensor)

                output = self._image_processor.numpy_to_pil(self._image_processor.pt_to_numpy(image))

        return output

    def _step(
        self,
        *,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        latents: List[ttnn.Tensor],  # device tensor
        timestep: List[ttnn.Tensor],  # host tensor
        pooled_prompt_embeds: List[ttnn.Tensor],  # device tensor
        prompt_embeds: List[ttnn.Tensor],  # device tensor
        sigma_difference: List[ttnn.Tensor],  # device tensor
        prompt_sequence_length: int,
        spatial_sequence_length: int,
        traced: bool,
    ) -> None:
        def inner(latent, prompt, pooled_projection, timestep, parameters, cfg_index):
            if do_classifier_free_guidance and not self.parallel_manager.is_cfg_parallel:
                latent_model_input = ttnn.concat([latent, latent])
            else:
                latent_model_input = latent
            noise_pred = sd_transformer(
                spatial=latent_model_input,
                prompt=prompt,
                pooled_projection=pooled_projection,
                timestep=timestep,
                parameters=parameters,
                parallel_manager=self.parallel_manager,
                num_heads=self.num_heads,
                N=spatial_sequence_length,
                L=prompt_sequence_length,
                cfg_index=cfg_index,
            )

            noise_pred = _reshape_noise_pred(
                noise_pred,
                height=latent.shape[-3] * self.parallel_manager.dit_parallel_config.sequence_parallel.factor,
                width=latent.shape[-2],
                patch_size=self.patch_size,
            )
            return noise_pred

        if traced and self._trace is None:
            if not self.quiet:
                logger.info("Tracing...")
            self._trace = [None for _ in self.parallel_manager.submesh_devices]
            for submesh_id, submesh_device in enumerate(self.parallel_manager.submesh_devices):
                if not self.quiet:
                    logger.info(f"Tracing submesh {submesh_id}")
                latent_device = latents[submesh_id]  # already on device
                prompt_device = prompt_embeds[submesh_id]  # already on device
                pooled_projection_device = pooled_prompt_embeds[submesh_id]  # already on device
                timestep_device = timestep[submesh_id].to(submesh_device)

                if not self.quiet:
                    logger.info("compile run")
                pred = inner(
                    latent_device,
                    prompt_device,
                    pooled_projection_device,
                    timestep_device,
                    self.tt_transformer_parameters[submesh_id],
                    submesh_id,
                )

                ttnn.synchronize_device(self.parallel_manager.submesh_devices[0])
                ttnn.synchronize_device(self.parallel_manager.submesh_devices[1])

                if not self.quiet:
                    logger.info("begin trace capture")
                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = inner(
                    latent_device,
                    prompt_device,
                    pooled_projection_device,
                    timestep_device,
                    self.tt_transformer_parameters[submesh_id],
                    submesh_id,
                )
                ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(self.parallel_manager.submesh_devices[0])
                ttnn.synchronize_device(self.parallel_manager.submesh_devices[1])
                if not self.quiet:
                    logger.info("done sync after trace capture")

                self._trace[submesh_id] = PipelineTrace(
                    spatial_input=latent_device,
                    prompt_input=prompt_device,
                    pooled_projection_input=pooled_projection_device,
                    timestep_input=timestep_device,
                    latents_output=pred,
                    tid=trace_id,
                )

        noise_pred_list = []
        if traced:
            for submesh_id, submesh_device in enumerate(self.parallel_manager.submesh_devices):
                ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._trace[submesh_id].timestep_input)
                ttnn.execute_trace(submesh_device, self._trace[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._trace[submesh_id].latents_output)
        else:
            for submesh_id, submesh_device in enumerate(self.parallel_manager.submesh_devices):
                noise_pred = inner(
                    latents[submesh_id],
                    prompt_embeds[submesh_id],
                    pooled_prompt_embeds[submesh_id],
                    timestep[submesh_id],
                    self.tt_transformer_parameters[submesh_id],
                    submesh_id,
                )
                noise_pred_list.append(noise_pred)

        if do_classifier_free_guidance:
            if not self.parallel_manager.is_cfg_parallel:
                split_pos = noise_pred_list[0].shape[0] // 2
                uncond = noise_pred_list[0][0:split_pos]
                cond = noise_pred_list[0][split_pos:]
                noise_pred_list[0] = uncond + guidance_scale * (cond - uncond)
            else:
                # uncond and cond are replicated, so it is fine to get a single tensor from each
                uncond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[0])[0].cpu(blocking=True)).to(
                    torch.float32
                )
                cond = ttnn.to_torch(ttnn.get_device_tensors(noise_pred_list[1])[0].cpu(blocking=True)).to(
                    torch.float32
                )

                torch_noise_pred = uncond + guidance_scale * (cond - uncond)

                shard_latents_dims = [None, None]
                shard_latents_dims[
                    self.parallel_manager.dit_parallel_config.sequence_parallel.mesh_axis
                ] = 1  # height of latents
                noise_pred_list[0] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=self.parallel_manager.submesh_devices[0],
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.parallel_manager.submesh_devices[0],
                        self.parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
                        dims=shard_latents_dims,
                    ),
                )

                noise_pred_list[1] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=self.parallel_manager.submesh_devices[1],
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.parallel_manager.submesh_devices[1],
                        self.parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape,
                        dims=shard_latents_dims,
                    ),
                )

        for submesh_id, submesh_device in enumerate(self.parallel_manager.submesh_devices):
            ttnn.add_(latents[submesh_id], sigma_difference[submesh_id] * noise_pred_list[submesh_id])

        return latents

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
        clip_skip: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timer = self.timing_collector

        tokenizer_max_length = self._tokenizer_1.model_max_length

        with timer.time_section("clip_encoding") if timer else nullcontext():
            prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
                prompt=prompt_1,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_1,
                text_encoder=self._text_encoder_1,
                tokenizer_max_length=tokenizer_max_length,
                ttnn_device=self.encoder_parallel_manager.mesh_device,
                encoder_parallel_manager=self.encoder_parallel_manager,
                clip_skip=clip_skip,
            )

            prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_2,
                text_encoder=self._text_encoder_2,
                tokenizer_max_length=tokenizer_max_length,
                ttnn_device=self.encoder_parallel_manager.mesh_device,
                encoder_parallel_manager=self.encoder_parallel_manager,
                clip_skip=clip_skip,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        with timer.time_section("t5_encoding") if timer else nullcontext():
            t5_prompt_embed = _get_t5_prompt_embeds(
                device=self.encoder_parallel_manager.mesh_device,
                encoder_parallel_manager=self.encoder_parallel_manager,
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

        with timer.time_section("clip_encoding") if timer else nullcontext():
            negative_prompt_embed, negative_pooled_prompt_embed = _get_clip_prompt_embeds(
                prompt=negative_prompt_1,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_1,
                text_encoder=self._text_encoder_1,
                tokenizer_max_length=tokenizer_max_length,
                encoder_parallel_manager=self.encoder_parallel_manager,
                ttnn_device=self.encoder_parallel_manager.mesh_device,
                clip_skip=clip_skip,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = _get_clip_prompt_embeds(
                prompt=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_2,
                text_encoder=self._text_encoder_2,
                tokenizer_max_length=tokenizer_max_length,
                encoder_parallel_manager=self.encoder_parallel_manager,
                ttnn_device=self.encoder_parallel_manager.mesh_device,
                clip_skip=clip_skip,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

        with timer.time_section("t5_encoding") if timer else nullcontext():
            t5_negative_prompt_embed = _get_t5_prompt_embeds(
                device=self.encoder_parallel_manager.mesh_device,
                encoder_parallel_manager=self.encoder_parallel_manager,
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


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    *,
    clip_skip: int | None = None,
    device: torch.device | None = None,
    ttnn_device: ttnn.Device | None = None,
    encoder_parallel_manager: EncoderParallelManager | None = None,
    num_images_per_prompt: int,
    prompt: list[str],
    text_encoder: TtCLIPTextTransformer,
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

    tt_text_input_ids = ttnn.from_torch(
        text_input_ids,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device),
    )

    encoder_output, projected_output = text_encoder(
        tt_text_input_ids,
        ttnn_device,
        parallel_manager=encoder_parallel_manager,
        clip_skip=clip_skip,
        output_hidden_states=True,
    )

    if clip_skip is None:
        sequence_embeddings = encoder_output.hidden_states[-2]
    else:
        layer_index = -(clip_skip + 2)
        if abs(layer_index) > len(encoder_output.hidden_states):
            layer_index = -2
        sequence_embeddings = encoder_output.hidden_states[layer_index]

    prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(sequence_embeddings)[0])

    pooled_prompt_embeds = ttnn.to_torch(ttnn.get_device_tensors(projected_output)[0])

    prompt_embeds = prompt_embeds.to(device=device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device)

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
    encoder_parallel_manager: EncoderParallelManager | None = None,
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

    tt_text_input_ids = ttnn.from_torch(
        text_input_ids,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    tt_prompt_embeds = text_encoder(tt_text_input_ids, device, parallel_manager=encoder_parallel_manager)
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

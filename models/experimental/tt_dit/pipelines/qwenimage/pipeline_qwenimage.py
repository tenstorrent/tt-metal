# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List
from PIL import Image

import torch
import tqdm
import ttnn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel as TorchQwenImageTransformer2DModel,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from loguru import logger
from transformers import AutoTokenizer
from contextlib import contextmanager, nullcontext

from ....encoders.qwen25vl.encoder_pair import Qwen25VlTokenizerEncoderPair

from ...models.transformers.transformer_qwenimage import QwenImageTransformer
from ...models.vae.vae_qwenimage import QwenImageVaeDecoder
from ...parallel.manager import CCLManager
from ...parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig, ParallelFactor
from ...utils.padding import PaddingConfig
from ...utils.cache import save_cache_dict, load_cache_dict, cache_dict_exists, get_and_create_cache_path

TILE_SIZE = 32


@dataclass
class TimingData:
    clip_encoding_time: float = 0.0
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
            total_encoding_time=self.timings.get("total_encoding", 0.0),
            denoising_step_times=self.step_timings.get("denoising_step", []),
            vae_decoding_time=self.timings.get("vae_decoding", 0.0),
            total_time=self.timings.get("total", 0.0),
        )

    def reset(self):
        self.timings = {}
        self.step_timings = {}
        return self


@dataclass
class PipelineTrace:
    spatial_input: ttnn.Tensor
    prompt_input: ttnn.Tensor
    spatial_rope_input: tuple[ttnn.Tensor, ttnn.Tensor]
    prompt_rope_input: tuple[ttnn.Tensor, ttnn.Tensor]
    timestep_input: ttnn.Tensor
    latents_output: ttnn.Tensor
    tid: int


class QwenImagePipeline:
    def __init__(
        self,
        *,
        mesh_device: ttnn.MeshDevice,
        guidance_cond: int,
        parallel_config: DiTParallelConfig,
        num_links: int,
        height: int,
        width: int,
        model_checkpoint_path: str,
        use_cache=False,
    ) -> None:
        self._mesh_device = mesh_device

        self.dit_parallel_config = parallel_config
        self.hf_model_checkpoint_path = "Qwen/Qwen-Image"
        self.hf_text_tokenizer_checkpoint = "Qwen/Qwen2.5-7B-Instruct"
        self.hf_text_encoder_model_checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"

        self.latents_height = 128
        self.latents_width = 128

        # Create submeshes
        submesh_shape = list(mesh_device.shape)
        submesh_shape[parallel_config.cfg_parallel.mesh_axis] //= parallel_config.cfg_parallel.factor
        logger.info(f"Parallel config: {parallel_config}")
        logger.info(f"Original mesh shape: {mesh_device.shape}")
        logger.info(f"Creating submeshes with shape {submesh_shape}")
        self.submesh_devices = self._mesh_device.create_submeshes(ttnn.MeshShape(*submesh_shape))

        self.ccl_managers = [
            CCLManager(submesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
            for submesh_device in self.submesh_devices
        ]
        # Hacky submesh reshapes and assignment to parallelize encoders and VAE
        encoder_device = self.submesh_devices[0]
        self.original_submesh_shape = tuple(encoder_device.shape)
        self.desired_encoder_submesh_shape = tuple(encoder_device.shape)

        if encoder_device.shape[1] != 4:
            # If reshaping, vae_device must be on submesh 0.
            vae_submesh_idx = 0

            cfg_shape = tuple(encoder_device.shape)
            assert cfg_shape[0] * cfg_shape[1] == 4, f"Cannot reshape {cfg_shape} to a 1x4 mesh"
            logger.info(f"Reshaping submesh device 0 from {cfg_shape} to (1, 4) for CLIP")
            self.desired_encoder_submesh_shape = (1, 4)

        else:
            # vae_device can only be on submesh 1 if submesh is not getting reshaped.
            vae_submesh_idx = 1
        vae_device = self.submesh_devices[vae_submesh_idx]

        # Create encoder parallel config
        encoder_parallel_config = EncoderParallelConfig(
            tensor_parallel=ParallelFactor(factor=4, mesh_axis=1)  # 1x4 submesh, parallel on axis 1
        )

        self.encoder_parallel_config = encoder_parallel_config
        self.encoder_device = encoder_device

        vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))
        self.vae_parallel_config = vae_parallel_config
        self.vae_device = vae_device
        self.vae_submesh_idx = vae_submesh_idx

        logger.info("loading models...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_text_tokenizer_checkpoint)
        self._text_encoder = Qwen25VlTokenizerEncoderPair(
            self.hf_text_encoder_model_checkpoint,
            max_sequence_length=512,
            max_batch_size=32,
            device=encoder_device,
            use_torch=False,
        )
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_checkpoint_path, subfolder="scheduler")
        self._torch_vae = AutoencoderKL.from_pretrained(model_checkpoint_path, subfolder="vae")

        self.torch_transformer = TorchQwenImageTransformer2DModel.from_pretrained(
            model_checkpoint_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,  # bfloat16 is the native datatype of the model
        )
        self.torch_transformer.eval()

        assert isinstance(self._tokenizer, AutoTokenizer)
        assert isinstance(self._text_encoder, Qwen25VlTokenizerEncoderPair)
        assert isinstance(self._scheduler, FlowMatchEulerDiscreteScheduler)
        assert isinstance(self._torch_vae, AutoencoderKL)
        assert isinstance(self.torch_transformer, TorchQwenImageTransformer2DModel)

        logger.info("creating TT-NN transformer...")

        assert "Qwen/Qwen-Image" in str(model_checkpoint_path)

        if self.torch_transformer.config.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                self.torch_transformer.config.num_attention_heads,
                self.torch_transformer.config.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        self.transformers = []
        for i, submesh_device in enumerate(self.submesh_devices):
            tt_transformer = QwenImageTransformer(
                patch_size=2,
                in_channels=64,
                num_layers=60,
                attention_head_dim=128,
                num_attention_heads=24,
                joint_attention_dim=3584,
                out_channels=16,
                mesh_device=submesh_device,
                ccl_manager=self.ccl_managers[i],
                parallel_config=self.dit_parallel_config,
                padding_config=padding_config,
            )

            if use_cache:
                cache_path = get_and_create_cache_path(
                    model_name="Qwen/Qwen-Image",
                    subfolder="transformer",
                    parallel_config=self.dit_parallel_config,
                    mesh_shape=tuple(submesh_device.shape),
                    dtype="bf16",
                )
                # create cache if it doesn't exist
                if not cache_dict_exists(cache_path):
                    logger.info(
                        f"Cache does not exist. Creating cache: {cache_path} and loading transformer weights from PyTorch state dict"
                    )
                    tt_transformer.load_state_dict(self.torch_transformer.state_dict())
                    save_cache_dict(tt_transformer.to_cached_state_dict(cache_path), cache_path)
                else:
                    logger.info(f"Loading transformer weights from cache: {cache_path}")
                    tt_transformer.from_cached_state_dict(load_cache_dict(cache_path))
            else:
                logger.info("Loading transformer weights from PyTorch state dict")
                tt_transformer.load_state_dict(self.torch_transformer.state_dict())

            self.transformers.append(tt_transformer)
            ttnn.synchronize_device(submesh_device)

        self._num_channels_latents = self.torch_transformer.config.in_channels
        self._joint_attention_dim = self.torch_transformer.config.joint_attention_dim
        self.patch_size = 2  # SD3.5 uses patch_size of 2

        self._block_out_channels = self._torch_vae.config.block_out_channels
        self._torch_vae_scaling_factor = self._torch_vae.config.scaling_factor
        self._torch_vae_shift_factor = self._torch_vae.config.shift_factor

        self._torch_vae_scale_factor = 2 ** (len(self._block_out_channels) - 1)
        self._image_processor = VaeImageProcessor(vae_scale_factor=self._torch_vae_scale_factor)

        if self.desired_encoder_submesh_shape != self.original_submesh_shape:
            # HACK: reshape submesh device 0 to 1D
            self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))

        self.timing_collector = None  # Set externally when timing is needed

        self._trace = None

        # intermediate buffers for safe tracing
        self._intermediate_noise_list = []
        self._sigma_difference_list = []
        self._vae_input_latents = None

        ttnn.synchronize_device(self.encoder_device)

        self._vae_decoder = QwenImageVaeDecoder(
            base_dim=96,
            z_dim=16,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            temperal_downsample=(False, True, True),
            device=self.vae_device,
            parallel_config=self.vae_parallel_config,
            ccl_manager=self.ccl_managers[self.vae_submesh_idx],
        )

        if self.desired_encoder_submesh_shape != self.original_submesh_shape:
            # HACK: reshape submesh device 0 to 1D
            # If reshaping, vae device is same as encoder device
            self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))

    def prepare(
        self,
        *,
        batch_size: int,
        num_images_per_prompt: int = 1,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 4.5,
        prompt_sequence_length: int = 333,
        spatial_sequence_length: int = 4096,
    ) -> None:
        self._prepared_batch_size = batch_size
        self._prepared_num_images_per_prompt = num_images_per_prompt
        self._prepared_width = width
        self._prepared_height = height
        self._prepared_guidance_scale = guidance_scale
        self._prepared_prompt_sequence_length = prompt_sequence_length

    @staticmethod
    def create_pipeline(
        mesh_device,
        batch_size=1,
        image_w=1024,
        image_h=1024,
        guidance_scale=3.5,
        num_images_per_prompt=1,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
        cfg_config=None,
        sp_config=None,
        tp_config=None,
        num_links=None,
        model_checkpoint_path=f"Qwen/Qwen-Image",
        use_cache=False,
    ):
        # defatult config per mesh shape
        default_config = {
            (2, 4): {"cfg_config": (2, 1), "sp_config": (2, 0), "tp_config": (2, 1), "num_links": 1},
            (4, 8): {"cfg_config": (2, 1), "sp_config": (4, 0), "tp_config": (4, 1), "num_links": 4},
        }

        # get config from user or default if not provided
        cfg_factor, cfg_axis = cfg_config or default_config[tuple(mesh_device.shape)]["cfg_config"]
        sp_factor, sp_axis = sp_config or default_config[tuple(mesh_device.shape)]["sp_config"]
        tp_factor, tp_axis = tp_config or default_config[tuple(mesh_device.shape)]["tp_config"]
        num_links = num_links or default_config[tuple(mesh_device.shape)]["num_links"]

        parallel_config = DiTParallelConfig(
            cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
            tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
            sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
        )

        guidance_cond = 2 if (guidance_scale > 1 and cfg_factor == 1) else 1

        logger.info(f"Mesh device shape: {mesh_device.shape}")
        logger.info(f"Parallel config: {parallel_config}")

        # Create pipeline
        pipeline = QwenImagePipeline(
            mesh_device=mesh_device,
            guidance_cond=guidance_cond,
            parallel_config=parallel_config,
            num_links=num_links,
            height=image_h,
            width=image_w,
            model_checkpoint_path=model_checkpoint_path,
            use_cache=use_cache,
        )

        pipeline.prepare(
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            width=image_w,
            height=image_h,
            guidance_scale=guidance_scale,
            prompt_sequence_length=prompt_sequence_length,
            spatial_sequence_length=spatial_sequence_length,
        )

        return pipeline

    def run_single_prompt(self, prompt, negative_prompt="", num_inference_steps=40, seed=None):
        return self.__call__(
            prompt=[prompt],
            negative_prompt=[negative_prompt or ""],
            num_inference_steps=num_inference_steps,
            seed=seed,
            traced=True,
        )

    def __call__(
        self,
        prompt: list[str],
        negative_prompt: list[str],
        num_inference_steps: int = 40,
        seed: int | None = None,
        traced: bool = False,
    ) -> List[Image.Image]:
        timer = self.timing_collector.reset() if self.timing_collector else None

        with timer.time_section("total") if timer else nullcontext():
            start_time = time.time()

            do_classifier_free_guidance = guidance_scale > 1
            patch_size = 2

            latents_shape = (
                1,
                self.latents_height // self.patch_size,
                self.latents_width // self.patch_size,
                self._num_channels_latents,
            )

            logger.info("encoding prompts...")

            with timer.time_section("total_encoding") if timer else nullcontext():
                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # HACK: reshape submesh device 0 from 2D to 1D
                    self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))
                prompt_encoding_start_time = time.time()
                prompt_embeds = self._encode_prompts(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    prompt_sequence_length=prompt_sequence_length,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # HACK: reshape submesh device 0 from 1D to 2D
                    self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))
                prompt_encoding_end_time = time.time()
                logger.info("preparing timesteps...")

            self._scheduler.set_timesteps(num_inference_steps)
            timesteps = self._scheduler.timesteps

            logger.info("preparing latents...")

            if seed is not None:
                torch.manual_seed(seed)
            latents = torch.randn(latents_shape, dtype=prompt_embeds.dtype)  # .permute([0, 2, 3, 1])
            latents = self.transformers[0].pack_latents(latents)

            timestep = torch.full([1], fill_value=500)

            patched_shape = [[(1, self.latents_height // self.patch_size, self.latents_width // self.patch_size)]] * 1
            txt_seq_lens = [len(prompt_embeds)] * 1
            spatial_rope, prompt_rope = self.torch_transformer.pos_embed.forward(patched_shape, txt_seq_lens, "cpu")

            spatial_rope_cos = spatial_rope.real.repeat_interleave(2, dim=-1)
            spatial_rope_sin = spatial_rope.imag.repeat_interleave(2, dim=-1)
            prompt_rope_cos = prompt_rope.real.repeat_interleave(2, dim=-1)
            prompt_rope_sin = prompt_rope.imag.repeat_interleave(2, dim=-1)

            tt_prompt_embeds_list = []
            tt_latents_step_list = []
            for i, submesh_device in enumerate(self.submesh_devices):
                tt_prompt_embeds = ttnn.from_torch(
                    prompt_embeds[i].unsqueeze(0).unsqueeze(0)
                    if self.dit_parallel_config.cfg_parallel.factor == 2
                    else prompt_embeds,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=[None, None],
                    ),
                )

                shard_latents_dims = [None, None]
                shard_latents_dims[self.dit_parallel_config.sequence_parallel.mesh_axis] = 2
                tt_initial_latents = ttnn.from_torch(
                    latents,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=submesh_device if not traced else None,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        submesh_device,
                        tuple(submesh_device.shape),
                        dims=shard_latents_dims,
                    ),
                )

                if len(self._intermediate_noise_list) <= i:
                    self._intermediate_noise_list.append(
                        ttnn.from_torch(
                            latents,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16,
                            device=submesh_device,
                            mesh_mapper=ttnn.ShardTensor2dMesh(
                                submesh_device,
                                tuple(submesh_device.shape),
                                dims=shard_latents_dims,
                            ),
                        )
                    )
                    self._sigma_difference_list.append(ttnn.clone(self._intermediate_noise_list[-1]))
                if traced:
                    if self._trace is None:
                        # Push inputs to device
                        tt_initial_latents = tt_initial_latents.to(submesh_device)
                        tt_prompt_embeds = tt_prompt_embeds.to(submesh_device)
                    else:
                        # Copy inputs to trace
                        ttnn.copy_host_to_device_tensor(tt_initial_latents, self._trace[i].spatial_input)
                        ttnn.copy_host_to_device_tensor(tt_prompt_embeds, self._trace[i].prompt_input)
                        # Ensure trace inputs are passed to function
                        tt_initial_latents = self._trace[i].spatial_input
                        tt_prompt_embeds = self._trace[i].prompt_input

                tt_prompt_embeds_list.append(tt_prompt_embeds)
                tt_latents_step_list.append(tt_initial_latents)

            logger.info("denoising...")
            denoising_start_time = time.time()

            for i, t in enumerate(tqdm.tqdm(timesteps)):
                with timer.time_step("denoising_step") if timer else nullcontext():
                    sigma_difference = self._scheduler.sigmas[i + 1] - self._scheduler.sigmas[i]

                    tt_timestep_list = []
                    tt_sigma_difference_list = []
                    for submesh_device in self.submesh_devices:
                        tt_timestep = ttnn.full(
                            [1, 1, 1, 1],
                            fill_value=t,
                            layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.float32,
                            device=submesh_device if not traced else None,
                        )
                        tt_timestep_list.append(tt_timestep)

                        tt_sigma_difference_list.append(
                            ttnn.full(
                                tt_latents_step_list[0].shape,
                                fill_value=sigma_difference,
                                layout=ttnn.TILE_LAYOUT,
                                dtype=ttnn.bfloat16,
                                device=None,  # We'll copy to device when needed
                            )
                        )

                    tt_latents_step_list = self._step(
                        timestep=tt_timestep_list,
                        latents=tt_latents_step_list,  # tt_latents,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        prompt_embeds=tt_prompt_embeds_list,
                        spatial_rope_device=(tt_spatial_rope_cos, tt_spatial_rope_sin),
                        prompt_rope_device=(tt_prompt_rope_cos, tt_prompt_rope_sin),
                        guidance_scale=guidance_scale,
                        sigma_difference=tt_sigma_difference_list,
                        prompt_sequence_length=333,
                        spatial_sequence_length=4096,
                        traced=traced,
                    )

            denoising_end_time = time.time()

            logger.info("decoding image...")

            with timer.time_section("vae_decoding") if timer else nullcontext():
                image_decoding_start_time = time.time()
                decoded_output = self._vae_decode(tt_latents_step_list[self.vae_submesh_idx], width, height)
                decoded_output = ttnn.to_torch(ttnn.get_device_tensors(decoded_output)[0]).permute(0, 3, 1, 2)

                # HACK: reshape submesh device 0 from 1D to 2D
                if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                    # If reshaping, vae device is same as encoder device
                    self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))
                # image = self._torch_vae.decoder(tt_latents)
                image = self._image_processor.postprocess(decoded_output, output_type="pt")
                print(f"postprocessed image shape: {image.shape}")
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
        latents: List[ttnn.Tensor],  # device tensor
        timestep: List[ttnn.Tensor],  # host tensor
        prompt_embeds: List[ttnn.Tensor],  # device tensor
        spatial_rope_device: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope_device: tuple[ttnn.Tensor, ttnn.Tensor],
        sigma_difference: List[ttnn.Tensor],  # device tensor
        prompt_sequence_length: int,
        spatial_sequence_length: int,
        traced: bool,
    ) -> List[ttnn.Tensor]:
        def inner(latent, prompt, timestep, spatial_rope, prompt_rope, cfg_index):
            if do_classifier_free_guidance and not self.dit_parallel_config.cfg_parallel.factor > 1:
                latent_model_input = ttnn.concat([latent, latent])
            else:
                latent_model_input = latent

            return self.transformers[cfg_index](
                spatial=latent_model_input,
                prompt=prompt,
                timestep=timestep,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
                prompt_sequence_length=prompt_sequence_length,
            )

        if traced and self._trace is None:
            print(f"Tracing...")
            self._trace = [None for _ in self.submesh_devices]
            for submesh_id, submesh_device in enumerate(self.submesh_devices):
                print(f"Tracing submesh {submesh_id}")
                latent_device = latents[submesh_id]  # already on device
                prompt_device = prompt_embeds[submesh_id]  # already on device
                timestep_device = timestep[submesh_id].to(submesh_device)

                print("compile run")
                pred = inner(
                    latent_device,
                    prompt_device,
                    timestep_device,
                    spatial_rope_device,
                    prompt_rope_device,
                    submesh_id,
                )

                if submesh_id == self.vae_submesh_idx:
                    print("Initializing VAE buffers for safe tracing...")
                    self._vae_decode(latent_device, self._prepared_width, self._prepared_height)
                    if self.desired_encoder_submesh_shape != self.original_submesh_shape:
                        self.encoder_device.reshape(ttnn.MeshShape(*self.original_submesh_shape))

                ttnn.synchronize_device(submesh_device)
                # ttnn.synchronize_device(self.submesh_devices[0])
                # ttnn.synchronize_device(self.submesh_devices[1])

                print("begin trace capture")
                trace_id = ttnn.begin_trace_capture(submesh_device, cq_id=0)
                pred = inner(
                    latent_device,
                    prompt_device,
                    timestep_device,
                    spatial_rope_device,
                    prompt_rope_device,
                    submesh_id,
                )
                ttnn.end_trace_capture(submesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(submesh_device)
                # ttnn.synchronize_device(self.submesh_devices[0])
                # ttnn.synchronize_device(self.submesh_devices[1])
                print("done sync after trace capture")

                self._trace[submesh_id] = PipelineTrace(
                    spatial_input=latent_device,
                    prompt_input=prompt_device,
                    spatial_rope_input=(spatial_rope_device, prompt_rope_device),
                    timestep_input=timestep_device,
                    latents_output=pred,
                    tid=trace_id,
                )

        noise_pred_list = []
        if traced:
            for submesh_id, submesh_device in enumerate(self.submesh_devices):
                ttnn.copy_host_to_device_tensor(timestep[submesh_id], self._trace[submesh_id].timestep_input)
                ttnn.execute_trace(submesh_device, self._trace[submesh_id].tid, cq_id=0, blocking=False)
                noise_pred_list.append(self._trace[submesh_id].latents_output)
        else:
            for submesh_id, submesh_device in enumerate(self.submesh_devices):
                noise_pred = inner(
                    latents[submesh_id],
                    prompt_embeds[submesh_id],
                    timestep[submesh_id],
                    spatial_rope_device,
                    prompt_rope_device,
                    submesh_id,
                )
                noise_pred_list.append(noise_pred)

        if do_classifier_free_guidance:
            if not self.dit_parallel_config.cfg_parallel.factor > 1:
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
                shard_latents_dims[self.dit_parallel_config.sequence_parallel.mesh_axis] = 2
                noise_pred_list[0] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=None,  # self.submesh_devices[0],
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.submesh_devices[0],
                        tuple(self.submesh_devices[0].shape),
                        dims=shard_latents_dims,
                    ),
                )

                noise_pred_list[1] = ttnn.from_torch(
                    torch_noise_pred,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    device=None,  # self.submesh_devices[1],
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.submesh_devices[1],
                        tuple(self.submesh_devices[1].shape),
                        dims=shard_latents_dims,
                    ),
                )

        for submesh_id, submesh_device in enumerate(self.submesh_devices):
            ttnn.copy_host_to_device_tensor(noise_pred_list[submesh_id], self._intermediate_noise_list[submesh_id])
            ttnn.copy_host_to_device_tensor(sigma_difference[submesh_id], self._sigma_difference_list[submesh_id])
            ttnn.multiply_(
                self._sigma_difference_list[submesh_id], self._intermediate_noise_list[submesh_id]
            )  # This allocates during trace. Need to investigate
            ttnn.add_(latents[submesh_id], self._sigma_difference_list[submesh_id])

        return latents

    def _vae_decode(self, tt_latents, width, height):
        ttnn.synchronize_device(self.vae_device)

        tt_latents = self.ccl_managers[self.vae_submesh_idx].all_gather_persistent_buffer(
            tt_latents, dim=2, mesh_axis=self.dit_parallel_config.sequence_parallel.mesh_axis
        )

        torch_latents = ttnn.to_torch(ttnn.get_device_tensors(tt_latents)[0])
        torch_latents = (torch_latents / self._torch_vae_scaling_factor) + self._torch_vae_shift_factor
        torch_latents = self.transformers[0].unpack_latents(
            torch_latents,
            width=width // self._torch_vae_scale_factor,
            height=height // self._torch_vae_scale_factor,
        )

        if self.desired_encoder_submesh_shape != self.original_submesh_shape:
            # HACK: reshape submesh device 0 from 2D to 1D
            # If reshaping, vae device is same as encoder device
            self.encoder_device.reshape(ttnn.MeshShape(*self.desired_encoder_submesh_shape))

        tt_latents = ttnn.from_torch(
            torch_latents,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=None,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.vae_device),
        )

        if self._vae_input_latents is None:
            self._vae_input_latents = tt_latents.to(self.vae_device)
        else:
            ttnn.copy_host_to_device_tensor(tt_latents, self._vae_input_latents)

        decoded_output = self._vae_decoder(self._vae_input_latents)
        return decoded_output

    def _encode_prompts(
        self,
        *,
        prompt: list[str],
        negative_prompt: list[str],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        clip_skip: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timer = self.timing_collector

        tokenizer_max_length = self._tokenizer_1.model_max_length

        with timer.time_section("clip_encoding") if timer else nullcontext():
            prompt_embed = _get_clip_prompt_embeds(
                prompt=prompt_1,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_1,
                text_encoder=self._text_encoder_1,
                tokenizer_max_length=tokenizer_max_length,
                ttnn_device=self.encoder_device,
                encoder_parallel_config=self.encoder_parallel_config,
                clip_skip=clip_skip,
            )

            prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_2,
                text_encoder=self._text_encoder_2,
                tokenizer_max_length=tokenizer_max_length,
                ttnn_device=self.encoder_device,
                encoder_parallel_config=self.encoder_parallel_config,
                clip_skip=clip_skip,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

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
                encoder_parallel_config=self.encoder_parallel_config,
                ttnn_device=self.encoder_device,
                clip_skip=clip_skip,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = _get_clip_prompt_embeds(
                prompt=negative_prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                tokenizer=self._tokenizer_2,
                text_encoder=self._text_encoder_2,
                tokenizer_max_length=tokenizer_max_length,
                encoder_parallel_config=self.encoder_parallel_config,
                ttnn_device=self.encoder_device,
                clip_skip=clip_skip,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

        negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, prompt_embeds], dim=0)

        return prompt_embeds

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from dataclasses import dataclass

from diffusers import StableDiffusionXLPipeline
from loguru import logger
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_autoencoder_kl import TtAutoencoderKL
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.experimental.stable_diffusion_xl_base.refiner.tt.model_configs import (
    ModelOptimisations,
    RefinerModelOptimisations,
)
from transformers import CLIPTextModelWithProjection, CLIPTextModel
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    create_tt_clip_text_encoders,
    warmup_tt_text_encoders,
    batch_encode_prompt_on_device,
    retrieve_timesteps,
    run_tt_image_gen,
)
from models.common.utility_functions import profiler


@dataclass
class TtSDXLPipelineConfig:
    num_inference_steps: int
    guidance_scale: float
    is_galaxy: bool
    capture_trace: bool = True
    vae_on_device: bool = True
    encoders_on_device: bool = True
    use_cfg_parallel: bool = False
    crop_coords_top_left: tuple = (0, 0)
    guidance_rescale: float = 0.0
    _debug_mode: bool = False
    _torch_pipeline_type = StableDiffusionXLPipeline

    @property
    def pipeline_type(self):
        return self._torch_pipeline_type


class TtSDXLPipeline(LightweightModule):
    # TtSDXLPipeline is a wrapper class for the Stable Diffusion XL pipeline.
    # Class contains encoder, scheduler, unet and vae decoder modules.
    # Additionally, methods for text encoding, image generation, and input preparation are provided.
    # Compile and trace methods are also included for model optimization.
    # The class will fallback on the CPU implementetion for the non critical components.

    def __init__(self, ttnn_device, torch_pipeline, pipeline_config: TtSDXLPipelineConfig):
        super().__init__()

        assert isinstance(
            torch_pipeline, pipeline_config.pipeline_type
        ), f"torch_pipeline must be an instance of {pipeline_config.pipeline_type.__name__}, but got {type(torch_pipeline).__name__}"
        assert (
            isinstance(torch_pipeline.text_encoder, CLIPTextModel) or torch_pipeline.text_encoder is None
        ), "pipeline.text_encoder is not a CLIPTextModel or None"
        assert isinstance(
            torch_pipeline.text_encoder_2, CLIPTextModelWithProjection
        ), "pipeline.text_encoder_2 is not a CLIPTextModelWithProjection"
        assert not (pipeline_config._debug_mode and pipeline_config.capture_trace), (
            "`_debug_mode` and `capture_trace` cannot both be enabled at the same time. "
            "Please set only one of them to True."
        )

        self.ttnn_device = ttnn_device
        self.cpu_device = "cpu"
        self.batch_size = (
            list(self.ttnn_device.shape)[1] if pipeline_config.use_cfg_parallel else ttnn_device.get_num_devices()
        )
        self.torch_pipeline = torch_pipeline
        self.pipeline_config = pipeline_config
        self._reset_num_inference_steps()

        # Validate config parameters once at initialization
        self.__validate_config()

        self.encoders_compiled = False
        self.image_processing_compiled = False
        self.allocated_device_tensors = False
        self.generated_input_tensors = False

        if pipeline_config.is_galaxy:
            logger.info("Setting TT_MM_THROTTLE_PERF for Galaxy")
            os.environ["TT_MM_THROTTLE_PERF"] = "5"
            # assert (
            #     os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] == "7,7"
            # ), "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE is not set to 7,7, and it needs to be set for Galaxy"

        logger.info("Loading TT components...")
        self.__load_tt_components(pipeline_config)
        logger.info("TT components loaded")

        self.scaling_factor = ttnn.from_torch(
            torch.Tensor([self.torch_pipeline.vae.config.scaling_factor]),
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )

        self.guidance_scale = ttnn.from_torch(
            torch.Tensor([self.pipeline_config.guidance_scale]),
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )

        self.guidance_rescale = ttnn.from_torch(
            torch.Tensor([self.pipeline_config.guidance_rescale]),
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )

        self.one_minus_guidance_rescale = ttnn.from_torch(
            torch.Tensor([1.0 - self.pipeline_config.guidance_rescale]),
            dtype=ttnn.bfloat16,
            device=self.ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )

        compute_grid_size = self.ttnn_device.compute_with_storage_grid_size()
        ccl_sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        self.ag_semaphores = [ttnn.create_global_semaphore(ttnn_device, ccl_sub_device_crs, 0) for _ in range(2)]

        self.ag_persistent_buffer = ttnn.from_torch(
            torch.zeros((2, 1, 16384, 32)),
            device=ttnn_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(ttnn_device),
            dtype=ttnn.bfloat16,
        )

        self.num_in_channels_unet = 4
        # Hardcoded input tensor parameters

        # Tensor shapes
        B, C, H, W = 1, self.num_in_channels_unet, 128, 128
        self.tt_latents_shape = [B, C, H, W]

    def set_num_inference_steps(self, num_inference_steps: int):
        # When changing num_inference_steps, the timesteps and latents need to be recreated.
        self.pipeline_config.num_inference_steps = num_inference_steps
        self.generated_input_tensors = False

    def _reset_num_inference_steps(self):
        self.num_inference_steps = self.pipeline_config.num_inference_steps

    def set_guidance_scale(self, guidance_scale: float):
        self.pipeline_config.guidance_scale = guidance_scale
        host_guidance_scale = ttnn.from_torch(
            torch.Tensor([self.pipeline_config.guidance_scale]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )
        ttnn.copy_host_to_device_tensor(host_guidance_scale, self.guidance_scale)

    def set_guidance_rescale(self, guidance_rescale: float):
        self.pipeline_config.guidance_rescale = guidance_rescale
        host_guidance_rescale = ttnn.from_torch(
            torch.Tensor([self.pipeline_config.guidance_rescale]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )
        host_one_minus_guidance_rescale = ttnn.from_torch(
            torch.Tensor([1.0 - self.pipeline_config.guidance_rescale]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )
        ttnn.copy_host_to_device_tensor(host_guidance_rescale, self.guidance_rescale)
        ttnn.copy_host_to_device_tensor(host_one_minus_guidance_rescale, self.one_minus_guidance_rescale)

    def set_crop_coords_top_left(self, crop_coords_top_left: tuple):
        self.pipeline_config.crop_coords_top_left = crop_coords_top_left

    def __validate_config(self):
        """
        Validates pipeline configuration parameters.
        Called once during initialization.
        """
        # Validate guidance_rescale
        guidance_rescale = self.pipeline_config.guidance_rescale
        assert isinstance(
            guidance_rescale, (int, float)
        ), f"`guidance_rescale` must be numeric but is {type(guidance_rescale)}"
        assert 0.0 <= guidance_rescale <= 1.0, (
            f"`guidance_rescale` must be in range [0.0, 1.0] but is {guidance_rescale}. "
            "Typical values are 0.0 (no rescale) to 0.7."
        )

        # Validate crop_coords_top_left
        crop_coords_top_left = self.pipeline_config.crop_coords_top_left
        assert isinstance(
            crop_coords_top_left, (tuple, list)
        ), f"`crop_coords_top_left` must be a tuple or list but is {type(crop_coords_top_left)}"
        assert (
            len(crop_coords_top_left) == 2
        ), f"`crop_coords_top_left` must have exactly 2 elements (y, x) but has {len(crop_coords_top_left)}"
        y, x = crop_coords_top_left
        assert isinstance(y, (int, float)) and isinstance(
            x, (int, float)
        ), f"`crop_coords_top_left` values must be numeric but got ({type(y)}, {type(x)})"
        assert (
            y >= 0 and x >= 0
        ), f"`crop_coords_top_left` = ({y}, {x}) has negative values. Both coordinates must be non-negative."
        assert (
            y <= 2048 and x <= 2048
        ), f"`crop_coords_top_left` = ({y}, {x}) is unreasonably large. Coordinates should typically be within [0, 1024] for SDXL."

    def _validate_timesteps_sigmas(self, timesteps=None, sigmas=None):
        """
        Validates timesteps and sigmas parameters.
        """
        assert not (
            timesteps is not None and sigmas is not None
        ), "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"

        # Validate timesteps
        if timesteps is not None:
            assert isinstance(timesteps, (list, tuple)), f"`timesteps` must be a list or tuple but is {type(timesteps)}"
            assert len(timesteps) > 0, "`timesteps` cannot be empty"
            assert (
                len(timesteps) <= 1000
            ), f"`timesteps` has {len(timesteps)} steps which is too many. Maximum recommended is 1000 steps."

            # Check all timesteps are non-negative and within reasonable range
            num_train_timesteps = getattr(self.torch_pipeline.scheduler, "num_train_timesteps", 1000)
            for i, ts in enumerate(timesteps):
                assert isinstance(ts, (int, float)), f"`timesteps[{i}]` must be numeric but is {type(ts)}"
                assert ts >= 0, f"`timesteps[{i}]` = {ts} is negative. All timesteps must be non-negative."
                assert (
                    ts <= num_train_timesteps
                ), f"`timesteps[{i}]` = {ts} exceeds num_train_timesteps ({num_train_timesteps})"

        # Validate sigmas
        if sigmas is not None:
            assert isinstance(sigmas, (list, tuple)), f"`sigmas` must be a list or tuple but is {type(sigmas)}"
            assert len(sigmas) > 0, "`sigmas` cannot be empty"
            assert (
                len(sigmas) <= 1000
            ), f"`sigmas` has {len(sigmas)} steps which is too many. Maximum recommended is 1000 steps."

            # Check all sigmas are non-negative
            for i, sigma in enumerate(sigmas):
                assert isinstance(sigma, (int, float)), f"`sigmas[{i}]` must be numeric but is {type(sigma)}"
                assert sigma >= 0, f"`sigmas[{i}]` = {sigma} is negative. All sigmas must be non-negative."
                assert (
                    sigma <= 1000
                ), f"`sigmas[{i}]` = {sigma} is unreasonably large. Sigmas should typically be in range [0, ~20]."

    def check_inputs(
        self,
        prompts,
        negative_prompts,
        prompt_2=None,
        negative_prompt_2=None,
    ):
        """
        Validates prompt-related input parameters before generation.
        Raises ValueError if any parameter is invalid.
        """
        assert prompts is not None, "Provide `prompts`. Cannot be None."

        if prompt_2 is not None:
            assert isinstance(
                prompt_2, (str, list)
            ), f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}"

        # Validate negative prompts
        if negative_prompts is not None:
            if isinstance(negative_prompts, list):
                assert len(negative_prompts) == len(prompts), (
                    f"`negative_prompts` list length ({len(negative_prompts)}) must match "
                    f"`prompts` list length ({len(prompts)})"
                )
            else:
                assert isinstance(
                    negative_prompts, str
                ), f"`negative_prompts` has to be of type `str` or `list` but is {type(negative_prompts)}"

        if negative_prompt_2 is not None:
            assert isinstance(
                negative_prompt_2, (str, list)
            ), f"`negative_prompt_2` has to be of type `str` or `list` but is {type(negative_prompt_2)}"

    def compile_text_encoding(self):
        # Compilation of text encoders on the device.
        if not self.encoders_compiled:
            assert self.pipeline_config.encoders_on_device, "Host text encoders are used; compile is not needed"
            assert self.tt_text_encoder_2 is not None, "Text encoder is not loaded on the device"

            warmup_tt_text_encoders(
                self.tt_text_encoder,
                self.tt_text_encoder_2,
                self.torch_pipeline.tokenizer,
                self.torch_pipeline.tokenizer_2,
                self.ttnn_device,
                self.batch_size,
            )

            self.encoders_compiled = True

    def compile_image_processing(self):
        # Compile/trace run for denoising loop and vae decoder.
        if not self.image_processing_compiled:
            assert self.allocated_device_tensors, "Input tensors are not allocated"

            profiler.start("warmup_run")
            logger.info("Performing warmup run on denoising, to make use of program caching in actual inference...")

            _, _, _, self.output_shape, _ = run_tt_image_gen(
                self.ttnn_device,
                self.tt_unet,
                self.tt_scheduler,
                self.tt_latents_device,
                self.tt_prompt_embeds_device,
                self.tt_time_ids_device,
                self.tt_text_embeds_device,
                1,
                self.extra_step_kwargs,
                self.guidance_scale,
                self.scaling_factor,
                self.tt_latents_shape,
                self.tt_vae if self.pipeline_config.vae_on_device else self.torch_pipeline.vae,
                self.batch_size,
                self.ag_persistent_buffer,
                self.ag_semaphores,
                capture_trace=False,
                use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
                guidance_rescale=self.guidance_rescale,
                one_minus_guidance_rescale=self.one_minus_guidance_rescale,
            )
            ttnn.synchronize_device(self.ttnn_device)
            profiler.end("warmup_run")

            if self.pipeline_config.capture_trace:
                self.__trace_image_processing()

            self._reset_num_inference_steps()
            self.image_processing_compiled = True

    def encode_prompts(self, prompts, negative_prompts, prompt_2=None, negative_prompt_2=None):
        # Encode prompts using the text encoders.

        # Validate prompt inputs at the beginning of execution
        self.check_inputs(
            prompts=prompts,
            negative_prompts=negative_prompts,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
        )

        if self.pipeline_config.encoders_on_device:
            # Prompt encode on device
            assert self.encoders_compiled, "Text encoders are not compiled"
            logger.info("Encoding prompts on device...")
            profiler.start("encode_prompts")

            all_embeds = []
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i : i + self.batch_size]
                current_negative_prompts = (
                    negative_prompts[i : i + self.batch_size]
                    if isinstance(negative_prompts, list)
                    else negative_prompts
                )
                batch_prompts_2 = prompt_2[i : i + self.batch_size] if isinstance(prompt_2, list) else prompt_2
                current_negative_prompts_2 = (
                    negative_prompt_2[i : i + self.batch_size]
                    if isinstance(negative_prompt_2, list)
                    else negative_prompt_2
                )
                batch_embeds = batch_encode_prompt_on_device(
                    self.torch_pipeline,
                    self.tt_text_encoder,
                    self.tt_text_encoder_2,
                    self.ttnn_device,
                    prompt=batch_prompts,  # Pass the entire batch
                    prompt_2=batch_prompts_2,
                    device=self.cpu_device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=current_negative_prompts,
                    negative_prompt_2=current_negative_prompts_2,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    negative_pooled_prompt_embeds=None,
                    lora_scale=None,
                    clip_skip=None,
                    use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
                )
                # batch_encode_prompt_on_device returns a single tuple of 4 tensors,
                # but we need individual tuples for each prompt
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = batch_embeds
                # Split the tensors by batch dimension and create individual tuples
                for j in range(len(batch_prompts)):
                    all_embeds.append(
                        (
                            prompt_embeds[j : j + 1],  # Keep batch dimension
                            negative_prompt_embeds[j : j + 1] if negative_prompt_embeds is not None else None,
                            pooled_prompt_embeds[j : j + 1],
                            negative_pooled_prompt_embeds[j : j + 1]
                            if negative_pooled_prompt_embeds is not None
                            else None,
                        )
                    )
        else:
            # Prompt encode on host
            logger.info("Encoding prompts on host...")

            # batched impl of host encoding
            profiler.start("encode_prompts")
            all_embeds = self.torch_pipeline.encode_prompt(
                prompt=prompts,  # Pass the entire list at once
                prompt_2=prompt_2,
                device=self.cpu_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompts,
                negative_prompt_2=negative_prompt_2,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=None,
                clip_skip=None,
            )
            (
                prompt_embeds_batch,
                negative_prompt_embeds_batch,
                pooled_prompt_embeds_batch,
                negative_pooled_prompt_embeds_batch,
            ) = all_embeds
            all_embeds = list(
                zip(
                    torch.split(prompt_embeds_batch, 1, dim=0),
                    torch.split(negative_prompt_embeds_batch, 1, dim=0),
                    torch.split(pooled_prompt_embeds_batch, 1, dim=0),
                    torch.split(negative_pooled_prompt_embeds_batch, 1, dim=0),
                )
            )
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = zip(*all_embeds)

        all_prompt_embeds_torch = torch.cat(
            [torch.stack(negative_prompt_embeds, dim=0), torch.stack(prompt_embeds, dim=0)], dim=1
        )
        torch_add_text_embeds = torch.cat(
            [torch.stack(negative_pooled_prompt_embeds, dim=0), torch.stack(pooled_prompt_embeds, dim=0)], dim=1
        )

        profiler.end("encode_prompts")
        logger.info(f"Encoded prompts")
        return (
            all_prompt_embeds_torch,
            torch_add_text_embeds,
        )

    def generate_input_tensors(
        self,
        all_prompt_embeds_torch,
        torch_add_text_embeds,
        start_latent_seed=None,
        fixed_seed_for_batch=False,
        timesteps=None,
        sigmas=None,
    ):
        # Generate user input tensors for the TT model.

        # Validate timesteps/sigmas at the beginning if custom values are provided
        if timesteps is not None or sigmas is not None:
            self._validate_timesteps_sigmas(timesteps, sigmas)

        logger.info("Generating input tensors...")
        profiler.start("prepare_latents")

        self._prepare_timesteps(timesteps, sigmas)

        num_channels_latents = self.torch_pipeline.unet.config.in_channels
        height = width = 1024
        assert (
            num_channels_latents == self.num_in_channels_unet
        ), f"num_channels_latents is {num_channels_latents}, but it should be 4"
        assert start_latent_seed is None or isinstance(
            start_latent_seed, int
        ), "start_latent_seed must be an integer or None"

        latents_list = []
        for index in range(self.batch_size):
            if start_latent_seed is not None:
                torch.manual_seed(start_latent_seed if fixed_seed_for_batch else start_latent_seed + index)
            latents = self.torch_pipeline.prepare_latents(
                1,
                num_channels_latents,
                height,
                width,
                all_prompt_embeds_torch.dtype,
                self.cpu_device,
                None,
                None,
            )
            B, C, H, W = latents.shape  # 1, 4, 128, 128
            latents = torch.permute(latents, (0, 2, 3, 1))  # [1, H, W, C]
            latents = latents.reshape(B, 1, H * W, C)  # [1, 1, H*W, C]
            latents_list.append(latents)
        tt_latents = torch.cat(latents_list, dim=0)  # [batch_size, 1, H*W, C]

        self.extra_step_kwargs = self.torch_pipeline.prepare_extra_step_kwargs(None, 0.0)
        text_encoder_projection_dim = self.torch_pipeline.text_encoder_2.config.projection_dim
        assert (
            text_encoder_projection_dim == 1280
        ), f"text_encoder_projection_dim is {text_encoder_projection_dim}, but it should be 1280"

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = self.pipeline_config.crop_coords_top_left
        add_time_ids = self.torch_pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=all_prompt_embeds_torch.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids
        torch_add_time_ids = torch.stack([negative_add_time_ids.squeeze(0), add_time_ids.squeeze(0)], dim=0)

        tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.__create_user_tensors(
            latents=tt_latents,
            all_prompt_embeds_torch=all_prompt_embeds_torch,
            torch_add_text_embeds=torch_add_text_embeds,
        )

        self.__allocate_device_tensors(
            tt_latents=tt_latents,
            tt_prompt_embeds=tt_prompt_embeds,
            tt_text_embeds=tt_add_text_embeds,
            tt_time_ids=torch_add_time_ids,
        )
        ttnn.synchronize_device(self.ttnn_device)
        profiler.end("prepare_latents")
        logger.info("Input tensors generated")

        self.generated_input_tensors = True
        return tt_latents, tt_prompt_embeds, tt_add_text_embeds

    def prepare_input_tensors(self, host_tensors):
        # Tensor device transfer for the current users.

        assert self.allocated_device_tensors, "Device tensors are not allocated"

        logger.info("Preparing input tensors for TT model...")
        profiler.start("prepare_input_tensors")
        device_tensors = [self.tt_latents_device, self.tt_prompt_embeds_device, self.tt_text_embeds_device]

        for host_tensor, device_tensor in zip(host_tensors, device_tensors):
            ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

        ttnn.synchronize_device(self.ttnn_device)
        profiler.end("prepare_input_tensors")

    def generate_images(self):
        # SDXL inference run.
        assert self.image_processing_compiled, "Image processing is not compiled"
        assert self.generated_input_tensors, "Input tensors are not re/generated"

        logger.info("Generating images...")
        imgs, self.tid, self.output_device, self.output_shape, self.tid_vae = run_tt_image_gen(
            self.ttnn_device,
            self.tt_unet,
            self.tt_scheduler,
            self.tt_latents_device,
            self.tt_prompt_embeds_device,
            self.tt_time_ids_device,
            self.tt_text_embeds_device,
            self.num_inference_steps,
            self.extra_step_kwargs,
            self.guidance_scale,
            self.scaling_factor,
            self.tt_latents_shape,
            self.tt_vae if self.pipeline_config.vae_on_device else self.torch_pipeline.vae,
            self.batch_size,
            self.ag_persistent_buffer,
            self.ag_semaphores,
            tid=self.tid if hasattr(self, "tid") else None,
            output_device=self.output_device if hasattr(self, "output_device") else None,
            output_shape=self.output_shape,
            tid_vae=self.tid_vae if hasattr(self, "tid_vae") else None,
            use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
            guidance_rescale=self.guidance_rescale,
            one_minus_guidance_rescale=self.one_minus_guidance_rescale,
        )
        self._reset_num_inference_steps()
        return imgs

    def _prepare_timesteps(self, timesteps=None, sigmas=None):
        # Helper method for timestep preparation.

        self.ttnn_timesteps, self.num_inference_steps = retrieve_timesteps(
            self.torch_pipeline.scheduler, self.pipeline_config.num_inference_steps, self.cpu_device, timesteps, sigmas
        )

    def __load_tt_components(self, pipeline_config):
        # Method for instantiating TT components based on the torch pipeline.
        # Included components are TtUNet2DConditionModel, TtAutoencoderKL, and TtEulerDiscreteScheduler.
        # TODO: move encoder here

        profiler.start("load_tt_componenets")
        with ttnn.distribute(ttnn.ReplicateTensorToMesh(self.ttnn_device)):
            # 2. Load tt_unet, tt_vae and tt_scheduler
            self.tt_unet_model_config = (
                ModelOptimisations()
                if not self.torch_pipeline.unet.state_dict()["conv_in.weight"].shape[0] == 384
                else RefinerModelOptimisations()
            )
            self.tt_vae_model_config = ModelOptimisations()
            self.tt_unet = TtUNet2DConditionModel(
                self.ttnn_device,
                self.torch_pipeline.unet.state_dict(),
                "unet",
                model_config=self.tt_unet_model_config,
                debug_mode=pipeline_config._debug_mode,
            )
            self.tt_vae = (
                TtAutoencoderKL(
                    self.ttnn_device,
                    self.torch_pipeline.vae.state_dict(),
                    self.tt_vae_model_config,
                    debug_mode=pipeline_config._debug_mode,
                )
                if pipeline_config.vae_on_device
                else None
            )
            self.tt_scheduler = TtEulerDiscreteScheduler(
                self.ttnn_device,
                self.torch_pipeline.scheduler.config.num_train_timesteps,
                self.torch_pipeline.scheduler.config.beta_start,
                self.torch_pipeline.scheduler.config.beta_end,
                self.torch_pipeline.scheduler.config.beta_schedule,
                self.torch_pipeline.scheduler.config.trained_betas,
                self.torch_pipeline.scheduler.config.prediction_type,
                self.torch_pipeline.scheduler.config.interpolation_type,
                self.torch_pipeline.scheduler.config.use_karras_sigmas,
                self.torch_pipeline.scheduler.config.use_exponential_sigmas,
                self.torch_pipeline.scheduler.config.use_beta_sigmas,
                self.torch_pipeline.scheduler.config.sigma_min,
                self.torch_pipeline.scheduler.config.sigma_max,
                self.torch_pipeline.scheduler.config.timestep_spacing,
                self.torch_pipeline.scheduler.config.timestep_type,
                self.torch_pipeline.scheduler.config.steps_offset,
                self.torch_pipeline.scheduler.config.rescale_betas_zero_snr,
                self.torch_pipeline.scheduler.config.final_sigmas_type,
            )
        self.torch_pipeline.scheduler = self.tt_scheduler

        if pipeline_config.encoders_on_device:
            self.tt_text_encoder, self.tt_text_encoder_2 = create_tt_clip_text_encoders(
                self.torch_pipeline, self.ttnn_device
            )
        else:
            self.tt_text_encoder, self.tt_text_encoder_2 = None, None

        ttnn.synchronize_device(self.ttnn_device)
        profiler.end("load_tt_componenets")

    def __allocate_device_tensors(self, tt_latents, tt_prompt_embeds, tt_text_embeds, tt_time_ids):
        # Allocation of device tensors for the input data.
        if not self.allocated_device_tensors:
            profiler.start("allocate_input_tensors")

            self.tt_latents_device = ttnn.allocate_tensor_on_device(
                tt_latents.shape,
                tt_latents.dtype,
                tt_latents.layout,
                self.ttnn_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

            self.tt_prompt_embeds_device = ttnn.allocate_tensor_on_device(
                tt_prompt_embeds[0].shape,
                tt_prompt_embeds[0].dtype,
                tt_prompt_embeds[0].layout,
                self.ttnn_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

            self.tt_text_embeds_device = ttnn.allocate_tensor_on_device(
                tt_text_embeds[0].shape,
                tt_text_embeds[0].dtype,
                tt_text_embeds[0].layout,
                self.ttnn_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

            self.tt_time_ids_device = ttnn.from_torch(
                tt_time_ids,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=self.ttnn_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.ttnn_device, list(self.ttnn_device.shape), dims=(0, None)),
            )
            self.tt_time_ids_device = ttnn.squeeze(self.tt_time_ids_device, dim=0)
            ttnn.synchronize_device(self.ttnn_device)
            profiler.end("prepare_input_tensors")

            self.allocated_device_tensors = True
        else:
            tt_time_ids_host = ttnn.from_torch(
                tt_time_ids,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.ttnn_device, list(self.ttnn_device.shape), dims=(0, None)),
            )
            tt_time_ids_host = ttnn.squeeze(tt_time_ids_host, dim=0)

            ttnn.copy_host_to_device_tensor(tt_time_ids_host, self.tt_time_ids_device)

    def __create_user_tensors(self, latents, all_prompt_embeds_torch, torch_add_text_embeds):
        # Instantiation of user host input tensors for the TT model.

        profiler.start("create_user_tensors")
        tt_latents = ttnn.from_torch(
            latents,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.ttnn_device, list(self.ttnn_device.shape), dims=(None, 0)),
        )

        tt_prompt_embeds = ttnn.from_torch(
            all_prompt_embeds_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.ttnn_device, list(self.ttnn_device.shape), dims=(1, 0)),
        )

        tt_add_text_embeds = ttnn.from_torch(
            torch_add_text_embeds,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.ttnn_device, list(self.ttnn_device.shape), dims=(1, 0)),
        )
        ttnn.synchronize_device(self.ttnn_device)
        profiler.end("create_user_tensors")
        return tt_latents, tt_prompt_embeds, tt_add_text_embeds

    def __trace_image_processing(self):
        # Helper method for image processing trace capture.

        self.__release_trace()

        logger.info("Capturing model trace...")
        profiler.start("capture_model_trace")

        _, self.tid, self.output_device, self.output_shape, self.tid_vae = run_tt_image_gen(
            self.ttnn_device,
            self.tt_unet,
            self.tt_scheduler,
            self.tt_latents_device,
            self.tt_prompt_embeds_device,
            self.tt_time_ids_device,
            self.tt_text_embeds_device,
            1,
            self.extra_step_kwargs,
            self.guidance_scale,
            self.scaling_factor,
            self.tt_latents_shape,
            self.tt_vae if self.pipeline_config.vae_on_device else self.torch_pipeline.vae,
            self.batch_size,
            self.ag_persistent_buffer,
            self.ag_semaphores,
            capture_trace=True,
            use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
            guidance_rescale=self.guidance_rescale,
            one_minus_guidance_rescale=self.one_minus_guidance_rescale,
        )
        ttnn.synchronize_device(self.ttnn_device)
        profiler.end("capture_model_trace")

    def __release_trace(self):
        # Helper method for trace release.
        if self.pipeline_config.capture_trace and hasattr(self, "tid") and self.tid is not None:
            ttnn.release_trace(self.ttnn_device, self.tid)
            delattr(self, "tid")

        if self.pipeline_config.vae_on_device and hasattr(self, "tid_vae") and self.tid_vae is not None:
            ttnn.release_trace(self.ttnn_device, self.tid_vae)
            delattr(self, "tid_vae")

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use individual methods for text encoding and image generation.")

    def __del__(self):
        self.__release_trace()

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
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from transformers import CLIPTextModelWithProjection, CLIPTextModel
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    create_tt_clip_text_encoders,
    warmup_tt_text_encoders,
    batch_encode_prompt_on_device,
    retrieve_timesteps,
    run_tt_image_gen,
)
from models.utility_functions import profiler


@dataclass
class TtSDXLPipelineConfig:
    num_inference_steps: int
    guidance_scale: float
    is_galaxy: bool
    capture_trace: bool = True
    vae_on_device: bool = True
    encoders_on_device: bool = True


class TtSDXLPipeline(LightweightModule):
    # TtSDXLPipeline is a wrapper class for the Stable Diffusion XL pipeline.
    # Class contains encoder, scheduler, unet and vae decoder modules.
    # Additionally, methods for text encoding, image generation, and input preparation are provided.
    # Compile and trace methods are also included for model optimization.
    # The class will fallback on the CPU implementetion for the non critical components.

    def __init__(self, ttnn_device, torch_pipeline, pipeline_config: TtSDXLPipelineConfig):
        super().__init__()

        assert isinstance(
            torch_pipeline, StableDiffusionXLPipeline
        ), "torch_pipeline must be an instance of StableDiffusionXLPipeline"
        assert isinstance(torch_pipeline.text_encoder, CLIPTextModel), "pipeline.text_encoder is not a CLIPTextModel"
        assert isinstance(
            torch_pipeline.text_encoder_2, CLIPTextModelWithProjection
        ), "pipeline.text_encoder_2 is not a CLIPTextModelWithProjection"

        self.ttnn_device = ttnn_device
        self.cpu_device = "cpu"
        self.batch_size = ttnn_device.get_num_devices()
        self.torch_pipeline = torch_pipeline
        self.pipeline_config = pipeline_config

        self.encoders_compiled = False
        self.image_processing_compiled = False
        self.allocated_device_tensors = False
        self.generated_input_tensors = False

        if pipeline_config.is_galaxy:
            logger.info("Setting TT_MM_THROTTLE_PERF for Galaxy")
            os.environ["TT_MM_THROTTLE_PERF"] = "5"

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

        # Hardcoded input tensor parameters

        # Tensor shapes
        B, C, H, W = 1, 4, 128, 128
        self.tt_latents_shape = [B, C, H, W]

    def set_num_inference_steps(self, num_inference_steps: int):
        # When changing num_inference_steps, the timesteps and latents need to be recreated.
        self.pipeline_config.num_inference_steps = num_inference_steps
        self.generated_input_tensors = False

    def set_guidance_scale(self, guidance_scale: float):
        self.pipeline_config.guidance_scale = guidance_scale
        host_guidance_scale = ttnn.from_torch(
            torch.Tensor([self.pipeline_config.guidance_scale]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device),
        )
        ttnn.copy_host_to_device_tensor(host_guidance_scale, self.guidance_scale)

    def compile_text_encoding(self):
        # Compilation of text encoders on the device.
        if not self.encoders_compiled:
            assert self.pipeline_config.encoders_on_device, "Host text encoders are used; compile is not needed"
            assert self.tt_text_encoder is not None, "Text encoder is not loaded on the device"

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
                [self.ttnn_timesteps[0]],
                self.extra_step_kwargs,
                self.guidance_scale,
                self.scaling_factor,
                self.tt_latents_shape,
                self.tt_vae if self.pipeline_config.vae_on_device else self.torch_pipeline.vae,
                self.batch_size,
                capture_trace=False,
            )
            ttnn.synchronize_device(self.ttnn_device)
            profiler.end("warmup_run")

            if self.pipeline_config.capture_trace:
                self.__trace_image_processing()

            self.image_processing_compiled = True

    def encode_prompts(self, prompts, negative_prompts):
        # Encode prompts using the text encoders.

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
                batch_embeds = batch_encode_prompt_on_device(
                    self.torch_pipeline,
                    self.tt_text_encoder,
                    self.tt_text_encoder_2,
                    self.ttnn_device,
                    prompt=batch_prompts,  # Pass the entire batch
                    prompt_2=None,
                    device=self.cpu_device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=current_negative_prompts,
                    negative_prompt_2=None,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    pooled_prompt_embeds=None,
                    negative_pooled_prompt_embeds=None,
                    lora_scale=None,
                    clip_skip=None,
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
                prompt_2=None,
                device=self.cpu_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompts,
                negative_prompt_2=None,
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

        prompt_embeds_torch = torch.split(torch.cat(prompt_embeds, dim=0), self.batch_size, dim=0)
        negative_prompt_embeds_torch = torch.split(torch.cat(negative_prompt_embeds, dim=0), self.batch_size, dim=0)
        pooled_prompt_embeds_torch = torch.split(torch.cat(pooled_prompt_embeds, dim=0), self.batch_size, dim=0)
        negative_pooled_prompt_embeds_torch = torch.split(
            torch.cat(negative_pooled_prompt_embeds, dim=0), self.batch_size, dim=0
        )

        profiler.end("encode_prompts")
        logger.info(f"Encoded prompts")
        return (
            prompt_embeds_torch,
            negative_prompt_embeds_torch,
            pooled_prompt_embeds_torch,
            negative_pooled_prompt_embeds_torch,
        )

    def generate_input_tensors(
        self,
        prompt_embeds_torch,
        negative_prompt_embeds_torch,
        pooled_prompt_embeds_torch,
        negative_pooled_prompt_embeds_torch,
    ):
        # Generate user input tensors for the TT model.

        logger.info("Generating input tensors...")
        profiler.start("prepare_latents")
        self.__prepare_timesteps()

        num_channels_latents = self.torch_pipeline.unet.config.in_channels
        height = width = 1024
        assert num_channels_latents == 4, f"num_channels_latents is {num_channels_latents}, but it should be 4"

        latents = self.torch_pipeline.prepare_latents(
            1,
            num_channels_latents,
            height,
            width,
            prompt_embeds_torch[0].dtype,
            self.cpu_device,
            None,
            None,
        )

        self.extra_step_kwargs = self.torch_pipeline.prepare_extra_step_kwargs(None, 0.0)

        text_encoder_projection_dim = self.torch_pipeline.text_encoder_2.config.projection_dim
        assert (
            text_encoder_projection_dim == 1280
        ), f"text_encoder_projection_dim is {text_encoder_projection_dim}, but it should be 1280"

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = self.torch_pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds_torch[0].dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = add_time_ids

        B, C, H, W = latents.shape

        # All device code will work with channel last tensors
        tt_latents = torch.permute(latents, (0, 2, 3, 1))
        tt_latents = tt_latents.reshape(1, 1, B * H * W, C)
        tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.__create_user_tensors(
            latents=tt_latents,
            negative_prompt_embeds=negative_prompt_embeds_torch,
            prompt_embeds=prompt_embeds_torch,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_torch,
            add_text_embeds=pooled_prompt_embeds_torch,
        )

        self.__allocate_device_tensors(
            tt_latents=tt_latents,
            tt_prompt_embeds=tt_prompt_embeds,
            tt_text_embeds=tt_add_text_embeds,
            tt_time_ids=[negative_add_time_ids, add_time_ids],
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
        device_tensors = [self.tt_latents_device, *self.tt_prompt_embeds_device, *self.tt_text_embeds_device]

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
            self.ttnn_timesteps,
            self.extra_step_kwargs,
            self.guidance_scale,
            self.scaling_factor,
            self.tt_latents_shape,
            self.tt_vae if self.pipeline_config.vae_on_device else self.torch_pipeline.vae,
            self.batch_size,
            tid=self.tid if hasattr(self, "tid") else None,
            output_device=self.output_device if hasattr(self, "output_device") else None,
            output_shape=self.output_shape,
            tid_vae=self.tid_vae if hasattr(self, "tid_vae") else None,
        )
        return imgs

    def __load_tt_components(self, pipeline_config):
        # Method for instantiating TT components based on the torch pipeline.
        # Included components are TtUNet2DConditionModel, TtAutoencoderKL, and TtEulerDiscreteScheduler.
        # TODO: move encoder here

        profiler.start("load_tt_componenets")
        with ttnn.distribute(ttnn.ReplicateTensorToMesh(self.ttnn_device)):
            # 2. Load tt_unet, tt_vae and tt_scheduler
            self.tt_model_config = ModelOptimisations()
            self.tt_unet = TtUNet2DConditionModel(
                self.ttnn_device,
                self.torch_pipeline.unet.state_dict(),
                "unet",
                model_config=self.tt_model_config,
            )
            self.tt_vae = (
                TtAutoencoderKL(self.ttnn_device, self.torch_pipeline.vae.state_dict(), self.tt_model_config)
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
        is_mesh_device = isinstance(self.ttnn_device, ttnn._ttnn.multi_device.MeshDevice)
        if not self.allocated_device_tensors:
            profiler.start("allocate_input_tensors")

            self.tt_latents_device = ttnn.allocate_tensor_on_device(
                tt_latents.shape,
                tt_latents.dtype,
                tt_latents.layout,
                self.ttnn_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

            self.tt_prompt_embeds_device = [
                ttnn.allocate_tensor_on_device(
                    tt_prompt_embeds[0][0].shape,
                    tt_prompt_embeds[0][0].dtype,
                    tt_prompt_embeds[0][0].layout,
                    self.ttnn_device,
                    ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.allocate_tensor_on_device(
                    tt_prompt_embeds[0][1].shape,
                    tt_prompt_embeds[0][1].dtype,
                    tt_prompt_embeds[0][1].layout,
                    self.ttnn_device,
                    ttnn.DRAM_MEMORY_CONFIG,
                ),
            ]

            self.tt_text_embeds_device = [
                ttnn.allocate_tensor_on_device(
                    tt_text_embeds[0][0].shape,
                    tt_text_embeds[0][0].dtype,
                    tt_text_embeds[0][0].layout,
                    self.ttnn_device,
                    ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.allocate_tensor_on_device(
                    tt_text_embeds[0][1].shape,
                    tt_text_embeds[0][1].dtype,
                    tt_text_embeds[0][1].layout,
                    self.ttnn_device,
                    ttnn.DRAM_MEMORY_CONFIG,
                ),
            ]

            self.tt_time_ids_device = [
                ttnn.from_torch(
                    tt_time_ids[0].squeeze(0),
                    dtype=ttnn.bfloat16,
                    device=self.ttnn_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device) if is_mesh_device else None,
                ),
                ttnn.from_torch(
                    tt_time_ids[1].squeeze(0),
                    dtype=ttnn.bfloat16,
                    device=self.ttnn_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device) if is_mesh_device else None,
                ),
            ]
            ttnn.synchronize_device(self.ttnn_device)
            profiler.end("prepare_input_tensors")

            self.allocated_device_tensors = True
        else:
            tt_time_ids_host = [
                ttnn.from_torch(
                    tt_time_ids[0].squeeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device) if is_mesh_device else None,
                ),
                ttnn.from_torch(
                    tt_time_ids[1].squeeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device) if is_mesh_device else None,
                ),
            ]

            for host_tensor, device_tensor in zip(tt_time_ids_host, self.tt_time_ids_device):
                ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

    def __create_user_tensors(
        self, latents, negative_prompt_embeds, prompt_embeds, negative_pooled_prompt_embeds, add_text_embeds
    ):
        # Instantiation of user host input tensors for the TT model.

        profiler.start("create_user_tensors")
        is_mesh_device = isinstance(self.ttnn_device, ttnn._ttnn.multi_device.MeshDevice)
        tt_latents = ttnn.from_torch(
            latents,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.ttnn_device) if is_mesh_device else None,
        )

        tt_prompt_embeds = [
            [
                ttnn.from_torch(
                    negative_prompt_embed,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.ttnn_device, dim=0) if is_mesh_device else None,
                ),
                ttnn.from_torch(
                    prompt_embed,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.ttnn_device, dim=0) if is_mesh_device else None,
                ),
            ]
            for negative_prompt_embed, prompt_embed in zip(negative_prompt_embeds, prompt_embeds)
        ]

        tt_add_text_embeds = [
            [
                ttnn.from_torch(
                    negative_pooled_prompt_embed,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.ttnn_device, dim=0) if is_mesh_device else None,
                ),
                ttnn.from_torch(
                    add_text_embed,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.ttnn_device, dim=0) if is_mesh_device else None,
                ),
            ]
            for negative_pooled_prompt_embed, add_text_embed in zip(negative_pooled_prompt_embeds, add_text_embeds)
        ]
        ttnn.synchronize_device(self.ttnn_device)
        profiler.end("create_user_tensors")
        return tt_latents, tt_prompt_embeds, tt_add_text_embeds

    def __prepare_timesteps(self):
        # Helper method for timestep preparation.

        self.ttnn_timesteps, self.pipeline_config.num_inference_steps = retrieve_timesteps(
            self.torch_pipeline.scheduler, self.pipeline_config.num_inference_steps, self.cpu_device, None, None
        )

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
            [self.ttnn_timesteps[0]],
            self.extra_step_kwargs,
            self.guidance_scale,
            self.scaling_factor,
            self.tt_latents_shape,
            self.tt_vae if self.pipeline_config.vae_on_device else self.torch_pipeline.vae,
            self.batch_size,
            capture_trace=True,
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

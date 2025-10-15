# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from diffusers import StableDiffusionXLInpaintPipeline
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    get_timesteps,
    prepare_image_latents,
    prepare_mask_latents_inpainting,
    run_tt_image_gen_inpainting,
)
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline import (
    TtSDXLPipeline,
    TtSDXLImg2ImgPipelineConfig,
)
import torch
from loguru import logger
from models.common.utility_functions import profiler

import ttnn


@dataclass
class TtSDXLInpaintingPipelineConfig(TtSDXLImg2ImgPipelineConfig):
    strength = 0.99
    _torch_pipeline_type = StableDiffusionXLInpaintPipeline


class TtSDXLInpaintingPipeline(TtSDXLPipeline):
    def __init__(self, ttnn_device, torch_pipeline, pipeline_config: TtSDXLInpaintingPipelineConfig):
        super().__init__(ttnn_device, torch_pipeline, pipeline_config)

        self.num_in_channels_unet = 9
        self.num_channels_image_latents = 4

        # Refers to combined latents shape:
        # [1, 4, H, W] of image latents
        # [1, 1, H, W] of mask
        # [1, 4, H, W] of masked image latents
        # These are concatenated together over channel dimension to form the latent input to the unet of shape [1, 9, H, W]
        B, C, H, W = 1, self.num_in_channels_unet, 128, 128
        self.tt_latents_shape = [B, C, H, W]

        B, C, H, W = 1, self.num_channels_image_latents, 128, 128
        self.tt_image_latents_shape = [B, C, H, W]

    def _prepare_timesteps(self):
        super()._prepare_timesteps()

        self.ttnn_timesteps, self.pipeline_config.num_inference_steps = get_timesteps(
            self.tt_scheduler, self.pipeline_config.num_inference_steps, self.pipeline_config.strength, None
        )

        if self.pipeline_config.num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {self.pipeline_config.strength}, the number of pipeline"
                f"steps is {self.pipeline_config.num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )

    def generate_input_tensors(
        self,
        all_prompt_embeds_torch,
        torch_add_text_embeds,
        torch_image,
        torch_masked_image,
        torch_mask,
        start_latent_seed=None,  # need this to generate noise tensors, and in the future if we want to support strength_max == 1.0
        fixed_seed_for_batch=False,
    ):
        # Generate user input tensors for the TT model.

        logger.info("Generating input tensors...")
        profiler.start("prepare_latents")

        # This cuts number of inference steps relative by strength parameter
        self._prepare_timesteps()

        num_channels_image_latents = self.torch_pipeline.vae.config.latent_channels
        height = width = 1024
        assert (
            num_channels_image_latents == self.num_channels_image_latents
        ), f"num_channels_latents is {num_channels_image_latents}, but it should be {self.num_channels_image_latents}"
        assert start_latent_seed is None or isinstance(
            start_latent_seed, int
        ), "start_latent_seed must be an integer or None"

        img_latents_list = []
        mask_list = []
        masked_image_latents_list = []
        for index in range(self.batch_size):
            if start_latent_seed is not None:
                torch.manual_seed(start_latent_seed if fixed_seed_for_batch else start_latent_seed + index)
            # the function returns img_latents, noise but we don't use noise at the moment, so discard it
            img_latents = prepare_image_latents(
                self.torch_pipeline,
                self,
                1,
                num_channels_image_latents,
                height,
                width,
                self.cpu_device,
                all_prompt_embeds_torch.dtype,
                torch_image,
                self.pipeline_config.strength == 1,
                True,  # Make this configurable
                None,  # passed in latents
            )
            B, C, H, W = img_latents.shape  # 1, 4, 128, 128
            img_latents = torch.permute(img_latents, (0, 2, 3, 1))  # [1, H, W, C]
            img_latents = img_latents.reshape(B, 1, H * W, C)  # [1, 1, H*W, C]
            img_latents_list.append(img_latents)

            mask, masked_image_latents = prepare_mask_latents_inpainting(
                self,
                torch_mask,
                torch_masked_image,
                1,
                height,
                width,
                all_prompt_embeds_torch.dtype,
                self.cpu_device,
                None,
            )

            B, C, H, W = mask.shape
            mask = torch.permute(mask, (0, 2, 3, 1))  # [1, H, W, C]
            mask = mask.reshape(B, 1, H * W, C)  # [1, 1, H*W, C]
            mask_list.append(mask)

            B, C, H, W = masked_image_latents.shape
            masked_image_latents = torch.permute(masked_image_latents, (0, 2, 3, 1))  # [1, H, W, C]
            masked_image_latents = masked_image_latents.reshape(B, 1, H * W, C)  # [1, 1, H*W, C]
            masked_image_latents_list.append(masked_image_latents)

        tt_img_latents = torch.cat(img_latents_list, dim=0)  # [batch_size, 1, H*W, C]
        tt_mask = torch.cat(mask_list, dim=0)  # [batch_size, 1, H*W, C]
        tt_masked_image_latents = torch.cat(masked_image_latents_list, dim=0)  # [batch_size, 1, H*W, C]

        self.extra_step_kwargs = self.torch_pipeline.prepare_extra_step_kwargs(None, 0.0)
        text_encoder_projection_dim = self.torch_pipeline.text_encoder_2.config.projection_dim
        assert (
            text_encoder_projection_dim == 1280
        ), f"text_encoder_projection_dim is {text_encoder_projection_dim}, but it should be 1280"

        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        add_time_ids, negative_add_time_ids = self.torch_pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            self.pipeline_config.aesthetic_score,
            self.pipeline_config.negative_aesthetic_score,
            original_size,  # negative_original_size, assume the same as positive
            crops_coords_top_left,  # negative_crops_coords_top_left, assume the same as positive
            target_size,  # negative_target_size, assume the same as positive
            dtype=all_prompt_embeds_torch.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        torch_add_time_ids = torch.stack([negative_add_time_ids.squeeze(0), add_time_ids.squeeze(0)], dim=0)

        (
            tt_image_latents,
            tt_masked_image_latents,
            tt_mask,
            tt_prompt_embeds,
            tt_add_text_embeds,
        ) = self.__create_user_tensors(
            img_latents=tt_img_latents,
            masked_image_latents=tt_masked_image_latents,
            mask=tt_mask,
            all_prompt_embeds_torch=all_prompt_embeds_torch,
            torch_add_text_embeds=torch_add_text_embeds,
        )

        self.__allocate_device_tensors(
            tt_image_latents=tt_image_latents,
            tt_masked_image_latents=tt_masked_image_latents,
            tt_mask=tt_mask,
            tt_prompt_embeds=tt_prompt_embeds,
            tt_text_embeds=tt_add_text_embeds,
            tt_time_ids=torch_add_time_ids,
        )
        ttnn.synchronize_device(self.ttnn_device)
        profiler.end("prepare_latents")
        logger.info("Input tensors generated")

        self.generated_input_tensors = True
        return tt_image_latents, tt_masked_image_latents, tt_mask, tt_prompt_embeds, tt_add_text_embeds

    def compile_image_processing(self):
        # Compile/trace run for denoising loop and vae decoder.
        if not self.image_processing_compiled:
            assert self.allocated_device_tensors, "Input tensors are not allocated"

            profiler.start("warmup_run")
            logger.info("Performing warmup run on denoising, to make use of program caching in actual inference...")

            _, _, _, self.output_shape, _ = run_tt_image_gen_inpainting(
                self.ttnn_device,
                self.tt_unet,
                self.tt_scheduler,
                self.tt_latents_device,
                self.tt_masked_image_latents_device,
                self.tt_mask_device,
                self.tt_prompt_embeds_device,
                self.tt_time_ids_device,
                self.tt_text_embeds_device,
                [self.ttnn_timesteps[0]],
                self.extra_step_kwargs,
                self.guidance_scale,
                self.scaling_factor,
                self.tt_latents_shape,
                self.tt_image_latents_shape,
                self.tt_vae if self.pipeline_config.vae_on_device else self.torch_pipeline.vae,
                self.batch_size,
                self.ag_persistent_buffer,
                self.ag_semaphores,
                capture_trace=False,
                use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
            )
            ttnn.synchronize_device(self.ttnn_device)
            profiler.end("warmup_run")

            if self.pipeline_config.capture_trace:
                self._TtSDXLPipeline__trace_image_processing()

            self.image_processing_compiled = True

    def prepare_input_tensors(self, host_tensors):
        # Tensor device transfer for the current users.

        assert self.allocated_device_tensors, "Device tensors are not allocated"

        logger.info("Preparing input tensors for TT model...")
        profiler.start("prepare_input_tensors")
        device_tensors = [
            self.tt_latents_device,
            self.tt_masked_image_latents_device,
            self.tt_mask_device,
            self.tt_prompt_embeds_device,
            self.tt_text_embeds_device,
        ]

        for host_tensor, device_tensor in zip(host_tensors, device_tensors):
            ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

        ttnn.synchronize_device(self.ttnn_device)
        profiler.end("prepare_input_tensors")

    def generate_images(self):
        # SDXL inference run.
        assert self.image_processing_compiled, "Image processing is not compiled"
        assert self.generated_input_tensors, "Input tensors are not re/generated"

        logger.info("Generating images...")
        imgs, self.tid, self.output_device, self.output_shape, self.tid_vae = run_tt_image_gen_inpainting(
            self.ttnn_device,
            self.tt_unet,
            self.tt_scheduler,
            self.tt_latents_device,
            self.tt_masked_image_latents_device,
            self.tt_mask_device,
            self.tt_prompt_embeds_device,
            self.tt_time_ids_device,
            self.tt_text_embeds_device,
            self.ttnn_timesteps,
            self.extra_step_kwargs,
            self.guidance_scale,
            self.scaling_factor,
            self.tt_latents_shape,
            self.tt_image_latents_shape,
            self.tt_vae if self.pipeline_config.vae_on_device else self.torch_pipeline.vae,
            self.batch_size,
            self.ag_persistent_buffer,
            self.ag_semaphores,
            tid=self.tid if hasattr(self, "tid") else None,
            output_device=self.output_device if hasattr(self, "output_device") else None,
            output_shape=self.output_shape,
            tid_vae=self.tid_vae if hasattr(self, "tid_vae") else None,
            use_cfg_parallel=self.pipeline_config.use_cfg_parallel,
        )
        return imgs

    def __allocate_device_tensors(
        self, tt_image_latents, tt_masked_image_latents, tt_mask, tt_prompt_embeds, tt_text_embeds, tt_time_ids
    ):
        # Allocation of device tensors for the input data.
        if not self.allocated_device_tensors:
            profiler.start("allocate_input_tensors")

            self.tt_latents_device = ttnn.allocate_tensor_on_device(
                tt_image_latents.shape,
                tt_image_latents.dtype,
                tt_image_latents.layout,
                self.ttnn_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

            self.tt_masked_image_latents_device = ttnn.allocate_tensor_on_device(
                tt_masked_image_latents.shape,
                tt_masked_image_latents.dtype,
                tt_masked_image_latents.layout,
                self.ttnn_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )

            self.tt_mask_device = ttnn.allocate_tensor_on_device(
                tt_mask.shape,
                tt_mask.dtype,
                tt_mask.layout,
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

            for host_tensor, device_tensor in zip(tt_time_ids_host, self.tt_time_ids_device):
                ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

    def __create_user_tensors(
        self, img_latents, masked_image_latents, mask, all_prompt_embeds_torch, torch_add_text_embeds
    ):
        # Instantiation of user host input tensors for the TT model.

        profiler.start("create_user_tensors")
        tt_img_latents = ttnn.from_torch(
            img_latents,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.ttnn_device, list(self.ttnn_device.shape), dims=(None, 0)),
        )

        tt_masked_image_latents = ttnn.from_torch(
            masked_image_latents,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.ttnn_device, list(self.ttnn_device.shape), dims=(None, 0)),
        )

        tt_mask = ttnn.from_torch(
            mask,
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
        return tt_img_latents, tt_masked_image_latents, tt_mask, tt_prompt_embeds, tt_add_text_embeds

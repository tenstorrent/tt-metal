# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from diffusers import StableDiffusionXLImg2ImgPipeline
from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    get_timesteps,
    prepare_image_latents,
)
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig
import torch
from loguru import logger
from models.common.utility_functions import profiler

import ttnn


@dataclass
class TtSDXLImg2ImgPipelineConfig(TtSDXLPipelineConfig):
    strength: float = 0.3
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5
    _torch_pipeline_type = StableDiffusionXLImg2ImgPipeline


class TtSDXLImg2ImgPipeline(TtSDXLPipeline):
    def __init__(self, ttnn_device, torch_pipeline, pipeline_config: TtSDXLImg2ImgPipelineConfig):
        super().__init__(ttnn_device, torch_pipeline, pipeline_config)

        self.num_in_channels_unet = 4
        self.num_channels_image_latents = 4
        B, C, H, W = 1, self.num_in_channels_unet, 128, 128
        self.tt_latents_shape = [B, C, H, W]

    def set_strength(self, strength: int):
        # When changing strength, the timesteps and latents need to be recreated.
        self.pipeline_config.strength = strength
        self.generated_input_tensors = False

    def set_aesthetic_score(self, aesthetic_score: int):
        # When changing strength, the timesteps and latents need to be recreated.
        self.pipeline_config.aesthetic_score = aesthetic_score
        self.generated_input_tensors = False

    def set_negative_aesthetic_score(self, negative_aesthetic_score: int):
        # When changing strength, the timesteps and latents need to be recreated.
        self.pipeline_config.negative_aesthetic_score = negative_aesthetic_score
        self.generated_input_tensors = False

    def _prepare_timesteps(self):
        super()._prepare_timesteps()

        self.ttnn_timesteps, self.pipeline_config.num_inference_steps = get_timesteps(
            self.torch_pipeline.scheduler, self.pipeline_config.num_inference_steps, self.pipeline_config.strength, None
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

        if start_latent_seed is not None:
            torch.manual_seed(start_latent_seed if fixed_seed_for_batch else start_latent_seed)
        img_latents = prepare_image_latents(
            self.torch_pipeline,
            self,
            torch_image.shape[0],
            num_channels_image_latents,
            height,
            width,
            self.cpu_device,
            all_prompt_embeds_torch.dtype,
            torch_image,
            False,  # No max strength path in ref img2img implementation
            True,  # Make this configurable
            None,  # passed in latents
        )
        B, C, H, W = img_latents.shape  # 1, 4, 128, 128
        img_latents = torch.permute(img_latents, (0, 2, 3, 1))  # [1, H, W, C]
        tt_img_latents = img_latents.reshape(B, 1, H * W, C)  # [1, 1, H*W, C]

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
            tt_prompt_embeds,
            tt_add_text_embeds,
        ) = super()._TtSDXLPipeline__create_user_tensors(
            latents=tt_img_latents,
            all_prompt_embeds_torch=all_prompt_embeds_torch,
            torch_add_text_embeds=torch_add_text_embeds,
        )

        super()._TtSDXLPipeline__allocate_device_tensors(
            tt_latents=tt_image_latents,
            tt_prompt_embeds=tt_prompt_embeds,
            tt_text_embeds=tt_add_text_embeds,
            tt_time_ids=torch_add_time_ids,
        )
        ttnn.synchronize_device(self.ttnn_device)
        profiler.end("prepare_latents")
        logger.info("Input tensors generated")

        self.generated_input_tensors = True
        return tt_image_latents, tt_prompt_embeds, tt_add_text_embeds

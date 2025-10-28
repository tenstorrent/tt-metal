# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from loguru import logger
import torch

from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline import (
    TtSDXLImg2ImgPipeline,
    TtSDXLImg2ImgPipelineConfig,
)
from models.common.utility_functions import profiler

MAX_SEQUENCE_LENGTH = 77
TEXT_ENCODER_2_PROJECTION_DIM = 1280
CONCATENATED_TEXT_EMBEDINGS_SIZE = 2048  # text_encoder_1_hidden_size + text_encoder_2_hidden_size (768 + 1280)


@dataclass
class TtSDXLCombinedPipelineConfig:
    base_config: TtSDXLPipelineConfig
    refiner_config: TtSDXLImg2ImgPipelineConfig
    denoising_split: float = 0.8  # Base does 80%, refiner does 20%. Set to 1.0 to disable refiner

    def __post_init__(self):
        # Validate denoising_split
        assert isinstance(
            self.denoising_split, (int, float)
        ), f"denoising_split must be numeric but is {type(self.denoising_split)}"
        assert (
            0.0 <= self.denoising_split <= 1.0
        ), f"denoising_split must be in range [0.0, 1.0] but is {self.denoising_split}"

    @property
    def use_refiner(self):
        """Refiner is used when denoising_split < 1.0"""
        return self.denoising_split < 1.0


class TtSDXLCombinedPipeline:
    """
    Combined pipeline that orchestrates SDXL base and img2img refiner models.

    The base model (TtSDXLPipeline) performs initial denoising up to a specified split point,
    then the refiner model (TtSDXLImg2ImgPipeline) takes over to complete the denoising and decode the final image.
    Both models share the same scheduler instance for coordinated timestep management.
    """

    def __init__(self, ttnn_device, tt_base_pipeline, tt_refiner_pipeline, config: TtSDXLCombinedPipelineConfig):
        assert isinstance(tt_base_pipeline, TtSDXLPipeline), "tt_base_pipeline must be an instance of TtSDXLPipeline"
        assert isinstance(
            tt_refiner_pipeline, TtSDXLImg2ImgPipeline
        ), "tt_refiner_pipeline must be an instance of TtSDXLImg2ImgPipeline"

        self.ttnn_device = ttnn_device
        self.config = config
        self.batch_size = (
            list(ttnn_device.shape)[1] if config.base_config.use_cfg_parallel else ttnn_device.get_num_devices()
        )

        logger.info("Initializing base pipeline...")
        self.base_pipeline = tt_base_pipeline

        if config.use_refiner:
            logger.info("Initializing refiner pipeline...")
            self.refiner_pipeline = tt_refiner_pipeline

            # Share the scheduler instance between base and refiner
            self.refiner_pipeline.tt_scheduler = self.base_pipeline.tt_scheduler
            self.refiner_pipeline.torch_pipeline.scheduler = self.base_pipeline.tt_scheduler
        else:
            self.refiner_pipeline = None

        # Track compilation state
        self._compiled = False

        logger.info("Combined pipeline initialized successfully")

    def _auto_compile_if_needed(self):
        if self._compiled:
            return

        logger.info("=" * 80)
        logger.info("Auto-compiling pipelines (first-time setup)...")
        logger.info("=" * 80)

        # 1. Compile text encoders if on device
        if self.config.base_config.encoders_on_device:
            logger.info("Compiling text encoders...")
            self.base_pipeline.compile_text_encoding()

        # 2. Allocate device tensors with dummy data for warmup
        logger.info("Allocating device tensors for base pipeline...")
        dummy_embeds = torch.randn(self.batch_size, 2, MAX_SEQUENCE_LENGTH, CONCATENATED_TEXT_EMBEDINGS_SIZE)
        dummy_text_embeds = torch.randn(self.batch_size, 2, TEXT_ENCODER_2_PROJECTION_DIM)

        _, _, _ = self.base_pipeline.generate_input_tensors(
            all_prompt_embeds_torch=dummy_embeds,
            torch_add_text_embeds=dummy_text_embeds,
        )

        # 3. Compile base image processing
        logger.info("Compiling base pipeline image processing...")
        self.base_pipeline.compile_image_processing()

        # 4. Compile refiner if enabled
        if self.config.use_refiner:
            logger.info("Allocating device tensors for refiner pipeline...")
            # Create dummy image tensor for img2img pipeline
            dummy_image = torch.randn(self.batch_size, 3, 1024, 1024)

            _, _, _ = self.refiner_pipeline.generate_input_tensors(
                all_prompt_embeds_torch=dummy_embeds,
                torch_add_text_embeds=dummy_text_embeds,
                torch_image=dummy_image,
            )

            logger.info("Compiling refiner pipeline image processing...")
            self.refiner_pipeline.compile_image_processing()

        self._compiled = True
        logger.info("=" * 80)
        logger.info("Pipeline compilation complete!")
        logger.info("=" * 80)

    def _calculate_timestep_split(self, num_inference_steps):
        split_idx = int(num_inference_steps * self.config.denoising_split)

        # Ensure at least 1 step for each pipeline if both are active
        if self.config.use_refiner:
            split_idx = max(1, min(split_idx, num_inference_steps - 1))

        return split_idx

    def generate(
        self,
        prompts,
        negative_prompts,
        prompt_2=None,
        negative_prompt_2=None,
        start_latent_seed=None,
        fixed_seed_for_batch=False,
        timesteps=None,
        sigmas=None,
        num_inference_steps=None,
        guidance_scale=None,
        crop_coords_top_left=None,
        guidance_rescale=None,
    ):
        self._auto_compile_if_needed()

        # Apply runtime parameters if provided and different from config
        if num_inference_steps is not None and num_inference_steps != self.config.base_config.num_inference_steps:
            self.base_pipeline.set_num_inference_steps(num_inference_steps)
            if self.config.use_refiner:
                self.refiner_pipeline.set_num_inference_steps(num_inference_steps)

        if guidance_scale is not None and guidance_scale != self.config.base_config.guidance_scale:
            self.base_pipeline.set_guidance_scale(guidance_scale)
            if self.config.use_refiner:
                self.refiner_pipeline.set_guidance_scale(guidance_scale)

        if crop_coords_top_left is not None and crop_coords_top_left != self.config.base_config.crop_coords_top_left:
            self.base_pipeline.set_crop_coords_top_left(crop_coords_top_left)
            if self.config.use_refiner:
                self.refiner_pipeline.set_crop_coords_top_left(crop_coords_top_left)

        if guidance_rescale is not None and guidance_rescale != self.config.base_config.guidance_rescale:
            self.base_pipeline.set_guidance_rescale(guidance_rescale)
            if self.config.use_refiner:
                self.refiner_pipeline.set_guidance_rescale(guidance_rescale)

        profiler.start("combined_generation")

        # 1. Encode prompts using base pipeline
        logger.info("Encoding prompts...")
        all_prompt_embeds_torch, torch_add_text_embeds = self.base_pipeline.encode_prompts(
            prompts=prompts,
            negative_prompts=negative_prompts,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
        )

        # 2. Generate input tensors for base pipeline
        logger.info("Generating input tensors for base pipeline...")
        tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.base_pipeline.generate_input_tensors(
            all_prompt_embeds_torch=all_prompt_embeds_torch,
            torch_add_text_embeds=torch_add_text_embeds,
            start_latent_seed=start_latent_seed,
            fixed_seed_for_batch=fixed_seed_for_batch,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        # 3. Calculate timestep split
        num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else self.config.base_config.num_inference_steps
        )
        split_idx = self._calculate_timestep_split(num_inference_steps)

        logger.info(
            f"Denoising split: base will perform {split_idx}/{num_inference_steps} steps, "
            f"refiner will perform {num_inference_steps - split_idx}/{num_inference_steps} steps"
        )

        # 4. Run base pipeline with timesteps[0:split_idx]
        if self.config.use_refiner and split_idx < num_inference_steps:
            # Temporarily adjust base pipeline timesteps
            original_timesteps = self.base_pipeline.ttnn_timesteps
            base_timesteps = original_timesteps[:split_idx]
            self.base_pipeline.ttnn_timesteps = base_timesteps

            logger.info(f"Running base pipeline for {split_idx} steps...")
            self.base_pipeline.prepare_input_tensors([tt_latents, tt_prompt_embeds[0], tt_add_text_embeds[0]])
            base_latents = self.base_pipeline.generate_images()

            # Restore original timesteps
            self.base_pipeline.ttnn_timesteps = original_timesteps

            # 5. Run refiner pipeline with timesteps[split_idx:] and base latents
            logger.info(f"Running refiner pipeline for {num_inference_steps - split_idx} steps...")

            # Refiner uses same timesteps but starts from split_idx
            refiner_timesteps = original_timesteps[split_idx:]
            self.refiner_pipeline.ttnn_timesteps = refiner_timesteps

            # Generate input tensors for refiner with base latents
            _, refiner_prompt_embeds, refiner_add_text_embeds = self.refiner_pipeline.generate_input_tensors(
                all_prompt_embeds_torch=all_prompt_embeds_torch,
                torch_add_text_embeds=torch_add_text_embeds,
                input_latents=base_latents,
                timesteps=timesteps,
                sigmas=sigmas,
            )

            # The scheduler needs to start from the correct step index
            self.refiner_pipeline.tt_scheduler.set_step_index(split_idx)

            self.refiner_pipeline.prepare_input_tensors(
                [
                    base_latents if hasattr(base_latents, "shape") else tt_latents,
                    refiner_prompt_embeds[0],
                    refiner_add_text_embeds[0],
                ]
            )
            images = self.refiner_pipeline.generate_images()

            # Reset scheduler for next generation
            self.refiner_pipeline.tt_scheduler.set_step_index(0)
        else:
            # No refiner or split_idx == num_inference_steps, just run base
            logger.info(f"Running base pipeline only for {num_inference_steps} steps...")
            self.base_pipeline.prepare_input_tensors([tt_latents, tt_prompt_embeds[0], tt_add_text_embeds[0]])
            images = self.base_pipeline.generate_images()

        profiler.end("combined_generation")
        logger.info("Combined generation completed")

        return images

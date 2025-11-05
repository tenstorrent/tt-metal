# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any
from loguru import logger
import os
import torch
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline import (
    TtSDXLImg2ImgPipeline,
    TtSDXLImg2ImgPipelineConfig,
)
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
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

    The combined pipeline ALWAYS creates and manages the shared scheduler internally,
    ensuring proper coordination between base and refiner models.

    Usage:
        combined = TtSDXLCombinedPipeline(
            ttnn_device=mesh_device,
            torch_base_pipeline=base,
            torch_refiner_pipeline=refiner,
            base_config=TtSDXLPipelineConfig(...),
            refiner_config=TtSDXLImg2ImgPipelineConfig(...),
            denoising_split=0.8
        )
    """

    def __init__(
        self,
        ttnn_device,
        torch_base_pipeline,
        torch_refiner_pipeline,
        base_config: TtSDXLPipelineConfig,
        refiner_config: TtSDXLImg2ImgPipelineConfig,
        denoising_split: float = 0.8,
    ):
        """
        Create a combined pipeline with automatic scheduler sharing.

        The combined pipeline creates a single shared scheduler and uses it for both
        base and refiner pipelines, ensuring optimal memory usage and proper coordination.

        Args:
            ttnn_device: The TTNN device
            torch_base_pipeline: Torch DiffusionPipeline for base model
            torch_refiner_pipeline: Torch StableDiffusionXLImg2ImgPipeline for refiner
            base_config: Configuration for base pipeline
            refiner_config: Configuration for refiner pipeline
            denoising_split: Fraction of denoising done by base (0.0-1.0), default 0.8
        """
        logger.info("Creating combined pipeline with shared scheduler...")

        self.ttnn_device = ttnn_device

        # Create combined pipeline config
        self.config = TtSDXLCombinedPipelineConfig(
            base_config=base_config,
            refiner_config=refiner_config,
            denoising_split=denoising_split,
        )

        self.batch_size = list(ttnn_device.shape)[1] if base_config.use_cfg_parallel else ttnn_device.get_num_devices()

        # Create a single shared scheduler based on base pipeline's scheduler config
        logger.info("Creating shared scheduler...")
        with ttnn.distribute(ttnn.ReplicateTensorToMesh(ttnn_device)):
            self.shared_scheduler = TtEulerDiscreteScheduler(
                ttnn_device,
                torch_base_pipeline.scheduler.config.num_train_timesteps,
                torch_base_pipeline.scheduler.config.beta_start,
                torch_base_pipeline.scheduler.config.beta_end,
                torch_base_pipeline.scheduler.config.beta_schedule,
                torch_base_pipeline.scheduler.config.trained_betas,
                torch_base_pipeline.scheduler.config.prediction_type,
                torch_base_pipeline.scheduler.config.interpolation_type,
                torch_base_pipeline.scheduler.config.use_karras_sigmas,
                torch_base_pipeline.scheduler.config.use_exponential_sigmas,
                torch_base_pipeline.scheduler.config.use_beta_sigmas,
                torch_base_pipeline.scheduler.config.sigma_min,
                torch_base_pipeline.scheduler.config.sigma_max,
                torch_base_pipeline.scheduler.config.timestep_spacing,
                torch_base_pipeline.scheduler.config.timestep_type,
                torch_base_pipeline.scheduler.config.steps_offset,
                torch_base_pipeline.scheduler.config.rescale_betas_zero_snr,
                torch_base_pipeline.scheduler.config.final_sigmas_type,
            )
        logger.info("Shared scheduler created")

        # Create base pipeline with shared scheduler
        logger.info("Creating base pipeline with shared scheduler...")
        self.base_pipeline = TtSDXLPipeline(
            ttnn_device=ttnn_device,
            torch_pipeline=torch_base_pipeline,
            pipeline_config=base_config,
            tt_scheduler=self.shared_scheduler,
        )

        # Create refiner pipeline with shared scheduler if needed
        if self.config.use_refiner:
            logger.info("Creating refiner pipeline with shared scheduler...")
            self.refiner_pipeline = TtSDXLImg2ImgPipeline(
                ttnn_device=ttnn_device,
                torch_pipeline=torch_refiner_pipeline,
                pipeline_config=refiner_config,
                tt_scheduler=self.shared_scheduler,
            )

            # When using refiner, disable VAE on base pipeline since refiner will handle final decoding
            if self.base_pipeline.pipeline_config.vae_on_device:
                logger.info("Disabling VAE on base pipeline since refiner will handle final decoding")
                self.base_pipeline.pipeline_config.vae_on_device = False
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
            fixed_seed_for_batch=True,
            start_latent_seed=0,
        )

        # 3. Compile base image processing
        logger.info("Compiling base pipeline image processing...")
        self.base_pipeline.compile_image_processing()

        # 4. Compile refiner if enabled
        if self.config.use_refiner:
            logger.info("Allocating device tensors for refiner pipeline...")
            # Create dummy image tensor for img2img pipeline
            dummy_latents = torch.randn(self.batch_size, 4, 128, 128)

            _, _, _ = self.refiner_pipeline.generate_input_tensors(
                all_prompt_embeds_torch=dummy_embeds,
                torch_add_text_embeds=dummy_text_embeds,
                input_latents=dummy_latents,
                fixed_seed_for_batch=True,
                start_latent_seed=0,
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
        batch_size = (
            list[Any](self.ttnn_device.shape)[1]
            if self.config.base_config.use_cfg_parallel
            else self.ttnn_device.get_num_devices()
        )
        if isinstance(prompts, str):
            prompts = [prompts]

        if prompt_2 is not None and isinstance(prompt_2, str):
            prompt_2 = [prompt_2]

        needed_padding = (batch_size - len(prompts) % batch_size) % batch_size
        if isinstance(negative_prompts, list):
            assert len(negative_prompts) == len(prompts), "prompts and negative_prompt lists must be the same length"

        prompts = prompts + [""] * needed_padding
        if prompt_2 is not None:
            prompt_2 = prompt_2 + [""] * needed_padding
        if isinstance(negative_prompts, list):
            negative_prompts = negative_prompts + [""] * needed_padding

        # Apply runtime parameters if provided and different from config
        # These must be set BEFORE auto-compile to ensure compilation happens with correct settings
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

        # Handle num_inference_steps BEFORE auto-compile
        # When using refiner, we'll set config but manually split timesteps later
        # When not using refiner, we can set it normally
        if num_inference_steps is not None and num_inference_steps != self.config.base_config.num_inference_steps:
            self.base_pipeline.set_num_inference_steps(num_inference_steps)

            if self.config.use_refiner:
                self.refiner_pipeline.set_num_inference_steps(num_inference_steps)

        # Auto-compile after all runtime parameters are set
        self._auto_compile_if_needed()

        profiler.start("combined_generation")

        # Determine effective num_inference_steps
        effective_num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else self.config.base_config.num_inference_steps
        )

        # Calculate split for base/refiner
        split_idx = (
            self._calculate_timestep_split(effective_num_inference_steps)
            if self.config.use_refiner
            else effective_num_inference_steps
        )

        # For refiner case, we need to mark that input tensors need regeneration since we updated the config
        if (
            self.config.use_refiner
            and num_inference_steps is not None
            and num_inference_steps != self.config.base_config.num_inference_steps
        ):
            logger.info(
                f"Configuring pipelines: base will use {split_idx} steps, refiner will use {effective_num_inference_steps - split_idx} steps from {effective_num_inference_steps} total"
            )
            self.base_pipeline.generated_input_tensors = False
            # Save originals for restoration later
            original_base_config_steps = self.base_pipeline.pipeline_config.num_inference_steps
            original_refiner_config_steps = self.refiner_pipeline.pipeline_config.num_inference_steps

        # 1. Encode prompts using base pipeline
        logger.info("Encoding prompts...")
        all_prompt_embeds_torch, torch_add_text_embeds = self.base_pipeline.encode_prompts(
            prompts=prompts,
            negative_prompts=negative_prompts,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
        )

        # 2. Generate input tensors for base pipeline (generates full timestep schedule)
        logger.info("Generating input tensors for base pipeline...")
        tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.base_pipeline.generate_input_tensors(
            all_prompt_embeds_torch=all_prompt_embeds_torch,
            torch_add_text_embeds=torch_add_text_embeds,
            start_latent_seed=start_latent_seed,
            fixed_seed_for_batch=fixed_seed_for_batch,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        # 3. Log the split information
        if self.config.use_refiner:
            logger.info(
                f"Denoising split: base will perform {split_idx}/{effective_num_inference_steps} steps, "
                f"refiner will perform {effective_num_inference_steps - split_idx}/{effective_num_inference_steps} steps"
            )

        # 4. Run base pipeline with timesteps[0:split_idx]
        if self.config.use_refiner and split_idx < effective_num_inference_steps:
            # Temporarily adjust base pipeline timesteps and num_inference_steps
            original_timesteps = self.base_pipeline.ttnn_timesteps
            original_num_inference_steps = self.base_pipeline.num_inference_steps
            base_timesteps = original_timesteps[:split_idx]
            refiner_timesteps = original_timesteps[split_idx:]
            self.base_pipeline.ttnn_timesteps = base_timesteps

            # Set base pipeline to only run split_idx steps
            # We set the attribute directly since we're manually managing timesteps
            self.base_pipeline.num_inference_steps = split_idx

            logger.info(f"Running base pipeline for {split_idx} steps...")
            self.base_pipeline.prepare_input_tensors([tt_latents, tt_prompt_embeds[0], tt_add_text_embeds[0]])
            base_latents = self.base_pipeline.generate_images(return_latents=True)  # Skip VAE decoding for refiner

            # Restore original timesteps and num_inference_steps for base
            self.base_pipeline.ttnn_timesteps = original_timesteps
            self.base_pipeline.num_inference_steps = original_num_inference_steps

            # 5. Run refiner pipeline with timesteps[split_idx:] and base latents
            logger.info(f"Running refiner pipeline for {effective_num_inference_steps - split_idx} steps...")

            # Generate input tensors for refiner with base latents
            (
                refiner_latents,
                refiner_prompt_embeds,
                refiner_add_text_embeds,
            ) = self.refiner_pipeline.generate_input_tensors(
                all_prompt_embeds_torch=all_prompt_embeds_torch,
                torch_add_text_embeds=torch_add_text_embeds,
                input_latents=base_latents,
                fixed_seed_for_batch=True,
                start_latent_seed=0,
                timesteps=timesteps,
                sigmas=sigmas,
            )

            self.refiner_pipeline.tt_scheduler.set_step_index(split_idx)

            # Now override timesteps and num_inference_steps AFTER generate_input_tensors
            # to avoid them being reset by _prepare_timesteps
            self.refiner_pipeline.ttnn_timesteps = refiner_timesteps
            self.refiner_pipeline.num_inference_steps = effective_num_inference_steps - split_idx

            self.refiner_pipeline.prepare_input_tensors(
                [
                    refiner_latents,
                    refiner_prompt_embeds[0],
                    refiner_add_text_embeds[0],
                ]
            )
            images = self.refiner_pipeline.generate_images()

            # Reset scheduler and restore timesteps for next generation
            self.refiner_pipeline.tt_scheduler.set_step_index(0)
            self.refiner_pipeline.ttnn_timesteps = original_timesteps

            # Restore original configs if they were modified
            if num_inference_steps is not None and num_inference_steps != self.config.base_config.num_inference_steps:
                self.base_pipeline.pipeline_config.num_inference_steps = original_base_config_steps
                self.refiner_pipeline.pipeline_config.num_inference_steps = original_refiner_config_steps
        else:
            # No refiner or split_idx == effective_num_inference_steps, just run base
            logger.info(f"Running base pipeline only for {effective_num_inference_steps} steps...")
            self.base_pipeline.prepare_input_tensors([tt_latents, tt_prompt_embeds[0], tt_add_text_embeds[0]])
            images = self.base_pipeline.generate_images()

        profiler.end("combined_generation")
        logger.info("Combined generation completed")

        # Save images to output folder
        if not os.path.exists("output"):
            os.mkdir("output")

        for idx, img in enumerate(images):
            img = img.unsqueeze(0)
            img = self.base_pipeline.torch_pipeline.image_processor.postprocess(img, output_type="pil")[0]
            img.save(f"output/output{idx}.png")
            logger.info(f"Image saved to output/output{idx}.png")

        return images

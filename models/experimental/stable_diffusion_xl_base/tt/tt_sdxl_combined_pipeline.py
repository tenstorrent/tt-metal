# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from loguru import logger
import torch
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline import (
    TtSDXLImg2ImgPipeline,
    TtSDXLImg2ImgPipelineConfig,
)
from models.experimental.stable_diffusion_xl_base.tt.tt_euler_discrete_scheduler import TtEulerDiscreteScheduler
from models.common.utility_functions import profiler

from models.experimental.stable_diffusion_xl_base.tests.test_common import (
    MAX_SEQUENCE_LENGTH,
    TEXT_ENCODER_2_PROJECTION_DIM,
    CONCATENATED_TEXT_EMBEDINGS_SIZE,
    CONCATENATED_TEXT_EMBEDINGS_SIZE_REFINER,
)


@dataclass
class TtSDXLCombinedPipelineConfig:
    """
    Unified configuration for the combined SDXL pipeline.
    This config contains all parameters and creates configs for base and refiner pipelines.
    """

    # Common parameters (apply to both base and refiner)
    num_inference_steps: int
    guidance_scale: float
    is_galaxy: bool
    use_cfg_parallel: bool
    vae_on_device: bool
    encoders_on_device: bool
    capture_trace: bool
    crop_coords_top_left: tuple = (0, 0)
    guidance_rescale: float = 0.0

    # Refiner-specific parameters
    use_refiner: bool = False
    strength: float = 0.3
    aesthetic_score: float = 6.0
    negative_aesthetic_score: float = 2.5

    denoising_split: float = 1.0

    def __post_init__(self):
        # Validate denoising_split
        assert isinstance(
            self.denoising_split, (int, float)
        ), f"denoising_split must be numeric but is {type(self.denoising_split)}"
        assert (
            0.0 <= self.denoising_split <= 1.0
        ), f"denoising_split must be in range [0.0, 1.0] but is {self.denoising_split}"

        if not self.use_refiner and self.denoising_split != 1.0:
            logger.warning(
                f"use_refiner=False but denoising_split={self.denoising_split}. Setting denoising_split to 1.0"
            )
            self.denoising_split = 1.0

    def create_base_config(self) -> TtSDXLPipelineConfig:
        return TtSDXLPipelineConfig(
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            is_galaxy=self.is_galaxy,
            use_cfg_parallel=self.use_cfg_parallel,
            encoders_on_device=self.encoders_on_device,
            vae_on_device=self.vae_on_device,
            capture_trace=self.capture_trace,
            crop_coords_top_left=self.crop_coords_top_left,
            guidance_rescale=self.guidance_rescale,
        )

    def create_refiner_config(self) -> TtSDXLImg2ImgPipelineConfig:
        if not self.use_refiner:
            raise ValueError("Cannot create refiner config when use_refiner=False")

        return TtSDXLImg2ImgPipelineConfig(
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            is_galaxy=self.is_galaxy,
            use_cfg_parallel=self.use_cfg_parallel,
            vae_on_device=self.vae_on_device,
            encoders_on_device=self.encoders_on_device,
            strength=self.strength,
            aesthetic_score=self.aesthetic_score,
            negative_aesthetic_score=self.negative_aesthetic_score,
            capture_trace=self.capture_trace,
            crop_coords_top_left=self.crop_coords_top_left,
            guidance_rescale=self.guidance_rescale,
        )


class TtSDXLCombinedPipeline:
    """
    Combined pipeline that orchestrates SDXL base and optionally img2img refiner models.

    The base model (TtSDXLPipeline) performs initial denoising up to a specified split point,
    then the refiner model (TtSDXLImg2ImgPipeline) takes over to complete the denoising and decode the final image.
    Both models share the same scheduler instance for coordinated timestep management.

    The combined pipeline ALWAYS creates and manages the shared scheduler internally,
    ensuring proper coordination between base and refiner models.

    Usage (base-only):
        config = TtSDXLCombinedPipelineConfig(
            num_inference_steps=50,
            guidance_scale=5.0,
            is_galaxy=True,
            use_cfg_parallel=True,
            vae_on_device=True,
            encoders_on_device=True,
            capture_trace=True,
        )
        combined = TtSDXLCombinedPipeline(
            ttnn_device=mesh_device,
            torch_base_pipeline=base,
            config=config,
        )

    Usage (with refiner):
        config = TtSDXLCombinedPipelineConfig(
            num_inference_steps=50,
            guidance_scale=5.0,
            is_galaxy=True,
            use_cfg_parallel=True,
            vae_on_device=True,
            encoders_on_device=True,
            capture_trace=True,
            use_refiner=True,
            denoising_split=0.8,
            strength=0.3,
            aesthetic_score=6.0,
            negative_aesthetic_score=2.5,
        )
        combined = TtSDXLCombinedPipeline(
            ttnn_device=mesh_device,
            torch_base_pipeline=base,
            torch_refiner_pipeline=refiner,
            config=config,
        )
    """

    def __init__(
        self,
        ttnn_device,
        torch_base_pipeline,
        config: TtSDXLCombinedPipelineConfig,
        torch_refiner_pipeline=None,
    ):
        """
        Create a combined pipeline with automatic scheduler sharing.

        The combined pipeline creates a single shared scheduler and uses it for both
        base and refiner pipelines, ensuring optimal memory usage and proper coordination.

        Args:
            ttnn_device: The TTNN device
            torch_base_pipeline: Torch DiffusionPipeline for base model (required)
            config: Unified configuration for the combined pipeline (required)
            torch_refiner_pipeline: Optional Torch StableDiffusionXLImg2ImgPipeline for refiner
                                   (required if config.use_refiner=True)
        """
        logger.info("Creating combined pipeline with shared scheduler...")

        if config.use_cfg_parallel:
            assert ttnn_device.get_num_devices() % 2 == 0, "TT device must have even number of devices"
            ttnn_device.reshape(ttnn.MeshShape(2, ttnn_device.get_num_devices() // 2))

        self.ttnn_device = ttnn_device
        self.config = config

        if config.use_refiner and torch_refiner_pipeline is None:
            raise ValueError("config.use_refiner=True but torch_refiner_pipeline is None")
        if not config.use_refiner and torch_refiner_pipeline is not None:
            logger.warning("torch_refiner_pipeline provided but config.use_refiner=False, refiner will not be used")

        self.batch_size = list(ttnn_device.shape)[1] if config.use_cfg_parallel else ttnn_device.get_num_devices()

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

        logger.info("Creating base pipeline config from unified config...")
        base_config = config.create_base_config()

        if config.use_refiner and base_config.vae_on_device:
            logger.info("Disabling VAE on base pipeline since refiner will handle final decoding")
            base_config.vae_on_device = False

        logger.info("Creating base pipeline with shared scheduler...")
        self.base_pipeline = TtSDXLPipeline(
            ttnn_device=ttnn_device,
            torch_pipeline=torch_base_pipeline,
            pipeline_config=base_config,
            tt_scheduler=self.shared_scheduler,
        )

        if config.use_refiner:
            logger.info("Creating refiner pipeline config from unified config...")
            refiner_config = config.create_refiner_config()

            logger.info("Creating refiner pipeline with shared scheduler...")
            self.refiner_pipeline = TtSDXLImg2ImgPipeline(
                ttnn_device=ttnn_device,
                torch_pipeline=torch_refiner_pipeline,
                pipeline_config=refiner_config,
                tt_scheduler=self.shared_scheduler,
            )
        else:
            self.refiner_pipeline = None

        self._compiled = False

        logger.info("Combined pipeline initialized successfully")

    def set_denoising_split(self, denoising_split: float):
        assert self.config.use_refiner, "Cannot set denoising split when use_refiner=False"
        if not isinstance(denoising_split, (float)):
            raise ValueError(f"denoising_split must be float but is {type(denoising_split)}")
        if not (0.0 <= denoising_split <= 1.0):
            raise ValueError(f"denoising_split must be in range [0.0, 1.0] but is {denoising_split}")

        self.config.denoising_split = denoising_split

    def _auto_compile_if_needed(self):
        if self._compiled:
            return

        logger.info("=" * 80)
        logger.info("Auto-compiling pipelines (first-time setup)...")
        logger.info("=" * 80)

        # 1. Compile text encoders if on device
        if self.config.encoders_on_device:
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
        self.base_pipeline.compile_image_processing(force_no_trace=True)

        # 4. Compile refiner if enabled
        if self.config.use_refiner:
            if self.config.encoders_on_device:
                logger.info("Compiling text encoders...")
                self.refiner_pipeline.compile_text_encoding()

            logger.info("Allocating device tensors for refiner pipeline...")
            dummy_latents = torch.randn(self.batch_size, 4, 128, 128)
            refiner_dummy_embeds = torch.randn(
                self.batch_size, 2, MAX_SEQUENCE_LENGTH, CONCATENATED_TEXT_EMBEDINGS_SIZE_REFINER
            )

            _, _, _ = self.refiner_pipeline.generate_input_tensors(
                all_prompt_embeds_torch=refiner_dummy_embeds,
                torch_add_text_embeds=dummy_text_embeds,
                torch_image=dummy_latents,
                denoising_start=self.config.denoising_split,
            )

            logger.info("Compiling refiner pipeline image processing...")
            self.refiner_pipeline.compile_image_processing()

        if self.config.capture_trace:
            self.base_pipeline._TtSDXLPipeline__trace_image_processing()

        self._compiled = True
        logger.info("=" * 80)
        logger.info("Pipeline compilation complete!")
        logger.info("=" * 80)

    def _calculate_timestep_split(self, num_inference_steps, denoising_split):
        discrete_timestep_cutoff = int(
            round(
                self.shared_scheduler.num_train_timesteps
                - (denoising_split * self.shared_scheduler.num_train_timesteps)
            )
        )

        split_idx = len([t for t in self.shared_scheduler.torch_timesteps if t < discrete_timestep_cutoff])

        return num_inference_steps - split_idx

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
        denoising_split=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if prompt_2 is not None and isinstance(prompt_2, str):
            prompt_2 = [prompt_2]

        needed_padding = (self.batch_size - len(prompts) % self.batch_size) % self.batch_size
        if isinstance(negative_prompts, list):
            assert len(negative_prompts) == len(prompts), "prompts and negative_prompt lists must be the same length"

        prompts = prompts + [""] * needed_padding
        if prompt_2 is not None:
            prompt_2 = prompt_2 + [""] * needed_padding
        if isinstance(negative_prompts, list):
            negative_prompts = negative_prompts + [""] * needed_padding

        if guidance_scale is not None and guidance_scale != self.config.guidance_scale:
            self.base_pipeline.set_guidance_scale(guidance_scale)
            if self.config.use_refiner:
                self.refiner_pipeline.set_guidance_scale(guidance_scale)

        if crop_coords_top_left is not None and crop_coords_top_left != self.config.crop_coords_top_left:
            self.base_pipeline.set_crop_coords_top_left(crop_coords_top_left)
            if self.config.use_refiner:
                self.refiner_pipeline.set_crop_coords_top_left(crop_coords_top_left)

        if guidance_rescale is not None and guidance_rescale != self.config.guidance_rescale:
            self.base_pipeline.set_guidance_rescale(guidance_rescale)
            if self.config.use_refiner:
                self.refiner_pipeline.set_guidance_rescale(guidance_rescale)

        if num_inference_steps is not None and num_inference_steps != self.config.num_inference_steps:
            self.base_pipeline.set_num_inference_steps(num_inference_steps)

            if self.config.use_refiner:
                self.refiner_pipeline.set_num_inference_steps(num_inference_steps)

        if denoising_split is not None and denoising_split != self.config.denoising_split:
            self.set_denoising_split(denoising_split)

        profiler.start("auto_compile_if_needed")
        self._auto_compile_if_needed()
        profiler.end("auto_compile_if_needed")

        profiler.start("combined_generation")

        num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else self.config.num_inference_steps
        )

        split_idx = (
            self._calculate_timestep_split(num_inference_steps, self.config.denoising_split)
            if self.config.use_refiner
            else num_inference_steps
        )

        # For refiner case, we need to mark that input tensors need regeneration since we updated the config
        if (
            self.config.use_refiner
            and num_inference_steps is not None
            and num_inference_steps != self.config.num_inference_steps
        ):
            logger.info(
                f"Configuring pipelines: base will use {split_idx} steps, refiner will use {num_inference_steps - split_idx} steps from {num_inference_steps} total"
            )
            self.base_pipeline.generated_input_tensors = False

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

        # 3. Run base pipeline with timesteps[0:split_idx]
        if self.config.use_refiner:
            logger.info(
                f"Denoising split: base will perform {split_idx}/{num_inference_steps} steps, "
                f"refiner will perform {num_inference_steps - split_idx}/{num_inference_steps} steps"
            )
            self.base_pipeline.num_inference_steps = split_idx

            logger.info(f"Running base pipeline for {split_idx} steps...")
            self.base_pipeline.prepare_input_tensors([tt_latents, tt_prompt_embeds[0], tt_add_text_embeds[0]])
            base_latents = self.base_pipeline.generate_images(return_latents=True)  # Skip VAE decoding for refiner

            # 4. Run refiner pipeline with timesteps[split_idx:] and base latents
            logger.info(f"Running refiner pipeline for {num_inference_steps - split_idx} steps...")

            all_prompt_embeds_torch_refiner, torch_add_text_embeds_refiner = self.refiner_pipeline.encode_prompts(
                prompts=prompts,
                negative_prompts=negative_prompts,
                prompt_2=prompt_2,
                negative_prompt_2=negative_prompt_2,
            )
            (
                refiner_latents,
                refiner_prompt_embeds,
                refiner_add_text_embeds,
            ) = self.refiner_pipeline.generate_input_tensors(
                all_prompt_embeds_torch=all_prompt_embeds_torch_refiner,
                torch_add_text_embeds=torch_add_text_embeds_refiner,
                torch_image=base_latents,
                fixed_seed_for_batch=True,
                start_latent_seed=0,
                timesteps=timesteps,
                sigmas=sigmas,
                denoising_start=self.config.denoising_split,
            )

            self.refiner_pipeline.prepare_input_tensors(
                [
                    refiner_latents,
                    refiner_prompt_embeds[0],
                    refiner_add_text_embeds[0],
                ]
            )
            images = self.refiner_pipeline.generate_images()

            self.refiner_pipeline.tt_scheduler.set_step_index(0)
        else:
            logger.info(f"Running base pipeline only for {num_inference_steps} steps...")
            self.base_pipeline.prepare_input_tensors([tt_latents, tt_prompt_embeds[0], tt_add_text_embeds[0]])
            images = self.base_pipeline.generate_images()

        profiler.end("combined_generation")
        logger.info("Combined generation completed")

        return images

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import torch
import ttnn
from diffusers import DiffusionPipeline
from PIL import Image
from models.experimental.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE, SDXL_TRACE_REGION_SIZE
from sdxl_config import SDXLConfig
from utils.logger import setup_logger
from utils.image_utils import tensor_to_pil
from typing import List


class SDXLRunner:
    """
    Wrapper for TtSDXLPipeline that handles device initialization,
    model loading, warmup, and inference execution.

    Based on:
    - /home/tt-admin/tt-inference-server/tt-media-server/tt_model_runners/sdxl_generate_runner_trace.py
    - /home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/demo/demo.py
    """

    def __init__(self, worker_id: int, config: SDXLConfig):
        self.worker_id = worker_id
        self.config = config
        self.logger = setup_logger(f"SDXLRunner-{worker_id}")
        self.ttnn_device = None
        self.pipeline = None
        self.tt_sdxl = None

    def initialize_device(self):
        """Initialize mesh device with proper configuration"""
        self.logger.info(f"Initializing device for worker {self.worker_id}")

        # Skip validation - launch script already validated devices
        # ttnn will handle device availability during initialization

        # Set device visibility to only this worker's device
        # For T3K: each worker gets one device (1x1 mesh)
        worker_device_id = (
            str(self.config.device_ids[0])
            if len(self.config.device_ids) == 1
            else str(self.config.device_ids[self.worker_id % len(self.config.device_ids)])
        )
        os.environ["TT_VISIBLE_DEVICES"] = worker_device_id
        os.environ["TT_METAL_VISIBLE_DEVICES"] = worker_device_id

        self.logger.info(f"Worker {self.worker_id} assigned to device {worker_device_id}")

        # Set mesh configuration
        os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "7,7"

        # No fabric config needed for 1x1 mesh (single device)

        # Always use open_mesh_device like tt-media-server does (even for 1x1 mesh)
        # This ensures consistent behavior with the reference implementation
        dispatch_core_config = ttnn.DispatchCoreConfig()  # Uses system defaults
        rows, cols = self.config.device_mesh_shape

        # Use open_mesh_device for all mesh shapes (matching tt-media-server)
        self.ttnn_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(rows, cols),
            l1_small_size=self.config.l1_small_size,
            trace_region_size=self.config.trace_region_size,
            dispatch_core_config=dispatch_core_config,
        )

        self.logger.info(f"Device initialized with mesh shape {self.config.device_mesh_shape}")
        return self.ttnn_device

    def load_model(self, kernel_ready_queue=None):
        """Load HuggingFace pipeline and create TtSDXLPipeline with warmup

        Args:
            kernel_ready_queue: Optional queue to signal when kernel compilation is complete
                               (allows next worker to start before program cache warmup finishes)
        """
        self.logger.info("Loading SDXL model...")

        # Load torch pipeline (auto-downloads if not cached)
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
            cache_dir=self.config.hf_home,
        )
        self.logger.info("HuggingFace pipeline loaded")

        # Create TtSDXLPipeline
        self.tt_sdxl = TtSDXLPipeline(
            ttnn_device=self.ttnn_device,
            torch_pipeline=self.pipeline,
            pipeline_config=TtSDXLPipelineConfig(
                encoders_on_device=self.config.encoders_on_device,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                is_galaxy=self.config.is_galaxy,
                capture_trace=self.config.capture_trace,
                vae_on_device=self.config.vae_on_device,
                use_cfg_parallel=self.config.use_cfg_parallel,
            ),
        )
        self.logger.info("TtSDXLPipeline created")

        # Compile text encoders
        self.logger.info("Compiling text encoders...")
        self.tt_sdxl.compile_text_encoding()

        # Signal that kernel compilation is complete - next worker can start
        # Program cache warmup below can run in parallel with other workers
        if kernel_ready_queue is not None:
            self.logger.info(f"Worker {self.worker_id} kernel compilation complete, signaling...")
            kernel_ready_queue.put(self.worker_id)

        # Warmup inference (1 step)
        self.logger.info("Performing warmup inference...")
        warmup_prompt = ["Sunrise on a beach"]
        warmup_negative = [""]

        prompt_embeds, text_embeds = self.tt_sdxl.encode_prompts(warmup_prompt, warmup_negative)

        tt_latents, tt_prompts, tt_texts = self.tt_sdxl.generate_input_tensors(prompt_embeds, text_embeds)

        self.tt_sdxl.prepare_input_tensors([tt_latents, tt_prompts, tt_texts])

        # NOTE: DO NOT set guidance_rescale before trace capture.
        # tt-media-server traces with default 0.0 - device tensor values can be
        # updated before trace execution since trace captures addresses not values.

        # Compile with warmup
        self.logger.info("Compiling image processing...")
        self.tt_sdxl.compile_image_processing()

        self.logger.info("Model loaded and warmed up successfully")

    def run_inference(self, requests: List[dict]) -> List[Image.Image]:
        """
        Run inference for batch of requests

        Args:
            requests: List of request dictionaries with 'prompt', 'negative_prompt', etc.

        Returns:
            List of PIL Images
        """
        prompts = [req["prompt"] for req in requests]
        prompts_2 = [req.get("prompt_2") for req in requests]
        negative_prompts = [req.get("negative_prompt", "") for req in requests]
        negative_prompts_2 = [req.get("negative_prompt_2") for req in requests]

        # Update num_inference_steps if specified in request
        if requests[0].get("num_inference_steps") is not None:
            self.tt_sdxl.set_num_inference_steps(requests[0]["num_inference_steps"])

        # Update guidance_scale if specified in request
        if requests[0].get("guidance_scale") is not None:
            self.tt_sdxl.set_guidance_scale(requests[0]["guidance_scale"])

        # Update guidance_rescale if specified in request
        if requests[0].get("guidance_rescale") is not None:
            self.tt_sdxl.set_guidance_rescale(requests[0]["guidance_rescale"])

        # Reset scheduler state for new inference run (fixes progress bar and ensures correct timesteps)
        self.tt_sdxl.tt_scheduler.set_begin_index(0)

        # Encode prompts (including prompt_2 for SDXL's dual text encoder)
        prompt_embeds, text_embeds = self.tt_sdxl.encode_prompts(
            prompts, negative_prompts, prompt_2=prompts_2, negative_prompt_2=negative_prompts_2
        )

        # Extract seed from request for reproducibility
        seed = requests[0].get("seed")

        # Generate tensors with seed
        tt_latents, tt_prompts, tt_texts = self.tt_sdxl.generate_input_tensors(
            prompt_embeds, text_embeds, start_latent_seed=seed
        )

        self.tt_sdxl.prepare_input_tensors([tt_latents, tt_prompts, tt_texts])
        img_tensors = self.tt_sdxl.generate_images()

        images = []
        for img_tensor in img_tensors:
            pil_img = tensor_to_pil(img_tensor, self.pipeline.image_processor)
            images.append(pil_img)
        return images

    def close_device(self):
        """Close device and cleanup"""
        # Release traces BEFORE closing device to prevent segfault
        if self.tt_sdxl:
            self.logger.info("Releasing traces before device closure")
            self.tt_sdxl.release_traces()

        if self.ttnn_device:
            # Always use close_mesh_device since we always use open_mesh_device
            self.logger.info("Closing mesh device")
            ttnn.close_mesh_device(self.ttnn_device)

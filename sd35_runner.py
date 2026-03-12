# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import ttnn
from PIL import Image
from typing import List

from sd35_config import SD35Config
from utils.logger import setup_logger


class SD35Runner:
    """
    Wrapper for StableDiffusion3Pipeline that handles device initialization,
    model loading, warmup, and inference execution.

    Key architectural note: SD3.5 guidance_scale is a startup parameter baked in
    at create_pipeline() time via pipeline.prepare(). There is no per-request
    setter (unlike SDXL's set_guidance_scale()). A warning is logged if a
    per-request guidance_scale is requested that differs from the configured value.
    """

    def __init__(self, worker_id: int, config: SD35Config):
        self.worker_id = worker_id
        self.config = config
        self.logger = setup_logger(f"SD35Runner-{worker_id}")
        self.mesh_device = None
        self.pipeline = None

    def initialize_device(self):
        """Initialize 2x4 mesh device with FABRIC_1D and SD3.5-specific parameters"""
        self.logger.info(f"Initializing 2x4 mesh device for worker {self.worker_id}")

        rows, cols = self.config.device_mesh_shape

        # Map the fabric_config_name string to the ttnn enum value
        fabric_config = getattr(ttnn.FabricConfig, self.config.fabric_config_name)

        self.mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(rows, cols),
            l1_small_size=self.config.l1_small_size,
            trace_region_size=self.config.trace_region_size,
            dispatch_core_config=ttnn.DispatchCoreConfig(),
            fabric_config=fabric_config,
        )

        self.logger.info(
            f"Mesh device initialized: shape={self.config.device_mesh_shape}, "
            f"fabric={self.config.fabric_config_name}, "
            f"l1_small_size={self.config.l1_small_size}, "
            f"trace_region_size={self.config.trace_region_size}"
        )
        return self.mesh_device

    def load_model(self, kernel_ready_queue=None):
        """Create SD3.5 pipeline and run warmup inference.

        The pipeline is created via StableDiffusion3Pipeline.create_pipeline() which
        handles all model loading and device setup. A single warmup run is performed
        to compile kernels and optionally capture execution traces.

        Args:
            kernel_ready_queue: Optional queue to signal when the pipeline is ready
                                for inference (warmup complete). For SD3.5 with
                                num_workers=1, the server never waits on this queue,
                                but the signal is sent for protocol consistency.
        """
        # Deferred import: avoid importing ttnn/pipeline in the main process
        from models.tt_dit.pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
            StableDiffusion3Pipeline,
        )

        self.logger.info("Creating SD3.5 Large pipeline...")
        self.logger.info(f"  model_name: {self.config.model_name}")
        self.logger.info(f"  image size: {self.config.image_width}x{self.config.image_height}")
        self.logger.info(f"  guidance_scale: {self.config.guidance_scale}")
        self.logger.info(f"  use_trace: {self.config.use_trace}")
        self.logger.info(f"  num_inference_steps: {self.config.num_inference_steps}")

        self.pipeline = StableDiffusion3Pipeline.create_pipeline(
            mesh_device=self.mesh_device,
            batch_size=1,
            image_w=self.config.image_width,
            image_h=self.config.image_height,
            guidance_scale=self.config.guidance_scale,
            cfg_config=self.config.cfg_config,
            sp_config=self.config.sp_config,
            tp_config=self.config.tp_config,
            num_links=self.config.num_links,
            checkpoint_name=self.config.model_name,
        )
        self.logger.info("SD3.5 pipeline created")

        # Warmup inference: compiles kernels and (if use_trace=True) captures traces.
        # For trace capture this can take 10-15 minutes on first run.
        self.logger.info("Running warmup inference (compiles kernels / captures trace)...")
        self.pipeline.run_single_prompt(
            prompt="A golden sunrise over mountain peaks",
            negative_prompt="",
            num_inference_steps=self.config.num_inference_steps,
            seed=42,
            traced=self.config.use_trace,
        )
        self.logger.info("Warmup complete")

        # Signal kernel ready — for SD3.5 num_workers=1 so the server's overlapped
        # startup loop (range(num_workers - 1) == range(0)) never waits here, but
        # we send the signal for protocol consistency.
        if kernel_ready_queue is not None:
            self.logger.info(f"Worker {self.worker_id} signaling kernel ready")
            kernel_ready_queue.put(self.worker_id)

    def run_inference(self, requests: List[dict]) -> List[Image.Image]:
        """Run inference for a single SD3.5 request.

        SD3.5 processes one request at a time (batch_size=1 across 2x4 mesh).
        Only the first request in the list is used. guidance_scale is baked in at
        create_pipeline() time and cannot be changed per-request; a warning is
        logged if the caller requests a different value.

        Args:
            requests: List of request dicts. Only requests[0] is used.

        Returns:
            List containing one PIL Image.
        """
        request = requests[0]

        prompt = request["prompt"]
        negative_prompt = request.get("negative_prompt", "")
        num_inference_steps = request.get("num_inference_steps") or self.config.num_inference_steps
        seed = request.get("seed")

        # Warn if caller requests a guidance_scale that differs from the baked-in value
        requested_guidance = request.get("guidance_scale")
        if requested_guidance is not None and requested_guidance != self.config.guidance_scale:
            self.logger.warning(
                f"Per-request guidance_scale={requested_guidance} requested, but SD3.5 guidance_scale "
                f"is fixed at {self.config.guidance_scale} (set at pipeline creation time). "
                f"The configured value will be used."
            )

        self.logger.info(f"Running inference: prompt='{prompt[:80]}', steps={num_inference_steps}, seed={seed}")

        images = self.pipeline.run_single_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            num_inference_steps=num_inference_steps,
            seed=seed,
            traced=self.config.use_trace,
        )

        return images

    def close_device(self):
        """Synchronize all submesh devices then close the mesh device"""
        if self.pipeline is not None:
            self.logger.info("Synchronizing submesh devices before closure")
            self.pipeline.synchronize_devices()

        if self.mesh_device is not None:
            self.logger.info("Closing mesh device")
            ttnn.close_mesh_device(self.mesh_device)
            self.mesh_device = None

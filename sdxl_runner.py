# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import numpy as np
import torch
import ttnn
from diffusers import DiffusionPipeline
from PIL import Image
from models.demos.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig
from models.demos.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE, SDXL_FABRIC_CONFIG
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
    - /home/tt-admin/tt-metal/models/demos/stable_diffusion_xl_base/demo/demo.py
    """

    def __init__(self, worker_id: int, config: SDXLConfig):
        self.worker_id = worker_id
        self.config = config
        self.logger = setup_logger(f"SDXLRunner-{worker_id}")
        self.ttnn_device = None
        self.pipeline = None
        self.tt_sdxl = None
        # Currently fused LoRA, tracked as (lora_path, lora_scale). None = base
        # weights. Lets repeated requests with the same adapter skip the
        # unload/reload/fuse cycle.
        self._current_lora = None
        # Status of the most recently applied LoRA (surfaced to the client so the
        # UI can warn when an adapter is skipped/partial). None = no LoRA active.
        self._last_lora_status = None

    def initialize_device(self):
        """Initialize mesh device with proper configuration"""
        self.logger.info(f"Initializing device for worker {self.worker_id}")

        # Skip validation - launch script already validated devices
        # ttnn will handle device availability during initialization

        # When num_workers > 1 (e.g. T3K: 4 workers × 1×1 mesh), each worker
        # owns one chip and we re-pin TT_VISIBLE_DEVICES to that single chip.
        # When num_workers == 1, the worker uses all chips in config.device_ids
        # for its mesh — leave the visibility env vars as worker.py set them.
        if self.config.num_workers > 1:
            worker_device_id = str(self.config.device_ids[self.worker_id % len(self.config.device_ids)])
            os.environ["TT_VISIBLE_DEVICES"] = worker_device_id
            os.environ["TT_METAL_VISIBLE_DEVICES"] = worker_device_id
            self.logger.info(f"Worker {self.worker_id} pinned to device {worker_device_id}")
        else:
            self.logger.info(
                f"Worker {self.worker_id} using all devices "
                f"{self.config.device_ids} for {self.config.device_mesh_shape} mesh"
            )

        # Per-board compute-grid override. Wormhole-tuned boards (T3K) need
        # the legacy "7,7" cap; Blackhole P-series must use the device's
        # native compute grid so model_configs_1024x1024BH.py's (10,8)=80-core
        # matmul sharding fits. Mirrors tt-inference-server's
        # tt-media-server/utils/runner_utils.py:_setup_grid_override.
        from device_specs import get_board_spec

        override = get_board_spec(self.config.board).core_grid_override
        if override:
            os.environ["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = override
            self.logger.info(f"Set TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE={override}")
        else:
            os.environ.pop("TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE", None)
            self.logger.info("Using native compute grid (no TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE)")

        dispatch_core_config = ttnn.DispatchCoreConfig()  # Uses system defaults
        rows, cols = self.config.device_mesh_shape

        # Multi-chip tensor-parallel layouts (rows > 1) need fabric for the
        # all_gather_async ops inside cfg_parallel mode. Mirrors
        # tt-media-server/tt_model_runners/base_sdxl_runner.py:35-58 +
        # base_metal_device_runner.py: ttnn.set_fabric_config(SDXL_FABRIC_CONFIG)
        # when is_tensor_parallel.
        if rows > 1:
            ttnn.set_fabric_config(SDXL_FABRIC_CONFIG)
            self.logger.info(f"Set fabric config: {SDXL_FABRIC_CONFIG}")

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

        # Warmup inference — mirror tt-media-server's _warmup_inference_block
        # (sdxl_generate_runner_trace.py:45-62 + base_sdxl_runner.py:198-212).
        # The reference applies request settings (steps=1, guidance=5.0,
        # rescale=0.7, crop=(0,0)) and encodes prompt_2/negative_prompt_2
        # BEFORE compile_image_processing captures the trace. Diverging from
        # this baked stale values into the trace and produced SSIM ~0 images.
        self.logger.info("Performing warmup inference...")
        self.tt_sdxl.set_num_inference_steps(1)
        self.tt_sdxl.set_guidance_scale(5.0)
        self.tt_sdxl.set_guidance_rescale(0.7)
        self.tt_sdxl.set_crop_coords_top_left((0, 0))

        prompt_embeds, text_embeds = self.tt_sdxl.encode_prompts(
            ["Sunrise on a beach"],
            ["low resolution"],
            prompt_2=["Mountains in the background"],
            negative_prompt_2=["blurry"],
        )

        tt_latents, tt_prompts, tt_texts = self.tt_sdxl.generate_input_tensors(prompt_embeds, text_embeds)

        self.tt_sdxl.prepare_input_tensors([tt_latents, tt_prompts[0], tt_texts[0]])

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
        prompts_2_raw = [req.get("prompt_2") for req in requests]
        prompts_2 = (
            None
            if all(p in (None, "") for p in prompts_2_raw)
            else [(p if (p is not None and p != "") else "") for p in prompts_2_raw]
        )
        neg_raw = [req.get("negative_prompt") for req in requests]
        negative_prompts = (
            None
            if all(np in (None, "") for np in neg_raw)
            else [(np if (np is not None and np != "") else "") for np in neg_raw]
        )
        neg2_raw = [req.get("negative_prompt_2") for req in requests]
        negative_prompts_2 = (
            None
            if all(np in (None, "") for np in neg2_raw)
            else [(np if (np is not None and np != "") else "") for np in neg2_raw]
        )

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

        # Mirror reference's per-request defensive call (idempotent in SDK).
        self.tt_sdxl.compile_text_encoding()

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

        self.tt_sdxl.prepare_input_tensors([tt_latents, tt_prompts[0], tt_texts[0]])
        img_tensors = self.tt_sdxl.generate_images()

        images = []
        for img_tensor in img_tensors:
            pil_img = tensor_to_pil(img_tensor, self.pipeline.image_processor)
            images.append(pil_img)
        return images

    # ------------------------------------------------------------------
    # Staged operations (additive — used by the ComfyUI HTTP nodes).
    # The full-pipeline run_inference() path above is left untouched.
    # ------------------------------------------------------------------

    def _apply_request_lora(self, request: dict):
        """Apply the per-request LoRA adapter on device (load + fuse), caching the
        currently-fused adapter so repeated requests skip the reload.

        The adapter is validated and fused into the base UNet weights via the
        pipeline's TtLoRAWeightsManager. Switching adapters first unloads the
        previous one (restoring base weights) so deltas never stack. An empty
        ``lora_path`` restores base weights.
        """
        lora_path = (request.get("lora_path") or "").strip()
        scale = request.get("lora_scale")
        scale = float(scale) if scale is not None else 1.0

        if not lora_path:
            if self._current_lora is not None:
                self.logger.info("Clearing active LoRA (request has no adapter)")
                self.tt_sdxl.unload_lora_weights()
                self._current_lora = None
            self._last_lora_status = None
            return

        key = (lora_path, scale)
        if key == self._current_lora:
            return  # already fused on device; keep the existing status

        # Switching adapters (or changing scale): restore base weights first so
        # the new delta is applied to the unmodified base, not a stacked weight.
        if self._current_lora is not None:
            self.tt_sdxl.unload_lora_weights()
            self._current_lora = None

        self.logger.info(f"Loading LoRA lora_path={lora_path!r}, lora_scale={scale}")
        self.tt_sdxl.load_lora_weights(lora_path)
        self.tt_sdxl.fuse_lora(lora_scale=scale)

        status = self.tt_sdxl.get_lora_status()
        applied = bool(status.get("unet") or status.get("text_encoder"))
        self._last_lora_status = {
            "requested": lora_path,
            "scale": scale,
            "applied": applied,
            "unet": bool(status.get("unet")),
            "text_encoder": bool(status.get("text_encoder")),
            "skipped_reason": status.get("skipped_reason"),
        }
        if not applied:
            self.logger.warning(f"LoRA {lora_path!r} had no effect (skipped_reason={status.get('skipped_reason')})")
        elif status.get("skipped_reason"):
            self.logger.warning(f"LoRA {lora_path!r} partially applied: {status.get('skipped_reason')}")
        self._current_lora = key

    def _prompt_lists(self, request: dict):
        """Build the (prompts, negative, prompt_2, negative_2) lists from one request,
        mirroring run_inference()'s normalization."""
        prompts = [request.get("prompt", "")]
        p2 = request.get("prompt_2")
        prompts_2 = None if p2 in (None, "") else [p2]
        neg = request.get("negative_prompt")
        negative_prompts = None if neg in (None, "") else [neg]
        neg2 = request.get("negative_prompt_2")
        negative_prompts_2 = None if neg2 in (None, "") else [neg2]
        return prompts, negative_prompts, prompts_2, negative_prompts_2

    def _apply_request_settings(self, request: dict):
        """Apply per-request scheduler/guidance settings (subset of run_inference)."""
        if request.get("num_inference_steps") is not None:
            self.tt_sdxl.set_num_inference_steps(request["num_inference_steps"])
        if request.get("guidance_scale") is not None:
            self.tt_sdxl.set_guidance_scale(request["guidance_scale"])
        if request.get("guidance_rescale") is not None:
            self.tt_sdxl.set_guidance_rescale(request["guidance_rescale"])
        self.tt_sdxl.tt_scheduler.set_begin_index(0)
        self.tt_sdxl.compile_text_encoding()

    def _latents_to_numpy(self, latents) -> np.ndarray:
        """Convert device/host latents to standard diffusers layout [B, C, H, W] float32.

        generate_images(return_latents=True) returns the raw scheduler latents (NOT
        divided by the VAE scaling factor) in the TT layout [N, 1, H*W, C].
        """
        if not torch.is_tensor(latents):
            latents = ttnn.to_torch(latents, mesh_composer=ttnn.ConcatMeshToTensor(self.ttnn_device, dim=0)).float()
        else:
            latents = latents.float()

        c = self.tt_sdxl.num_in_channels_unet
        h = self.tt_sdxl.height // self.pipeline.vae_scale_factor
        w = self.tt_sdxl.width // self.pipeline.vae_scale_factor
        n = latents.shape[0]
        latents = latents.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()
        latents = latents[: self.tt_sdxl.batch_size]
        return latents.cpu().numpy()

    def denoise(self, request: dict, on_event=None) -> np.ndarray:
        """Staged: encode prompts + run the (traced) UNet denoise loop on device.

        Returns raw scheduler latents as numpy [B, C, H, W] (consumed by vae_decode).

        If ``on_event`` is provided, a DenoiseStep event is emitted once per
        denoise iteration so the server can stream per-step progress.
        """
        self._apply_request_lora(request)
        self._apply_request_settings(request)
        prompts, negative_prompts, prompts_2, negative_prompts_2 = self._prompt_lists(request)
        prompt_embeds, text_embeds = self.tt_sdxl.encode_prompts(
            prompts, negative_prompts, prompt_2=prompts_2, negative_prompt_2=negative_prompts_2
        )
        seed = request.get("seed")
        tt_latents, tt_prompts, tt_texts = self.tt_sdxl.generate_input_tensors(
            prompt_embeds, text_embeds, start_latent_seed=seed
        )
        self.tt_sdxl.prepare_input_tensors([tt_latents, tt_prompts[0], tt_texts[0]])

        on_step = None
        if on_event is not None:
            from models.tt_dit.pipelines.events import DenoiseStep

            def on_step(step, total):
                on_event(DenoiseStep(step=step, total=total, sigma=0.0))

        latents = self.tt_sdxl.generate_images(return_latents=True, on_step=on_step)
        return self._latents_to_numpy(latents)

    def vae_decode(self, latents_np: np.ndarray) -> np.ndarray:
        """Staged: decode raw scheduler latents [B, C, H, W] -> image [B, H, W, C] in [0, 1].

        POC uses the host torch VAE for a stable, additive contract; on-device VAE
        decode of arbitrary latents is a follow-up optimization.
        """
        latents = torch.from_numpy(np.ascontiguousarray(latents_np)).float()
        vae = self.pipeline.vae
        latents = latents / vae.config.scaling_factor
        with torch.no_grad():
            image = vae.decode(latents.to(vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0.0, 1.0)
        image = image.permute(0, 2, 3, 1).contiguous().float()
        return image.cpu().numpy()

    def vae_encode(self, image_np: np.ndarray) -> np.ndarray:
        """Staged: encode image [B, H, W, C] in [0, 1] -> raw scheduler latents [B, C, H, W].

        Output matches the denoise() latent contract (scaled by the VAE scaling factor).
        """
        image = torch.from_numpy(np.ascontiguousarray(image_np)).float()
        if image.ndim != 4:
            raise ValueError(f"vae_encode expects [B, H, W, C], got shape {tuple(image.shape)}")
        image = image.permute(0, 3, 1, 2).contiguous()
        image = 2.0 * image - 1.0
        vae = self.pipeline.vae
        with torch.no_grad():
            latents = vae.encode(image.to(vae.dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        return latents.float().cpu().numpy()

    def close_device(self):
        """Close device and cleanup"""
        # Release traces BEFORE closing the device to prevent a segfault.
        # main's TtSDXLPipeline has no public release_traces(); trace cleanup runs
        # in __del__ -> __release_trace (ttnn.release_trace for the unet/vae tids).
        # Drop our only reference and force collection so that runs now,
        # deterministically, before close_mesh_device.
        if self.tt_sdxl is not None:
            self.logger.info("Releasing SDXL pipeline (trace cleanup) before device closure")
            self.tt_sdxl = None
            import gc

            gc.collect()

        if self.ttnn_device:
            # Always use close_mesh_device since we always use open_mesh_device
            self.logger.info("Closing mesh device")
            ttnn.close_mesh_device(self.ttnn_device)

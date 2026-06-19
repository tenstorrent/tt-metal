# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import List

import numpy as np
from utils.logger import setup_logger
from wan_config import WanConfig

import ttnn


class WanRunner:
    """Wrapper for tt-metal WanPipeline (Wan2.2 T2V).

    Mirrors the runner contract: initialize_device → load_model → run_inference → close_device.
    Returns numpy uint8 frames of shape (T, H, W, 3) per request.
    """

    def __init__(self, worker_id: int, config: WanConfig):
        self.worker_id = worker_id
        self.config = config
        self.logger = setup_logger(f"WanRunner-{worker_id}")
        self.mesh_device = None
        self.pipeline = None
        self._fabric_config = None
        # Per-worker LoRA adapter cache: (high_path, low_path, scale) -> registered name.
        # Repeated requests with the same adapter skip the (slow) reload/register and
        # only re-bind. ``_active_lora_key`` tracks what is currently fused on device.
        self._lora_cache: dict = {}
        self._active_lora_key = None

    def initialize_device(self):
        rows, cols = self.config.device_mesh_shape
        self.logger.info(
            f"Initializing {rows}x{cols} mesh device for worker {self.worker_id} "
            f"on board {self.config.board.name.lower()}"
        )

        fabric_config = getattr(ttnn.FabricConfig, self.config.fabric_config_name)
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
        )
        self._fabric_config = fabric_config

        self.mesh_device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(rows, cols),
            l1_small_size=self.config.l1_small_size,
            trace_region_size=self.config.trace_region_size,
            dispatch_core_config=ttnn.DispatchCoreConfig(),
        )
        self.logger.info(
            f"Mesh device initialized: shape={tuple(self.mesh_device.shape)}, "
            f"fabric={self.config.fabric_config_name}, "
            f"l1_small_size={self.config.l1_small_size}, "
            f"trace_region_size={self.config.trace_region_size}"
        )
        return self.mesh_device

    def load_model(self, kernel_ready_queue=None):
        # When LoRA is enabled the transformer is built with LoRA-aware Linears so
        # adapters can be hot-swapped per request. Fuse mode keeps forward/trace
        # overhead at zero, so non-LoRA requests are unaffected.
        if self.config.lora_enabled:
            from models.tt_dit.experimental.pipelines.pipeline_wan_runtime_lora import WanPipelineRuntimeLoRA

            pipeline_cls = WanPipelineRuntimeLoRA
        else:
            from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline

            pipeline_cls = WanPipeline

        self.logger.info("Creating Wan2.2 T2V pipeline...")
        self.logger.info(f"  model_name: {self.config.model_name}")
        self.logger.info(f"  size: {self.config.width}x{self.config.height}, frames: {self.config.num_frames}")
        self.logger.info(f"  steps: {self.config.num_inference_steps}, use_trace: {self.config.use_trace}")
        self.logger.info(f"  lora_enabled: {self.config.lora_enabled}")

        # Geometry (height/width/num_frames) and CFG are fixed at creation; guidance_scale
        # is applied per call. Enable CFG so per-request guidance_scale > 1 is accepted.
        self.pipeline = pipeline_cls.create_pipeline(
            mesh_device=self.mesh_device,
            checkpoint_name=self.config.model_name,
            height=self.config.height,
            width=self.config.width,
            num_frames=self.config.num_frames,
            cfg_enabled=self.config.guidance_scale > 1,
        )
        self.logger.info("Wan2.2 pipeline created")

        self.logger.info("Running warmup inference (compiles kernels / captures trace)...")
        # num_frames/height/width are fixed at creation; __call__ does not accept them.
        self.pipeline(
            prompts=["A golden sunrise over mountain peaks"],
            num_inference_steps=2,
            guidance_scale=self.config.guidance_scale,
            guidance_scale_2=self.config.guidance_scale_2,
            seed=42,
            output_type="uint8",
            traced=self.config.use_trace,
        )
        self.logger.info("Warmup complete")

        if kernel_ready_queue is not None:
            kernel_ready_queue.put(self.worker_id)

    def _apply_request_lora(self, request: dict) -> None:
        """Register (if new) and bind the per-request LoRA adapter on both experts.

        ``high_lora_path`` targets the high-noise expert (transformer) and
        ``low_lora_path`` the low-noise expert (transformer_2); either or both may
        be supplied. Adapters are cached per (high, low, scale) so repeated requests
        with the same adapter only re-bind (sub-second) rather than reload weights.
        Passing no LoRA fields restores the base weights.
        """
        if not hasattr(self.pipeline, "register_lora"):
            if request.get("high_lora_path") or request.get("low_lora_path"):
                self.logger.warning(
                    "LoRA requested but pipeline was built without lora_enabled; ignoring. "
                    "Restart the server with WanConfig.lora_enabled=True to use adapters."
                )
            return

        high = request.get("high_lora_path") or None
        low = request.get("low_lora_path") or None
        scale = request.get("lora_scale")
        scale = float(scale) if scale is not None else 1.0
        self.logger.info(f"_apply_request_lora: high={high!r}, low={low!r}, scale={scale}")

        if (not high and not low) or scale == 0.0:
            if self._active_lora_key is not None:
                reason = "lora_scale=0.0" if scale == 0.0 else "request has no adapter"
                self.logger.info(f"Clearing active LoRA ({reason})")
                self.pipeline.set_active_lora(None)
                self._active_lora_key = None
            return

        key = (high, low, scale)
        if key == self._active_lora_key:
            return  # already fused on device — no work

        name = self._lora_cache.get(key)
        if name is None:
            name = f"lora_{len(self._lora_cache)}"
            self.pipeline.register_lora(name, high_path=high, low_path=low, scale=scale)
            self._lora_cache[key] = name
            self.logger.info(f"Registered LoRA '{name}': high={high}, low={low}, scale={scale}")

        self.pipeline.set_active_lora(name)
        self._active_lora_key = key
        self.logger.info(f"Activated LoRA '{name}'")

    def run_inference(self, requests: List[dict]) -> List[np.ndarray]:
        request = requests[0]
        self._apply_request_lora(request)

        prompt = request["prompt"]
        negative_prompt = request.get("negative_prompt") or None
        num_inference_steps = request.get("num_inference_steps") or self.config.num_inference_steps
        # num_frames / height / width are fixed at pipeline creation; __call__ reads
        # self._num_frames/_height/_width and does not accept per-request overrides.
        guidance_scale = request.get("guidance_scale") or self.config.guidance_scale
        guidance_scale_2 = request.get("guidance_scale_2") or self.config.guidance_scale_2
        # flow_shift / boundary_ratio are host-side schedule/expert-selection knobs; pass
        # through only when supplied so the pipeline keeps its construction-time defaults.
        flow_shift = request.get("flow_shift")
        boundary_ratio = request.get("boundary_ratio")
        seed = request.get("seed")

        self.logger.info(
            f"Running inference: prompt='{prompt[:80]}', steps={num_inference_steps}, "
            f"frames={self.config.num_frames}, size={self.config.width}x{self.config.height}, seed={seed}"
        )

        kwargs = dict(
            prompts=[prompt],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            seed=int(seed) if seed is not None else 0,
            output_type="uint8",
            traced=self.config.use_trace,
        )
        if negative_prompt:
            kwargs["negative_prompts"] = [negative_prompt]
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)
        if boundary_ratio is not None:
            kwargs["boundary_ratio"] = float(boundary_ratio)

        output = self.pipeline(**kwargs)
        # WanPipelineOutput.frames: numpy uint8 of shape (B, T, H, W, C). Strip batch.
        frames = output.frames if hasattr(output, "frames") else output
        if frames.ndim == 5:
            frames = frames[0]
        return [frames]

    # -- staged ops (mirror SDXLRunner.denoise / vae_decode) ---------------
    #
    # The Wan pipeline already separates the two stages: __call__(output_type="latent")
    # runs the (traced) denoise loop and returns host latents without VAE; _decode_latents
    # runs the VAE decode independently. Both are warmed during load_model's full-pipeline
    # warmup. Geometry (height/width/num_frames) stays fixed at construction time.

    def _build_call_kwargs(self, request: dict) -> dict:
        """Shared kwarg builder for run_inference / denoise (everything but output_type)."""
        prompt = request["prompt"]
        negative_prompt = request.get("negative_prompt") or None
        num_inference_steps = request.get("num_inference_steps") or self.config.num_inference_steps
        guidance_scale = request.get("guidance_scale") or self.config.guidance_scale
        guidance_scale_2 = request.get("guidance_scale_2") or self.config.guidance_scale_2
        flow_shift = request.get("flow_shift")
        boundary_ratio = request.get("boundary_ratio")
        seed = request.get("seed")

        kwargs = dict(
            prompts=[prompt],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            seed=int(seed) if seed is not None else 0,
            traced=self.config.use_trace,
        )
        if negative_prompt:
            kwargs["negative_prompts"] = [negative_prompt]
        if flow_shift is not None:
            kwargs["flow_shift"] = float(flow_shift)
        if boundary_ratio is not None:
            kwargs["boundary_ratio"] = float(boundary_ratio)
        return kwargs

    def denoise(self, request: dict, on_event=None) -> np.ndarray:
        """Staged: encode prompt (umT5) + run the (traced) denoise loop on device.

        Returns raw denoised latents as numpy [B, z_dim, F, H, W] (consumed by vae_decode).
        ``on_event`` is an optional PipelineEventCallback used to stream progress
        (SectionStart/SectionEnd/DenoiseStep) back to the caller.
        """
        self._apply_request_lora(request)
        kwargs = self._build_call_kwargs(request)
        kwargs["output_type"] = "latent"
        if on_event is not None:
            kwargs["on_event"] = on_event
        self.logger.info(
            f"Staged denoise: prompt='{request['prompt'][:80]}', steps={kwargs['num_inference_steps']}, "
            f"size={self.config.width}x{self.config.height}, seed={kwargs['seed']}"
        )
        latents = self.pipeline(**kwargs)  # torch tensor on host
        return latents.float().cpu().numpy()

    def vae_decode(self, latents_np: np.ndarray) -> np.ndarray:
        """Staged: decode raw latents [B, z_dim, F, H, W] -> frames [T, H, W, C] in [0, 1].

        Mirrors the SDXL contract (float32 image in [0, 1]); reuses the pipeline's own VAE
        decode path so device/trace setup from warmup is shared.
        """
        import torch

        from models.tt_dit.pipelines.events import null_callback

        latents = torch.from_numpy(np.ascontiguousarray(latents_np)).float()
        video = self.pipeline._decode_latents(latents, output_type="uint8", on_event=null_callback)
        # _decode_latents("uint8") returns numpy (B, T, H, W, C) uint8. Strip batch + normalize.
        if video.ndim == 5:
            video = video[0]
        return video.astype(np.float32) / 255.0

    def close_device(self):
        if self.pipeline is not None and hasattr(self.pipeline, "synchronize_devices"):
            self.logger.info("Synchronizing submesh devices before closure")
            try:
                self.pipeline.synchronize_devices()
            except Exception as e:
                self.logger.warning(f"synchronize_devices failed: {e}")

        if self.mesh_device is not None:
            self.logger.info("Closing mesh device")
            ttnn.close_mesh_device(self.mesh_device)
            self.mesh_device = None
            if self._fabric_config is not None:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                self._fabric_config = None

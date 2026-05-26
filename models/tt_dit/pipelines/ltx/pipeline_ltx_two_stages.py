# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 22B two-stage audio-video pipeline.

Mirrors the reference ``ltx_pipelines.ti2vid_two_stages.TI2VidTwoStagesPipeline``:

- **Stage 1**: half-res AV denoise on the base 22B checkpoint with full
  ``MultiModalGuider`` guidance (CFG + STG + AV-modality).
- **Spatial upsample**: stage-1 latent is x2-upsampled on CPU via the
  reference ``VideoUpsampler`` block (re-uses the helper in ``LTXFastPipeline``).
- **Stage 2**: full-res AV refine with the distilled LoRA fused into the
  transformer and no guidance (``SimpleDenoiser`` equivalent), starting from
  the upsampled video latent + stage-1 audio latent renoised at
  ``sigmas[0] = 0.909375``.

Text-only; image conditioning is not wired here yet.
"""

from __future__ import annotations

import gc
import os
import time

import torch
from loguru import logger

import ttnn

from ...utils.lora import LoraSpec
from .pipeline_ltx import LTXPipeline
from .pipeline_ltx_av import LTXAVPipeline
from .pipeline_ltx_fast import LTXFastPipeline

STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


class LTXAVTwoStagesPipeline(LTXAVPipeline):
    """Two-stage AV pipeline: full-guidance stage 1 + distilled-LoRA stage 2 refine."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("mode", "av")
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice, **kwargs) -> "LTXAVTwoStagesPipeline":
        kwargs.setdefault("mode", "av")
        kwargs["pipeline_class"] = LTXAVTwoStagesPipeline
        return LTXPipeline.create_pipeline(mesh_device, **kwargs)

    def generate(
        self,
        prompt: str,
        *,
        output_path: str,
        upsampler_path: str,
        distilled_lora_path: str,
        distilled_lora_strength: float = 1.0,
        negative_prompt: str | None = None,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        # Stage 1 (guided) knobs — same defaults as LTXAVPipeline.generate.
        num_inference_steps: int = 30,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        video_stg_scale: float = 1.0,
        audio_stg_scale: float = 1.0,
        video_modality_scale: float = 3.0,
        audio_modality_scale: float = 3.0,
        rescale_scale: float = 0.7,
        stg_block: int = 28,
        ge_gamma: float | None = None,
        # Stage 2 (refine) knobs.
        stage_2_sigma_values: list[float] | None = None,
        seed: int = 10,
        fps: int = 24,
    ) -> str:
        """Run the 2-stage AV pipeline (full-guidance s1 + distilled s2) and write an MP4."""
        assert height % 64 == 0, f"Height must be divisible by 64 (got {height})"
        assert width % 64 == 0, f"Width must be divisible by 64 (got {width})"

        s1_h, s1_w = height // 2, width // 2
        s2_sigma_values = list(stage_2_sigma_values) if stage_2_sigma_values else STAGE_2_DISTILLED_SIGMA_VALUES

        if ge_gamma is None:
            ge_gamma = 2.0 if ttnn.device.is_blackhole() else 0.0
            logger.info(f"ge_gamma={ge_gamma} (arch default)")

        # Env overrides — match LTXAVPipeline.generate semantics.
        def _env_float(name: str, default: float) -> float:
            val = os.environ.get(name)
            return float(val) if val is not None else default

        video_cfg_scale = _env_float("VIDEO_CFG_SCALE", video_cfg_scale)
        audio_cfg_scale = _env_float("AUDIO_CFG_SCALE", audio_cfg_scale)
        video_stg_scale = _env_float("VIDEO_STG_SCALE", video_stg_scale)
        audio_stg_scale = _env_float("AUDIO_STG_SCALE", audio_stg_scale)
        video_modality_scale = _env_float("VIDEO_MODALITY_SCALE", video_modality_scale)
        audio_modality_scale = _env_float("AUDIO_MODALITY_SCALE", audio_modality_scale)
        rescale_scale = _env_float("RESCALE_SCALE", rescale_scale)
        if os.environ.get("GE_GAMMA") is not None:
            ge_gamma = float(os.environ["GE_GAMMA"])
        distilled_lora_strength = _env_float("DISTILLED_LORA_STRENGTH", distilled_lora_strength)

        import sys

        sys.path.insert(0, "LTX-2/packages/ltx-core/src")
        sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
        torch.cuda.synchronize = lambda *a, **kw: None  # noqa: ARG005
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

        neg = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

        total_t0 = time.time()

        # 1) Encode prompts once — same context is reused by both stages, matching
        #    the reference which uses the same ``ctx_p`` for stage 1 and stage 2.
        t0 = time.time()
        results = self.encode_prompts_reference([prompt, neg])
        logger.info(f"Encoding: {time.time() - t0:.1f}s")
        v_p = results[0].video_encoding.float()
        a_p = results[0].audio_encoding.float()
        v_n = results[1].video_encoding.float()
        a_n = results[1].audio_encoding.float()

        # 2) Stage 1: base weights, half resolution, full guidance.
        self._lora_specs = []
        self._prepare_transformer()
        gc.collect()
        logger.info(f"Stage 1: {s1_h}x{s1_w}, {num_inference_steps} guided steps")
        t0 = time.time()
        s1_video, s1_audio = self.call_av(
            video_prompt_embeds=v_p,
            audio_prompt_embeds=a_p,
            neg_video_prompt_embeds=v_n,
            neg_audio_prompt_embeds=a_n,
            num_frames=num_frames,
            height=s1_h,
            width=s1_w,
            num_inference_steps=num_inference_steps,
            video_cfg_scale=video_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            video_stg_scale=video_stg_scale,
            audio_stg_scale=audio_stg_scale,
            video_modality_scale=video_modality_scale,
            audio_modality_scale=audio_modality_scale,
            rescale_scale=rescale_scale,
            stg_block=stg_block,
            seed=seed,
            ge_gamma=ge_gamma,
        )
        logger.info(f"Stage 1: {time.time() - t0:.1f}s")

        # 3) Free transformer (BH LB can't hold transformer + upsampler at once).
        self.transformer = None
        gc.collect()

        latent_frames = (num_frames - 1) // 8 + 1
        s1_lh, s1_lw = s1_h // 32, s1_w // 32
        s1_spatial = s1_video.reshape(1, latent_frames, s1_lh, s1_lw, 128).permute(0, 4, 1, 2, 3)
        t0 = time.time()
        upsampled = LTXFastPipeline._upsample_latent_reference(self, s1_spatial, upsampler_path)
        logger.info(f"Upsample: {time.time() - t0:.1f}s")
        upsampled_flat = upsampled.permute(0, 2, 3, 4, 1).reshape(
            1, latent_frames * (height // 32) * (width // 32), 128
        )

        # 4) Stage 2: distilled LoRA fused into transformer; no guidance; refine
        #    from the upsampled video latent + stage-1 audio latent renoised at
        #    sigmas[0]. Reuses call_av by neutralising all guidance scales.
        self._lora_specs = [LoraSpec(path=distilled_lora_path, strength=distilled_lora_strength)]
        self._prepare_transformer()
        gc.collect()
        s2_sigmas = torch.tensor(s2_sigma_values, dtype=torch.float32)
        n_s2_steps = len(s2_sigma_values) - 1
        logger.info(
            f"Stage 2: {height}x{width}, {n_s2_steps} distilled refine steps (LoRA strength={distilled_lora_strength})"
        )
        t0 = time.time()
        s2_audio_init = s1_audio.unsqueeze(0) if s1_audio.dim() == 2 else s1_audio
        s2_video, s2_audio = self.call_av(
            video_prompt_embeds=v_p,
            audio_prompt_embeds=a_p,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=n_s2_steps,
            video_cfg_scale=1.0,
            audio_cfg_scale=1.0,
            video_stg_scale=0.0,
            audio_stg_scale=0.0,
            video_modality_scale=1.0,
            audio_modality_scale=1.0,
            rescale_scale=0.0,
            seed=seed,
            ge_gamma=0.0,
            sigmas=s2_sigmas,
            initial_video_latent=upsampled_flat,
            initial_audio_latent=s2_audio_init,
            noise_scale=s2_sigma_values[0],
        )
        logger.info(f"Stage 2: {time.time() - t0:.1f}s")

        # 5) Free transformer; load VAE; decode + export.
        self.transformer = None
        gc.collect()

        t0 = time.time()
        self._prepare_vae()
        logger.info(f"VAE loaded in {time.time() - t0:.0f}s")

        latent_h, latent_w = height // 32, width // 32
        t0 = time.time()
        video_pixels = self.decode_latents(s2_video, latent_frames, latent_h, latent_w)
        logger.info(f"VAE decode: {time.time() - t0:.1f}s — {video_pixels.shape}")

        audio_obj = self.decode_audio_reference(s2_audio, num_frames, fps=fps)
        self.export_video(video_pixels, output_path, fps=fps, audio=audio_obj)

        total_time = time.time() - total_t0
        logger.info(f"Total: {total_time:.1f}s | Output: {output_path}")
        return output_path

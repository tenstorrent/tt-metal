# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 22B two-stage audio-video pipeline.

Mirrors the reference ``ltx_pipelines.ti2vid_two_stages.TI2VidTwoStagesPipeline``:
stage 1 denoises half-res AV on the base 22B checkpoint with full MultiModalGuider
guidance; the stage-1 video latent is x2-upsampled on device; stage 2 refines at full
res with the distilled LoRA fused in and no guidance, starting from the upsampled
video + renoised stage-1 audio.

Text-only; image conditioning is not wired here yet.
"""

from __future__ import annotations

import os
import time

import torch
from loguru import logger

import ttnn

from ...utils.fuse_loras import LoraSpec
from ...utils.patchifiers import AudioLatentShape, VideoPixelShape
from ...utils.video import export_video_audio
from .pipeline_ltx import DEFAULT_NEGATIVE_PROMPT, SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION, LTXPipeline, latent_grid

# Stage-2 distilled sigma schedule (stage 1 renoises the s1 latent at sigmas[0]).
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


class LTXTwoStagesPipeline(LTXPipeline):
    """Two-stage AV pipeline: full-guidance s1 (variant 0 = base 22B) +
    distilled-LoRA s2 refine (variant 1 = LoRA-fused base)."""

    HAS_UPSAMPLER = True

    def __init__(
        self,
        *args,
        distilled_lora_path: str | None = None,
        distilled_lora_strength: float = 1.0,
        **kwargs,
    ) -> None:
        kwargs.setdefault("mode", "av")
        if distilled_lora_path is not None:
            kwargs.setdefault(
                "extra_transformer_variants",
                [
                    (
                        f"distilled_lora_strength_{distilled_lora_strength}",
                        [LoraSpec(path=distilled_lora_path, strength=distilled_lora_strength)],
                    )
                ],
            )
        self._distilled_lora_path = distilled_lora_path
        self._distilled_lora_strength = distilled_lora_strength
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice, **kwargs) -> "LTXTwoStagesPipeline":
        kwargs.setdefault("mode", "av")
        kwargs["pipeline_class"] = LTXTwoStagesPipeline
        return LTXPipeline.create_pipeline(mesh_device, **kwargs)

    def warmup_buffers(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int = 2,
    ) -> None:
        """Compile every program both stages will hit. Stage 1: variant 0
        (base, half-res, full guidance). Stage 2: variant 1 (LoRA-fused,
        full-res, neutral guidance). Stage 2 is skipped if ``__init__`` was
        called without a ``distilled_lora_path``."""
        assert height % 64 == 0 and width % 64 == 0, f"H/W must be div by 64 (got {height}x{width})"
        assert num_frames > 0, f"num_frames must be > 0 (got {num_frames})"

        has_s2 = len(self.transformer_states) > 1
        t0 = time.time()
        logger.info(
            f"warmup (2-stage): {num_frames}f@{height}x{width}, "
            f"s1={num_inference_steps} steps" + (f" + s2={num_inference_steps} steps" if has_s2 else " (s2 skipped)")
        )

        s1_h, s1_w = height // 2, width // 2
        self._prepare_transformer(0)

        # Dummy zero embeddings at the real shapes — warmup only needs to compile the
        # (shape-driven) call_av kernels, not real prompt content. Avoids loading the
        # encoder here (it would coresident-evict the DiT just loaded above); the encoder
        # kernels compile on the first generate().
        v_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.video_dim)
        a_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.audio_dim)
        v_n, a_n = v_p, a_p

        logger.info(f"warmup stage 1: {s1_h}x{s1_w}")
        self.call_av(
            video_prompt_embeds=v_p,
            audio_prompt_embeds=a_p,
            neg_video_prompt_embeds=v_n,
            neg_audio_prompt_embeds=a_n,
            num_frames=num_frames,
            height=s1_h,
            width=s1_w,
            num_inference_steps=num_inference_steps,
            seed=0,
            ge_gamma=0.0,
        )

        if not has_s2:
            # Decode runs at s1 half-res when there's no s2 stage.
            self._warmup_decode(num_frames, s1_h, s1_w)
            self._prepare_transformer(0)
            logger.info(f"warmup (s1-only) done in {time.time() - t0:.1f}s")
            return

        # Upsample runs between stage 1 and stage 2; compile its kernels here.
        logger.info(f"warmup upsample: {s1_h}x{s1_w} → {height}x{width}")
        self._warmup_upsample(num_frames, height, width)

        self._prepare_transformer(1)

        s2_sigmas = list(STAGE_2_DISTILLED_SIGMA_VALUES)[:num_inference_steps] + [0.0]
        s2_sigmas_t = torch.tensor(s2_sigmas, dtype=torch.float32)

        latent_frames, full_lh, full_lw = latent_grid(num_frames, height, width)
        full_latent_count = latent_frames * full_lh * full_lw
        dummy_v_init = torch.zeros(1, full_latent_count, self.in_channels)
        vps = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
        als = AudioLatentShape.from_video_pixel_shape(vps)
        dummy_a_init = torch.zeros(1, als.frames, self.in_channels)

        logger.info(f"warmup stage 2: {height}x{width}, σ={s2_sigmas}")
        self.call_av(
            video_prompt_embeds=v_p,
            audio_prompt_embeds=a_p,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            video_cfg_scale=1.0,
            audio_cfg_scale=1.0,
            video_stg_scale=0.0,
            audio_stg_scale=0.0,
            video_modality_scale=1.0,
            audio_modality_scale=1.0,
            rescale_scale=0.0,
            seed=0,
            ge_gamma=0.0,
            sigmas=s2_sigmas_t,
            initial_video_latent=dummy_v_init,
            initial_audio_latent=dummy_a_init,
            noise_scale=s2_sigmas[0],
        )
        # Compile VAE decode at full-res (only s2 feeds decode in generate).
        self._warmup_decode(num_frames, height, width)
        # Re-prime variant 0 so it's resident when the real generate starts stage 1.
        self._prepare_transformer(0)
        logger.info(f"warmup (2-stage) done in {time.time() - t0:.1f}s")

    def generate(
        self,
        prompt: str,
        *,
        output_path: str,
        # LoRA path / strength are consumed by ``__init__`` (variant 1 is built
        # there). Optional here for back-compat; if passed they must match what
        # ``__init__`` saw or generate raises.
        distilled_lora_path: str | None = None,
        distilled_lora_strength: float | None = None,
        negative_prompt: str | None = None,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        # Stage 1 (guided) knobs — same defaults as LTXPipeline.generate.
        num_inference_steps: int = 30,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        video_stg_scale: float = 1.0,
        audio_stg_scale: float = 1.0,
        video_modality_scale: float = 3.0,
        audio_modality_scale: float = 3.0,
        rescale_scale: float = 0.7,
        stg_block: int = 28,
        ge_gamma: float = 0.0,
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

        if len(self.transformer_states) < 2:
            raise RuntimeError("Variant 1 (LoRA-fused) not registered — pass distilled_lora_path to create_pipeline")
        if distilled_lora_path is not None and distilled_lora_path != self._distilled_lora_path:
            raise ValueError(
                f"distilled_lora_path mismatch: generate got {distilled_lora_path!r}, "
                f"__init__ used {self._distilled_lora_path!r}"
            )
        if distilled_lora_strength is None:
            distilled_lora_strength = self._distilled_lora_strength
        if distilled_lora_strength != self._distilled_lora_strength:
            raise ValueError(
                f"distilled_lora_strength mismatch: generate got {distilled_lora_strength}, "
                f"__init__ used {self._distilled_lora_strength}"
            )

        neg = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

        total_t0 = time.time()

        # Encode prompts once — both stages reuse the same context (matching the
        # reference's shared ``ctx_p``). On-device Gemma encode is coresident-excluded
        # with the DiT/VAE, so it auto-evicts them; load only on a cache miss.
        t0 = time.time()
        cached = os.path.exists(self._device_embed_cache_path([prompt, neg]))
        if not cached:
            self.gemma_encoder_pair.ensure_loaded()
        enc = self.encode_prompts([prompt, neg])
        logger.info(f"Encoding ({'cache' if cached else 'device'}): {time.time() - t0:.1f}s")
        v_p, a_p = enc[0][0].float(), enc[0][1].float()
        v_n, a_n = enc[1][0].float(), enc[1][1].float()

        # Stage 1: variant 0 (base), half-res, full guidance.
        self._prepare_transformer(0)
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

        latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
        s1_lh, s1_lw = s1_h // SPATIAL_COMPRESSION, s1_w // SPATIAL_COMPRESSION
        s1_spatial = s1_video.reshape(1, latent_frames, s1_lh, s1_lw, 128).permute(0, 4, 1, 2, 3)
        t0 = time.time()
        upsampled = self._upsample_latent(s1_spatial)
        logger.info(f"Upsample: {time.time() - t0:.1f}s")
        upsampled_flat = upsampled.permute(0, 2, 3, 4, 1).reshape(
            1, latent_frames * (height // SPATIAL_COMPRESSION) * (width // SPATIAL_COMPRESSION), 128
        )

        # Stage 2: variant 1 (LoRA-fused), full-res, neutral guidance. Refines
        # upsampled video + renoised stage-1 audio.
        self._prepare_transformer(1)
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

        t0 = time.time()
        self._prepare_vae()
        logger.info(f"VAE loaded in {time.time() - t0:.0f}s")

        latent_h, latent_w = height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION
        t0 = time.time()
        video_pixels = self.decode_latents(s2_video, latent_frames, latent_h, latent_w)
        logger.info(f"VAE decode: {time.time() - t0:.1f}s — {video_pixels.shape}")

        audio_obj = self.decode_audio(s2_audio, num_frames, fps=fps)
        export_video_audio(video_pixels, output_path, fps=fps, audio=audio_obj)

        total_time = time.time() - total_t0
        logger.info(f"Total: {total_time:.1f}s | Output: {output_path}")
        return output_path

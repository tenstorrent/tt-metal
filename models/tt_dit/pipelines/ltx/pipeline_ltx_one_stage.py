# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 one-stage audio-video pipeline (Pro).

Mirrors ``ltx_pipelines.ti2vid_one_stage.TI2VidOneStagePipeline``.
"""

from __future__ import annotations

import os
import time

import torch
from loguru import logger

from ...utils.patchifiers import AudioLatentShape, VideoPixelShape
from ...utils.video import export_video_audio
from .pipeline_ltx import DEFAULT_NEGATIVE_PROMPT, LTXPipeline, latent_grid


class LTXOneStagePipeline(LTXPipeline):
    """One-stage AV pipeline: full MultiModalGuider guidance on the base 22B checkpoint."""

    def warmup_buffers(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int = 2,
    ) -> None:
        """Compile the ``call_av`` + VAE-decode programs (``ge_gamma=0`` skips the GE branch)."""
        t0 = time.time()
        logger.info(f"warmup (AV): {num_frames}f@{height}x{width}, {num_inference_steps} steps")

        # Zeros at the real shapes compile the shape-driven call_av kernels without loading
        # the encoder, which would coresident-evict the DiT.
        v_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.video_dim)
        a_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.audio_dim)
        v_n, a_n = v_p, a_p

        self.call_av(
            video_prompt_embeds=v_p,
            audio_prompt_embeds=a_p,
            neg_video_prompt_embeds=v_n,
            neg_audio_prompt_embeds=a_n,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=0,
            ge_gamma=0.0,
        )

        self._warmup_decode(num_frames, height, width)

        # Warm the on-device audio decode eagerly so the first real (traced) decode captures on
        # warm state at a deterministic free-list.
        als = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
        )
        self._warmup_audio_decode(torch.zeros(1, als.frames, self.in_channels), num_frames)

        self._prepare_transformer(0)
        logger.info(f"warmup (AV) done in {time.time() - t0:.1f}s")

    def generate(
        self,
        prompt: str,
        *,
        output_path: str,
        negative_prompt: str | None = None,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 30,
        video_cfg_scale: float = 3.0,
        audio_cfg_scale: float = 7.0,
        video_stg_scale: float = 1.0,
        audio_stg_scale: float = 1.0,
        video_modality_scale: float = 3.0,
        audio_modality_scale: float = 3.0,
        rescale_scale: float = 0.7,
        stg_block: int = 28,
        seed: int = 10,
        ge_gamma: float = 0.0,
        fps: int = 24,
    ) -> str:
        """Run LTX-2.3 Pro AV generation and write an MP4. Guidance defaults match the
        reference ``LTX_2_3_PARAMS``; ``ge_gamma`` > 0 enables gradient-estimation sampling."""
        neg = negative_prompt if negative_prompt is not None else DEFAULT_NEGATIVE_PROMPT

        total_t0 = time.time()
        timings: list[tuple[str, float]] = []

        t0 = time.time()
        # Gemma is coresident-excluded with the DiT/VAE; load it only on a cache miss.
        cached = os.path.exists(self._device_embed_cache_path([prompt, neg]))
        if not cached:
            self.gemma_encoder_pair.ensure_loaded()
        enc = self.encode_prompts([prompt, neg])
        v_embeds, a_embeds = enc[0][0].float(), enc[0][1].float()
        neg_v, neg_a = enc[1][0].float(), enc[1][1].float()
        t_encode = time.time() - t0
        timings.append(("Encoder (cache)" if cached else "Encoder", t_encode))
        logger.info(f"Encoding ({'cache' if cached else 'device'}): {t_encode:.1f}s")

        self._prepare_transformer(0)

        t0 = time.time()
        video_latent, audio_latent = self.call_av(
            video_prompt_embeds=v_embeds,
            audio_prompt_embeds=a_embeds,
            neg_video_prompt_embeds=neg_v,
            neg_audio_prompt_embeds=neg_a,
            num_frames=num_frames,
            height=height,
            width=width,
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
        denoise_time = time.time() - t0
        timings.append(("Denoise", denoise_time))
        logger.info(f"Denoising: {denoise_time:.1f}s ({denoise_time / num_inference_steps:.1f}s/step)")

        t0 = time.time()
        self._prepare_vae()
        logger.info(f"VAE loaded in {time.time() - t0:.0f}s")

        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)

        t0 = time.time()
        video_pixels = self.decode_latents(video_latent, latent_frames, latent_h, latent_w)
        t_vae = time.time() - t0
        timings.append(("VAE decode", t_vae))
        logger.info(f"VAE decode: {t_vae:.1f}s — {video_pixels.shape}")

        t0 = time.time()
        audio_obj = self.decode_audio(audio_latent, num_frames, fps=fps)
        timings.append(("Audio decode", time.time() - t0))
        export_video_audio(video_pixels, output_path, fps=fps, audio=audio_obj)

        self.last_timings = list(timings)
        total_time = time.time() - total_t0
        logger.info(f"Total: {total_time:.1f}s | Output: {output_path}")
        return output_path

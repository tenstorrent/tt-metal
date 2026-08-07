# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 distilled two-stage audio-video pipeline."""

from __future__ import annotations

import os
import time

import torch
from loguru import logger

from ...models.vae.vae_ltx import upsample_latent
from ...utils.ltx import load_conditioning_image
from ...utils.patchifiers import AudioLatentShape, VideoPixelShape
from ...utils.video import export_video_audio, export_video_audio_yuv
from .pipeline_ltx import SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION, LTXPipeline, latent_grid

# Distilled sigma schedules for the two stages.
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


class LTXDistilledPipeline(LTXPipeline):
    """Distilled 2-stage AV pipeline: half-res denoise → upsample → full-res refine."""

    HAS_UPSAMPLER = True

    def warmup_buffers(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int = 2,
        stages: tuple[str, ...] = ("s1", "s2"),
    ) -> None:
        """Compile both stages' programs (variant 0 for both); ``stages=("s1",)`` skips s2."""
        assert height % 64 == 0 and width % 64 == 0, f"H/W must be div by 64 (got {height}x{width})"
        assert num_frames > 0, f"num_frames must be > 0 (got {num_frames})"
        valid = {"s1", "s2"}
        assert set(stages).issubset(valid), f"stages must be subset of {valid} (got {stages})"

        t0 = time.time()
        logger.info(
            f"warmup (distilled 2-stage): {num_frames}f@{height}x{width}, "
            f"stages={stages}, {num_inference_steps} steps/stage"
        )

        v_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.video_dim)
        a_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.audio_dim)

        # Allocate both stages' persistent trace I/O before any capture so all held inputs sit
        # below both traces' activation regions and neither replay overwrites the other's inputs.
        if self._traced:
            self._prealloc_trace_io("s1", num_frames=num_frames, height=height // 2, width=width // 2)
            self._prealloc_trace_io("s2", num_frames=num_frames, height=height, width=width)

        # Warm the encoder before any capture so its connector workspace isn't in a trace's
        # activation region (zeroed on replay). dynamic_load reloads per request → warms last.
        if self._traced and not self.dynamic_load:
            self.gemma_encoder_pair.ensure_loaded()
            self.encode_prompts(["warmup"], use_cache=False)

        # Real distilled sigmas so warmup hits the same branches (incl. sigma_next == 0 final step).
        s1_sigmas = list(DISTILLED_SIGMA_VALUES)[:num_inference_steps] + [0.0]
        s2_sigmas = list(STAGE_2_DISTILLED_SIGMA_VALUES)[:num_inference_steps] + [0.0]

        if "s1" in stages:
            s1_h, s1_w = height // 2, width // 2

            logger.info(f"warmup stage 1: {s1_h}x{s1_w}, σ={s1_sigmas}")
            self._denoise(
                v_p,
                a_p,
                num_frames=num_frames,
                height=s1_h,
                width=s1_w,
                sigma_values=s1_sigmas,
                seed=0,
            )

        if "s2" in stages:
            # Upsample runs between stage 1 and stage 2; compile its kernels here.
            logger.info(f"warmup upsample → {height}x{width}")
            self._warmup_upsample(num_frames, height, width)

            # Zero-dummies at the exact shapes the real stage-2 call uses.
            latent_frames, full_lh, full_lw = latent_grid(num_frames, height, width)
            dummy_v_init = torch.zeros(1, latent_frames * full_lh * full_lw, self.in_channels)
            als = AudioLatentShape.from_video_pixel_shape(
                VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
            )
            dummy_a_init = torch.zeros(1, als.frames, self.in_channels)

            logger.info(f"warmup stage 2: {height}x{width}, σ={s2_sigmas}")
            self._denoise(
                v_p,
                a_p,
                num_frames=num_frames,
                height=height,
                width=width,
                sigma_values=s2_sigmas,
                seed=0,
                initial_video_latent=dummy_v_init,
                initial_audio_latent=dummy_a_init,
            )

            # Compile VAE decode at full-res (only s2 feeds decode in generate).
            self._warmup_decode(num_frames, height, width)

            # Programs are now compiled
            if self._traced and self.vae_decoder is not None:
                self.vae_decoder._vae_traced = True

            # Warm the on-device audio decode eagerly at the real latent shape: compiles kernels,
            # initializes lazy device state, and frees back to a deterministic allocator free-list,
            # so the first real (traced) decode captures cleanly on warm state.
            logger.info("warmup audio decode (on-device, eager)")
            self._warmup_audio_decode(torch.zeros(1, als.frames, self.in_channels), num_frames)

            self._prepare_transformer(0)

        # Warm the encoders last: they coresident-evict the DiT/VAE, so gen #0 re-loads the DiT.
        # Warm whenever the checkpoint has an encoder (not only when an I2V image is staged): a first
        # I2V request after capture would otherwise load the encoder, evict the DiT, and clobber the
        # captured traces' activation regions — corrupting every subsequent gen to static.
        if self.vae_encoder is not None:
            logger.info(f"warmup image encoder: {height // 2}x{width // 2} + {height}x{width}")
            self._warmup_encode(height // 2, width // 2)
            self._warmup_encode(height, width)

        # use_cache=False forces a real encode so the Gemma/connector kernels compile. traced-static
        # already warmed before capture (above); dynamic_load / untraced warm last.
        if self.dynamic_load or not self._traced:
            self.gemma_encoder_pair.ensure_loaded()
            self.encode_prompts(["warmup"], use_cache=False)

        logger.info(f"warmup (distilled 2-stage) done in {time.time() - t0:.1f}s")

    def generate(
        self,
        prompt: str,
        *,
        output_path: str | None = None,
        output_type: str = "rgb",
        # I2V: list of (image_path, frame_idx, strength). Only frame_idx==0 is supported.
        images: list[tuple[str, int, float]] | None = None,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        seed: int = 10,
        fps: int = 24,
    ):
        """Run the distilled 2-stage AV pipeline.

        output_path given → encode an AV MP4 and return its path (str).
        output_path None  → return ``(frames, audio)`` for the caller to encode.
        """
        assert height % 64 == 0, f"Height must be divisible by 64 (got {height})"
        assert width % 64 == 0, f"Width must be divisible by 64 (got {width})"

        s1_height = height // 2
        s1_width = width // 2

        # (label, seconds) rows counted toward the total; prepares and export excluded.
        timings: list[tuple[str, float]] = []

        t0 = time.time()
        # Only load the Gemma encoder (coresident-evicts DiT/VAE) on a cache miss.
        cached = os.path.exists(self._device_embed_cache_path([prompt]))
        if not cached:
            self.gemma_encoder_pair.ensure_loaded()
        enc = self.encode_prompts([prompt])
        v_embeds, a_embeds = enc[0][0].float(), enc[0][1].float()
        t_encode = time.time() - t0
        timings.append(("Encoder (cache)" if cached else "Encoder", t_encode))
        logger.info(f"Encoding ({'cache' if cached else 'device'}): {t_encode:.1f}s")

        s1_cond_latent = full_cond_latent = None
        cond_strength = 1.0
        if images:
            cond_imgs = [img for img in images if img[1] == 0]
            if len(cond_imgs) != len(images):
                logger.warning("Distilled I2V only supports frame_idx==0 conditioning; ignoring keyframe images")
            if cond_imgs:
                assert self.vae_encoder is not None, "checkpoint has no VAE encoder; cannot run I2V conditioning"
                img_path, _, cond_strength = cond_imgs[0]
                # Conditioning latent depends only on (image, resolution): encode once and memoize.
                # Skips re-running the eager VAE encoder on later gens (re-encoding under traced
                # replay has been observed to hang the device).
                s1_key = (img_path, s1_height, s1_width)
                full_key = (img_path, height, width)
                cache = self._i2v_cond_cache
                if s1_key in cache and full_key in cache:
                    s1_cond_latent, full_cond_latent = cache[s1_key], cache[full_key]
                    logger.info(f"I2V: reusing cached conditioning latents for {img_path} (strength={cond_strength})")
                else:
                    logger.info(f"I2V: encoding conditioning image {img_path} (strength={cond_strength})")
                    t0 = time.time()
                    img_s1 = load_conditioning_image(img_path, s1_height, s1_width)
                    img_full = load_conditioning_image(img_path, height, width)
                    s1_cond_latent = cache[s1_key] = self.encode_image(img_s1)
                    full_cond_latent = cache[full_key] = self.encode_image(img_full)
                    timings.append(("Image encode", time.time() - t0))
                    logger.info(f"Image encode: {time.time() - t0:.1f}s")

        t0 = time.time()
        self._prepare_transformer(0)
        if self.dynamic_load:
            logger.info(f"Transformer prepare: {time.time() - t0:.1f}s")

        logger.info(f"Stage 1: {s1_height}x{s1_width}, {len(DISTILLED_SIGMA_VALUES) - 1} steps")
        t0 = time.time()
        s1_video, s1_audio = self._denoise(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=s1_height,
            width=s1_width,
            sigma_values=DISTILLED_SIGMA_VALUES,
            seed=seed,
            image_cond_latent=s1_cond_latent,
            image_cond_strength=cond_strength,
            traced=self._traced,
            trace_key="s1",
        )
        t_stage1 = time.time() - t0
        timings.append(("Stage 1 denoise", t_stage1))
        logger.info(f"Stage 1 denoise: {t_stage1:.1f}s")

        latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
        s1_h, s1_w = s1_height // SPATIAL_COMPRESSION, s1_width // SPATIAL_COMPRESSION
        s1_spatial = s1_video.reshape(1, latent_frames, s1_h, s1_w, 128).permute(0, 4, 1, 2, 3)
        t0 = time.time()
        self._prepare_upsampler()
        upsampled = upsample_latent(self.upsampler, s1_spatial, *self._vae_per_channel_stats())
        t_upsample = time.time() - t0
        timings.append(("Latent upsample", t_upsample))
        logger.info(f"Latent upsample: {t_upsample:.1f}s")
        upsampled_flat = upsampled.permute(0, 2, 3, 4, 1).reshape(
            1, latent_frames * (height // SPATIAL_COMPRESSION) * (width // SPATIAL_COMPRESSION), 128
        )

        logger.info(f"Stage 2: {height}x{width}, {len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1} steps")
        t0 = time.time()
        s2_video, s2_audio = self._denoise(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=height,
            width=width,
            sigma_values=STAGE_2_DISTILLED_SIGMA_VALUES,
            seed=seed,
            initial_video_latent=upsampled_flat,
            initial_audio_latent=s1_audio.unsqueeze(0) if s1_audio.dim() == 2 else s1_audio,
            image_cond_latent=full_cond_latent,
            image_cond_strength=cond_strength,
            traced=self._traced,
            trace_key="s2",
        )
        t_stage2 = time.time() - t0
        timings.append(("Stage 2 denoise", t_stage2))
        logger.info(f"Stage 2 denoise: {t_stage2:.1f}s")

        t0 = time.time()
        self._prepare_vae()
        if self.dynamic_load:
            logger.info(f"VAE prepare: {time.time() - t0:.1f}s")

        latent_h, latent_w = height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION
        # LTX_YUV_EXPORT routes the mp4 path through the on-device YUV 4:2:0 fast gather
        yuv_export = output_path is not None and os.environ.get("LTX_YUV_EXPORT", "0") != "0"
        # export_video_audio needs float [-1,1]; the frame-return path uses the requested output_type.
        decode_type = ("yuv" if yuv_export else "float") if output_path is not None else output_type
        t0 = time.time()
        video_pixels = self.decode_latents(s2_video, latent_frames, latent_h, latent_w, output_type=decode_type)
        t_vae_decode = time.time() - t0
        timings.append(("VAE decode", t_vae_decode))
        logger.info(f"VAE decode (forward): {t_vae_decode:.1f}s — {tuple(video_pixels.shape)}")

        t0 = time.time()
        audio_obj = self.decode_audio(s2_audio, num_frames, fps=fps)
        t_audio_decode = time.time() - t0
        timings.append(("Audio decode", t_audio_decode))
        logger.info(f"Audio decode: {t_audio_decode:.1f}s")

        self.last_timings = list(timings)
        if output_path is None:
            logger.info(f"Total (compute): {sum(s for _, s in timings):.1f}s | frames={tuple(video_pixels.shape)}")
            return video_pixels, audio_obj

        t0 = time.time()
        if yuv_export:
            export_video_audio_yuv(video_pixels, output_path, fps=fps, audio=audio_obj)
        else:
            export_video_audio(video_pixels, output_path, fps=fps, audio=audio_obj)
        logger.info(f"Video export: {time.time() - t0:.1f}s")
        logger.info(f"Total (compute): {sum(s for _, s in timings):.1f}s | Output: {output_path}")
        return output_path

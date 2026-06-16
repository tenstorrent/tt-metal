# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 distilled two-stage audio-video pipeline."""

from __future__ import annotations

import os
import time

import torch
from loguru import logger

import ttnn

from ...models.transformers.ltx.transformer_ltx import LTXTransformerModel
from ...utils.patchifiers import AudioLatentShape, VideoPixelShape
from ...utils.tensor import bf16_tensor
from ...utils.video import export_video_audio
from .pipeline_ltx import SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION, LTXPipeline, LTXTransformerState, latent_grid

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
        """Compile both stages' programs. Both stages use variant 0 (distilled doesn't
        swap weights — only the sequence length differs); ``stages=("s1",)`` skips s2."""
        assert height % 64 == 0 and width % 64 == 0, f"H/W must be div by 64 (got {height}x{width})"
        assert num_frames > 0, f"num_frames must be > 0 (got {num_frames})"
        valid = {"s1", "s2"}
        assert set(stages).issubset(valid), f"stages must be subset of {valid} (got {stages})"

        t0 = time.time()
        logger.info(
            f"warmup (distilled 2-stage): {num_frames}f@{height}x{width}, "
            f"stages={stages}, {num_inference_steps} steps/stage"
        )

        # Zeros at the real shapes compile the shape-driven kernels; the encoder is warmed
        # separately at the end of this method (it coresident-evicts the DiT/VAE).
        v_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.video_dim)
        a_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.audio_dim)

        # Allocate both stages' persistent trace I/O before any capture, so all held inputs
        # sit below both traces' activation regions and neither trace's replay can overwrite
        # the other's inputs. Held for the session; generate() refreshes contents in place.
        if self._traced:
            self._prealloc_trace_io("s1", num_frames=num_frames, height=height // 2, width=width // 2)
            self._prealloc_trace_io("s2", num_frames=num_frames, height=height, width=width)

        # Sigma schedules from the real distilled paths so warmup exercises the
        # same math branches generate() does (incl. the sigma_next == 0 final step).
        s1_sigmas = list(DISTILLED_SIGMA_VALUES)[:num_inference_steps] + [0.0]
        s2_sigmas = list(STAGE_2_DISTILLED_SIGMA_VALUES)[:num_inference_steps] + [0.0]

        if "s1" in stages:
            s1_h, s1_w = height // 2, width // 2
            logger.info(f"warmup stage 1: {s1_h}x{s1_w}, σ={s1_sigmas}")
            self._denoise_no_guidance(
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
            self._denoise_no_guidance(
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

            # Build + JIT-compile the on-device audio decode on the exact latent
            # shape generate() produces, so the first real audio decode loads
            # from cache instead of building from the checkpoint (cold ~64s).
            logger.info("warmup audio decode (on-device)")
            self.decode_audio(torch.zeros(1, als.frames, self.in_channels), num_frames, fps=24.0)
            # The BWE trace captures on the first POST-cold decode (not the cold one above), and that
            # capturing pass emits clipped audio. A second warmup decode absorbs the capture so the
            # first real generate is a clean replay.
            if self._traced:
                self.decode_audio(torch.zeros(1, als.frames, self.in_channels), num_frames, fps=24.0)

            self._prepare_transformer(0)

        # Warm the encoder last: it coresident-evicts the DiT/VAE, so gen #0 then re-loads the DiT.
        # use_cache=False forces a real encode so the Gemma/connector kernels actually compile.
        self.gemma_encoder_pair.ensure_loaded()
        self.encode_prompts(["warmup"], use_cache=False)

        logger.info(f"warmup (distilled 2-stage) done in {time.time() - t0:.1f}s")

    def _prepare_stage_statics(
        self, state, *, latent_frames, latent_h, latent_w, video_N, video_N_real, audio_N, audio_N_real, sp_axis
    ):
        """Build a stage's static per-shape inputs once (rope/cross-PE/masks/trans_mat)."""
        if state.tt_video_rope_cos is not None:
            return
        v_cos, v_sin = self._prepare_rope(latent_frames, latent_h, latent_w)
        a_cos, a_sin = self._prepare_audio_rope(audio_N, audio_N_real)
        # Cross-PE is required: without it A↔V cross-attention has no positional info and
        # audio-video temporal sync (lip sync) breaks.
        (
            v_xpe_cos,
            v_xpe_sin,
            a_xpe_cos,
            a_xpe_sin,
            v_xpe_cos_full,
            v_xpe_sin_full,
            a_xpe_cos_full,
            a_xpe_sin_full,
        ) = self._prepare_av_cross_pe(latent_frames, latent_h, latent_w, audio_N, audio_N_real)
        tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full = self._prepare_audio_masks(audio_N, audio_N_real)
        state._tt_video_rope_cos.update(v_cos, False)
        state._tt_video_rope_sin.update(v_sin, False)
        state._tt_audio_rope_cos.update(a_cos, False)
        state._tt_audio_rope_sin.update(a_sin, False)
        state._tt_trans_mat.update(self._prepare_trans_mat(), False)
        state._tt_video_cross_pe_cos.update(v_xpe_cos, False)
        state._tt_video_cross_pe_sin.update(v_xpe_sin, False)
        state._tt_audio_cross_pe_cos.update(a_xpe_cos, False)
        state._tt_audio_cross_pe_sin.update(a_xpe_sin, False)
        state._tt_video_cross_pe_cos_full.update(v_xpe_cos_full, False)
        state._tt_video_cross_pe_sin_full.update(v_xpe_sin_full, False)
        state._tt_audio_cross_pe_cos_full.update(a_xpe_cos_full, False)
        state._tt_audio_cross_pe_sin_full.update(a_xpe_sin_full, False)
        state._tt_audio_attn_mask.update(tt_attn_mask, False)
        state._tt_audio_padding_mask.update(tt_pad_mask_sp, False)
        state._tt_audio_padding_mask_full.update(tt_pad_mask_full, False)
        state._tt_video_padding_mask.update(self._prepare_video_masks(video_N, video_N_real), False)
        v_mask = torch.ones(1, 1, video_N, self.in_channels)
        v_mask[:, :, video_N_real:, :] = 0.0
        a_mask = torch.ones(1, 1, audio_N, self.in_channels)
        a_mask[:, :, audio_N_real:, :] = 0.0
        state._tt_video_pad_mask.update(v_mask, False, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device)
        state._tt_audio_pad_mask.update(a_mask, False, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device)

    def _prealloc_trace_io(self, trace_key, *, num_frames, height, width):
        """Allocate a stage's persistent trace inputs (constants, latent buffers, masks) up front,
        before any capture. A ttnn trace bakes absolute tensor addresses; activations allocated
        during capture are freed afterward and reused by the next capture, so a held input sitting
        in another trace's activation region is overwritten on replay. Allocating every held input
        for both stages first keeps them below both traces' activations. (The prompt is built
        separately in _denoise.)"""
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        video_N_real = latent_frames * latent_h * latent_w
        video_N = self._sp_pad_len(video_N_real)
        als = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
        )
        audio_N_real = als.frames
        audio_N = self._sp_pad_len(audio_N_real)
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        state = self._trace_state.setdefault(trace_key, LTXTransformerState())
        self._prepare_stage_statics(
            state,
            latent_frames=latent_frames,
            latent_h=latent_h,
            latent_w=latent_w,
            video_N=video_N,
            video_N_real=video_N_real,
            audio_N=audio_N,
            audio_N_real=audio_N_real,
            sp_axis=sp_axis,
        )
        # Reserve the latent buffers before capture so the trace bakes their addresses.
        if state.tt_video_lat is None:
            state._tt_video_lat.update(
                torch.zeros(1, 1, video_N, self.in_channels),
                False,
                mesh_axes=[None, None, sp_axis, None],
                device=self.mesh_device,
            )
            state._tt_audio_lat.update(
                torch.zeros(1, 1, audio_N, self.in_channels),
                False,
                mesh_axes=[None, None, sp_axis, None],
                device=self.mesh_device,
            )

    def _denoise_no_guidance(
        self,
        v_embeds: torch.Tensor,
        a_embeds: torch.Tensor,
        *,
        num_frames: int,
        height: int,
        width: int,
        sigma_values: list[float],
        seed: int,
        initial_video_latent: torch.Tensor | None = None,
        initial_audio_latent: torch.Tensor | None = None,
        traced: bool = False,
        trace_key: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = 1
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        video_N_real = latent_frames * latent_h * latent_w
        # SP padding: round the video seq dim up to TILE_SIZE * sp_factor for ring SDPA.
        video_N = self._sp_pad_len(video_N_real)
        als = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
        )
        audio_N_real = als.frames
        audio_N = self._sp_pad_len(audio_N_real)
        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        logger.info(f"  shapes: vN={video_N}(real={video_N_real}), aN={audio_N}(real={audio_N_real}) [sp={sp_factor}]")

        # Persist buffers only when traced (baked addresses, fixed shape); untraced uses a transient
        # state so statics rebuild — resolution can differ across generates.
        state = self._trace_state.setdefault(trace_key, LTXTransformerState()) if traced else LTXTransformerState()
        self._prepare_stage_statics(
            state,
            latent_frames=latent_frames,
            latent_h=latent_h,
            latent_w=latent_w,
            video_N=video_N,
            video_N_real=video_N_real,
            audio_N=audio_N,
            audio_N_real=audio_N_real,
            sp_axis=sp_axis,
        )

        prompt_v = self._prepare_prompt(v_embeds)
        prompt_a = bf16_tensor(a_embeds.unsqueeze(0), device=self.mesh_device)
        # Traced persists the shared prompt (baked address); untraced keeps the locals so they
        # don't fragment DRAM for the downstream VAE decode.
        if traced:
            self._prompt_v.update(prompt_v, traced)
            self._prompt_a.update(prompt_a, traced)
            prompt_v, prompt_a = self._prompt_v.value, self._prompt_a.value

        sigmas = torch.tensor(sigma_values, dtype=torch.float32)

        # ----- Video latent init (always end up with shape (B, video_N, C)) -----
        if initial_video_latent is not None:
            # Stage-2 path: upsampled latent comes in at (B, video_N_real, C).
            video_lat_real = initial_video_latent.float()
            assert video_lat_real.shape[1] == video_N_real, (
                f"initial_video_latent seq dim {video_lat_real.shape[1]} != " f"video_N_real {video_N_real}"
            )
            torch.manual_seed(seed)
            noise_v = torch.randn_like(video_lat_real)
            video_lat_real = video_lat_real * (1 - sigmas[0]) + noise_v * sigmas[0]
        else:
            torch.manual_seed(seed)
            video_lat_real = torch.randn(B, video_N_real, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]

        if video_N > video_N_real:
            video_lat = torch.zeros(B, video_N, self.in_channels)
            video_lat[:, :video_N_real, :] = video_lat_real
        else:
            video_lat = video_lat_real

        # ----- Audio latent init (unchanged: already padded to audio_N) -----
        if initial_audio_latent is not None:
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = initial_audio_latent[:, :audio_N_real, :].float()
            torch.manual_seed(seed + 1)
            noise_a = torch.randn_like(audio_lat)
            audio_lat = audio_lat * (1 - sigmas[0]) + noise_a * sigmas[0]
        else:
            torch.manual_seed(seed)
            # Consume the same number of RNG draws the previous (video_N-sized) randn
            # consumed, so the audio RNG stream is bit-identical to prior runs.
            _ = torch.randn(B, video_N_real, self.in_channels, dtype=torch.bfloat16)
            audio_lat_real = torch.randn(B, audio_N_real, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = audio_lat_real

        num_steps = len(sigma_values) - 1

        # Device-resident loop: inner_step returns SP-sharded velocity (gather_output=False),
        # stepped in place by an on-device Euler. update() copies into the address-baked buffer when
        # traced, rebinds otherwise.
        state._tt_video_lat.update(
            video_lat.unsqueeze(0), traced, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device
        )
        state._tt_audio_lat.update(
            audio_lat.unsqueeze(0), traced, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device
        )

        for step_idx in range(num_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()
            state._tt_timestep.update(
                torch.tensor([sigma]).reshape(1, 1, B, 1) * 1000.0, traced, device=self.mesh_device
            )
            # video_N is the LOGICAL (unpadded) count, forwarded as logical_n so ring SDPA masks
            # padded K positions.
            v_out, a_out = self.transformer.inner_step(
                video_1BNI=state.tt_video_lat,
                timestep=state.tt_timestep,
                audio_1BNI=state.tt_audio_lat,
                video_prompt_1BLP=prompt_v,
                video_rope_cos=state.tt_video_rope_cos,
                video_rope_sin=state.tt_video_rope_sin,
                video_N=video_N_real,
                trans_mat=state.tt_trans_mat,
                audio_prompt_1BLP=prompt_a,
                audio_rope_cos=state.tt_audio_rope_cos,
                audio_rope_sin=state.tt_audio_rope_sin,
                audio_N=audio_N,
                video_cross_pe_cos=state.tt_video_cross_pe_cos,
                video_cross_pe_sin=state.tt_video_cross_pe_sin,
                audio_cross_pe_cos=state.tt_audio_cross_pe_cos,
                audio_cross_pe_sin=state.tt_audio_cross_pe_sin,
                video_cross_pe_cos_full=state.tt_video_cross_pe_cos_full,
                video_cross_pe_sin_full=state.tt_video_cross_pe_sin_full,
                audio_cross_pe_cos_full=state.tt_audio_cross_pe_cos_full,
                audio_cross_pe_sin_full=state.tt_audio_cross_pe_sin_full,
                audio_attn_mask=state.tt_audio_attn_mask,
                audio_padding_mask=state.tt_audio_padding_mask,
                audio_padding_mask_full=state.tt_audio_padding_mask_full,
                video_padding_mask=state.tt_video_padding_mask,
                gather_output=False,
                traced=traced,
                tracer_trace_key=trace_key,
            )
            # In-place flow-matching Euler (latents += dt*velocity); SP-padding slots zeroed.
            # In place so the trace's baked latent address holds across replays.
            dt = sigma_next - sigma
            v_vel = ttnn.typecast(v_out, ttnn.bfloat16)
            ttnn.multiply_(v_vel, state.tt_video_pad_mask)
            ttnn.multiply_(v_vel, dt)
            ttnn.add_(state.tt_video_lat, v_vel)
            ttnn.multiply_(state.tt_video_lat, state.tt_video_pad_mask)
            a_vel = ttnn.typecast(a_out, ttnn.bfloat16)
            ttnn.multiply_(a_vel, state.tt_audio_pad_mask)
            ttnn.multiply_(a_vel, dt)
            ttnn.add_(state.tt_audio_lat, a_vel)
            ttnn.multiply_(state.tt_audio_lat, state.tt_audio_pad_mask)
            logger.info(f"  Step {step_idx + 1}/{num_steps}: σ {sigma:.4f} → {sigma_next:.4f}")

        v_final = LTXTransformerModel.device_to_host(
            state.tt_video_lat,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            sp_already_gathered=False,
            tp_already_gathered=True,
        ).squeeze(0)
        a_final = LTXTransformerModel.device_to_host(
            state.tt_audio_lat,
            ccl_manager=self.ccl_manager,
            parallel_config=self.parallel_config,
            sp_already_gathered=False,
            tp_already_gathered=True,
        ).squeeze(0)
        return v_final[:, :video_N_real, :], a_final[:, :audio_N_real, :]

    @staticmethod
    def _write_stage1_wav(audio_obj, path: str) -> None:
        """Write decoded audio as stereo WAV (2, T) → (T, 2) for soundfile."""
        import soundfile as sf

        wav = audio_obj.waveform
        if wav.dim() == 3:
            wav = wav[0]
        if wav.dim() == 2:
            wav = wav.transpose(0, 1)
        sf.write(path, wav.cpu().numpy(), int(audio_obj.sampling_rate))

    def generate_stage1_only(
        self,
        prompt: str,
        *,
        output_path: str,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        seed: int = 10,
        fps: int = 24,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run stage-1 denoise only; optionally dump ``<output_path>_s1.wav``."""
        assert height % 64 == 0, f"Height must be divisible by 64 (got {height})"
        assert width % 64 == 0, f"Width must be divisible by 64 (got {width})"

        s1_height, s1_width = height // 2, width // 2
        # On-device Gemma encode (coresident-excluded with the DiT/VAE, so it auto-evicts
        # them and _prepare_transformer(0) evicts the encoder back). Only load on a cache miss.
        if not os.path.exists(self._device_embed_cache_path([prompt])):
            self.gemma_encoder_pair.ensure_loaded()
        enc = self.encode_prompts([prompt])
        v_embeds, a_embeds = enc[0][0].float(), enc[0][1].float()

        self._prepare_transformer(0)

        s1_video, s1_audio = self._denoise_no_guidance(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=s1_height,
            width=s1_width,
            sigma_values=DISTILLED_SIGMA_VALUES,
            seed=seed,
        )

        if os.environ.get("LTX_DECODE_S1_AUDIO", "").lower() in ("1", "true", "yes") or os.environ.get(
            "LTX_DUMP_STAGE1_AUDIO", ""
        ).lower() in ("1", "true", "yes"):
            s1_path = output_path.replace(".mp4", "_s1.wav")
            audio_obj = self.decode_audio(s1_audio, num_frames, fps=fps)
            if audio_obj is not None:
                self._write_stage1_wav(audio_obj, s1_path)
                logger.info(f"Stage-1 audio dump: wrote {s1_path}")

        return s1_video, s1_audio

    def generate(
        self,
        prompt: str,
        *,
        output_path: str,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        seed: int = 10,
        fps: int = 24,
    ) -> str:
        """Run the distilled 2-stage AV pipeline and write an MP4."""
        assert height % 64 == 0, f"Height must be divisible by 64 (got {height})"
        assert width % 64 == 0, f"Width must be divisible by 64 (got {width})"

        s1_height = height // 2
        s1_width = width // 2

        # (label, seconds) rows counted toward the total; prepares and export are excluded.
        timings: list[tuple[str, float]] = []

        t0 = time.time()
        # On-device Gemma encode. Only load the encoder (coresident-evicts DiT/VAE) on a cache
        # miss — a cached prompt skips the encoder entirely.
        cached = os.path.exists(self._device_embed_cache_path([prompt]))
        if not cached:
            self.gemma_encoder_pair.ensure_loaded()
        enc = self.encode_prompts([prompt])
        v_embeds, a_embeds = enc[0][0].float(), enc[0][1].float()
        t_encode = time.time() - t0
        timings.append(("Encoder (cache)" if cached else "Encoder", t_encode))
        logger.info(f"Encoding ({'cache' if cached else 'device'}): {t_encode:.1f}s")

        t0 = time.time()
        self._prepare_transformer(0)
        if self.dynamic_load:
            logger.info(f"Transformer prepare: {time.time() - t0:.1f}s")

        logger.info(f"Stage 1: {s1_height}x{s1_width}, {len(DISTILLED_SIGMA_VALUES) - 1} steps")
        t0 = time.time()
        s1_video, s1_audio = self._denoise_no_guidance(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=s1_height,
            width=s1_width,
            sigma_values=DISTILLED_SIGMA_VALUES,
            seed=seed,
            traced=self._traced,
            trace_key="s1",
        )
        t_stage1 = time.time() - t0
        timings.append(("Stage 1 denoise", t_stage1))
        logger.info(f"Stage 1 denoise: {t_stage1:.1f}s")

        if os.environ.get("LTX_DECODE_S1_AUDIO", "").lower() in ("1", "true", "yes"):
            s1_path = output_path.replace(".mp4", "_s1.wav")
            audio_obj = self.decode_audio(s1_audio, num_frames, fps=fps)
            if audio_obj is not None:
                self._write_stage1_wav(audio_obj, s1_path)
                logger.info(f"LTX_DECODE_S1_AUDIO: wrote {s1_path}")
            return output_path

        latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
        s1_h, s1_w = s1_height // SPATIAL_COMPRESSION, s1_width // SPATIAL_COMPRESSION
        s1_spatial = s1_video.reshape(1, latent_frames, s1_h, s1_w, 128).permute(0, 4, 1, 2, 3)
        t0 = time.time()
        upsampled = self._upsample_latent(s1_spatial)
        t_upsample = time.time() - t0
        timings.append(("Latent upsample", t_upsample))
        logger.info(f"Latent upsample: {t_upsample:.1f}s")
        upsampled_flat = upsampled.permute(0, 2, 3, 4, 1).reshape(
            1, latent_frames * (height // SPATIAL_COMPRESSION) * (width // SPATIAL_COMPRESSION), 128
        )

        logger.info(f"Stage 2: {height}x{width}, {len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1} steps")
        t0 = time.time()
        s2_video, s2_audio = self._denoise_no_guidance(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=height,
            width=width,
            sigma_values=STAGE_2_DISTILLED_SIGMA_VALUES,
            seed=seed,
            initial_video_latent=upsampled_flat,
            initial_audio_latent=s1_audio.unsqueeze(0) if s1_audio.dim() == 2 else s1_audio,
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
        t0 = time.time()
        video_pixels = self.decode_latents(s2_video, latent_frames, latent_h, latent_w)
        t_vae_decode = time.time() - t0
        timings.append(("VAE decode", t_vae_decode))
        logger.info(f"VAE decode (forward): {t_vae_decode:.1f}s — {tuple(video_pixels.shape)}")

        t0 = time.time()
        audio_obj = self.decode_audio(s2_audio, num_frames, fps=fps)
        t_audio_decode = time.time() - t0
        timings.append(("Audio decode", t_audio_decode))
        logger.info(f"Audio decode: {t_audio_decode:.1f}s")

        t0 = time.time()
        export_video_audio(video_pixels, output_path, fps=fps, audio=audio_obj)
        logger.info(f"Video export: {time.time() - t0:.1f}s")

        self.last_timings = list(timings)
        logger.info(f"Total (compute): {sum(s for _, s in timings):.1f}s | Output: {output_path}")
        return output_path

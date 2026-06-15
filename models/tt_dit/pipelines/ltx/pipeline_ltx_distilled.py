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
from .pipeline_ltx import SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION, LTXPipeline, euler_step, latent_grid

# Distilled sigma schedules for the two stages.
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


class LTXDistilledPipeline(LTXPipeline):
    """Distilled 2-stage AV pipeline: half-res denoise → upsample → full-res refine."""

    HAS_UPSAMPLER = True

    @staticmethod
    def create_pipeline(mesh_device: ttnn.MeshDevice, **kwargs) -> "LTXDistilledPipeline":
        kwargs["pipeline_class"] = LTXDistilledPipeline
        return LTXPipeline.create_pipeline(mesh_device, **kwargs)

    def warmup_buffers(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int = 2,
        stages: tuple[str, ...] = ("s1", "s2"),
    ) -> None:
        """Compile every program both stages will hit. Both stages use variant 0
        (distilled doesn't swap weights between stages); only the sequence length
        differs. Pass ``stages=("s1",)`` to skip the full-res s2 warmup."""
        assert height % 64 == 0 and width % 64 == 0, f"H/W must be div by 64 (got {height}x{width})"
        assert num_frames > 0, f"num_frames must be > 0 (got {num_frames})"
        valid = {"s1", "s2"}
        assert set(stages).issubset(valid), f"stages must be subset of {valid} (got {stages})"

        t0 = time.time()
        logger.info(
            f"warmup (distilled 2-stage): {num_frames}f@{height}x{width}, "
            f"stages={stages}, {num_inference_steps} steps/stage"
        )

        # Dummy zero embeddings at the real shapes — the denoise warmup below only needs
        # to compile the (shape-driven) kernels, not real prompt content. The encoder is
        # warmed separately at the end of this method (it coresident-evicts the DiT/VAE).
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
            full_latent_count = latent_frames * full_lh * full_lw
            dummy_v_init = torch.zeros(1, full_latent_count, self.in_channels)

            vps = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
            als = AudioLatentShape.from_video_pixel_shape(vps)
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

            self._prepare_transformer(0)

        # Warm the encoder last: it coresident-evicts the DiT/VAE, so gen #0 then re-loads the DiT.
        # use_cache=False forces a real encode so the Gemma/connector kernels actually compile.
        self.gemma_encoder_pair.ensure_loaded()
        self.encode_prompts(["warmup"], use_cache=False)

        logger.info(f"warmup (distilled 2-stage) done in {time.time() - t0:.1f}s")

    def _prealloc_trace_io(self, trace_key, *, num_frames, height, width):
        """Allocate and cache a stage's persistent trace inputs (constants, latent buffers,
        padding masks) up front, before any capture. A ttnn trace bakes absolute tensor
        addresses; activations allocated during capture are freed afterward and reused by the
        next capture, so a held input sitting in another trace's activation region is
        overwritten on replay. Allocating every held input for both stages first keeps them
        below both traces' activations. (The prompt is built separately in _denoise.)"""
        B = 1
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        video_N_real = latent_frames * latent_h * latent_w
        video_N = self._sp_pad_len(video_N_real)
        vps = VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
        als = AudioLatentShape.from_video_pixel_shape(vps)
        audio_N_real = als.frames
        audio_N = self._sp_pad_len(audio_N_real)
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        if trace_key not in self._trace_consts:
            v_cos, v_sin = self._prepare_rope(latent_frames, latent_h, latent_w)
            a_cos, a_sin = self._prepare_audio_rope(audio_N, audio_N_real)
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
            trans_mat = self._prepare_trans_mat()
            tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full = self._prepare_audio_masks(audio_N, audio_N_real)
            tt_v_pad_mask_sp = self._prepare_video_masks(video_N, video_N_real)
            self._trace_consts[trace_key] = (
                v_cos,
                v_sin,
                a_cos,
                a_sin,
                v_xpe_cos,
                v_xpe_sin,
                a_xpe_cos,
                a_xpe_sin,
                v_xpe_cos_full,
                v_xpe_sin_full,
                a_xpe_cos_full,
                a_xpe_sin_full,
                trans_mat,
                tt_attn_mask,
                tt_pad_mask_sp,
                tt_pad_mask_full,
                tt_v_pad_mask_sp,
            )

        if trace_key not in self._trace_latents:
            video_lat_dev = bf16_tensor(
                torch.zeros(1, B, video_N, self.in_channels),
                device=self.mesh_device,
                mesh_axis=sp_axis,
                shard_dim=-2,
            )
            audio_lat_dev = bf16_tensor(
                torch.zeros(1, B, audio_N, self.in_channels),
                device=self.mesh_device,
                mesh_axis=sp_axis,
                shard_dim=-2,
            )
            v_mask = torch.ones(1, B, video_N, self.in_channels)
            v_mask[:, :, video_N_real:, :] = 0.0
            a_mask = torch.ones(1, B, audio_N, self.in_channels)
            a_mask[:, :, audio_N_real:, :] = 0.0
            video_pad_mask = bf16_tensor(v_mask, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2)
            audio_pad_mask = bf16_tensor(a_mask, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2)
            self._trace_latents[trace_key] = (video_lat_dev, audio_lat_dev, video_pad_mask, audio_pad_mask)

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
        # SP padding: round video seq dim up to TILE_SIZE * sp_factor so ring SDPA's
        # N_local % TILE_HEIGHT and N_global == N_local * ring_size checks pass.
        video_N = self._sp_pad_len(video_N_real)

        vps = VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
        als = AudioLatentShape.from_video_pixel_shape(vps)
        audio_N_real = als.frames
        sp_factor = self.parallel_config.sequence_parallel.factor
        audio_N = self._sp_pad_len(audio_N_real)

        logger.info(
            f"  shapes: vN={video_N}(real={video_N_real}), aN={audio_N}(real={audio_N_real}) " f"[sp={sp_factor}]"
        )

        # Reuse the pre-allocated constants when traced; only build them on the eager path.
        # Rebuilding them per generate would place fresh tensors at addresses the persisted
        # trace expects to own, so the held copies (allocated at warmup) are reused as-is.
        if traced and trace_key in self._trace_consts:
            (
                v_cos,
                v_sin,
                a_cos,
                a_sin,
                v_xpe_cos,
                v_xpe_sin,
                a_xpe_cos,
                a_xpe_sin,
                v_xpe_cos_full,
                v_xpe_sin_full,
                a_xpe_cos_full,
                a_xpe_sin_full,
                trans_mat,
                tt_attn_mask,
                tt_pad_mask_sp,
                tt_pad_mask_full,
                tt_v_pad_mask_sp,
            ) = self._trace_consts[trace_key]
        else:
            v_cos, v_sin = self._prepare_rope(latent_frames, latent_h, latent_w)
            a_cos, a_sin = self._prepare_audio_rope(audio_N, audio_N_real)
            # Without cross-PE the A↔V cross-attention runs with no positional info,
            # destroying audio-video temporal sync (lip sync). Reference: pipeline_ltx.py.
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
            trans_mat = self._prepare_trans_mat()
            tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full = self._prepare_audio_masks(audio_N, audio_N_real)
            tt_v_pad_mask_sp = self._prepare_video_masks(video_N, video_N_real)

            if traced:
                self._trace_consts[trace_key] = (
                    v_cos,
                    v_sin,
                    a_cos,
                    a_sin,
                    v_xpe_cos,
                    v_xpe_sin,
                    a_xpe_cos,
                    a_xpe_sin,
                    v_xpe_cos_full,
                    v_xpe_sin_full,
                    a_xpe_cos_full,
                    a_xpe_sin_full,
                    trans_mat,
                    tt_attn_mask,
                    tt_pad_mask_sp,
                    tt_pad_mask_full,
                    tt_v_pad_mask_sp,
                )

        # One prompt buffer shared by both stages (the text embedding is identical). Built on
        # the first traced step — before s1's capture, so it sits below both traces'
        # activations rather than inside a prior trace's footprint. Refreshed in place each
        # generate (WAN's prepare_text_conditioning pattern) so a new prompt is read without
        # reallocating the buffer whose address the trace baked.
        if traced:
            prompt_buf = self._trace_prompt.get("shared")
            if prompt_buf is None:
                tt_vp = self._prepare_prompt(v_embeds)
                tt_ap = bf16_tensor(a_embeds.unsqueeze(0), device=self.mesh_device)
                self._trace_prompt["shared"] = (tt_vp, tt_ap)
            else:
                tt_vp, tt_ap = prompt_buf
                ttnn.copy(self._prepare_prompt(v_embeds), tt_vp)
                ttnn.copy(bf16_tensor(a_embeds.unsqueeze(0), device=self.mesh_device), tt_ap)
        else:
            tt_vp = self._prepare_prompt(v_embeds)
            tt_ap = bf16_tensor(a_embeds.unsqueeze(0), device=self.mesh_device)

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
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        if traced:
            # Device-resident denoise: the SP-sharded latent stays on device for the whole
            # loop, inner_step returns sharded velocity (gather_output=False), and an
            # on-device Euler steps it in place. No per-step host round-trip or fresh device
            # allocation, so a persisted trace's baked buffer addresses stay valid across
            # generates (a ttnn trace bakes absolute tensor addresses into its command stream).
            if trace_key in self._trace_latents:
                video_lat_dev, audio_lat_dev, video_pad_mask, audio_pad_mask = self._trace_latents[trace_key]
                ttnn.copy(
                    bf16_tensor(video_lat.unsqueeze(0), device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2),
                    video_lat_dev,
                )
                ttnn.copy(
                    bf16_tensor(audio_lat.unsqueeze(0), device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2),
                    audio_lat_dev,
                )
            else:
                video_lat_dev = bf16_tensor(
                    video_lat.unsqueeze(0), device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2
                )
                audio_lat_dev = bf16_tensor(
                    audio_lat.unsqueeze(0), device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2
                )
                # Full-channel 0/1 masks zero the SP-padding slots (replaces host _zero_*_padding).
                v_mask = torch.ones(1, B, video_N, self.in_channels)
                v_mask[:, :, video_N_real:, :] = 0.0
                a_mask = torch.ones(1, B, audio_N, self.in_channels)
                a_mask[:, :, audio_N_real:, :] = 0.0
                video_pad_mask = bf16_tensor(v_mask, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2)
                audio_pad_mask = bf16_tensor(a_mask, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=-2)
                self._trace_latents[trace_key] = (video_lat_dev, audio_lat_dev, video_pad_mask, audio_pad_mask)

            for step_idx in range(num_steps):
                sigma = sigmas[step_idx].item()
                dt = sigmas[step_idx + 1].item() - sigma
                timestep = bf16_tensor(
                    torch.tensor([sigma]).reshape(1, 1, B, 1) * 1000.0, device=self.mesh_device, on_host=True
                )
                # video_lat_dev/audio_lat_dev are the SAME persistent objects every call;
                # updated in place below, so the trace (same buffer address) reads fresh data.
                v_out, a_out = self._traced_step(
                    trace_key,
                    self.transformer.inner_step,
                    capture_inputs=dict(
                        video_1BNI=video_lat_dev,
                        timestep=timestep,
                        audio_1BNI=audio_lat_dev,
                        video_prompt_1BLP=tt_vp,
                        video_rope_cos=v_cos,
                        video_rope_sin=v_sin,
                        video_N=video_N_real,
                        trans_mat=trans_mat,
                        audio_prompt_1BLP=tt_ap,
                        audio_rope_cos=a_cos,
                        audio_rope_sin=a_sin,
                        audio_N=audio_N,
                        video_cross_pe_cos=v_xpe_cos,
                        video_cross_pe_sin=v_xpe_sin,
                        audio_cross_pe_cos=a_xpe_cos,
                        audio_cross_pe_sin=a_xpe_sin,
                        video_cross_pe_cos_full=v_xpe_cos_full,
                        video_cross_pe_sin_full=v_xpe_sin_full,
                        audio_cross_pe_cos_full=a_xpe_cos_full,
                        audio_cross_pe_sin_full=a_xpe_sin_full,
                        audio_attn_mask=tt_attn_mask,
                        audio_padding_mask=tt_pad_mask_sp,
                        audio_padding_mask_full=tt_pad_mask_full,
                        video_padding_mask=tt_v_pad_mask_sp,
                        gather_output=False,
                    ),
                    replay_inputs=dict(
                        video_1BNI=video_lat_dev,
                        timestep=timestep,
                        audio_1BNI=audio_lat_dev,
                        video_prompt_1BLP=tt_vp,
                        audio_prompt_1BLP=tt_ap,
                    ),
                )
                # On-device flow-matching Euler x += dt*vel, padded slots zeroed (matches
                # the host euler_step + _zero_*_padding). Step tensors free each iteration;
                # only the latent buffers persist.
                v_vel = ttnn.mul(ttnn.typecast(v_out, ttnn.bfloat16), video_pad_mask)
                ttnn.copy(ttnn.mul(ttnn.add(video_lat_dev, ttnn.mul(v_vel, dt)), video_pad_mask), video_lat_dev)
                a_vel = ttnn.mul(ttnn.typecast(a_out, ttnn.bfloat16), audio_pad_mask)
                ttnn.copy(ttnn.mul(ttnn.add(audio_lat_dev, ttnn.mul(a_vel, dt)), audio_pad_mask), audio_lat_dev)
                logger.info(f"  Step {step_idx + 1}/{num_steps}: σ {sigma:.4f} → {sigma + dt:.4f}")

            v_final = LTXTransformerModel.device_to_host(
                video_lat_dev,
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
                sp_already_gathered=False,
                tp_already_gathered=True,
            ).squeeze(0)
            a_final = LTXTransformerModel.device_to_host(
                audio_lat_dev,
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
                sp_already_gathered=False,
                tp_already_gathered=True,
            ).squeeze(0)
            return v_final[:, :video_N_real, :], a_final[:, :audio_N_real, :]

        for step_idx in range(num_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()

            # video_N / audio_N here are the LOGICAL (unpadded) counts forwarded as
            # ``logical_n`` to ring SDPA so padded K positions get masked.
            v_out, a_out = self.transformer.forward(
                video_1BNI_torch=video_lat.unsqueeze(0),
                video_prompt_1BLP=tt_vp,
                video_rope_cos=v_cos,
                video_rope_sin=v_sin,
                video_N=video_N_real,
                audio_1BNI_torch=audio_lat.unsqueeze(0),
                audio_prompt_1BLP=tt_ap,
                audio_rope_cos=a_cos,
                audio_rope_sin=a_sin,
                audio_N=audio_N,
                trans_mat=trans_mat,
                timestep_torch=torch.tensor([sigma]),
                video_cross_pe_cos=v_xpe_cos,
                video_cross_pe_sin=v_xpe_sin,
                audio_cross_pe_cos=a_xpe_cos,
                audio_cross_pe_sin=a_xpe_sin,
                video_cross_pe_cos_full=v_xpe_cos_full,
                video_cross_pe_sin_full=v_xpe_sin_full,
                audio_cross_pe_cos_full=a_xpe_cos_full,
                audio_cross_pe_sin_full=a_xpe_sin_full,
                audio_attn_mask=tt_attn_mask,
                audio_padding_mask=tt_pad_mask_sp,
                audio_padding_mask_full=tt_pad_mask_full,
                video_padding_mask=tt_v_pad_mask_sp,
            )
            v_vel = LTXTransformerModel.device_to_host(
                v_out,
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
                sp_already_gathered=True,
                tp_already_gathered=True,
            ).squeeze(0)
            a_vel = LTXTransformerModel.device_to_host(
                a_out,
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
                sp_already_gathered=True,
                tp_already_gathered=True,
            ).squeeze(0)
            # Zero padded velocity slots so they don't drift the latent in the padded region.
            v_vel = self._zero_sp_padding(v_vel, video_N_real)

            v_den = (video_lat.bfloat16().float() - v_vel.float() * sigma).bfloat16()
            a_den = (audio_lat.bfloat16().float() - a_vel.float() * sigma).bfloat16()

            if sigma_next == 0.0:
                v_new = v_den.float()
                a_new = a_den.float()
            else:
                v_new = euler_step(video_lat, v_den.float(), sigma, sigma_next).bfloat16().float()
                a_new = euler_step(audio_lat, a_den.float(), sigma, sigma_next).bfloat16().float()

            # Re-zero padded slots in both modalities after each step.
            video_lat = self._zero_sp_padding(v_new, video_N_real)
            audio_lat = self._zero_sp_padding(a_new, audio_N_real)
            logger.info(f"  Step {step_idx + 1}/{num_steps}: σ {sigma:.4f} → {sigma_next:.4f}")

        # Traces are kept resident across generate() calls and freed by release_traces().
        return video_lat[:, :video_N_real, :], audio_lat[:, :audio_N_real, :]

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
        total_t0 = time.time()

        t0 = time.time()
        # On-device Gemma encode. Only load the encoder (coresident-evicts DiT/VAE) on a cache
        # miss — a cached prompt skips the encoder entirely.
        cached = os.path.exists(self._device_embed_cache_path([prompt]))
        if not cached:
            self.gemma_encoder_pair.ensure_loaded()
        enc = self.encode_prompts([prompt])
        v_embeds, a_embeds = enc[0][0].float(), enc[0][1].float()
        logger.info(f"Encoding ({'cache' if cached else 'device'}): {time.time() - t0:.1f}s")

        # Both distilled stages share variant 0 (no weight swap between stages).
        t0 = time.time()
        self._prepare_transformer(0)
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
        logger.info(f"Stage 1 denoise: {time.time() - t0:.1f}s")

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
        logger.info(f"Latent upsample: {time.time() - t0:.1f}s")
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
        logger.info(f"Stage 2 denoise: {time.time() - t0:.1f}s")

        t0 = time.time()
        self._prepare_vae()
        logger.info(f"VAE prepare: {time.time() - t0:.1f}s")

        latent_h, latent_w = height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION
        t0 = time.time()
        video_pixels = self.decode_latents(s2_video, latent_frames, latent_h, latent_w)
        logger.info(f"VAE decode (forward): {time.time() - t0:.1f}s — {tuple(video_pixels.shape)}")

        t0 = time.time()
        audio_obj = self.decode_audio(s2_audio, num_frames, fps=fps)
        logger.info(f"Audio decode: {time.time() - t0:.1f}s")

        t0 = time.time()
        export_video_audio(video_pixels, output_path, fps=fps, audio=audio_obj)
        logger.info(f"Video export: {time.time() - t0:.1f}s")

        logger.info(f"Total: {time.time() - total_t0:.1f}s | Output: {output_path}")
        return output_path

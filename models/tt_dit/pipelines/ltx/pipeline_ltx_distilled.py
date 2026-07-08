# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 distilled two-stage audio-video pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn

from ...models.transformers.ltx.rope_ltx import prepare_audio_rope, prepare_av_cross_pe, prepare_video_rope
from ...models.transformers.ltx.transformer_ltx import (
    LTX_DIT_PREP_RUN,
    LTXTransformerModel,
    build_audio_masks,
    build_video_pad_mask,
)
from ...models.vae.vae_ltx import upsample_latent
from ...utils import walltime
from ...utils.patchifiers import AudioLatentShape, VideoPixelShape
from ...utils.tensor import bf16_tensor
from ...utils.video import export_video_audio
from .pipeline_ltx import SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION, LTXPipeline, LTXTransformerState, latent_grid

# Distilled sigma schedules for the two stages. The defaults are the shipped 8-step (stage 1)
# and 3-step (stage 2) schedules. LTX_S1_SIGMAS / LTX_S2_SIGMAS override with a comma-separated
# list to A/B fewer-step schedules (L2 step cut). Unset = byte-identical to the shipped baseline.
_DEFAULT_S1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
_DEFAULT_S2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


def _sigma_override(env_name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(default)
    vals = [float(x) for x in raw.split(",") if x.strip() != ""]
    assert len(vals) >= 2, f"{env_name} needs >=2 sigmas (got {vals})"
    assert vals[-1] == 0.0, f"{env_name} must end at 0.0 (got {vals[-1]})"
    assert all(a > b for a, b in zip(vals, vals[1:])), f"{env_name} must be strictly decreasing: {vals}"
    return vals


DISTILLED_SIGMA_VALUES = _sigma_override("LTX_S1_SIGMAS", _DEFAULT_S1_SIGMAS)
STAGE_2_DISTILLED_SIGMA_VALUES = _sigma_override("LTX_S2_SIGMAS", _DEFAULT_S2_SIGMAS)

# Interior (non-edge) keyframe i2v on a few-step schedule: a full-strength pin injects a noise-free
# latent into one interior frame whose two-sided neighbors start from noise, and the coarse stage
# can't reconcile that wall on nodes that are off the distilled 8-step trajectory — the neighbors
# land off-distribution and chromatically scramble (worse, non-monotonically, on medium's off-trajectory
# 6-step). Stage 1 for such gens uses a strict subset of the high sigma nodes and a looser interior
# pin; stage 2 re-locks image identity at full strength (PCC stays ~0.99). Only engages below the
# shipped node count — the 8-step high schedule already carries the reconciliation budget.
_KEYFRAME_S1_ALIGNED = [1.0, 0.909375, 0.725, 0.421875, 0.0]
_KEYFRAME_S1_ALIGNED_LONG = [1.0, 0.975, 0.909375, 0.725, 0.421875, 0.0]
KEYFRAME_S1_STRENGTH = float(os.environ.get("LTX_KEYFRAME_S1_STRENGTH", "0.5"))


def _keyframe_s1_sigmas(latent_frames: int) -> list[float]:
    """Trajectory-aligned S1 nodes for a non-frame-0 keyframe. A longer clip's interior pin has more
    neighbors to reconcile, so a clip past ~6s gets one extra aligned node. LTX_KEYFRAME_S1_SIGMAS
    overrides both."""
    if os.environ.get("LTX_KEYFRAME_S1_SIGMAS", "").strip():
        return _sigma_override("LTX_KEYFRAME_S1_SIGMAS", _KEYFRAME_S1_ALIGNED)
    return _KEYFRAME_S1_ALIGNED_LONG if latent_frames > 20 else _KEYFRAME_S1_ALIGNED


def pixel_to_latent_frame(pixel_idx: int, num_frames: int) -> int:
    """Map a pixel-frame index to the latent frame that owns it.

    The causal temporal VAE encodes pixel frame 0 to latent frame 0, then every
    ``TEMPORAL_COMPRESSION`` pixel frames to one latent frame — so the last pixel frame
    (``num_frames - 1``) lands in the last latent frame. A single encoded conditioning image is one
    latent frame, pinned into that slot. Out-of-range indices clamp into ``[0, latent_frames-1]``."""
    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    if pixel_idx <= 0:
        return 0
    return min((pixel_idx - 1) // TEMPORAL_COMPRESSION + 1, latent_frames - 1)


def build_conditioning_tensors(conds, latent_frames, latent_h, latent_w, in_channels):
    """Place per-frame conditioning latents into the flat token sequence.

    Tokens are frame-major: latent frame ``k`` owns rows ``[k*hw : (k+1)*hw)`` (``hw = latent_h *
    latent_w``). ``conds`` is a list of ``(latent_frame_index, cond_latent, strength)`` where
    ``cond_latent`` is one encoded frame ``(1, C, 1, latent_h, latent_w)``. Returns
    ``(clean_latent [1, N, C], denoise_mask [1, N, 1], pin_rows [N] bool)``: the mask is
    ``1 - strength`` on a pinned frame's rows (1.0 pins it fully to the image) and 1.0 elsewhere.
    Generalizes frame-0 i2v to first/last/arbitrary keyframes without changing the trace (the
    pin op and the pre-allocated pin buffers are position-agnostic and full-length)."""
    hw = latent_h * latent_w
    video_N_real = latent_frames * hw
    clean_latent = torch.zeros(1, video_N_real, in_channels)
    denoise_mask = torch.ones(1, video_N_real, 1)
    pin_rows = torch.zeros(video_N_real, dtype=torch.bool)
    for lat_idx, cond_latent, strength in conds:
        assert 0 <= lat_idx < latent_frames, f"latent frame {lat_idx} out of range [0,{latent_frames})"
        tokens = cond_latent.float().permute(0, 2, 3, 4, 1).reshape(1, -1, in_channels)
        n = tokens.shape[1]
        assert n == hw, f"cond latent has {n} tokens, expected one latent frame = {hw}"
        off = lat_idx * hw
        clean_latent[:, off : off + n, :] = tokens
        denoise_mask[:, off : off + n, :] = 1.0 - strength
        pin_rows[off : off + n] = True
    return clean_latent, denoise_mask, pin_rows


@dataclass
class _I2VConditioning:
    """Image conditioning for one denoise stage — the tt analog of the reference LatentState fields
    written by ``VideoConditionByLatentIndex.apply_to``, generalized from frame-0 to arbitrary
    latent frames (first / last / keyframe)."""

    denoise_mask: torch.Tensor | None  # (B, N, 1): 1−strength at cond tokens else 1; None = plain T2V
    clean_latent: torch.Tensor | None  # (B, N, C): cond tokens at their latent frames else 0
    pin_rows: torch.Tensor | None  # (N,) bool: True at pinned tokens (any latent frame); None = no image


class LTXDistilledPipeline(LTXPipeline):
    """Distilled 2-stage AV pipeline: half-res denoise → upsample → full-res refine."""

    HAS_UPSAMPLER = True
    SUPPORTS_IMAGE_CONDITIONING = True

    @staticmethod
    def _post_process_latent_tt(
        denoised: ttnn.Tensor,
        denoise_mask: ttnn.Tensor,
        clean_latent: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Blend the denoised estimate toward the clean latent: ``denoised * mask + clean * (1 - mask)``.
        Mirrors the reference ``post_process_latent``; the caller applies it to the x0 estimate before
        the Euler step so partial ``image_cond_strength`` integrates like the reference (hard pin at 1.0)."""
        one_minus = ttnn.subtract(
            ttnn.full_like(denoise_mask, 1.0, dtype=ttnn.bfloat16),
            denoise_mask,
        )
        return ttnn.add(ttnn.multiply(denoised, denoise_mask), ttnn.multiply(clean_latent, one_minus))

    @staticmethod
    def _noise_video_latent(
        base: torch.Tensor,
        denoise_mask: torch.Tensor | None,
        sigma: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """GaussianNoiser (ltx_core): ``noise·(mask·σ) + base·(1−mask·σ)``.

        ``denoise_mask`` None is the plain forward step (mask ≡ 1); a per-token mask holding
        ``1−strength`` at the conditioning tokens pins them toward ``base``. Noise is drawn at
        bf16 — the device latent dtype — matching the reference, which draws at ``latent.dtype``."""
        torch.manual_seed(seed)
        noise = torch.randn(base.shape, dtype=torch.bfloat16).to(base.dtype)
        scaled_mask = sigma if denoise_mask is None else denoise_mask * sigma
        return noise * scaled_mask + base * (1.0 - scaled_mask)

    def _build_i2v_conditioning(
        self,
        image_conds: list | None,
        needs_video_ts: bool,
        B: int,
        video_N_real: int,
        latent_frames: int,
        latent_h: int,
        latent_w: int,
    ) -> _I2VConditioning:
        """Build the per-token conditioning (denoise mask + clean latent + pinned-row mask) — tt
        analog of the reference ``create_initial_state`` + ``VideoConditionByLatentIndex.apply_to``,
        one entry per conditioned latent frame. ``denoise_mask`` is None only when the transformer
        needs no per-token video timestep and no image is staged (the plain-T2V forward-noise path)."""
        if not image_conds and not needs_video_ts:
            return _I2VConditioning(denoise_mask=None, clean_latent=None, pin_rows=None)
        if image_conds:
            clean_latent, denoise_mask, pin_rows = build_conditioning_tensors(
                image_conds, latent_frames, latent_h, latent_w, self.in_channels
            )
            logger.info(
                f"I2V: pinning {int(pin_rows.sum())} tokens across {len(image_conds)} frame(s) at "
                f"latent indices {sorted(int(i) for i, _, _ in image_conds)} "
                f"(strength≈{image_conds[0][2]})"
            )
            return _I2VConditioning(denoise_mask=denoise_mask, clean_latent=clean_latent, pin_rows=pin_rows)
        return _I2VConditioning(denoise_mask=torch.ones(B, video_N_real, 1), clean_latent=None, pin_rows=None)

    def warmup_buffers(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int = 2,
        stages: tuple[str, ...] = ("s1", "s2"),
        capture_all: bool = False,
        in_capture_pass: bool = False,
    ) -> None:
        """Compile both stages' programs. Both stages use variant 0 (distilled doesn't
        swap weights — only the sequence length differs); ``stages=("s1",)`` skips s2.

        ``capture_all`` forces a COMPLETE warmup (all eager warmups + the s1/s2 denoise), overriding
        the iter_fast / prep_run skips, so the cold-start capture pass records every kernel the real
        run will use — otherwise a deferred kernel compiles cold in gen#0.

        ``in_capture_pass`` marks the kernel-prewarm capture pass (running under capture-only, dispatch
        skipped). The audio vocoder captures its trace with prep_run=False and lazily initializes device
        statics on first dispatch; a dispatch-skipped capture-only run would leave it half-initialized
        and crash the real run. So skip audio decode here — the real warmup initializes and captures it
        cleanly, its small kernel set compiling in-window."""
        assert height % 64 == 0 and width % 64 == 0, f"H/W must be div by 64 (got {height}x{width})"
        assert num_frames > 0, f"num_frames must be > 0 (got {num_frames})"
        valid = {"s1", "s2"}
        assert set(stages).issubset(valid), f"stages must be subset of {valid} (got {stages})"

        # LTX_ITER_FAST=1 runs a single real generate, so warmup only needs to warm each stage's
        # inner_step kernels + graph state once for the trace capture: one step exercises the same
        # inner_step/Euler ops as two. The eager upsample/VAE/audio warmups are dropped there too
        # since gen#0 compiles them on its lone real use (warming would just run them twice). Full
        # (multi-gen) runs keep the wider warmup so every gen replays a warm set.
        #
        # LTX_DIT_PREP_RUN (inner_step captured with prep_run=True, set when LTX_ITER_FAST is in the
        # env at import time) makes gen#0's first traced step self-warm — its eager prep forward
        # compiles the s1/s2 kernels + builds graph state using gen#0's own persistent statics. The
        # s1/s2 mini-denoise below then does nothing gen#0 doesn't already do, so skip it entirely
        # and keep only the prealloc trace-io + stage statics that gen#0 reuses. When prep_run is off
        # (CI / quality runs), keep the mini-denoise so the prep_run=False capture finds warm kernels.
        iter_fast = (not capture_all) and os.environ.get("LTX_ITER_FAST", "0") in ("1", "true", "True")
        warmup_steps = 1 if iter_fast else num_inference_steps
        skip_dit_warmup = LTX_DIT_PREP_RUN and not capture_all

        t0 = time.time()
        logger.info(
            f"warmup (distilled 2-stage): {num_frames}f@{height}x{width}, "
            f"stages={stages}, {warmup_steps} steps/stage"
        )

        # Zeros at the real shapes compile the shape-driven kernels; the encoder is warmed
        # separately at the end of this method (it coresident-evicts the DiT/VAE).
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
        s1_sigmas = list(DISTILLED_SIGMA_VALUES)[:warmup_steps] + [0.0]
        s2_sigmas = list(STAGE_2_DISTILLED_SIGMA_VALUES)[:warmup_steps] + [0.0]

        if "s1" in stages and not skip_dit_warmup:
            s1_h, s1_w = height // 2, width // 2

            logger.info(f"warmup stage 1: {s1_h}x{s1_w}, σ={s1_sigmas}")
            # Cold kernel compile for the stage-1 denoise — a dominant, otherwise-untracked
            # slice of warmup wall time.
            with walltime.timed("warmup", "stage1 build"):
                self._denoise_no_guidance(
                    v_p,
                    a_p,
                    num_frames=num_frames,
                    height=s1_h,
                    width=s1_w,
                    sigma_values=s1_sigmas,
                    seed=0,
                )
        elif skip_dit_warmup:
            logger.info("LTX_DIT_PREP_RUN: skipping stage-1 warmup denoise (gen#0 self-warms via prep_run)")

        if "s2" in stages:
            # Upsample runs between stage 1 and stage 2; compile its kernels here.
            if not iter_fast:
                logger.info(f"warmup upsample → {height}x{width}")
                self._warmup_upsample(num_frames, height, width)

            # Zero-dummies at the exact shapes the real stage-2 call uses.
            latent_frames, full_lh, full_lw = latent_grid(num_frames, height, width)
            dummy_v_init = torch.zeros(1, latent_frames * full_lh * full_lw, self.in_channels)
            als = AudioLatentShape.from_video_pixel_shape(
                VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
            )
            dummy_a_init = torch.zeros(1, als.frames, self.in_channels)

            if skip_dit_warmup:
                logger.info("LTX_DIT_PREP_RUN: skipping stage-2 warmup denoise (gen#0 self-warms via prep_run)")
            else:
                logger.info(f"warmup stage 2: {height}x{width}, σ={s2_sigmas}")
                # Cold kernel compile for the stage-2 denoise (full-res) — the other dominant
                # warmup slice.
                with walltime.timed("warmup", "stage2 build"):
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
            if not iter_fast:
                self._warmup_decode(num_frames, height, width)

            # Build + JIT-compile the on-device audio decode on the exact latent
            # shape generate() produces, so the first real audio decode loads
            # from cache instead of building from the checkpoint (cold ~64s).
            # The single-generate eager decode is bit-identical to the replay, so the AV mp4 is
            # unchanged; multi-gen runs keep the warmup capture so every real decode replays (else
            # gen#1 would decode into the garbage-emitting capture pass).
            if in_capture_pass:
                logger.info("capture pass: skipping audio decode (vocoder prep_run=False; real warmup inits it)")
            elif iter_fast:
                logger.info("LTX_ITER_FAST=1: skipping warmup audio decode (gen#0 decodes eagerly)")
            else:
                logger.info("warmup audio decode (on-device)")
                self.decode_audio(torch.zeros(1, als.frames, self.in_channels), num_frames, fps=24.0)
                # The vocoder trace captures on the first POST-cold decode (not the cold one above),
                # and that capturing pass emits clipped audio. A second warmup decode absorbs the
                # capture so the first real generate is a clean replay.
                if self._traced:
                    self.decode_audio(torch.zeros(1, als.frames, self.in_channels), num_frames, fps=24.0)

        # Warm the encoders last: they coresident-evict the VAE decoder (which already evicted the
        # DiT), so they never disturb the denoise/decode kernels compiled above.
        # The VAE image encoder is warmed here (not before the denoise warmups) because running it
        # evicts the transformer weights, which the denoise steps above still need resident.
        # Warming whenever the checkpoint has an encoder (not only when an I2V image is staged) keeps
        # a post-capture I2V request from loading the encoder for the first time, which would evict
        # the DiT and clobber the already-captured traces' activation regions — corrupting every
        # subsequent gen to static.
        # LTX_WARMUP_ENCODERS=0 skips both encoder warmups for the steady-state t2v iteration loop:
        # the image encoder is never exercised without an I2V request, and a t2v prompt whose device
        # embeddings are disk-cached never runs the Gemma encoder in generate(). Neither encoder is
        # loaded after capture in that loop, so skipping their warmup compiles is safe there; leave it
        # at the default (warm) for I2V or uncached-prompt runs that load an encoder post-capture.
        if os.environ.get("LTX_WARMUP_ENCODERS", "1") in ("1", "true", "True"):
            if self.vae_encoder is not None:
                logger.info(f"warmup image encoder: {height // 2}x{width // 2} + {height}x{width}")
                self._warmup_encode(height // 2, width // 2)
                self._warmup_encode(height, width)

            # use_cache=False forces a real encode so the Gemma/connector kernels compile. traced-static
            # already warmed before capture (above); dynamic_load / untraced warm last.
            if self.dynamic_load or not self._traced:
                self.gemma_encoder_pair.ensure_loaded()
                self.encode_prompts(["warmup"], use_cache=False)
        else:
            logger.info("LTX_WARMUP_ENCODERS=0: skipping image + gemma encoder warmup")

        logger.info(f"warmup (distilled 2-stage) done in {time.time() - t0:.1f}s")

    def _prepare_stage_statics(
        self, state, *, latent_frames, latent_h, latent_w, video_N, video_N_real, audio_N, audio_N_real, sp_axis
    ):
        """Build a stage's static per-shape inputs once (rope/cross-PE/masks/trans_mat)."""
        if state.tt_video_rope_cos is not None:
            return
        v_cos, v_sin = prepare_video_rope(
            latent_frames,
            latent_h,
            latent_w,
            inner_dim=self.inner_dim,
            num_attention_heads=self.num_attention_heads,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
        )
        a_cos, a_sin = prepare_audio_rope(
            audio_N,
            audio_N_real,
            theta=self.positional_embedding_theta,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
        )
        # Cross-PE required: without it A↔V cross-attention loses positional info and lip sync breaks.
        (
            v_xpe_cos,
            v_xpe_sin,
            a_xpe_cos,
            a_xpe_sin,
            a_xpe_cos_full,
            a_xpe_sin_full,
        ) = prepare_av_cross_pe(
            latent_frames,
            latent_h,
            latent_w,
            audio_N,
            audio_N_real,
            theta=self.positional_embedding_theta,
            mesh_device=self.mesh_device,
            parallel_config=self.parallel_config,
        )
        tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full = build_audio_masks(
            audio_N, audio_N_real, mesh_device=self.mesh_device, sp_axis=sp_axis
        )
        state._tt_video_rope_cos.update(v_cos, False)
        state._tt_video_rope_sin.update(v_sin, False)
        state._tt_audio_rope_cos.update(a_cos, False)
        state._tt_audio_rope_sin.update(a_sin, False)
        state._tt_trans_mat.update(self._prepare_trans_mat(), False)
        state._tt_video_cross_pe_cos.update(v_xpe_cos, False)
        state._tt_video_cross_pe_sin.update(v_xpe_sin, False)
        state._tt_audio_cross_pe_cos.update(a_xpe_cos, False)
        state._tt_audio_cross_pe_sin.update(a_xpe_sin, False)
        state._tt_audio_cross_pe_cos_full.update(a_xpe_cos_full, False)
        state._tt_audio_cross_pe_sin_full.update(a_xpe_sin_full, False)
        state._tt_audio_attn_mask.update(tt_attn_mask, False)
        state._tt_audio_padding_mask.update(tt_pad_mask_sp, False)
        state._tt_audio_padding_mask_full.update(tt_pad_mask_full, False)
        state._tt_video_padding_mask.update(
            build_video_pad_mask(video_N, video_N_real, mesh_device=self.mesh_device, sp_axis=sp_axis), False
        )
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
            # I2V pin buffers (mask + clean frame-0 latent): reserve here so replays can't clobber
            # them. Allocated for every stage even when unused by t2v — small and harmless. The mask
            # is per-token width-1 (broadcasts against the 128-ch latent in the pin); clean carries the
            # real per-channel cond latent, so it stays full width.
            state._tt_i2v_mask.update(
                torch.zeros(1, 1, video_N, 1),
                False,
                mesh_axes=[None, None, sp_axis, None],
                device=self.mesh_device,
            )
            state._tt_i2v_clean.update(
                torch.zeros(1, 1, video_N, self.in_channels),
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
        # I2V conditioning: list of (latent_frame_index, cond_latent (1,C,1,lh,lw), strength).
        image_conds: list | None = None,
        traced: bool = False,
        trace_key: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = 1
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        video_N_real = latent_frames * latent_h * latent_w
        # SP padding: round video seq dim up to TILE_SIZE * sp_factor for ring SDPA.
        video_N = self._sp_pad_len(video_N_real)
        als = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=24)
        )
        audio_N_real = als.frames
        audio_N = self._sp_pad_len(audio_N_real)
        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        image_cond = bool(image_conds)
        # AdaLN's 2-value timestep pair (in the step loop) carries one pinned noise level, so the
        # modulation uses a single shared strength; the per-token pin/denoise_mask below still honors
        # each frame's own strength. Uniform strengths (the server default) make this exact.
        image_cond_strength = image_conds[0][2] if image_cond else 1.0
        needs_video_ts = getattr(self.transformer, "image_conditioning", False)
        i2v = self._build_i2v_conditioning(
            image_conds, needs_video_ts, B, video_N_real, latent_frames, latent_h, latent_w
        )

        logger.info(f"  shapes: vN={video_N}(real={video_N_real}), aN={audio_N}(real={audio_N_real}) [sp={sp_factor}]")

        # Persist buffers only when traced (baked addresses); untraced uses a transient state so
        # statics rebuild — resolution can differ across generates.
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
        # Traced persists the shared prompt (baked address); untraced keeps locals to avoid
        # fragmenting DRAM for the downstream VAE decode.
        if traced:
            self._prompt_v.update(prompt_v, traced)
            self._prompt_a.update(prompt_a, traced)
            prompt_v, prompt_a = self._prompt_v.value, self._prompt_a.value

        sigmas = torch.tensor(sigma_values, dtype=torch.float32)

        # ----- Video latent init: one GaussianNoiser over three bases — I2V (frame-0 replaced by
        # the clean cond latent), T2V-S2 (upsampled latent), T2V-S1 (zeros). denoise_mask pins the
        # conditioning tokens; None ≡ full forward noise. Ends at (B, video_N, C). -----
        if image_cond:  # I2V: zeros (S1) or upsampled (S2), frame-0 overwritten by the cond latent
            if initial_video_latent is not None:
                base_v = initial_video_latent.float()
                if base_v.dim() == 2:
                    base_v = base_v.unsqueeze(0)
                base_v = base_v.clone()
            else:
                base_v = torch.zeros(B, video_N_real, self.in_channels)
            base_v[:, i2v.pin_rows, :] = i2v.clean_latent[:, i2v.pin_rows, :]
        elif initial_video_latent is not None:  # T2V S2: upsampled latent arrives at (B, video_N_real, C)
            base_v = initial_video_latent.float()
            assert (
                base_v.shape[1] == video_N_real
            ), f"initial_video_latent seq dim {base_v.shape[1]} != video_N_real {video_N_real}"
        else:  # T2V S1: pure noise from zeros
            base_v = torch.zeros(B, video_N_real, self.in_channels)
        video_lat_real = self._noise_video_latent(base_v, i2v.denoise_mask, sigmas[0], seed)

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
            # Consume the same RNG draws the prior video_N-sized randn did, keeping the audio
            # stream bit-identical to prior runs.
            _ = torch.randn(B, video_N_real, self.in_channels, dtype=torch.bfloat16)
            audio_lat_real = torch.randn(B, audio_N_real, self.in_channels, dtype=torch.bfloat16).float() * sigmas[0]
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = audio_lat_real

        num_steps = len(sigma_values) - 1

        # Device-resident loop: inner_step returns SP-sharded velocity, stepped in place by an
        # on-device Euler. update() copies into the address-baked buffer when traced, else rebinds.
        state._tt_video_lat.update(
            video_lat.unsqueeze(0), traced, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device
        )
        state._tt_audio_lat.update(
            audio_lat.unsqueeze(0), traced, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device
        )

        tt_i2v_mask = tt_i2v_clean = None
        if image_cond:
            # Mask stays width-1; ttnn broadcasts it against the (…,128) latent in the pin. Padded
            # tokens keep 1.0 (unpinned).
            mask_host = torch.ones(1, 1, video_N, 1)
            mask_host[:, :, :video_N_real, :] = i2v.denoise_mask[0, :, 0].unsqueeze(-1)
            clean_host = torch.zeros(1, 1, video_N, self.in_channels)
            clean_host[:, :, :video_N_real, :] = i2v.clean_latent
            # Copy into the pre-allocated trace-baked buffers so pin inputs keep stable addresses
            # across replays — never freshly allocated here.
            state._tt_i2v_mask.update(mask_host, traced, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device)
            state._tt_i2v_clean.update(
                clean_host, traced, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device
            )
            tt_i2v_mask, tt_i2v_clean = state.tt_i2v_mask, state.tt_i2v_clean

        for step_idx in range(num_steps):
            sigma = sigmas[step_idx].item()
            sigma_next = sigmas[step_idx + 1].item()
            state._tt_timestep.update(
                torch.tensor([sigma]).reshape(1, 1, B, 1) * 1000.0, traced, device=self.mesh_device
            )
            video_ts_pair_tt = video_pin_mask_tt = None
            if needs_video_ts:
                # Per-token timestep has 2 values (pinned frame-0 vs. sigma): pass the (2,) pair +
                # {0,1} pin mask so the transformer blends per token (avoids dense modulation OOM).
                has_pins = image_cond and bool(i2v.pin_rows.any())
                pinned_scale = (1.0 - image_cond_strength) if has_pins else 1.0
                ts_pair = torch.tensor([pinned_scale * sigma, sigma], dtype=torch.float32)
                state._tt_video_ts_pair.update(ts_pair.reshape(1, 1, 2, 1) * 1000.0, traced, device=self.mesh_device)
                pin_mask_host = torch.zeros(1, 1, video_N, 1)
                if has_pins:
                    pin_idx = torch.nonzero(i2v.pin_rows, as_tuple=False).squeeze(-1)
                    pin_mask_host[:, :, pin_idx, :] = 1.0
                state._tt_video_pin_mask.update(
                    pin_mask_host,
                    traced,
                    mesh_axes=[None, None, sp_axis, None],
                    device=self.mesh_device,
                )
                video_ts_pair_tt = state.tt_video_ts_pair
                video_pin_mask_tt = state.tt_video_pin_mask
            # video_N_real is the logical (unpadded) count so ring SDPA masks padded K positions.
            v_out, a_out = self.transformer.inner_step(
                video_1BNI=state.tt_video_lat,
                timestep=state.tt_timestep,
                video_ts_pair=video_ts_pair_tt,
                video_pin_mask=video_pin_mask_tt,
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
            # Flow-matching Euler (latents += dt*velocity, SP-padding slots zeroed) so the trace's
            # baked latent address holds across replays.
            dt = sigma_next - sigma
            v_vel = ttnn.typecast(v_out, ttnn.bfloat16)
            ttnn.multiply_(v_vel, state.tt_video_pad_mask)
            if image_cond:
                # Reference-parity: pin the x0 estimate pre-step, then Euler-step it. Stepping the
                # pinned x0 (not overwriting the latent after) tracks the reference under partial
                # image_cond_strength; equal at strength 1.0. sigma is never 0 in-loop, so dt/sigma is safe.
                x0 = ttnn.subtract(state.tt_video_lat, ttnn.multiply(v_vel, sigma))
                x0 = self._post_process_latent_tt(x0, tt_i2v_mask, tt_i2v_clean)
                v_pin = ttnn.multiply(ttnn.subtract(state.tt_video_lat, x0), dt / sigma)
                ttnn.add_(state.tt_video_lat, v_pin)
            else:
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

    def generate(
        self,
        prompt: str,
        *,
        output_path: str | None = None,
        output_type: str = "rgb",
        # I2V conditioning: list of (image_path, pixel_frame_idx, s1_strength[, s2_strength]).
        # frame_idx 0 = first frame, num_frames-1 = last, any value = a keyframe.
        images: list[tuple] | None = None,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        seed: int = 10,
        fps: int = 24,
    ):
        """Run the distilled 2-stage AV pipeline.

        output_path given → encode an AV MP4 and return its path (str).
        output_path None  → return ``(frames, audio)`` for the caller to encode: frames per
                            ``output_type`` (see ``decode_latents``), audio = decoded ``Audio``.
        """
        assert height % 64 == 0, f"Height must be divisible by 64 (got {height})"
        assert width % 64 == 0, f"Width must be divisible by 64 (got {width})"

        s1_height = height // 2
        s1_width = width // 2

        # Per-stage eager override. The DiT trace-capture cost is a FIXED ~2 forwards per stage
        # (one prep_run compile/alloc forward + one record forward — see @traced_function/Tracer),
        # independent of step count, so for a single-generate run a stage with few denoise steps can
        # be faster run EAGER than traced: the capture overhead isn't amortized over enough replays.
        # Stage 2 (3 steps, the large full-res forward) is the prime case; stage 1 (8 steps) still
        # amortizes. Eager runs the identical op sequence the trace records+replays, so output is
        # BYTE-IDENTICAL; only the dispatch mechanism differs. Mirrors the BWE/VAE trace default-OFF
        # policy for device-compute-bound, few-iteration ops.
        #
        # DEFAULT: under LTX_ITER_FAST (single-generate dev iteration — the test skips gen#1, so the
        # s2 trace is never replayed) default stage 2 to eager. Measured byte-identical (md5==golden)
        # and faster: job 112435-42 vs 080535-41 — Stage-2 denoise 21.6->17.5s, gen 34.6->30.2s,
        # broker ~98.5->93.2s. Full multi-gen runs (LTX_ITER_FAST unset — CI / VBench / CLIP quality)
        # default to all-traced so gen#1 stays a pure replay: byte-identical to the pre-existing path.
        # Override explicitly via LTX_GEN_EAGER_STAGES (e.g. "" forces all-traced; "s1,s2" eagers both).
        _iter_fast = os.environ.get("LTX_ITER_FAST", "0") in ("1", "true", "True")
        eager_stages = {
            s.strip()
            for s in os.environ.get("LTX_GEN_EAGER_STAGES", "s2" if _iter_fast else "").split(",")
            if s.strip()
        }
        if eager_stages:
            logger.info(f"LTX gen: running stages {sorted(eager_stages)} eager (untraced, byte-identical)")

        # (label, seconds) rows counted toward the total; prepares and export are excluded.
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

        s1_image_conds = full_image_conds = None
        s1_sigmas = DISTILLED_SIGMA_VALUES
        if images:
            assert self.vae_encoder is not None, "checkpoint has no VAE encoder; cannot run I2V conditioning"
            latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
            # A non-frame-0 keyframe (interior or last) on a below-high schedule triggers the
            # aligned-node + looser-pin path. Frame-0 i2v is the training-dominant case and stays
            # full-strength on its normal schedule.
            nonzero_kf = any(pixel_to_latent_frame(im[1], num_frames) != 0 for im in images)
            kf_fix = nonzero_kf and len(DISTILLED_SIGMA_VALUES) < len(_DEFAULT_S1_SIGMAS)
            if kf_fix:
                s1_sigmas = _keyframe_s1_sigmas(latent_frames)
            # Each image conditions the latent frame owning its pixel time (0 -> first,
            # num_frames-1 -> last, any value -> a keyframe). A later image at the same latent slot
            # wins — one encoded frame owns each slot. Tuple: (path, frame_idx, s1[, s2] strength); a
            # distinct s2 lets the coarse stage run looser while refine re-locks image identity.
            by_slot = {}
            for img in images:
                lat_idx = pixel_to_latent_frame(img[1], num_frames)
                s1s = img[2] if len(img) > 2 else 1.0
                s2s = img[3] if len(img) > 3 else s1s
                if kf_fix and lat_idx != 0:
                    s1s = min(s1s, KEYFRAME_S1_STRENGTH)  # soften the coarse non-frame-0 pin; s2 re-locks
                by_slot[lat_idx] = (img[0], s1s, s2s)
            # The conditioning latent depends only on (image, resolution), so encode once and memoize.
            # This skips re-running the eager VAE encoder on later gens (e.g. the traced steady-state
            # replay pass, where re-encoding has been observed to hang the device).
            cache = self._i2v_cond_cache
            t0 = time.time()
            s1_conds, full_conds, n_encoded = [], [], 0
            for lat_idx in sorted(by_slot):
                img_path, s1s, s2s = by_slot[lat_idx]
                s1_key = (img_path, s1_height, s1_width)
                full_key = (img_path, height, width)
                if s1_key in cache and full_key in cache:
                    s1_lat, full_lat = cache[s1_key], cache[full_key]
                    logger.info(
                        f"I2V: reusing cached conditioning latents for {img_path} at latent frame "
                        f"{lat_idx} (strength s1={s1s} s2={s2s})"
                    )
                else:
                    logger.info(
                        f"I2V: encoding conditioning image {img_path} at latent frame {lat_idx} "
                        f"(strength s1={s1s} s2={s2s})"
                    )
                    img_s1 = self._load_conditioning_image(img_path, s1_height, s1_width)
                    img_full = self._load_conditioning_image(img_path, height, width)
                    s1_lat = cache[s1_key] = self.encode_image(img_s1)
                    full_lat = cache[full_key] = self.encode_image(img_full)
                    n_encoded += 1
                s1_conds.append((lat_idx, s1_lat, s1s))
                full_conds.append((lat_idx, full_lat, s2s))
            if n_encoded:
                timings.append(("Image encode", time.time() - t0))
                logger.info(f"Image encode ({n_encoded} frame(s)): {time.time() - t0:.1f}s")
            s1_image_conds, full_image_conds = s1_conds, full_conds

        t0 = time.time()
        self._prepare_transformer(0)
        if self.dynamic_load:
            logger.info(f"Transformer prepare: {time.time() - t0:.1f}s")

        logger.info(
            f"Stage 1: {s1_height}x{s1_width}, {len(s1_sigmas) - 1} steps"
            + (" [interior-keyframe aligned]" if s1_sigmas is not DISTILLED_SIGMA_VALUES else "")
        )
        t0 = time.time()
        s1_video, s1_audio = self._denoise_no_guidance(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=s1_height,
            width=s1_width,
            sigma_values=s1_sigmas,
            seed=seed,
            image_conds=s1_image_conds,
            traced=self._traced and "s1" not in eager_stages,
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
            image_conds=full_image_conds,
            traced=self._traced and "s2" not in eager_stages,
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
        # export_video_audio needs float [-1,1]; the frame-return path uses the requested output_type.
        decode_type = "float" if output_path is not None else output_type
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
        # The denoise stages are the generation cost with no Watchdog of their own; surface
        # them from the timings already collected (vae/upsample/audio are tracked elsewhere).
        for _label, _secs in timings:
            if "denoise" in _label.lower():
                walltime.record("gen", _label, _secs)
        if output_path is None:
            logger.info(f"Total (compute): {sum(s for _, s in timings):.1f}s | frames={tuple(video_pixels.shape)}")
            return video_pixels, audio_obj

        t0 = time.time()
        export_video_audio(video_pixels, output_path, fps=fps, audio=audio_obj)
        logger.info(f"Video export: {time.time() - t0:.1f}s")
        logger.info(f"Total (compute): {sum(s for _, s in timings):.1f}s | Output: {output_path}")
        return output_path

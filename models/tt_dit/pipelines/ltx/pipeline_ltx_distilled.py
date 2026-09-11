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
from ...models.transformers.ltx.transformer_ltx import LTXTransformerModel, build_audio_masks, build_video_pad_mask
from ...models.upsampler.latent_upsampler_ltx import LTXLatentUpsampler
from ...models.vae.vae_ltx import upsample_latent
from ...utils.ltx import (
    LTX_AUDIO_N_BUCKET,
    LTX_BUCKET_LADDER,
    LTX_CANVASES,
    LTX_SERVED_CANVASES,
    LTXBucketRoute,
    ceil_to,
    load_conditioning_image,
    ltx_served_configs,
    ltx_stage_video_n_real,
    route_ltx_config,
    route_ltx_request,
    validate_bucket_ladder,
)
from ...utils.patchifiers import AudioLatentShape, VideoPixelShape
from ...utils.tensor import bf16_tensor
from ...utils.video import export_video_audio, export_video_audio_yuv
from .pipeline_ltx import SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION, LTXPipeline, LTXTransformerState, latent_grid

# Distilled sigma schedules for the two stages.
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


@dataclass
class _I2VConditioning:
    """Frame-0 image conditioning for one denoise stage — the tt analog of the reference
    LatentState fields written by ``VideoConditionByLatentIndex.apply_to`` (latent_idx=0)."""

    denoise_mask: torch.Tensor | None  # (B, N, 1): 1−strength at cond tokens else 1; None = plain T2V
    clean_latent: torch.Tensor | None  # (B, N, C): cond tokens at frame-0 else 0
    n_cond: int  # count of pinned frame-0 tokens


class LTXDistilledPipeline(LTXPipeline):
    """Distilled 2-stage AV pipeline: half-res denoise → upsample → full-res refine."""

    HAS_UPSAMPLER = True
    DEFERS_ENCODE_TRACE = True

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
        image_cond_latent: torch.Tensor | None,
        image_cond_strength: float,
        needs_video_ts: bool,
        B: int,
        video_N_real: int,
    ) -> _I2VConditioning:
        """Build the frame-0 conditioning (per-token denoise mask + clean latent) — tt analog of the
        reference ``create_initial_state`` + ``VideoConditionByLatentIndex.apply_to``. ``denoise_mask``
        is None only when the transformer needs no per-token video timestep and no image is staged
        (the plain-T2V forward-noise path)."""
        if image_cond_latent is None and not needs_video_ts:
            return _I2VConditioning(denoise_mask=None, clean_latent=None, n_cond=0)
        denoise_mask = torch.ones(B, video_N_real, 1)
        clean_latent = None
        n_cond = 0
        if image_cond_latent is not None:
            cond_tokens = image_cond_latent.float().permute(0, 2, 3, 4, 1).reshape(B, -1, self.in_channels)
            n_cond = cond_tokens.shape[1]
            assert n_cond <= video_N_real, f"image cond tokens {n_cond} exceed video tokens {video_N_real}"
            clean_latent = torch.zeros(B, video_N_real, self.in_channels)
            clean_latent[:, :n_cond, :] = cond_tokens
            denoise_mask[:, :n_cond, :] = 1.0 - image_cond_strength
            logger.info(f"I2V: pinning {n_cond} frame-0 tokens (strength={image_cond_strength})")
        return _I2VConditioning(denoise_mask=denoise_mask, clean_latent=clean_latent, n_cond=n_cond)

    # ------------------------------------------------------------------ trace buckets
    # Ladder / audio bucket the denoise traces are laid out on. Class defaults; ``warmup_buffers``
    # may override the ladder per deployment.
    bucket_ladder: tuple[int, ...] = LTX_BUCKET_LADDER
    audio_n_bucket: int = LTX_AUDIO_N_BUCKET
    # Filled by warmup: the routes this deployment serves, the canvases routing may accept (None =
    # any 64-aligned canvas; the ad-hoc single-shape warmup), the rungs holding preallocated I/O,
    # and the pixel shapes whose upsampler/VAE/audio decode were warmed (traced VAE decode).
    _served_routes: tuple[LTXBucketRoute, ...] = ()
    _served_canvases: tuple[str, ...] | None = None
    _warm_rungs: frozenset[int] = frozenset()
    _hot_shapes: frozenset[tuple[int, int, int, int]] = frozenset()
    _upsampler_shape: tuple[int, int, int] | None = None
    _upsampler_hot_shape: tuple[int, int, int] | None = None

    def _route(self, *, num_frames: int, height: int, width: int, fps: int) -> LTXBucketRoute:
        """Route one T2V/I2V request onto the ladder. I2V is not rejected here: this pipeline's
        transformer is built for one modulation class at construction, so every trace it captures is
        that class, and the frame-0 pin buffers are part of each rung's preallocated I/O."""
        return route_ltx_request(
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            sp_factor=self.parallel_config.sequence_parallel.factor,
            ladder=self.bucket_ladder,
            served_canvases=self._served_canvases,
            audio_n_bucket=self.audio_n_bucket,
        )

    def _resolve_served_routes(
        self,
        served_configs,
        *,
        num_frames: int,
        height: int,
        width: int,
        fps: int,
    ) -> tuple[LTXBucketRoute, ...]:
        """Turn the ``served_configs`` argument (or ``LTX_SERVED_CONFIGS``) into routes.

        Accepted forms: ``"all"`` (the full served grid: 720p/1080p x all fps x all durations),
        ``"hot"`` (only the warmup shape), or an iterable / comma-separated string of
        ``canvas:fps:duration`` entries. ``None`` reads ``LTX_SERVED_CONFIGS`` and otherwise defaults
        to ``"all"`` when traced -- the console pays for every rung once at startup so a resolution
        or duration change is a trace replay, never a capture -- and to ``"hot"`` untraced (nothing
        is captured, so there is nothing to pre-warm). The warmup shape itself is always served.
        """
        if served_configs is None:
            served_configs = os.environ.get("LTX_SERVED_CONFIGS") or ("all" if self._traced else "hot")
        if isinstance(served_configs, str):
            served_configs = served_configs.strip()
            served_configs = served_configs if served_configs in ("all", "hot") else served_configs.split(",")

        if served_configs == "hot":
            self._served_canvases = None  # ad hoc: any 64-aligned canvas routes, nothing else is warm
            grid: list[tuple[str, int, int]] = []
        else:
            self._served_canvases = LTX_SERVED_CANVASES
            if served_configs == "all":
                grid = list(ltx_served_configs())
            else:
                grid = []
                for entry in served_configs:
                    if isinstance(entry, str):
                        canvas, fps_s, dur_s = entry.strip().split(":")
                        entry = (canvas, int(fps_s), int(dur_s))
                    grid.append(tuple(entry))

        routes = [
            route_ltx_config(
                canvas,
                cfg_fps,
                duration,
                sp_factor=self.parallel_config.sequence_parallel.factor,
                ladder=self.bucket_ladder,
                served_canvases=self._served_canvases or LTX_SERVED_CANVASES,
                audio_n_bucket=self.audio_n_bucket,
            )
            for canvas, cfg_fps, duration in grid
        ]
        hot = self._route(num_frames=num_frames, height=height, width=width, fps=fps)
        # The hot shape first so its rungs get captured with its own statics bound.
        return (hot, *[r for r in routes if r != hot])

    @staticmethod
    def _representative_per_rung(routes: tuple[LTXBucketRoute, ...]) -> dict[int, tuple[LTXBucketRoute, str]]:
        """First (route, stage) that lands on each rung; a trace is stage-agnostic so any will do."""
        reps: dict[int, tuple[LTXBucketRoute, str]] = {}
        for route in routes:
            for stage in ("s1", "s2"):
                reps.setdefault(route.trace_key(stage), (route, stage))
        return reps

    @staticmethod
    def _stage_hw(route: LTXBucketRoute, stage: str) -> tuple[int, int]:
        height, width = LTX_CANVASES.get(route.canvas) or tuple(int(v) for v in route.canvas.split("x"))
        return (height // 2, width // 2) if stage == "s1" else (height, width)

    def warmup_buffers(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        fps: int = 24,
        num_inference_steps: int = 2,
        stages: tuple[str, ...] = ("s1", "s2"),
        served_configs=None,
        bucket_ladder: tuple[int, ...] | None = None,
        exact_hot_rungs: bool = True,
    ) -> None:
        """Compile both stages' programs and, when traced, capture one denoise trace per bucket rung.

        ``(num_frames, height, width, fps)`` is the hot shape: its upsampler/VAE/audio decode are
        warmed (and the VAE decode traced) in addition to its two denoise rungs. ``served_configs``
        selects the set of rungs captured (see ``_resolve_served_routes``; traced defaults to the
        whole served grid, ~1 min and ~110 MB of trace region per rung on the 4x8); every rung's
        persistent I/O is allocated before the first capture so no replay can overwrite another's
        inputs.
        ``exact_hot_rungs`` adds the hot shape's SP-padded lengths to the ladder as rungs of their
        own, so the hot config keeps its exact (unpadded-beyond-SP) matmul shapes and pays nothing
        for bucketing; other configs still land on the geometric ladder.
        ``stages=("s1",)`` skips the stage-2 side (upsample, VAE, audio) of the hot shape.
        """
        assert height % 64 == 0 and width % 64 == 0, f"H/W must be div by 64 (got {height}x{width})"
        assert num_frames > 0, f"num_frames must be > 0 (got {num_frames})"
        valid = {"s1", "s2"}
        assert set(stages).issubset(valid), f"stages must be subset of {valid} (got {stages})"
        if bucket_ladder is not None:
            self.bucket_ladder = tuple(bucket_ladder)
        sp_factor = self.parallel_config.sequence_parallel.factor
        if exact_hot_rungs:
            hot_exact = {self._sp_pad_len(n) for n in ltx_stage_video_n_real(num_frames, height, width).values()}
            self.bucket_ladder = tuple(sorted(set(self.bucket_ladder) | hot_exact))
        validate_bucket_ladder(self.bucket_ladder, ttnn.TILE_SIZE * sp_factor)

        t0 = time.time()
        routes = self._resolve_served_routes(served_configs, num_frames=num_frames, height=height, width=width, fps=fps)
        self._served_routes = routes
        self._hot_shapes = frozenset({(num_frames, height, width, fps)})
        hot = routes[0]
        # Representatives prefer s1, so a stages=("s1",) warmup drops the rungs only s2 would reach.
        reps = {rung: rep for rung, rep in self._representative_per_rung(routes).items() if rep[1] in stages}
        logger.info(
            f"warmup (distilled 2-stage): hot {num_frames}f@{height}x{width}@{fps}fps "
            f"(rungs s1={hot.trace_key('s1')}, s2={hot.trace_key('s2')}), {len(routes)} served config(s) "
            f"-> {len(reps)} rung(s) {sorted(reps)}, stages={stages}, {num_inference_steps} steps/stage"
        )

        v_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.video_dim)
        a_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.audio_dim)

        # Allocate every rung's persistent trace I/O before any capture so all held inputs sit
        # below every trace's activation region and no replay overwrites another's inputs.
        if self._traced:
            for rung in sorted(reps):
                route, stage = reps[rung]
                self._prealloc_trace_io(rung, route=route, stage=stage)
            self._warm_rungs = frozenset(reps)

        # Warm the encoder before any capture so its connector workspace isn't in a trace's
        # activation region (zeroed on replay). dynamic_load reloads per request → warms last.
        if self._traced and not self.dynamic_load:
            self.gemma_encoder_pair.ensure_loaded()
            self.encode_prompts(["warmup"], use_cache=False)

        # Real distilled sigmas so warmup hits the same branches (incl. sigma_next == 0 final step).
        stage_sigmas = {
            "s1": list(DISTILLED_SIGMA_VALUES)[:num_inference_steps] + [0.0],
            "s2": list(STAGE_2_DISTILLED_SIGMA_VALUES)[:num_inference_steps] + [0.0],
        }

        # Per rung: an eager pass at the rung's exact shapes to fill the program cache (inner_step
        # is prep_run=False, and the untraced path pads differently so it cannot compile for the
        # trace), then the capture. Untraced pipelines just compile. The hot shape's own stages run
        # first, in stage order, so eager warmup matches the old flow.
        ordered = sorted(reps.items(), key=lambda kv: (kv[1][0] is not hot, kv[1][1], kv[0]))
        for rung, (route, stage) in ordered:
            s_h, s_w = self._stage_hw(route, stage)
            logger.info(
                f"warmup rung {rung} via {route.canvas} {route.num_frames}f@{route.fps}fps stage {stage} "
                f"({s_h}x{s_w}, real N={route.video_n_real(stage)}), σ={stage_sigmas[stage]}"
            )
            passes = (True, False) if self._traced else (False,)
            for compile_only in passes:
                self._denoise_no_guidance(
                    v_p,
                    a_p,
                    num_frames=route.num_frames,
                    height=s_h,
                    width=s_w,
                    fps=route.fps,
                    sigma_values=stage_sigmas[stage],
                    seed=0,
                    traced=self._traced,
                    route=route,
                    stage=stage,
                    compile_only=compile_only,
                )

        if "s2" in stages:
            # Upsample runs between stage 1 and stage 2; compile its kernels at the hot shape.
            logger.info(f"warmup upsample → {height}x{width}")
            self._ensure_upsampler_shape(num_frames, height, width)
            self._upsampler_hot_shape = self._upsampler_shape
            self._warmup_upsample(num_frames, height, width)

            # Compile VAE decode at the hot full-res shape (only s2 feeds decode in generate).
            self._warmup_decode(num_frames, height, width)

            # Programs are now compiled; the first real decode at a hot shape captures its trace.
            if self._traced and self.vae_decoder is not None:
                self.vae_decoder._vae_traced = True

            # Warm the on-device audio decode eagerly at the bucketed latent shape: compiles kernels,
            # initializes lazy device state, and frees back to a deterministic allocator free-list,
            # so the first real (traced) decode captures cleanly on warm state. The audio latent is
            # always decoded at the bucket length and the waveform cropped to the clip duration.
            logger.info("warmup audio decode (on-device, eager)")
            self._warmup_audio_decode(torch.zeros(1, self.audio_n_bucket, self.in_channels), num_frames, fps=fps)

            self._prepare_transformer(0)

        # Warm the image encoder only for an I2V build. Pure T2V constructs the encoder shell from
        # the checkpoint too, but its transformer is not built for image conditioning and cannot
        # become I2V later; compiling both encoder resolutions there only adds startup latency.
        if self._needs_image_encoder_warmup():
            logger.info(f"warmup image encoder: {height // 2}x{width // 2} + {height}x{width}")
            self._warmup_encode(height // 2, width // 2)
            self._warmup_encode(height, width)

        # use_cache=False forces a real encode so the Gemma/connector kernels compile. traced-static
        # already warmed before capture (above); dynamic_load / untraced warm last.
        if self.dynamic_load or not self._traced:
            self.gemma_encoder_pair.ensure_loaded()
            self.encode_prompts(["warmup"], use_cache=False)

        logger.info(f"warmup (distilled 2-stage) done in {time.time() - t0:.1f}s")

    def release_traces(self) -> None:
        super().release_traces()
        self._warm_rungs = frozenset()
        self._hot_shapes = frozenset()

    def _needs_image_encoder_warmup(self) -> bool:
        return self.vae_encoder is not None and self._image_conditioning

    def _ensure_upsampler_shape(self, num_frames: int, height: int, width: int) -> bool:
        """Rebuild the spatial upsampler when the request's stage-1 latent shape differs from the one
        it was built for (its DRAM GroupNorm grid is pinned at construction). The hot shape never
        rebuilds; other served configs pay an eager weight reload here. Returns whether the
        upsampler now sits at the hot (warmed) shape."""
        if self.upsampler is None:
            return True
        latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
        hf = self.vae_parallel_config.height_parallel.factor
        wf = self.vae_parallel_config.width_parallel.factor
        input_hw = (ceil_to(height // (SPATIAL_COMPRESSION * 2), hf), ceil_to(width // (SPATIAL_COMPRESSION * 2), wf))
        shape = (latent_frames, *input_hw)
        if self._upsampler_shape is None:
            # Built by the base pipeline at the init shape.
            init_lf = (self._init_num_frames - 1) // TEMPORAL_COMPRESSION + 1
            self._upsampler_shape = (
                init_lf,
                ceil_to(self._init_height // (SPATIAL_COMPRESSION * 2), hf),
                ceil_to(self._init_width // (SPATIAL_COMPRESSION * 2), wf),
            )
        if shape == self._upsampler_shape:
            return shape == (self._upsampler_hot_shape or shape)
        logger.warning(
            f"upsampler rebuilt for latent shape {shape} (was {self._upsampler_shape}); "
            "not a hot shape, so this request pays the eager upsampler reload"
        )
        old = self.upsampler
        self.upsampler = LTXLatentUpsampler.from_checkpoint(
            self._upsampler_path,
            input_hw=input_hw,
            latent_frames=latent_frames,
            mesh_device=self.mesh_device,
            parallel_config=self.vae_parallel_config,
            ccl_manager=self.vae_ccl_manager,
            dit_parallel_config=self.parallel_config,
        )
        self._upsampler_shape = shape
        if old is not None:
            old.deallocate_weights()
        self._register_coresident_exclusions()
        return shape == self._upsampler_hot_shape

    def _set_video_logical_n(self, state: LTXTransformerState, video_N_real: int, traced: bool) -> None:
        """Write the live video length into the rung's one-element uint32 tensor (in place when traced)."""
        host = ttnn.from_torch(
            torch.tensor([video_N_real], dtype=torch.int64).reshape(1, 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        if traced and state.tt_video_logical_n is not None:
            ttnn.copy_host_to_device_tensor(host, state.tt_video_logical_n)
        else:
            state._tt_video_logical_n.update(host.to(self.mesh_device), False)

    def _prepare_stage_statics(
        self,
        state: LTXTransformerState,
        *,
        latent_frames,
        latent_h,
        latent_w,
        video_N,
        video_N_real,
        audio_N,
        audio_N_real,
        fps,
        sp_axis,
        traced,
    ):
        """Bind a state's static per-config inputs (rope/cross-PE/masks/trans_mat).

        Skipped when the state already holds this exact config; otherwise rebuilt on host and
        copied into the existing buffers when traced (their addresses are baked into the rung's
        trace), or rebound when untraced.
        """
        key = (latent_frames, latent_h, latent_w, video_N, video_N_real, audio_N, audio_N_real, fps)
        if state.bound_config == key:
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
            fps=fps,
            video_N=video_N,
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
            fps=fps,
            video_N=video_N,
        )
        tt_attn_mask, tt_pad_mask_sp, tt_pad_mask_full = build_audio_masks(
            audio_N, audio_N_real, mesh_device=self.mesh_device, sp_axis=sp_axis
        )
        state._tt_video_rope_cos.update(v_cos, traced)
        state._tt_video_rope_sin.update(v_sin, traced)
        state._tt_audio_rope_cos.update(a_cos, traced)
        state._tt_audio_rope_sin.update(a_sin, traced)
        state._tt_trans_mat.update(self._prepare_trans_mat(), traced)
        state._tt_video_cross_pe_cos.update(v_xpe_cos, traced)
        state._tt_video_cross_pe_sin.update(v_xpe_sin, traced)
        state._tt_audio_cross_pe_cos.update(a_xpe_cos, traced)
        state._tt_audio_cross_pe_sin.update(a_xpe_sin, traced)
        state._tt_audio_cross_pe_cos_full.update(a_xpe_cos_full, traced)
        state._tt_audio_cross_pe_sin_full.update(a_xpe_sin_full, traced)
        state._tt_audio_attn_mask.update(tt_attn_mask, traced)
        state._tt_audio_padding_mask.update(tt_pad_mask_sp, traced)
        state._tt_audio_padding_mask_full.update(tt_pad_mask_full, traced)
        state._tt_video_padding_mask.update(
            build_video_pad_mask(video_N, video_N_real, mesh_device=self.mesh_device, sp_axis=sp_axis), traced
        )
        v_mask = torch.ones(1, 1, video_N, self.in_channels)
        v_mask[:, :, video_N_real:, :] = 0.0
        a_mask = torch.ones(1, 1, audio_N, self.in_channels)
        a_mask[:, :, audio_N_real:, :] = 0.0
        state._tt_video_pad_mask.update(v_mask, traced, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device)
        state._tt_audio_pad_mask.update(a_mask, traced, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device)
        state.bound_config = key

    def _prealloc_trace_io(self, rung: int, *, route: LTXBucketRoute, stage: str) -> None:
        """Allocate a rung's persistent trace inputs (statics, latent buffers, masks, live length) up front.

        A ttnn trace bakes absolute tensor addresses; capture-time activations are freed and reused
        by the next capture, so a held input in another trace's activation region is overwritten on
        replay. Allocating every held input first keeps them below every trace's activations. The
        statics are filled for ``(route, stage)``; later requests on the rung refresh them in place.
        """
        s_h, s_w = self._stage_hw(route, stage)
        latent_frames, latent_h, latent_w = latent_grid(route.num_frames, s_h, s_w)
        video_N_real = latent_frames * latent_h * latent_w
        assert video_N_real == route.video_n_real(stage) and rung == route.trace_key(stage)
        video_N = rung
        audio_N = route.audio_n
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        state = self._trace_state.setdefault(rung, LTXTransformerState())
        self._prepare_stage_statics(
            state,
            latent_frames=latent_frames,
            latent_h=latent_h,
            latent_w=latent_w,
            video_N=video_N,
            video_N_real=video_N_real,
            audio_N=audio_N,
            audio_N_real=route.audio_n_real,
            fps=route.fps,
            sp_axis=sp_axis,
            traced=False,
        )
        # Reserve the latent buffers before capture so the trace bakes their addresses.
        if state.tt_video_lat is None:
            self._set_video_logical_n(state, video_N_real, traced=False)
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
            # them. Allocated for every rung even when unused by t2v — small and harmless. The mask
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
        image_cond_latent: torch.Tensor | None = None,
        image_cond_strength: float = 1.0,
        fps: int = 24,
        traced: bool = False,
        route: LTXBucketRoute | None = None,
        stage: str | None = None,
        compile_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One distilled denoise stage.

        Traced: ``route``/``stage`` pick the bucket rung; the video sequence is padded to the rung,
        the audio to the audio bucket, the rung's trace is replayed (captured on first use) and ring
        SDPA reads the real video length from the rung's ``video_logical_n`` tensor. Untraced: the
        sequence is padded only to the SP boundary and the real length is passed as a scalar.

        ``compile_only`` (traced only): run the rung-shaped step eagerly through the tracer instead
        of capturing, so every program the trace will contain is in the program cache first --
        ``inner_step`` is decorated with ``prep_run=False``, and the untraced path pads to different
        shapes so it cannot stand in for this pass.
        """
        assert not compile_only or traced, "compile_only is a traced-mode warmup pass"
        B = 1
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        video_N_real = latent_frames * latent_h * latent_w
        als = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=fps)
        )
        audio_N_real = als.frames
        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        if traced:
            assert route is not None and stage is not None, "traced denoise needs a bucket route and stage"
            assert route.video_n_real(stage) == video_N_real, (
                f"route {route.canvas} {route.num_frames}f stage {stage} real N {route.video_n_real(stage)} "
                f"!= denoise shape real N {video_N_real}"
            )
            assert route.audio_n_real == audio_N_real
            video_N, audio_N, trace_key = route.video_n(stage), route.audio_n, route.trace_key(stage)
            # Never allocate a new rung's I/O under live traces: it could land in another trace's
            # activation region and be overwritten on replay. Warm the rung in warmup_buffers instead.
            if trace_key not in self._trace_state:
                raise RuntimeError(
                    f"bucket rung {trace_key} (stage {stage}, real N={video_N_real}) has no preallocated trace "
                    f"I/O; warmed rungs: {sorted(self._warm_rungs)}. Add the config to served_configs "
                    "(or LTX_SERVED_CONFIGS) at warmup, or run the pipeline untraced."
                )
            state = self._trace_state[trace_key]
        else:
            # SP padding only: round the seq dims up to TILE_SIZE * sp_factor for ring SDPA. A
            # transient state so statics rebuild — resolution can differ across generates.
            video_N, audio_N, trace_key = self._sp_pad_len(video_N_real), self._sp_pad_len(audio_N_real), None
            state = LTXTransformerState()

        image_cond = image_cond_latent is not None
        needs_video_ts = getattr(self.transformer, "image_conditioning", False)
        i2v = self._build_i2v_conditioning(image_cond_latent, image_cond_strength, needs_video_ts, B, video_N_real)

        logger.info(
            f"  shapes: vN={video_N}(real={video_N_real}), aN={audio_N}(real={audio_N_real}) "
            f"[sp={sp_factor}, fps={fps}, trace={'rung ' + str(trace_key) if traced else 'eager'}]"
        )

        self._prepare_stage_statics(
            state,
            latent_frames=latent_frames,
            latent_h=latent_h,
            latent_w=latent_w,
            video_N=video_N,
            video_N_real=video_N_real,
            audio_N=audio_N,
            audio_N_real=audio_N_real,
            fps=fps,
            sp_axis=sp_axis,
            traced=traced,
        )
        if traced:
            self._set_video_logical_n(state, video_N_real, traced=True)

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
            base_v[:, : i2v.n_cond, :] = i2v.clean_latent[:, : i2v.n_cond, :]
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
            # Consume the same RNG draws a video_N-sized randn would, so the audio noise stream
            # stays independent of video sequence length.
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
                pinned_scale = (1.0 - image_cond_strength) if (image_cond and i2v.n_cond > 0) else 1.0
                ts_pair = torch.tensor([pinned_scale * sigma, sigma], dtype=torch.float32)
                state._tt_video_ts_pair.update(ts_pair.reshape(1, 1, 2, 1) * 1000.0, traced, device=self.mesh_device)
                pin_mask_host = torch.zeros(1, 1, video_N, 1)
                if i2v.n_cond > 0:
                    pin_mask_host[:, :, : i2v.n_cond, :] = 1.0
                state._tt_video_pin_mask.update(
                    pin_mask_host,
                    traced,
                    mesh_axes=[None, None, sp_axis, None],
                    device=self.mesh_device,
                )
                video_ts_pair_tt = state.tt_video_ts_pair
                video_pin_mask_tt = state.tt_video_pin_mask
            # Ring SDPA masks padded K positions to the logical count: traced, the padded rung is the
            # (trace-constant) scalar and the live length rides in video_logical_n_tensor; untraced,
            # the real count is passed directly.
            v_out, a_out = self.transformer.inner_step(
                video_1BNI=state.tt_video_lat,
                timestep=state.tt_timestep,
                video_ts_pair=video_ts_pair_tt,
                video_pin_mask=video_pin_mask_tt,
                audio_1BNI=state.tt_audio_lat,
                video_prompt_1BLP=prompt_v,
                video_rope_cos=state.tt_video_rope_cos,
                video_rope_sin=state.tt_video_rope_sin,
                video_N=video_N if traced else video_N_real,
                video_logical_n_tensor=state.tt_video_logical_n if traced else None,
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
                traced=traced and not compile_only,
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

        # Traced: route the request onto the bucket ladder up front so an unserved config fails
        # before the encoder runs, not after stage 1. Every rung it needs must have been warmed.
        route = None
        if self._traced:
            route = self._route(num_frames=num_frames, height=height, width=width, fps=fps)
            missing = sorted(set(route.stage_rung.values()) - self._warm_rungs)
            if missing:
                raise ValueError(
                    f"request {num_frames}f@{height}x{width}@{fps}fps routes to rung(s) {missing} that were not "
                    f"warmed (warm rungs: {sorted(self._warm_rungs)}); add it to served_configs / "
                    "LTX_SERVED_CONFIGS at warmup"
                )
            logger.info(
                f"bucket route: {route.canvas} {route.num_frames}f@{fps}fps -> "
                f"s1 N {route.video_n_real('s1')}->{route.video_n('s1')}, "
                f"s2 N {route.video_n_real('s2')}->{route.video_n('s2')}, audio {route.audio_n_real}->{route.audio_n}"
            )

        # (label, seconds) rows counted toward the total; prepares and export excluded.
        timings: list[tuple[str, float]] = []

        t0 = time.time()
        # A served request encodes a prompt nothing has seen, so a cache hit would drop the encoder
        # out of the reported total and out of the traced path it belongs in. dynamic_load keeps the
        # cache: there the encoder is coresident-excluded from the DiT, so re-encoding every request
        # would reload it and evict the captured model state.
        cached = self.dynamic_load and os.path.exists(self._device_embed_cache_path([prompt]))
        if not cached:
            self.gemma_encoder_pair.ensure_loaded()
        enc = self.encode_prompts([prompt], use_cache=self.dynamic_load)
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
        s1_video, s1_audio = self._denoise_no_guidance(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=s1_height,
            width=s1_width,
            sigma_values=DISTILLED_SIGMA_VALUES,
            seed=seed,
            image_cond_latent=s1_cond_latent,
            image_cond_strength=cond_strength,
            fps=fps,
            traced=self._traced,
            route=route,
            stage="s1",
        )
        t_stage1 = time.time() - t0
        timings.append(("Stage 1 denoise", t_stage1))
        logger.info(f"Stage 1 denoise: {t_stage1:.1f}s")

        latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
        s1_h, s1_w = s1_height // SPATIAL_COMPRESSION, s1_width // SPATIAL_COMPRESSION
        s1_spatial = s1_video.reshape(1, latent_frames, s1_h, s1_w, 128).permute(0, 4, 1, 2, 3)
        t0 = time.time()
        upsampler_hot = self._ensure_upsampler_shape(num_frames, height, width)
        self._prepare_upsampler()
        upsampled = upsample_latent(self.upsampler, s1_spatial, *self._vae_per_channel_stats())
        if self._traced and not upsampler_hot and self.upsampler is not None:
            # Weights loaded under live traces may sit in a trace's activation region; they were
            # consumed before any replay, so drop them now and reload fresh next request.
            self.upsampler.deallocate_weights()
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
            image_cond_latent=full_cond_latent,
            image_cond_strength=cond_strength,
            fps=fps,
            traced=self._traced,
            route=route,
            stage="s2",
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
        # VAE decode is traced per exact latent shape, so only the warmed (hot) shapes replay a trace;
        # any other served config decodes eagerly rather than capturing a trace per request.
        vae_hot = any(shape[:3] == (num_frames, height, width) for shape in self._hot_shapes)
        saved_vae_traced = getattr(self.vae_decoder, "_vae_traced", None)
        if self.vae_decoder is not None and self._traced:
            self.vae_decoder._vae_traced = vae_hot
        try:
            video_pixels = self.decode_latents(s2_video, latent_frames, latent_h, latent_w, output_type=decode_type)
        finally:
            if self.vae_decoder is not None and saved_vae_traced is not None:
                self.vae_decoder._vae_traced = saved_vae_traced
        t_vae_decode = time.time() - t0
        timings.append(("VAE decode", t_vae_decode))
        logger.info(f"VAE decode (forward): {t_vae_decode:.1f}s — {tuple(video_pixels.shape)}")

        t0 = time.time()
        # Decode the audio latent at the bucket length (one traced decode shape for every config;
        # the zero tail is beyond the clip) and let decode_audio crop the waveform to the duration.
        audio_for_decode = s2_audio
        if route is not None and s2_audio.shape[1] < route.audio_n:
            audio_for_decode = torch.zeros(s2_audio.shape[0], route.audio_n, s2_audio.shape[2], dtype=s2_audio.dtype)
            audio_for_decode[:, : s2_audio.shape[1], :] = s2_audio
        audio_obj = self.decode_audio(audio_for_decode, num_frames, fps=fps)
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
        # Every trace this pipeline takes is captured by now, so the encoder can start tracing: its
        # capture is last and nothing left will reclaim its activation region.
        if self._traced and not self.dynamic_load:
            self.gemma_encoder_pair.open_trace_gate()
        return output_path

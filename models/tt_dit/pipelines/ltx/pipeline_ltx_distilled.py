# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 distilled two-stage audio-video pipeline."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

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
from ...utils import timing_tree, walltime
from ...utils.ltx_euler import EulerTail
from ...utils.patchifiers import AudioLatentShape, VideoPixelShape
from ...utils.tensor import bf16_tensor
from ...utils.tracing import StateTensor, Tracer, traced_function
from ...utils.video import YuvVideoExport, export_video_audio, export_video_audio_yuv
from .pipeline_ltx import SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION, LTXPipeline, LTXTransformerState, latent_grid

# Distilled sigma schedules for the two stages. The defaults are the shipped 8-step (stage 1)
# and 3-step (stage 2) schedules. LTX_S1_SIGMAS / LTX_S2_SIGMAS override with a comma-separated
# list to A/B fewer-step schedules (L2 step cut). Unset = byte-identical to the shipped baseline.
_DEFAULT_S1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
_DEFAULT_S2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]

# Serial number for LTX_DUMP_LATENTS files; see the dump site in generate().
_LATENT_DUMP_N = 0


def _env_on(name: str, default: str = "0") -> bool:
    """Read at call time, so a serving process (or an A/B harness) can flip it between gens."""
    return os.environ.get(name, default).strip() in ("1", "true", "True")


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
# The trajectory splits in two: the tail nodes carry the actual denoise and are mandatory (dropping
# one is what puts the neighbors off-distribution), while the head nodes cluster near σ=1 and only
# refine how much budget the coarse stage spends reconciling the pin wall. So a tier buys its extra
# steps in head nodes, taking them from the σ=1 end of the high schedule inward. Anchoring on the
# tail is what keeps every tier ON the distilled trajectory; without it a tier-sized subset is just
# the off-trajectory schedule that scrambles.
_KEYFRAME_S1_TAIL = [0.909375, 0.725, 0.421875, 0.0]
_KEYFRAME_S1_HEAD = [0.99375, 0.9875, 0.98125, 0.975]
_KEYFRAME_S1_ALIGNED = [1.0] + _KEYFRAME_S1_TAIL
KEYFRAME_S1_STRENGTH = float(os.environ.get("LTX_KEYFRAME_S1_STRENGTH", "0.5"))
# A coarser S1 reconciles the pin-wall across fewer spatial tokens, so it needs a softer interior
# pin: 0.5 is clean at 1080p (s1 544) but smears the post-pin neighbors at 720p (s1 352), while 0.0
# fails to establish the keyframe at all. s2 re-locks identity at full strength either way.
KEYFRAME_S1_STRENGTH_COARSE = float(os.environ.get("LTX_KEYFRAME_S1_STRENGTH_COARSE", "0.25"))


def _keyframe_s1_strength(s1_height: int) -> float:
    """Interior-pin S1 strength for a non-frame-0 keyframe. The coarse 720p S1 (352) softens to 0.25;
    1080p (544) keeps 0.5. Threshold sits between the two supported S1 heights."""
    return KEYFRAME_S1_STRENGTH_COARSE if s1_height < 480 else KEYFRAME_S1_STRENGTH


def _keyframe_s1_sigmas(latent_frames: int) -> list[float]:
    """Trajectory-aligned S1 nodes for a non-frame-0 keyframe, scaled to the active tier's budget.

    The aligned tail is 4 steps, so a tier with fewer than that (fast, 3) pays a step to stay on the
    trajectory, and one with more (medium, 6) spends the surplus on head nodes rather than collapsing
    onto fast's schedule — which is what a fixed node list did: a keyframe gen rendered bit-identical
    on medium and fast, making the tier a no-op for every keyframe user. A longer clip's interior pin
    has more neighbors to reconcile, so past ~6s it takes a head node even when the tier wouldn't.
    LTX_KEYFRAME_S1_SIGMAS overrides the lot."""
    if os.environ.get("LTX_KEYFRAME_S1_SIGMAS", "").strip():
        return _sigma_override("LTX_KEYFRAME_S1_SIGMAS", _KEYFRAME_S1_ALIGNED)
    tier_steps = len(DISTILLED_SIGMA_VALUES) - 1
    extra = min(max(tier_steps - len(_KEYFRAME_S1_TAIL), 1 if latent_frames > 20 else 0), len(_KEYFRAME_S1_HEAD))
    return [1.0] + (_KEYFRAME_S1_HEAD[-extra:] if extra else []) + _KEYFRAME_S1_TAIL


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


def build_conditioning_tensors(conds, latent_frames, latent_h, latent_w, in_channels, append_interior=False):
    """Place per-frame conditioning latents into the flat token sequence.

    Tokens are frame-major: latent frame ``k`` owns rows ``[k*hw : (k+1)*hw)`` (``hw = latent_h *
    latent_w``). ``conds`` is a list of ``(latent_frame_index, cond_latent, strength)`` where
    ``cond_latent`` is one encoded frame ``(1, C, 1, latent_h, latent_w)``.

    With ``append_interior``, a non-frame-0 keyframe is NOT pinned in its own grid rows (that static
    in-place pin freezes the video and rings a decode halo); its grid frame stays free and the encoded
    keyframe is pinned into an hw ANCHOR block appended after the grid, so the free frame converges to
    it via attention while staying on the moving distribution. Frame-0 always stays in-place. Returns
    ``(clean_latent [1, N, C], denoise_mask [1, N, 1], pin_rows [N] bool, anchor_frame_indices,
    video_N_real_ext)`` where ``N = video_N_real_ext = latent_frames*hw + n_anchor*hw``."""
    hw = latent_h * latent_w
    video_N_grid = latent_frames * hw
    inplace = [c for c in conds if c[0] == 0 or not append_interior]
    append = [c for c in conds if c[0] != 0 and append_interior]
    video_N_real_ext = video_N_grid + len(append) * hw
    clean_latent = torch.zeros(1, video_N_real_ext, in_channels)
    denoise_mask = torch.ones(1, video_N_real_ext, 1)
    pin_rows = torch.zeros(video_N_real_ext, dtype=torch.bool)

    def _pin(off, cond_latent, strength):
        tokens = cond_latent.float().permute(0, 2, 3, 4, 1).reshape(1, -1, in_channels)
        assert tokens.shape[1] == hw, f"cond latent has {tokens.shape[1]} tokens, expected one latent frame = {hw}"
        clean_latent[:, off : off + hw, :] = tokens
        denoise_mask[:, off : off + hw, :] = 1.0 - strength
        pin_rows[off : off + hw] = True

    for lat_idx, cond_latent, strength in inplace:
        assert 0 <= lat_idx < latent_frames, f"latent frame {lat_idx} out of range [0,{latent_frames})"
        _pin(lat_idx * hw, cond_latent, strength)
    anchor_frame_indices = []
    for a, (lat_idx, cond_latent, strength) in enumerate(append):
        assert 0 < lat_idx < latent_frames, f"interior keyframe {lat_idx} out of range (0,{latent_frames})"
        _pin(video_N_grid + a * hw, cond_latent, strength)  # anchor block; grid frame lat_idx left free
        anchor_frame_indices.append(lat_idx)
    return clean_latent, denoise_mask, pin_rows, anchor_frame_indices, video_N_real_ext


# Warn when a post-denoise latent's whiteness exceeds this. 1.0 is indistinguishable from
# the Gaussian the sampler started from; 21 healthy 1080p generations measured 0.25-0.71
# (mean 0.49), so this sits above the observed good range while still catching a partially
# corrupted latent. These are log warnings, not request rejections -- a false positive is
# cheap, a missed corruption costs us the next occurrence.
WHITENESS_WARN_THRESHOLD = 0.85

# LTX-2.5+ distilled stage 1 uses ancestral Euler (eta=1, s_noise=1). Stage 2 stays deterministic —
# its 3-step schedule is too short to clear freshly injected noise. Seed offset keeps the loop's
# first randn distinct from the initial GaussianNoiser draw (same shape/dtype/seed would collide).
ANCESTRAL_ETA = 1.0
ANCESTRAL_S_NOISE = 1.0
ANCESTRAL_NOISE_SEED_OFFSET = 10000


def _euler_ancestral_step(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float,
    sigma_next: float,
    noise: torch.Tensor | None,
    *,
    eta: float = ANCESTRAL_ETA,
    s_noise: float = ANCESTRAL_S_NOISE,
) -> torch.Tensor:
    """Rectified-flow ancestral Euler step (ltx_core ``EulerAncestralDiffusionStep``).

    ``velocity`` is the flow-matching velocity; ``denoised = sample - sigma * velocity``.
    Host float32 in/out; caller owns dtype cast. Kept for parity checks; the denoise loop
    uses ``_euler_ancestral_step_tt`` on device.
    """
    x = sample.float()
    vel = velocity.float()
    denoised = x - sigma * vel
    if sigma_next == 0.0:
        return denoised
    if eta > 0.0 and noise is None:
        raise ValueError("ancestral Euler requires noise when eta > 0")
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    sigma_down_ratio = sigma_down / sigma
    x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * denoised
    if eta > 0.0:
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise_coeff = sigma_next**2 - sigma_down**2 * (alpha_next / alpha_down) ** 2
        renoise_coeff = max(0.0, renoise_coeff) ** 0.5
        x_next = (alpha_next / alpha_down) * x_next + noise.float() * s_noise * renoise_coeff
    return x_next


def _ancestral_step_coeffs(
    sigma: float,
    sigma_next: float,
    *,
    eta: float = ANCESTRAL_ETA,
    s_noise: float = ANCESTRAL_S_NOISE,
) -> tuple[float, float, float] | None:
    """Return ``(sigma_down_ratio, alpha_scale, noise_scale)``, or ``None`` when σ_next==0 (x0 only)."""
    if sigma_next == 0.0:
        return None
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    sigma_down_ratio = sigma_down / sigma
    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    renoise_coeff = max(0.0, sigma_next**2 - sigma_down**2 * (alpha_next / alpha_down) ** 2) ** 0.5
    return sigma_down_ratio, alpha_next / alpha_down, s_noise * renoise_coeff


def _euler_ancestral_step_tt(
    tt_x: ttnn.Tensor,
    tt_vel: ttnn.Tensor,
    pad_mask: ttnn.Tensor,
    sigma: float,
    sigma_next: float,
    tt_noise: ttnn.Tensor | None,
) -> None:
    """In-place ancestral Euler into ``tt_x`` (SP-sharded). Matches ``_euler_ancestral_step``."""
    vel = ttnn.typecast(tt_vel, ttnn.bfloat16)
    ttnn.multiply_(vel, pad_mask)
    coeffs = _ancestral_step_coeffs(sigma, sigma_next)
    if coeffs is None:
        # x0 estimate: x - sigma * vel
        x_next = ttnn.subtract(tt_x, ttnn.multiply(vel, sigma))
    else:
        sigma_down_ratio, alpha_scale, noise_scale = coeffs
        denoised = ttnn.subtract(tt_x, ttnn.multiply(vel, sigma))
        x_next = ttnn.add(
            ttnn.multiply(tt_x, sigma_down_ratio),
            ttnn.multiply(denoised, 1.0 - sigma_down_ratio),
        )
        ttnn.deallocate(denoised)
        scaled = ttnn.multiply(x_next, alpha_scale)
        ttnn.deallocate(x_next)
        if tt_noise is not None and noise_scale != 0.0:
            x_next = ttnn.add(scaled, ttnn.multiply(tt_noise, noise_scale))
            ttnn.deallocate(scaled)
        else:
            x_next = scaled
    ttnn.multiply_(x_next, pad_mask)
    ttnn.copy(x_next, tt_x)
    ttnn.deallocate(x_next)


@dataclass
class _I2VConditioning:
    """Image conditioning for one denoise stage — the tt analog of the reference LatentState fields
    written by ``VideoConditionByLatentIndex.apply_to``, generalized from frame-0 to arbitrary
    latent frames (first / last / keyframe)."""

    denoise_mask: torch.Tensor | None  # (B, N, 1): 1−strength at cond tokens else 1; None = plain T2V
    clean_latent: torch.Tensor | None  # (B, N, C): cond tokens at their latent frames else 0
    pin_rows: torch.Tensor | None  # (N,) bool: True at pinned tokens (any latent frame); None = no image
    # Append-token: interior keyframes ride hw anchor tokens appended after the grid (N = video_N_real_ext);
    # their main-grid frame stays free. anchor_frame_indices lists each anchor's target latent frame.
    anchor_frame_indices: list[int] = field(default_factory=list)
    video_N_real_ext: int = 0  # grid + n_anchor*hw; == grid length when no append


class LTXDistilledPipeline(LTXPipeline):
    """Distilled 2-stage AV pipeline: half-res denoise → upsample → full-res refine."""

    HAS_UPSAMPLER = True
    SUPPORTS_IMAGE_CONDITIONING = True
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
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """GaussianNoiser (ltx_core): ``noise·(mask·σ) + base·(1−mask·σ)``.

        ``denoise_mask`` None is the plain forward step (mask ≡ 1); a per-token mask holding
        ``1−strength`` at the conditioning tokens pins them toward ``base``. ``noise`` is the bf16
        seeded draw (``_seeded_noise``) — the device latent dtype — matching the reference, which
        draws at ``latent.dtype``."""
        noise = noise.to(base.dtype)
        scaled_mask = sigma if denoise_mask is None else denoise_mask * sigma
        return noise * scaled_mask + base * (1.0 - scaled_mask)

    @staticmethod
    def _draw_seeded_noise(seed: int, shapes: tuple) -> list[torch.Tensor]:
        """bf16 ``randn`` per shape, in order, from one generator seeded with ``seed``: the same values
        as ``torch.manual_seed(seed)`` followed by ``torch.randn(shape, dtype=torch.bfloat16)`` per shape."""
        gen = torch.Generator().manual_seed(seed)
        return [torch.randn(shape, dtype=torch.bfloat16, generator=gen) for shape in shapes]

    def _prefetch_noise(self, seed: int, draws: list[tuple]) -> None:
        """Start this gen's seeded draws on a host thread. bf16 ``randn`` is ~4x slower than fp32 on the
        host (~90 ms for a 1080p stage-2 latent) and sat serially in each stage's denoise init; drawn
        here it overlaps the prompt encode and stage 1 on the device."""
        pool = getattr(self, "_noise_pool", None)
        if pool is None:
            pool = self._noise_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltx-noise")
        self._noise_prefetch = {
            (seed, shapes): pool.submit(self._draw_seeded_noise, seed, shapes) for shapes in map(tuple, draws)
        }

    def _seeded_noise(self, seed: int, *shapes: tuple) -> list[torch.Tensor]:
        """``_draw_seeded_noise``, served from ``_prefetch_noise`` when it started the same draws."""
        shapes = tuple(tuple(s) for s in shapes)
        pending = getattr(self, "_noise_prefetch", {}).pop((seed, shapes), None)
        if pending is not None:
            return pending.result()
        return self._draw_seeded_noise(seed, shapes)

    def _build_i2v_conditioning(
        self,
        image_conds: list | None,
        needs_video_ts: bool,
        B: int,
        video_N_real: int,
        latent_frames: int,
        latent_h: int,
        latent_w: int,
        append_interior: bool = False,
    ) -> _I2VConditioning:
        """Build the per-token conditioning (denoise mask + clean latent + pinned-row mask) — tt
        analog of the reference ``create_initial_state`` + ``VideoConditionByLatentIndex.apply_to``,
        one entry per conditioned latent frame. ``denoise_mask`` is None only when the transformer
        needs no per-token video timestep and no image is staged (the plain-T2V forward-noise path)."""
        if not image_conds and not needs_video_ts:
            return _I2VConditioning(denoise_mask=None, clean_latent=None, pin_rows=None, video_N_real_ext=video_N_real)
        if image_conds:
            clean_latent, denoise_mask, pin_rows, anchor_frames, video_N_real_ext = build_conditioning_tensors(
                image_conds, latent_frames, latent_h, latent_w, self.in_channels, append_interior=append_interior
            )
            logger.info(
                f"I2V: pinning {int(pin_rows.sum())} tokens across {len(image_conds)} frame(s) at "
                f"latent indices {sorted(int(i) for i, _, _ in image_conds)} "
                f"(strength≈{image_conds[0][2]}; appended anchors at {anchor_frames})"
            )
            return _I2VConditioning(
                denoise_mask=denoise_mask,
                clean_latent=clean_latent,
                pin_rows=pin_rows,
                anchor_frame_indices=anchor_frames,
                video_N_real_ext=video_N_real_ext,
            )
        return _I2VConditioning(
            denoise_mask=torch.ones(B, video_N_real, 1), clean_latent=None, pin_rows=None, video_N_real_ext=video_N_real
        )

    def warmup_buffers(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int = 2,
        stages: tuple[str, ...] = ("s1", "s2"),
        ref_num_frames: int | None = None,
        capture_all: bool = False,
        in_capture_pass: bool = False,
        capture_traced: bool = False,
    ) -> None:
        """Compile both stages' programs. Both stages use variant 0 (distilled doesn't
        swap weights — only the sequence length differs); ``stages=("s1",)`` skips s2.

        ``ref_num_frames`` (IC-LoRA) additionally warms the wider s1_ref/s2_ref trace family and the
        multi-frame reference encoders; pass e.g. ``stages=("s1_ref","s2_ref")`` to warm ONLY the
        reference family (a dedicated ``-ref`` worker — mirrors the deployed ``-kf`` keyframe worker —
        which keeps the four-trace DRAM pressure down). The s*_ref traces are captured HERE in warmup
        (``traced=capture_traced``), exactly like s1/s2, never deferred to gen#0.

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
        valid = {"s1", "s2", "s1_ref", "s2_ref"}
        assert set(stages).issubset(valid), f"stages must be subset of {valid} (got {stages})"
        assert ref_num_frames is not None or not any(
            s.endswith("_ref") for s in stages
        ), "ref trace stages (s1_ref/s2_ref) require ref_num_frames"

        # Keyframe worker bakes the tail-padded shape (one extra latent frame) so a last-frame keyframe
        # decodes interior; generate() pads + trims to match. Every keyframe gen — interior or last —
        # replays this one padded shape, so the trace/upsampler/VAE below all warm at the padded count.
        if os.environ.get("LTX_KF_TRACE_PAD", "0") in ("1", "true", "True") and self._kf_trace_anchors():
            num_frames += TEMPORAL_COMPRESSION
            # The upsampler + VAE decoder bake latent-T at construction; rebuild them for the padded
            # count before the upsample/decode warmups JIT at that shape (generate() then finds them
            # ready, so its own _ensure_* calls are no-ops). Idempotent on a base worker.
            self._ensure_upsampler_frames(num_frames)
            self._ensure_vae_decoder_frames(num_frames)

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
        # generate() drops the audio decode entirely under LTX_VIDEO_ONLY, so warming the audio
        # decoder here compiles kernels that are never replayed.
        video_only = os.environ.get("LTX_VIDEO_ONLY", "0") in ("1", "true", "True")
        # Profiling-only: the denoise is the measurement target, and the VAE/audio decode warmups cost
        # ~200s of eager compile that blows the device reservation before the traced denoise is ever
        # reached. Skipping them leaves encode → S1 → upsample → S2 untouched.
        denoise_only = os.environ.get("LTX_PROFILE_DENOISE_ONLY", "0") in ("1", "true", "True")
        warmup_steps = 1 if iter_fast else num_inference_steps
        skip_dit_warmup = LTX_DIT_PREP_RUN and not capture_all

        t0 = time.time()
        logger.info(
            f"warmup (distilled 2-stage): {num_frames}f@{height}x{width}, "
            f"stages={stages}, ref_num_frames={ref_num_frames}, {warmup_steps} steps/stage"
        )

        # Every stage below runs the DiT denoise, so variant 0 must be resident before any of them.
        # It isn't guaranteed on entry: this method warms the coresident-excluded encoders/VAE LAST,
        # so an earlier warmup (or an encode) leaves the DiT evicted — a second warmup_buffers call
        # (e.g. the IC-LoRA s1_ref/s2_ref family after a base s1/s2 pass) would otherwise denoise on
        # evicted weights ("parameter has no data" at adaln_single). No-op when already resident.
        # The resident upsampler goes first (see _register_coresident_exclusions): every DiT reload
        # after a decode must land where the captures below saw the DiT.
        if self.upsampler is not None and os.environ.get("LTX_UPSAMPLER_RESIDENT", "1") != "0":
            self._prepare_upsampler()
        self._prepare_transformer(0)

        # Zeros at the real shapes compile the shape-driven kernels; the encoder is warmed
        # separately at the end of this method (it coresident-evicts the DiT/VAE).
        v_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.video_dim)
        a_p = torch.zeros(1, self.gemma_encoder_pair.sequence_length, self.gemma_encoder_pair.audio_dim)
        if self._device_prompt_handoff:
            self._allocate_device_prompt_buffers()

        # Allocate every requested stage's persistent trace I/O before any capture so all held inputs
        # sit below every trace's activation region and no replay overwrites another's inputs. The
        # s*_ref stages size their baked buffers to the COMBINED target+reference length.
        # Keyframe worker: capture the s1/s2 traces at the append-token sequence length by feeding dummy
        # conditioning at the configured anchor latents. Zeros compile the same pin/append kernels the
        # real gen records; only the captured shape (anchor count) matters. Empty on a base worker.
        kf_anchors = self._kf_trace_anchors()
        image_capable = bool(getattr(self.transformer, "image_conditioning", False))
        # Trace families per stage: a keyframe worker only ever pins, so its one capture is the per-token
        # family; a base worker that can pin captures both the scalar family (plain t2v gens) and the
        # per-token family (frame-0 gens); a scalar-only transformer has just the scalar family.
        kf_per_token = image_capable and bool(kf_anchors)
        i2v_families = image_capable and not kf_anchors and not skip_dit_warmup
        # LTX_WARMUP_T2V_ONLY=1: a worker that only ever serves plain t2v skips the per-token
        # family (~39 s of 4x8 warmup). A pinned gen on such a worker then refuses instead of
        # capturing at gen time (deferred capture is the blank-video bug).
        self._warmup_t2v_only = i2v_families and _env_on("LTX_WARMUP_T2V_ONLY")
        if self._warmup_t2v_only:
            logger.info("  LTX_WARMUP_T2V_ONLY=1: skipping the s1_i2v/s2_i2v trace families")
            i2v_families = False

        if self._traced:

            def _alloc(base, per_token, **kw):
                self._prealloc_trace_io(self._trace_variant_key(base, per_token), num_frames=num_frames, **kw)

            if "s1" in stages:
                _alloc("s1", kf_per_token, height=height // 2, width=width // 2)
                if i2v_families:
                    _alloc("s1", True, height=height // 2, width=width // 2)
            if "s2" in stages:
                _alloc("s2", kf_per_token, height=height, width=width)
                if i2v_families:
                    _alloc("s2", True, height=height, width=width)
            if "s1" in stages and "s2" in stages and self._device_resident(None):
                self._transition_io(num_frames, height, width)
            if "s1_ref" in stages:
                _alloc("s1_ref", image_capable, height=height // 2, width=width // 2, ref_num_frames=ref_num_frames)
            if "s2_ref" in stages:
                _alloc("s2_ref", image_capable, height=height, width=width, ref_num_frames=ref_num_frames)

        def _kf_conds(h, w):
            if not kf_anchors:
                return None
            _, lh, lw = latent_grid(num_frames, h, w)
            return [(a, torch.zeros(1, self.in_channels, 1, lh, lw), 1.0) for a in kf_anchors]

        def _frame0_cond(h, w):
            _, lh, lw = latent_grid(num_frames, h, w)
            return [(0, torch.zeros(1, self.in_channels, 1, lh, lw), 1.0)]

        # Warm the encoder before any capture so its connector workspace isn't in a trace's
        # activation region (zeroed on replay). dynamic_load reloads per request → warms last.
        warm_encoders = os.environ.get("LTX_WARMUP_ENCODERS", "1") in ("1", "true", "True")
        image_encoder_warmed_early = False
        if self._traced and not self.dynamic_load:
            self.gemma_encoder_pair.ensure_loaded()
            self.encode_prompts(["warmup"], use_cache=False)
            # The VAE image encoder too. On a static mesh it is loaded once and never reloaded
            # (load_model returns early once loaded), so where its weights land is where they stay.
            # Warmed last — after the audio trace capture below — they landed in that trace's
            # activation region: the first audio replay (gen #0's audio decode) overwrote them, and
            # every later I2V gen encoded its image to garbage (latent PCC 0.03 vs gen #0), pinning a
            # grey frame 0 and ignoring the image. Loading and running it here, before any capture,
            # puts its weights and anything its first forward allocates below every trace.
            if self.vae_encoder is not None and warm_encoders:
                logger.info(f"warmup image encoder (before capture): {height // 2}x{width // 2} + {height}x{width}")
                self._warmup_encode(height // 2, width // 2)
                self._warmup_encode(height, width)
                # The IC-LoRA reference encoders are also built once and never reloaded here, so the
                # same before-capture rule applies; the late warmup below skips them on this path.
                if ref_num_frames is not None:
                    logger.info(
                        f"warmup reference encoder (before capture): {ref_num_frames}f @ "
                        f"{height // 2}x{width // 2} + {height}x{width}"
                    )
                    self._warmup_ref_encode(ref_num_frames, height, width)
                image_encoder_warmed_early = True

        # Real distilled sigmas so warmup hits the same branches (incl. sigma_next == 0 final step).
        # Drop the schedule's terminal 0.0 before slicing: a short schedule (len-1 <= warmup_steps,
        # e.g. the fast 1-step S2) would otherwise keep that 0.0 AND get another appended, leaving a
        # sigma=0 step — which the image-cond pin path (dt/sigma) divides by. Kernels are shape-driven,
        # so one fewer warmup step still compiles the same inner_step the real gen replays.
        s1_sigmas = list(DISTILLED_SIGMA_VALUES[:-1])[:warmup_steps] + [0.0]
        s2_sigmas = list(STAGE_2_DISTILLED_SIGMA_VALUES[:-1])[:warmup_steps] + [0.0]

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
                    image_conds=_kf_conds(s1_h, s1_w),
                    traced=capture_traced,
                    trace_key="s1",
                )
            if i2v_families:
                logger.info(f"warmup stage 1 (per-token family): {s1_h}x{s1_w}, frame-0 pin")
                with walltime.timed("warmup", "stage1_i2v build"):
                    self._denoise_no_guidance(
                        v_p,
                        a_p,
                        num_frames=num_frames,
                        height=s1_h,
                        width=s1_w,
                        sigma_values=s1_sigmas,
                        seed=0,
                        image_conds=_frame0_cond(s1_h, s1_w),
                        traced=capture_traced,
                        trace_key="s1",
                    )
        elif skip_dit_warmup:
            logger.info("LTX_DIT_PREP_RUN: skipping stage-1 warmup denoise (gen#0 self-warms via prep_run)")

        if "s1_ref" in stages and not skip_dit_warmup:
            # IC-LoRA stage 1: capture the combined (target+reference) DiT trace at s1_ref. A zero
            # looped reference clip fixes the wider sequence shape; captured HERE (traced=capture_traced,
            # trace_key="s1_ref") exactly like s1 above — NOT deferred to gen#0 (deferred capture is the
            # known blank-video bug). Reference rides ref_latent, never keyframe conds (mutually excl).
            s1_h, s1_w = height // 2, width // 2
            ref_lf, s1_lh, s1_lw = latent_grid(ref_num_frames, s1_h, s1_w)
            dummy_ref_s1 = torch.zeros(1, self.in_channels, ref_lf, s1_lh, s1_lw)
            logger.info(f"warmup stage 1 (ref): {s1_h}x{s1_w} +{ref_lf} ref frames, σ={s1_sigmas}")
            with walltime.timed("warmup", "stage1_ref build"):
                self._denoise_no_guidance(
                    v_p,
                    a_p,
                    num_frames=num_frames,
                    height=s1_h,
                    width=s1_w,
                    sigma_values=s1_sigmas,
                    seed=0,
                    ref_latent=dummy_ref_s1,
                    traced=capture_traced,
                    trace_key="s1_ref",
                )
        elif "s1_ref" in stages and skip_dit_warmup:
            logger.info("LTX_DIT_PREP_RUN: skipping stage-1 (ref) warmup denoise (gen#0 self-warms via prep_run)")

        # The eager upsample/VAE-decode/audio warmups below are shared by the target-only and the
        # reference stage-2 families — decode/upsample never see reference — so run them if EITHER
        # full-res stage is requested.
        if "s2" in stages or "s2_ref" in stages:
            # Upsample runs between stage 1 and stage 2; compile its kernels here.
            if not iter_fast:
                logger.info(f"warmup upsample → {height}x{width}")
                self._warmup_upsample(num_frames, height, width)

            # Zero-dummies at the exact shapes the real stage-2 call uses.
            latent_frames, full_lh, full_lw = latent_grid(num_frames, height, width)
            als = AudioLatentShape.from_video_pixel_shape(
                VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=self.fps)
            )
            dummy_a_init = torch.zeros(1, als.frames, self.in_channels)

            if "s2" in stages and skip_dit_warmup:
                logger.info("LTX_DIT_PREP_RUN: skipping stage-2 warmup denoise (gen#0 self-warms via prep_run)")
            elif "s2" in stages:
                dummy_v_init = torch.zeros(1, latent_frames * full_lh * full_lw, self.in_channels)
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
                        image_conds=_kf_conds(height, width),
                        traced=capture_traced,
                        trace_key="s2",
                    )
                if i2v_families:
                    logger.info(f"warmup stage 2 (per-token family): {height}x{width}, frame-0 pin")
                    with walltime.timed("warmup", "stage2_i2v build"):
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
                            image_conds=_frame0_cond(height, width),
                            traced=capture_traced,
                            trace_key="s2",
                        )

            if "s2_ref" in stages and skip_dit_warmup:
                logger.info("LTX_DIT_PREP_RUN: skipping stage-2 (ref) warmup denoise (gen#0 self-warms via prep_run)")
            elif "s2_ref" in stages:
                # GRID-sized upsampled latent (target frames only), like the s2/keyframe path: _denoise
                # allocates base_v at the combined video_N_real and writes this into the [:video_N_grid]
                # target slice; the reference rows [T,T+R) are base zeros overwritten by the ref pin.
                # Captured HERE at trace_key="s2_ref" exactly like s2 — never deferred to gen#0.
                ref_lf = latent_grid(ref_num_frames, height, width)[0]
                grid_v_init = torch.zeros(1, latent_frames * full_lh * full_lw, self.in_channels)
                dummy_ref_full = torch.zeros(1, self.in_channels, ref_lf, full_lh, full_lw)
                logger.info(f"warmup stage 2 (ref): {height}x{width} +{ref_lf} ref frames, σ={s2_sigmas}")
                with walltime.timed("warmup", "stage2_ref build"):
                    self._denoise_no_guidance(
                        v_p,
                        a_p,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        sigma_values=s2_sigmas,
                        seed=0,
                        initial_video_latent=grid_v_init,
                        initial_audio_latent=dummy_a_init,
                        ref_latent=dummy_ref_full,
                        traced=capture_traced,
                        trace_key="s2_ref",
                    )

            # The device-resident stage transition: captured here, after the upsampler was loaded by its
            # warmup and the stage-2 family exists, so its activations sit above every held input.
            if "s1" in stages and "s2" in stages and self._device_resident(None) and not skip_dit_warmup:
                self._prepare_upsampler()
                logger.info("warmup stage transition (traced, device-resident)")
                with walltime.timed("warmup", "stage transition build"):
                    self._run_stage_transition(
                        self._trace_state[self._trace_variant_key("s1", kf_per_token)].tt_video_lat,
                        num_frames=num_frames,
                        s1_height=height // 2,
                        s1_width=width // 2,
                        height=height,
                        width=width,
                        seed=0,
                        traced=capture_traced,
                    )

            # Compile VAE decode at full-res (only s2 feeds decode in generate).
            if not iter_fast and not denoise_only:
                self._warmup_decode(num_frames, height, width)

            # Warm the on-device audio decode eagerly at the exact latent shape generate() produces:
            # it compiles kernels (cold ~64s otherwise), inits the lazy device state, and frees back
            # to a deterministic allocator free-list. The vocoder/mel decoder are
            # @traced_function(prep_run=False) and so never self-warm — this eager pass (trace flags
            # forced off) is what inits them. The audio trace itself is then captured here too, while
            # the video traces' held inputs are still the only thing pinned below the activation
            # regions: a capture deferred to the first real generate lays its activations over those
            # baked inputs and every video replay after it decodes to blank frames.
            if video_only:
                logger.info("LTX_VIDEO_ONLY=1: skipping warmup audio decode (generate() decodes no audio)")
            elif denoise_only:
                logger.info("LTX_PROFILE_DENOISE_ONLY=1: skipping VAE + audio decode warmup")
            elif in_capture_pass:
                logger.info("capture pass: skipping audio decode (vocoder prep_run=False; real warmup inits it)")
            elif iter_fast:
                logger.info("LTX_ITER_FAST=1: skipping warmup audio decode (gen#0 decodes eagerly)")
            else:
                logger.info("warmup audio decode (on-device, eager)")
                self._warmup_audio_decode(torch.zeros(1, als.frames, self.in_channels), num_frames)
                if self._traced:
                    logger.info("warmup audio decode (capture pass)")
                    self.decode_audio(torch.zeros(1, als.frames, self.in_channels), num_frames, fps=self.fps)
            # Programs are now compiled
            if self._vae_traced and self.vae_decoder is not None:
                self.vae_decoder._vae_traced = True

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
        if warm_encoders:
            if self.vae_encoder is not None and not image_encoder_warmed_early:
                logger.info(f"warmup image encoder: {height // 2}x{width // 2} + {height}x{width}")
                self._warmup_encode(height // 2, width // 2)
                self._warmup_encode(height, width)

                # IC-LoRA reference encoders (multi-frame, both stage res) — warmed here in the same
                # post-DiT ordering as the image encoder so their load evicts the DiT LAST (never
                # mid-trace-capture). Built lazily by _build_ref_vae_encoders; skipped without ref.
                if ref_num_frames is not None:
                    logger.info(
                        f"warmup reference encoder: {ref_num_frames}f @ {height // 2}x{width // 2} + {height}x{width}"
                    )
                    self._warmup_ref_encode(ref_num_frames, height, width)

            # use_cache=False forces a real encode so the Gemma/connector kernels compile. traced-static
            # already warmed before capture (above); dynamic_load / untraced warm last.
            if self.dynamic_load or not self._traced:
                self.gemma_encoder_pair.ensure_loaded()
                self.encode_prompts(["warmup"], use_cache=False)
        else:
            logger.info("LTX_WARMUP_ENCODERS=0: skipping image + gemma encoder warmup")

        # The video replay decodes through the VAE, so its weights must already be resident when the
        # denoise traces are captured: weights loaded after capture land in a trace's activation
        # region and the next replay overwrites them (flat colour bands). _warmup_decode loads them
        # and nothing frees them now, so this only pins that invariant for the video-only path — the
        # one path with no audio decode to reload them — and short-circuits on the is_loaded guard.
        if self._traced and not self.dynamic_load and os.environ.get("LTX_VIDEO_ONLY", "0") in ("1", "true", "True"):
            self._prepare_vae()

        logger.info(f"warmup (distilled 2-stage) done in {time.time() - t0:.1f}s")

    def _prepare_stage_statics(
        self,
        state,
        *,
        latent_frames,
        latent_h,
        latent_w,
        video_N,
        video_N_real,
        audio_N,
        audio_N_real,
        sp_axis,
        video_N_grid=None,
        anchor_frames=None,
        ref_latent_frames=0,
    ):
        """Build a stage's static per-shape inputs once (rope/cross-PE/masks/trans_mat).

        ``anchor_frames`` (append-token) extends the video RoPE + cross-PE by one hw block per interior
        keyframe (carrying that frame's phase); ``video_N_grid`` is the pre-append length used for the
        V→A mask so the audio never attends to the appended anchors.

        ``ref_latent_frames`` (R, IC-LoRA) instead extends the video RoPE + cross-PE by the reference
        block's OWN fresh 0-based grid of R frames (rows ``[T*hw, (T+R)*hw)``) — mutually exclusive with
        ``anchor_frames``. Unlike anchors, the reference IS in audio's view, so the caller passes the
        COMBINED length as ``video_N_grid`` here (that is what the V→A mask real region must be)."""
        if video_N_grid is None:
            video_N_grid = video_N_real
        anchor_frames = anchor_frames or []
        # ref_num_frames drives the RoPE two-grid concat (rope_ltx); 0/None is an exact no-op.
        ref_num_frames = ref_latent_frames or None
        built = state.tt_video_rope_cos is not None
        # Only the video RoPE/cross-PE carry each anchor's target-frame phase (prepare_*'s anchor_frames
        # extends the video temporal axis; audio positions and every mask are anchor-position-independent).
        # A traced keyframe worker therefore refreshes just those four in place per gen when the keyframe
        # positions move — same baked shape, new phases — and builds everything else once. A base worker
        # has no anchors, so positions never change and the built path returns at once.
        if built and anchor_frames == getattr(state, "_anchor_frames", []):
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
            anchor_frames=anchor_frames,
            ref_num_frames=ref_num_frames,
            fps=self.fps,
        )
        if built:
            (v_xpe_cos, v_xpe_sin, *_a_xpe) = prepare_av_cross_pe(
                latent_frames,
                latent_h,
                latent_w,
                audio_N,
                audio_N_real,
                theta=self.positional_embedding_theta,
                mesh_device=self.mesh_device,
                parallel_config=self.parallel_config,
                anchor_frames=anchor_frames,
                ref_num_frames=ref_num_frames,
            )
            # traced=True => ttnn.copy into the baked buffer, not a fresh allocation the captured trace
            # would never reference.
            state._tt_video_rope_cos.update(v_cos, True)
            state._tt_video_rope_sin.update(v_sin, True)
            state._tt_video_cross_pe_cos.update(v_xpe_cos, True)
            state._tt_video_cross_pe_sin.update(v_xpe_sin, True)
            state._anchor_frames = list(anchor_frames)
            return
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
            anchor_frames=anchor_frames,
            ref_num_frames=ref_num_frames,
            fps=self.fps,
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
        # V→A video context mask: real region = grid (pre-append), so appended anchor rows and SP-pad
        # are both zeroed — audio never syncs to the anchor.
        state._tt_video_padding_mask.update(
            build_video_pad_mask(video_N, video_N_grid, mesh_device=self.mesh_device, sp_axis=sp_axis), False
        )
        # Euler pad mask: anchor rows stay REAL (1.0) so the pin runs on them; only true SP-pad is zeroed.
        v_mask = torch.ones(1, 1, video_N, self.in_channels)
        v_mask[:, :, video_N_real:, :] = 0.0
        a_mask = torch.ones(1, 1, audio_N, self.in_channels)
        a_mask[:, :, audio_N_real:, :] = 0.0
        state._tt_video_pad_mask.update(v_mask, False, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device)
        state._tt_audio_pad_mask.update(a_mask, False, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device)
        state._anchor_frames = list(anchor_frames)

    @staticmethod
    def _kf_trace_anchors() -> list[int]:
        """Latent-frame indices to bake as append-token anchor blocks into the trace
        (``LTX_KF_TRACE_ANCHORS``, comma-separated). A ttnn trace captures one sequence length, so a
        keyframe worker fixes its anchor set here and every keyframe gen replays that shape. Empty
        (unset, or append-token off) keeps the base no-anchor trace. Frame 0 pins in-place, never an
        anchor. Gated on LTX_KF_APPEND_TOKEN so prealloc and the denoise agree on whether anchors
        extend the sequence."""
        if os.environ.get("LTX_KF_APPEND_TOKEN", "0") not in ("1", "true", "True"):
            return []
        raw = os.environ.get("LTX_KF_TRACE_ANCHORS", "").strip()
        return [int(x) for x in raw.split(",") if x.strip()] if raw else []

    @staticmethod
    def _trace_variant_key(trace_key, per_token):
        """A pinned (per-token) gen replays a different graph than a scalar gen, so each stage key
        names two trace + state families; the per-token one carries the ``_i2v`` suffix."""
        if trace_key is None or not per_token:
            return trace_key
        return f"{trace_key}_i2v"

    def _assert_trace_family_warmed(self, trace_key, traced):
        """Refuse a traced per-token gen whose family a T2V-only warmup skipped."""
        if (
            traced
            and getattr(self, "_warmup_t2v_only", False)
            and trace_key is not None
            and trace_key.endswith("_i2v")
            and trace_key not in self._trace_state
        ):
            raise RuntimeError(
                f"trace family {trace_key} was not captured: warmup ran with LTX_WARMUP_T2V_ONLY=1, "
                "so this worker serves plain t2v only (unset it to serve image-conditioned gens)"
            )

    def _prealloc_trace_io(self, trace_key, *, num_frames, height, width, ref_num_frames=0):
        """Allocate a stage's persistent trace inputs (constants, latent buffers, masks) up front,
        before any capture. A ttnn trace bakes absolute tensor addresses; activations allocated
        during capture are freed afterward and reused by the next capture, so a held input sitting
        in another trace's activation region is overwritten on replay. Allocating every held input
        for both stages first keeps them below both traces' activations. (The prompt is built
        separately in _denoise.)

        ``ref_num_frames`` (IC-LoRA) grows every trace-baked buffer to the COMBINED target+reference
        length so an s1_ref/s2_ref trace family bakes the wider addresses its replay needs. It feeds
        the SAME "extend video_N_real" machinery the keyframe anchors use; ``ref_num_frames=0`` is an
        exact no-op (video_N_real / video_N_grid collapse to the target-only lengths). Reference and
        keyframe anchors are mutually exclusive, so at most one extends the sequence."""
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        hw = latent_h * latent_w
        grid = latent_frames * hw
        # A keyframe trace reserves one hw anchor block per configured keyframe (see _kf_trace_anchors);
        # the grid length stays the decode/strip length, and video_N_real carries the appended anchors.
        anchor_frames = [a for a in self._kf_trace_anchors() if 0 < a < latent_frames]
        # IC-LoRA reference block: R frames appended after the grid (fresh 0-based grid, IN audio's view).
        ref_latent_frames = latent_grid(ref_num_frames, height, width)[0] if ref_num_frames else 0
        video_N_real = grid + len(anchor_frames) * hw + ref_latent_frames * hw
        video_N = self._sp_pad_len(video_N_real)
        # V→A pad-mask real region (video_N_grid passed to _prepare_stage_statics): the target grid for a
        # keyframe trace (anchors hidden from audio) but the COMBINED length for a reference trace (audio
        # attends to the reference) — matches _denoise_no_guidance's video_kv_logical_n decouple so the
        # baked mask replays correctly.
        video_N_grid = grid + ref_latent_frames * hw if ref_latent_frames else grid
        als = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=self.fps)
        )
        audio_N_real = als.frames
        audio_N = self._sp_pad_len(audio_N_real)
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        state = self._trace_state.setdefault(trace_key, LTXTransformerState())
        # Reserves the base grid shape by default; with LTX_KF_TRACE_ANCHORS the append-token anchor
        # blocks are baked here so a keyframe worker's trace matches the extended gen sequence, and with
        # ref_num_frames the reference block is baked so an s*_ref trace matches the wider gen sequence.
        self._prepare_stage_statics(
            state,
            latent_frames=latent_frames,
            latent_h=latent_h,
            latent_w=latent_w,
            video_N=video_N,
            video_N_real=video_N_real,
            video_N_grid=video_N_grid,
            audio_N=audio_N,
            audio_N_real=audio_N_real,
            sp_axis=sp_axis,
            anchor_frames=anchor_frames,
            ref_latent_frames=ref_latent_frames,
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

    def _allocate_device_prompt_buffers(self):
        """Reserve shared prompt sinks before encoder/DiT/VAE captures."""
        pair = self.gemma_encoder_pair
        assert not self.dynamic_load
        assert pair.video_dim == self.cross_attention_dim, "device handoff requires matching video prompt width"
        if self._prompt_v.value is None or self._prompt_a.value is None:
            assert not Tracer._traces_live.get(
                self.mesh_device.id(), 0
            ), "prompt buffers must be allocated before capture; release all encoder and consumer traces first"
        for state, width in ((self._prompt_v, pair.video_dim), (self._prompt_a, pair.audio_dim)):
            if state.value is None:
                state.update(
                    ttnn.zeros(
                        (1, 1, pair.sequence_length, width),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                    ),
                    True,
                )

    def _stage_prompts(self, v_embeds, a_embeds, traced: bool, device_prompts: bool, reuse_prompt: bool):
        if device_prompts:
            assert self._device_prompt_handoff and v_embeds is None and a_embeds is None
            assert self._prompt_v.value is not None and self._prompt_a.value is not None
            return self._prompt_v.value, self._prompt_a.value
        if reuse_prompt:
            assert traced and self._prompt_v.value is not None and self._prompt_a.value is not None
            return self._prompt_v.value, self._prompt_a.value
        if traced and _env_on("LTX_PROMPT_HOST_COPY") and self._write_prompts_in_place(v_embeds, a_embeds):
            return self._prompt_v.value, self._prompt_a.value
        prompt_v = self._prepare_prompt(v_embeds)
        prompt_a = bf16_tensor(a_embeds.unsqueeze(0), device=self.mesh_device)
        # Traced persists the shared prompt (baked address); untraced keeps locals to avoid
        # fragmenting DRAM for the downstream VAE decode.
        if traced:
            self._prompt_v.update(prompt_v, traced)
            self._prompt_a.update(prompt_a, traced)
            return self._prompt_v.value, self._prompt_a.value
        return prompt_v, prompt_a

    @staticmethod
    def _prompt_host_bf16(embeds: torch.Tensor, width: int) -> torch.Tensor:
        """Host twin of _prepare_prompt: same shape and padding, already bf16.

        Same values as the fp32 -> bf16 cast ttnn.from_torch does: the encoder output is bf16, so
        the fp32 copy is bf16-exact and the cast is lossless either way.
        """
        prompt = embeds.unsqueeze(0)
        if prompt.shape[-1] < width:
            prompt = torch.nn.functional.pad(prompt, (0, width - prompt.shape[-1]))
        elif prompt.shape[-1] > width:
            prompt = prompt[..., :width]
        return prompt.to(torch.bfloat16).contiguous()

    def _write_prompts_in_place(self, v_embeds, a_embeds) -> bool:
        """Tilize the prompts on host and write them straight into the persistent buffers.

        Skips the fresh device allocation and the device-side copy into the address-baked buffer that
        the default path pays per new prompt. Returns False (caller takes the default path) until the
        buffers exist with matching shapes, i.e. on the capture gen.
        """
        buffers = (self._prompt_v.value, self._prompt_a.value)
        if any(b is None for b in buffers):
            return False
        hosts = (
            self._prompt_host_bf16(v_embeds, self.cross_attention_dim),
            self._prompt_host_bf16(a_embeds, a_embeds.shape[-1]),
        )
        if any(tuple(h.shape) != tuple(b.shape) for h, b in zip(hosts, buffers)):
            return False
        for host, buf in zip(hosts, buffers):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), buf)
        if _env_on("LTX_PROMPT_STAGING_CHECK"):
            self._check_prompt_staging(v_embeds, a_embeds)
        return True

    def _check_prompt_staging(self, v_embeds, a_embeds) -> None:
        """Assert the in-place write matches the default upload bit for bit, on every device."""
        refs = (self._prepare_prompt(v_embeds), bf16_tensor(a_embeds.unsqueeze(0), device=self.mesh_device))
        for name, ref, buf in zip(("video", "audio"), refs, (self._prompt_v.value, self._prompt_a.value)):
            want = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(ref)]
            got = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(buf)]
            same = all(torch.equal(w, g) for w, g in zip(want, got))
            logger.info(f"LTX_PROMPT_STAGING_CHECK {name}: {len(got)} devices, bit-identical={same}")
            assert same, f"in-place prompt staging differs from the default upload ({name})"
            ttnn.deallocate(ref)

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
        initial_video_latent_device: "ttnn.Tensor | None" = None,
        return_device_video: bool = False,
        initial_audio_latent: torch.Tensor | None = None,
        # I2V conditioning: list of (latent_frame_index, cond_latent (1,C,1,lh,lw), strength).
        image_conds: list | None = None,
        # IC-LoRA reference block: an encoded looped reference (1, C, R, lh, lw) appended clean AFTER
        # the target grid. Mutually exclusive with the append-token keyframe path.
        ref_latent: torch.Tensor | None = None,
        ref_strength: float = 1.0,
        traced: bool = False,
        trace_key: str | None = None,
        # Draining the device profiler is only legal BETWEEN trace replays: a read issued while the
        # trace is still capturing trips !trace_id_.has_value(). Warmup captures, so only the replay
        # path (generate) may set this.
        profile_drain: bool = False,
        device_prompts: bool = False,
        ancestral: bool = False,
        ancestral_noise_seed: int | None = None,
        # The persistent prompt buffers already hold v_embeds/a_embeds (a traced stage of the same
        # generate() wrote them); skip the re-upload.
        reuse_prompt: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = 1
        _t_init = time.perf_counter()
        _t_marks = []
        latent_frames, latent_h, latent_w = latent_grid(num_frames, height, width)
        hw = latent_h * latent_w
        # THREE lengths that COINCIDE for base/i2v/keyframe but SPLIT for reference:
        #   * decode_N       — the return/decode strip: always TARGET-ONLY (latent_frames * hw).
        #   * video_N_grid   — the target grid; == decode_N. In the append-token keyframe path this
        #                      also drives the KV-logical / V→A pad-mask real region (anchors hidden).
        #   * video_N_real   — the extended logical count fed to self-attn (Q): grid + anchor blocks
        #                      (keyframe) OR grid + reference block (reference); video_N is its SP-pad.
        # For REFERENCE the audio cross-attention MUST see the reference block, so its KV-logical and
        # the V→A pad-mask real region become the COMBINED (target+reference) length, NOT video_N_grid
        # — see video_kv_logical_n below (the Item 3 silent-correctness decouple).
        decode_N = latent_frames * hw
        video_N_grid = decode_N
        ref_latent_frames = ref_latent.shape[2] if ref_latent is not None else 0
        has_ref = ref_latent_frames > 0
        als = AudioLatentShape.from_video_pixel_shape(
            VideoPixelShape(batch=B, frames=num_frames, height=height, width=width, fps=self.fps)
        )
        audio_N_real = als.frames
        audio_N = self._sp_pad_len(audio_N_real)
        sp_factor = self.parallel_config.sequence_parallel.factor
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis

        append_interior = os.environ.get("LTX_KF_APPEND_TOKEN", "0") in ("1", "true", "True")
        # Reference and keyframe append-token are mutually exclusive: reference rides in-place conds
        # (never an anchor block) and is visible to audio, whereas anchors are appended and hidden.
        assert not (
            has_ref and append_interior
        ), "reference conditioning and append-token keyframe conditioning are mutually exclusive"
        image_cond = bool(image_conds) or has_ref
        # Per-token modulation only when this gen pins something. A plain t2v gen on an image-capable
        # worker takes the scalar-AdaLN path, which is a different graph, so it owns its own trace family.
        needs_video_ts = getattr(self.transformer, "image_conditioning", False) and image_cond
        trace_key = self._trace_variant_key(trace_key, needs_video_ts)
        self._assert_trace_family_warmed(trace_key, traced)
        logger.info(f"  conditioning path: {'per-token' if needs_video_ts else 'scalar'} (trace family {trace_key})")
        # AdaLN's 2-value timestep pair (in the step loop) carries one pinned noise level, so the
        # modulation uses a single shared strength; the per-token pin/denoise_mask below still honors
        # each frame's own strength. Uniform strengths (the server default) make this exact.
        if has_ref:
            # Pin the R reference frames IN-PLACE at combined latent indices [T, T+R) at ref_strength
            # (1.0 = kept clean forever); they ride the same conds / pin machinery as ordinary i2v
            # conds. build_conditioning_tensors' in-place assert (0 <= idx < latent_frames) requires
            # the COMBINED frame count, so build with combined_latent_frames and append_interior=False.
            ref_conds = [
                (latent_frames + i, ref_latent[:, :, i : i + 1, :, :], ref_strength) for i in range(ref_latent_frames)
            ]
            all_conds = (list(image_conds) if image_conds else []) + ref_conds
            combined_latent_frames = latent_frames + ref_latent_frames
            video_N_real_combined = combined_latent_frames * hw
            image_cond_strength = all_conds[0][2]
            i2v = self._build_i2v_conditioning(
                all_conds,
                needs_video_ts,
                B,
                video_N_real_combined,
                combined_latent_frames,
                latent_h,
                latent_w,
                append_interior=False,
            )
        else:
            image_cond_strength = image_conds[0][2] if image_cond else 1.0
            i2v = self._build_i2v_conditioning(
                image_conds,
                needs_video_ts,
                B,
                video_N_grid,
                latent_frames,
                latent_h,
                latent_w,
                append_interior=append_interior,
            )
        video_N_real = i2v.video_N_real_ext
        video_N = self._sp_pad_len(video_N_real)
        anchor_frames = i2v.anchor_frame_indices
        # THE DECOUPLE. video_kv_logical_n is audio cross-attention's logical view of the video K/V
        # (and the V→A pad-mask real region). Keyframe: video_N_grid (anchors excluded from audio).
        # Reference: the COMBINED length video_N_real (== video_N_real_combined) so audio attends to
        # target+reference — matching the proven SOURCE, whose inner_step kv_logical_n == combined.
        # Leaving this at video_N_grid in the reference path COMPILES, RUNS, and LOGS CLEAN while
        # silently dropping the reference from audio cross-attention (the Item 3 trap).
        video_kv_logical_n = video_N_real if has_ref else video_N_grid

        logger.info(
            f"  shapes: vN={video_N}(real={video_N_real} grid={video_N_grid} decode={decode_N} "
            f"kv_logical={video_kv_logical_n} ref={ref_latent_frames}), "
            f"aN={audio_N}(real={audio_N_real}) [sp={sp_factor}]"
        )

        # Persist buffers only when traced (baked addresses); untraced uses a transient state so
        # statics rebuild — resolution can differ across generates.
        state = self._trace_state.setdefault(trace_key, LTXTransformerState()) if traced else LTXTransformerState()
        _t_marks.append(("cond", time.perf_counter()))
        self._prepare_stage_statics(
            state,
            latent_frames=latent_frames,
            latent_h=latent_h,
            latent_w=latent_w,
            video_N=video_N,
            video_N_real=video_N_real,
            # V→A pad-mask real region: grid for keyframe (hide anchors), combined for reference.
            video_N_grid=video_kv_logical_n,
            audio_N=audio_N,
            audio_N_real=audio_N_real,
            sp_axis=sp_axis,
            anchor_frames=anchor_frames,
            ref_latent_frames=ref_latent_frames,
        )

        _t_marks.append(("statics", time.perf_counter()))
        prompt_v, prompt_a = self._stage_prompts(v_embeds, a_embeds, traced, device_prompts, reuse_prompt)

        _t_marks.append(("prompt", time.perf_counter()))
        sigmas = torch.tensor(sigma_values, dtype=torch.float32)

        # ----- Video latent init: one GaussianNoiser over three bases — I2V (frame-0 replaced by
        # the clean cond latent), T2V-S2 (upsampled latent), T2V-S1 (zeros). denoise_mask pins the
        # conditioning tokens; None ≡ full forward noise. Ends at (B, video_N, C). -----
        if image_cond:  # I2V: zeros (S1) or upsampled (S2), cond frames overwritten by the clean latent
            base_v = torch.zeros(B, video_N_real, self.in_channels)
            if initial_video_latent is not None:  # S2: upsampled latent arrives at grid length
                grid_v = initial_video_latent.float()
                if grid_v.dim() == 2:
                    grid_v = grid_v.unsqueeze(0)
                base_v[:, :video_N_grid, :] = grid_v.clone()
            # pin_rows now indexes frame-0 (in-place) and/or the appended anchor block; a free interior
            # frame keeps its base value (zeros=noise at S1, the upsampled latent at S2).
            base_v[:, i2v.pin_rows, :] = i2v.clean_latent[:, i2v.pin_rows, :]
        elif initial_video_latent_device is not None:  # T2V S2, device-resident: handled below
            base_v = None
        elif initial_video_latent is not None:  # T2V S2: upsampled latent arrives at (B, video_N_real, C)
            base_v = initial_video_latent.float()
            assert (
                base_v.shape[1] == video_N_real
            ), f"initial_video_latent seq dim {base_v.shape[1]} != video_N_real {video_N_real}"
        else:  # T2V S1: pure noise from zeros
            base_v = torch.zeros(B, video_N_real, self.in_channels)
        video_lat_dev = noise_a = None
        if initial_video_latent_device is not None:
            assert not image_cond, "device-resident stage input is T2V only"
            video_lat_dev = initial_video_latent_device  # the traced transition already noised and sharded it
            video_lat = None
        else:
            v_shape = (B, video_N_real, self.in_channels)
            if initial_audio_latent is None:  # stage-1 audio noise continues the video's seeded stream
                noise_v, noise_a = self._seeded_noise(seed, v_shape, (B, audio_N_real, self.in_channels))
            else:
                (noise_v,) = self._seeded_noise(seed, v_shape)
            video_lat_real = self._noise_video_latent(base_v, i2v.denoise_mask, sigmas[0], noise_v)

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
            if noise_a is None:
                # The audio draws always follow a video-sized draw from the same seed.
                _, noise_a = self._seeded_noise(
                    seed, (B, video_N_real, self.in_channels), (B, audio_N_real, self.in_channels)
                )
            audio_lat_real = noise_a.float() * sigmas[0]
            audio_lat = torch.zeros(B, audio_N, self.in_channels)
            audio_lat[:, :audio_N_real, :] = audio_lat_real

        _t_marks.append(("noise", time.perf_counter()))
        num_steps = len(sigma_values) - 1

        # Device-resident loop: inner_step returns SP-sharded velocity, stepped in place by an
        # on-device Euler. update() copies into the address-baked buffer when traced, else rebinds.
        if video_lat_dev is not None:
            state._tt_video_lat.update(video_lat_dev, traced)
        else:
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

        _t_marks.append(("upload", time.perf_counter()))
        _t_prev, _t_parts = _t_init, []
        for _name, _t in _t_marks:
            _t_parts.append(f"{_name} {(_t - _t_prev) * 1000:.0f}")
            _t_prev = _t
        logger.info(
            f"  denoise init (latent/prompt/mask uploads): {(time.perf_counter() - _t_init) * 1000:.0f} ms "
            f"[{', '.join(_t_parts)} ms]"
        )
        ancestral_gen = None
        if ancestral:
            noise_seed = (
                ancestral_noise_seed if ancestral_noise_seed is not None else seed + ANCESTRAL_NOISE_SEED_OFFSET
            )
            ancestral_gen = torch.Generator().manual_seed(noise_seed)
            logger.info(f"  ancestral Euler on-device (eta={ANCESTRAL_ETA}) noise_seed={noise_seed}")

        # Replays are CQ-ordered, so the host need not wait for one step before enqueuing the next;
        # LTX_STEP_SYNC=1 blocks on each replay so STEP_MS reports device step time.
        step_sync = _env_on("LTX_STEP_SYNC")
        for step_idx in range(num_steps):
            _t_step = time.perf_counter()
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
            # Drain BEFORE the step: the profiler's DRAM buffer holds a fixed number of programs and the
            # eager warmup already filled it, so a post-step drain would find the traced programs
            # already dropped. Draining first resets the buffer, so this step's trace fills it clean and
            # the NEXT drain reads it back. The drain issues an event-sync, illegal while the command
            # queue is in kernel capture (prewarm's capture stage) — there is no live replay to sync
            # against — so it is confined to the real run.
            if (
                profile_drain
                and os.environ.get("LTX_PROFILE_FLUSH")
                and not os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY")
            ):
                ttnn.ReadDeviceProfiler(self.mesh_device)
            # video_N_real is the logical (unpadded) count so ring SDPA masks padded K positions.
            _t_step = time.perf_counter()
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
                video_kv_logical_n=video_kv_logical_n,
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
                tracer_blocking_execution=step_sync,
            )
            # Flow-matching Euler (latents += dt*velocity, SP-padding slots zeroed) so the trace's
            # baked latent address holds across replays.
            dt = sigma_next - sigma
            trace_euler = (
                not ancestral
                and self._trace_euler_tail
                and traced
                and not image_cond
                and os.environ.get("LTX_DEBUG_STATS", "0") != "1"
            )
            if trace_euler:
                producer = LTXTransformerModel.inner_step._tracers_keyed.get(self.transformer, {}).get(trace_key)
                # Recipe-only capture produces transient skipped-dispatch outputs:
                # never retain them or create a tail owner before a real DiT trace.
                trace_euler = producer is not None and producer.trace_captured
            if ancestral:
                # Noise is drawn on host (seeded RNG parity with upstream); the Euler math stays
                # on-device so latents/velocities never round-trip. Video then audio from one
                # generator (upstream euler_ancestral_denoising_loop order).
                tt_noise_v = tt_noise_a = None
                if sigma_next != 0.0:
                    noise_v = torch.randn(1, 1, video_N, self.in_channels, generator=ancestral_gen, dtype=torch.float32)
                    noise_a = torch.randn(1, 1, audio_N, self.in_channels, generator=ancestral_gen, dtype=torch.float32)
                    if video_N > video_N_real:
                        noise_v[:, :, video_N_real:, :] = 0.0
                    if audio_N > audio_N_real:
                        noise_a[:, :, audio_N_real:, :] = 0.0
                    tt_noise_v = bf16_tensor(noise_v, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=2)
                    tt_noise_a = bf16_tensor(noise_a, device=self.mesh_device, mesh_axis=sp_axis, shard_dim=2)
                _euler_ancestral_step_tt(
                    state.tt_video_lat, v_out, state.tt_video_pad_mask, sigma, sigma_next, tt_noise_v
                )
                _euler_ancestral_step_tt(
                    state.tt_audio_lat, a_out, state.tt_audio_pad_mask, sigma, sigma_next, tt_noise_a
                )
                if tt_noise_v is not None:
                    ttnn.deallocate(tt_noise_v)
                if tt_noise_a is not None:
                    ttnn.deallocate(tt_noise_a)
            elif trace_euler:
                if state._euler_tail is None:
                    state._euler_tail = EulerTail(self.mesh_device)
                state._euler_tail(
                    state.tt_video_lat,
                    state.tt_audio_lat,
                    v_out,
                    a_out,
                    state.tt_video_pad_mask,
                    state.tt_audio_pad_mask,
                    dt,
                    blocking=step_sync,
                )
            else:
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
            # STEP_MS covers the step body only. A delta between successive log lines would fold the
            # host-side profiler drain into the step wall and corrupt the measurement.
            _step_ms = (time.perf_counter() - _t_step) * 1000.0
            # Without the per-step sync the body only enqueues; a distinct key keeps STEP_MS parsers honest.
            _step_key = "STEP_MS" if step_sync or not traced else "ENQUEUE_MS"
            logger.info(
                f"  Step {step_idx + 1}/{num_steps}: σ {sigma:.4f} → {sigma_next:.4f} {_step_key}={_step_ms:.1f}"
            )
            if os.environ.get("LTX_DEBUG_STATS", "0") == "1":
                # Device-0 shard only: enough to see whether the DiT's velocity or the running latent
                # first leaves the finite range, without a mesh-wide gather inside the step loop.
                for _lbl, _t in (("v_vel", v_vel), ("video_lat", state.tt_video_lat)):
                    _f = ttnn.to_torch(ttnn.get_device_tensors(_t)[0]).float()
                    _fin = _f[torch.isfinite(_f)]
                    logger.info(
                        f"STATS step{step_idx + 1} {_lbl}(dev0): finite_abs_mean={_fin.abs().mean():.4g} "
                        f"finite_max={_fin.abs().max():.4g} inf_frac={torch.isinf(_f).float().mean():.4f} "
                        f"nan_frac={torch.isnan(_f).float().mean():.4f}"
                    )

        if return_device_video:
            a_final = LTXTransformerModel.device_to_host(
                state.tt_audio_lat,
                ccl_manager=self.ccl_manager,
                parallel_config=self.parallel_config,
                sp_already_gathered=False,
                tp_already_gathered=True,
            ).squeeze(0)
            return state.tt_video_lat, a_final[:, :audio_N_real, :]
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
        # Strip the appended anchor / reference tokens (and SP-pad): return exactly the target grid
        # for decode. decode_N is TARGET-ONLY, so the reference block never reaches decode/upsample.
        return v_final[:, :decode_N, :], a_final[:, :audio_N_real, :]

    # ----- device-resident stage transition (T2V) -----------------------------------------------------------
    def _sync_for_stage_timing(self, device_resident: bool) -> None:
        """LTX_TIME_STAGES=1: block so async device work is charged to the stage that enqueued it.
        Off by default: the sync would stall the host and change the e2e time."""
        if device_resident and _env_on("LTX_TIME_STAGES"):
            ttnn.synchronize_device(self.mesh_device)

    def _device_resident(self, images) -> bool:
        """Keep the video latent on device from stage 1 through the upsampler into stage 2 (T2V only).
        The transition is a trace of its own: eager collectives issued after a replay deadlock on this
        line (the encode met the same class), traced ones do not. Default on for a Linear mesh; the Ring
        mesh keeps the host round-trip unless ``LTX_DEVICE_RESIDENT=1`` forces the device path, and
        ``LTX_DEVICE_RESIDENT=0`` restores the host round-trip everywhere."""
        if images:
            return False
        flag = os.environ.get("LTX_DEVICE_RESIDENT")
        if flag is not None:
            return flag == "1"
        return self.ccl_manager.topology != ttnn.Topology.Ring

    def _transition_io(self, num_frames: int, height: int, width: int) -> dict:
        """Persistent inputs of the traced stage transition, allocated before any capture like the stage
        I/O: the seeded stage-2 noise (uploaded per gen into the baked buffer) and the VAE per-channel
        stats. ``weights`` records the upsampler's weight addresses at capture; the trace bakes them."""
        io = getattr(self, "_t12_io", None)
        if io is not None:
            return io
        latent_frames, lh, lw = latent_grid(num_frames, height, width)
        video_N = self._sp_pad_len(latent_frames * lh * lw)
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        mean, std = self._vae_per_channel_stats()
        io = {"noise": StateTensor(), "mean": StateTensor(), "std": StateTensor(), "weights": None}
        io["noise"].update(
            torch.zeros(1, 1, video_N, self.in_channels),
            False,
            mesh_axes=[None, None, sp_axis, None],
            device=self.mesh_device,
        )
        # fp32 per-channel stats: the host path applies them in fp32, and rounding them to bf16 alone costs
        # ~0.4 % per channel on the stage-2 input.
        for key, v in (("mean", mean), ("std", std)):
            io[key].update(v.reshape(1, 1, 1, -1).float(), False, dtype=ttnn.float32, device=self.mesh_device)
        self._t12_io = io
        return io

    def _upsampler_weight_addresses(self) -> tuple:
        return tuple(p.data.buffer_address() for _, p in self.upsampler.named_parameters() if p.data is not None)

    # prep_run: nothing else compiles the transition's collectives and reshards, and a kernel build
    # inside the capture trips the mesh workload's !is_capturing_trace. The eager pre-run happens at
    # warmup, right after the stage captures, in the same fresh state gen#0's eager transition ran in.
    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False, prep_run=True)
    def _stage_transition(
        self,
        *,
        tokens: ttnn.Tensor,
        noise: ttnn.Tensor,
        mean_t: ttnn.Tensor,
        std_t: ttnn.Tensor,
        latent_frames: int,
        s1_h: int,
        s1_w: int,
        video_N_real: int,
        video_N: int,
        sigma: float,
        sp_axis: int,
    ) -> ttnn.Tensor:
        """Stage-1 tokens (``(1, 1, video_N1, C)``, sequence-sharded on ``sp_axis``, normalized) -> the noised
        stage-2 input ``(1, 1, video_N, C)`` sharded like the stage-2 state buffer. Mirrors ``upsample_latent``
        (un-normalize -> replicate-pad H/W to the upsampler's mesh factors -> upsampler -> crop -> re-normalize)
        and ``_noise_video_latent`` (fp32 mix with the seeded draw) with on-device gathers/partitions."""
        ccl = self.ccl_manager
        ups = self.upsampler
        upc = ups.parallel_config
        C = self.in_channels
        n_real = latent_frames * s1_h * s1_w

        x = ccl.all_gather(tokens, dim=2, mesh_axis=sp_axis, use_hyperparams=False)  # replicated tokens
        # Same numerics as the host path: un-normalize in fp32, bf16 into the upsampler (the host uploads
        # bf16), fp32 again for the re-normalize and the noise mix.
        x = ttnn.typecast(x, ttnn.float32)
        x = ttnn.add(ttnn.multiply(x, std_t), mean_t)  # un-normalize (padded rows are garbage, sliced next)
        x = ttnn.typecast(x, ttnn.bfloat16)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, n_real, C])
        x = ttnn.reshape(x, (latent_frames, s1_h, s1_w, C))  # THWC (B=1 folded)

        hf, wf = upc.height_parallel.factor, upc.width_parallel.factor
        ph, pw = (-s1_h) % hf, (-s1_w) % wf
        if ph:
            last = ttnn.slice(x, [0, s1_h - 1, 0, 0], [latent_frames, s1_h, s1_w, C])
            x = ttnn.concat([x, ttnn.repeat(last, (1, ph, 1, 1))], dim=1)
        if pw:
            last = ttnn.slice(x, [0, 0, s1_w - 1, 0], [latent_frames, s1_h + ph, s1_w, C])
            x = ttnn.concat([x, ttnn.repeat(last, (1, 1, pw, 1))], dim=2)
        x = ttnn.mesh_partition(x, dim=1, cluster_axis=upc.height_parallel.mesh_axis)
        x = ttnn.mesh_partition(x, dim=2, cluster_axis=upc.width_parallel.mesh_axis)
        x = ttnn.reshape(x, (1, latent_frames, (s1_h + ph) // hf, (s1_w + pw) // wf, C))  # BTHWC shard

        x, _, _ = ups.forward_device(
            x, s1_h + ph, s1_w + pw
        )  # the host path pads first and reports the padded size as logical
        x = ttnn.reshape(x, (latent_frames, x.shape[2], x.shape[3], C))
        x = ccl.all_gather(x, dim=1, mesh_axis=upc.height_parallel.mesh_axis, use_hyperparams=False)
        x = ccl.all_gather(x, dim=2, mesh_axis=upc.width_parallel.mesh_axis, use_hyperparams=False)
        x = ttnn.slice(x, [0, 0, 0, 0], [latent_frames, 2 * s1_h, 2 * s1_w, C])
        x = ttnn.reshape(x, (1, 1, latent_frames * 2 * s1_h * 2 * s1_w, C))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.typecast(x, ttnn.float32)
        # Re-normalize with a true division, in fp32. Multiplying by the reciprocal is off by up to one fp32
        # ulp from the host path's (x - mean) / std on a quarter of the elements; enough of those flip a bf16
        # rounding in the noise mix that the 3-step distilled stage 2 renders a visibly different (though
        # equally valid) sample, 36 dB PSNR_Y against the host path. With the division the galaxy ring run
        # was bit-identical to it (PR #56839).
        base = ttnn.divide(ttnn.subtract(x, mean_t), std_t)
        if video_N > video_N_real:
            base = ttnn.pad(base, [(0, 0), (0, 0), (0, video_N - video_N_real), (0, 0)], 0.0)
        base = ttnn.mesh_partition(base, dim=2, cluster_axis=sp_axis)
        base32 = base if base.dtype == ttnn.float32 else ttnn.typecast(base, ttnn.float32)
        noise32 = ttnn.typecast(noise, ttnn.float32)
        lat = ttnn.add(ttnn.multiply(noise32, sigma), ttnn.multiply(base32, 1.0 - sigma))
        return ttnn.typecast(lat, ttnn.bfloat16)

    def _run_stage_transition(
        self,
        tokens: ttnn.Tensor,
        *,
        num_frames: int,
        s1_height: int,
        s1_width: int,
        height: int,
        width: int,
        seed: int,
        traced: bool,
    ) -> ttnn.Tensor | None:
        """Upload this gen's seeded stage-2 noise and run the (traced) transition. Returns None when the
        upsampler's weights no longer sit at the addresses the trace baked (a reload landed elsewhere),
        so the caller takes the host round-trip for this gen instead of replaying over stale weights."""
        io = self._transition_io(num_frames, height, width)
        if traced and io["weights"] is not None and io["weights"] != self._upsampler_weight_addresses():
            logger.warning("stage transition: upsampler weights moved since capture; host round-trip for this gen")
            return None
        latent_frames, s1_h, s1_w = latent_grid(num_frames, s1_height, s1_width)
        _, lh, lw = latent_grid(num_frames, height, width)
        video_N_real = latent_frames * lh * lw
        video_N = self._sp_pad_len(video_N_real)
        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        B, C = 1, self.in_channels
        _t0 = time.perf_counter()
        (noise,) = self._seeded_noise(seed, (B, video_N_real, C))  # identical draws to the host path
        noise_p = torch.zeros(1, B, video_N, C, dtype=torch.float32)
        noise_p[0, :, :video_N_real, :] = noise.float()
        _t1 = time.perf_counter()
        io["noise"].update(noise_p, True, mesh_axes=[None, None, sp_axis, None], device=self.mesh_device)
        _t2 = time.perf_counter()
        out = self._stage_transition(
            tokens=tokens,
            noise=io["noise"].value,
            mean_t=io["mean"].value,
            std_t=io["std"].value,
            latent_frames=latent_frames,
            s1_h=s1_h,
            s1_w=s1_w,
            video_N_real=video_N_real,
            video_N=video_N,
            sigma=float(STAGE_2_DISTILLED_SIGMA_VALUES[0]),
            sp_axis=sp_axis,
            traced=traced,
            tracer_trace_key="t12",
            tracer_blocking_execution=_env_on("LTX_STEP_SYNC"),
        )
        if traced and io["weights"] is None:
            io["weights"] = self._upsampler_weight_addresses()
        # Where the transition's wall time goes: host draw, host tilize + upload, device replay (enqueue).
        logger.info(
            f"  stage transition: draw {(_t1 - _t0) * 1000:.0f} ms, upload {(_t2 - _t1) * 1000:.0f} ms, "
            f"replay {(time.perf_counter() - _t2) * 1000:.0f} ms"
        )
        return out

    @staticmethod
    def _latent_stats(x: torch.Tensor) -> dict:
        """Numeric fingerprint of a latent, for triaging noise outputs after the fact.

        ``whiteness`` is the mean absolute difference between adjacent tokens divided by
        what iid Gaussian noise of the same std would give (2/sqrt(pi) * std). It is ~1.0
        when the tensor is statistically indistinguishable from the noise the sampler
        started from -- i.e. the transformer's output never landed -- and well below 1
        for a latent carrying real spatial structure. std alone cannot tell these apart:
        LTX latents are per-channel normalised, so a correctly denoised latent is also
        ~unit variance.

        ``zeros`` is the fraction of exactly-zero elements. Diffusion output is never
        exactly zero, so a non-trivial value means part of the buffer was never written --
        the flat rectangles seen in corrupted decodes.
        """
        t = x.detach().float()
        if t.dim() == 3:
            t = t[0]
        flat = t.reshape(-1)
        std = flat.std().item()
        whiteness = float("nan")
        if t.dim() == 2 and t.shape[0] > 1 and std > 0:
            adjacent = (t[1:] - t[:-1]).abs().mean().item()
            whiteness = adjacent / (std * 2.0 * math.sqrt(1.0 / math.pi))
        return {
            "mean": flat.mean().item(),
            "std": std,
            "whiteness": whiteness,
            "zeros": (flat == 0).float().mean().item(),
            "nonfinite": int((~torch.isfinite(flat)).sum().item()),
        }

    def _log_latent_stats(
        self, label: str, video: torch.Tensor, audio: torch.Tensor | None, *, warn: bool = False
    ) -> None:
        """Log a fingerprint per latent; optionally flag the corrupted-generation signature.

        Good and bad generations are otherwise indistinguishable in the log -- same steps,
        same sigmas, same timings -- so without this a noise report has nothing to correlate.
        """
        if not _env_on("LTX_LATENT_STATS", "0"):
            # Opt-in (LTX_LATENT_STATS=1): the fingerprints are host reductions over the whole latent
            # (~7 ms at S1, ~10 ms at S2 for 1080p/145f) on the critical path between stages.
            return
        for name, tensor in (("video", video), ("audio", audio)):
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                # Device-resident hand-off (a ttnn tensor): reading it back only to log would add the
                # host round-trip the device path exists to remove, so no fingerprint here.
                logger.debug(f"  latent[{label}/{name}]: device-resident, stats skipped")
                continue
            s = self._latent_stats(tensor)
            logger.info(
                f"  latent[{label}/{name}]: mean={s['mean']:+.3f} std={s['std']:.3f} "
                f"whiteness={s['whiteness']:.2f} zeros={s['zeros']:.2%} nonfinite={s['nonfinite']}"
            )
            if not warn:
                continue
            if s["whiteness"] > WHITENESS_WARN_THRESHOLD:
                logger.warning(
                    f"{label}/{name}: post-denoise latent is ~white noise "
                    f"(whiteness={s['whiteness']:.2f}); the transformer output never landed -- "
                    "expect a noise result"
                )
            if s["zeros"] > 0.01:
                logger.warning(
                    f"{label}/{name}: post-denoise latent is {s['zeros']:.2%} exactly zero; "
                    "part of the buffer was never written -- expect flat patches in the decode"
                )
            if s["nonfinite"]:
                logger.warning(f"{label}/{name}: {s['nonfinite']} non-finite elements in post-denoise latent")

    def generate(
        self,
        prompt: str,
        *,
        output_path: str | None = None,
        output_type: str = "rgb",
        # I2V conditioning: list of (image_path, pixel_frame_idx, s1_strength[, s2_strength]).
        # frame_idx 0 = first frame, num_frames-1 = last, any value = a keyframe.
        images: list[tuple] | None = None,
        # IC-LoRA sequence-extension conditioning: (reference_sheet_path, strength). The still is
        # looped to a static reference video, VAE-encoded per stage, and APPENDED (clean) as an extra
        # token block — the sequence grows by R frames; requires the fused IC-LoRA checkpoint. Mutually
        # exclusive with the append-token keyframe path.
        reference_video: tuple[str, float] | None = None,
        ref_frames: int | None = None,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        seed: int = 10,
        fps: float | None = None,
    ):
        """Run the distilled 2-stage AV pipeline.

        output_path given → encode an AV MP4 and return its path (str).
        output_path None  → return ``(frames, audio)`` for the caller to encode: frames per
                            ``output_type`` (see ``decode_latents``), audio = decoded ``Audio``.

        ``fps`` defaults to the pipeline's own rate and may not differ from it: the audio
        latent length and the A/V cross-PE are derived from it and are baked into the
        captured traces, so a per-request rate would replay a trace built for another shape.
        """
        assert height % 64 == 0, f"Height must be divisible by 64 (got {height})"
        assert width % 64 == 0, f"Width must be divisible by 64 (got {width})"

        fps = self._resolve_fps(fps)

        s1_height = height // 2
        s1_width = width // 2

        # IC-LoRA reference sheet: loop the still to the output length (model card >=121, capped to
        # num_frames). Mutually exclusive with the append-token keyframe path — they contradict on the
        # KV-logical/decode-strip roles (reference is IN audio's view; anchors are hidden), so refuse
        # to run both rather than silently pick one.
        # The reference is a STILL looped to ref_pixel_frames, so frames beyond the first carry no extra
        # signal — but they cost sequence: R=num_frames doubles the video tokens and OOMs from 720p up.
        # Callers pass a small ref_frames (17 px -> 3 latent) to keep the overhead ~20%. This MUST match
        # the ref_num_frames the s*_ref traces were warmed with, or trace replay shape-mismatches.
        ref_pixel_frames = (ref_frames or num_frames) if reference_video is not None else 0
        assert not (
            reference_video is not None and os.environ.get("LTX_KF_APPEND_TOKEN", "0") in ("1", "true", "True")
        ), "reference conditioning (reference_video) is mutually exclusive with append-token keyframe conditioning (LTX_KF_APPEND_TOKEN)"

        # Append-token tail-pad. A keyframe pinned to the LAST latent frame has no neighbor after it,
        # so its freed grid frame settles onto the static anchor and the causal VAE holds the clip's
        # final pixels (a frozen tail). Generate one extra latent frame (TEMPORAL_COMPRESSION more
        # pixel frames) so that last keyframe becomes INTERIOR — it regains a moving neighbor and flows
        # through append-token like any interior pin — then trim the padded tail back off before export.
        # Only engages for append-token last-frame i2v; a no-op (num_frames_out == num_frames) otherwise.
        append_token = os.environ.get("LTX_KF_APPEND_TOKEN", "0") in ("1", "true", "True")
        num_frames_out = num_frames
        if images and append_token:
            _last_lat = (num_frames - 1) // TEMPORAL_COMPRESSION  # index of the last latent frame
            # A keyframe worker (LTX_KF_TRACE_PAD) always pads so every keyframe gen matches its one
            # padded trace; the eager path pads only when a keyframe actually lands on the last frame.
            _always = os.environ.get("LTX_KF_TRACE_PAD", "0") in ("1", "true", "True")
            _last_kf = _last_lat > 0 and any(pixel_to_latent_frame(im[1], num_frames) == _last_lat for im in images)
            if _always or _last_kf:
                num_frames += TEMPORAL_COMPRESSION
                logger.info(
                    f"append-token tail-pad: generating {num_frames}f, trimming to {num_frames_out}f "
                    f"({'keyframe-worker' if _always else 'last-frame keyframe'} → interior)"
                )

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

        if not images and reference_video is None:
            # Plain T2V knows every seeded draw up front: stage 1 video + audio, stage 2 video.
            C = self.in_channels
            s1_n, s2_n = (math.prod(latent_grid(num_frames, h, w)) for h, w in ((s1_height, s1_width), (height, width)))
            audio_n = AudioLatentShape.from_video_pixel_shape(
                VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=self.fps)
            ).frames
            self._prefetch_noise(seed, [((1, s1_n, C), (1, audio_n, C)), ((1, s2_n, C),)])

        # (label, seconds) rows counted toward the total; prepares and export are excluded.
        timings: list[tuple[str, float]] = []

        t0 = time.time()
        # A served request encodes a prompt nothing has seen, so a cache hit would drop the encoder
        # out of the reported total and out of the traced path it belongs in. dynamic_load keeps the
        # cache: there the encoder is coresident-excluded from the DiT, so re-encoding every request
        # would reload it and evict the captured model state.
        _embed_path = self._device_embed_cache_path([prompt]) if self.dynamic_load else None
        cached = _embed_path is not None and os.path.exists(_embed_path)
        if not cached:
            self.gemma_encoder_pair.ensure_loaded()
        if self._device_prompt_handoff:
            self._allocate_device_prompt_buffers()
            self.gemma_encoder_pair.encode_to_device_buffers(prompt, self._prompt_v.value, self._prompt_a.value)
            v_embeds = a_embeds = None
        else:
            enc = self.encode_prompts([prompt], use_cache=self.dynamic_load)
            v_embeds, a_embeds = enc[0][0].float(), enc[0][1].float()
        t_encode = time.time() - t0

        def _stats(label, t):
            # LTX_DEBUG_STATS=1: host-side summary per stage, to tell a dead encoder from a dead denoise
            # from a dead decode when the output is flat — a grey mp4 carries no trace of where the
            # signal was lost.
            if os.environ.get("LTX_DEBUG_STATS", "0") != "1" or not torch.is_tensor(t):
                return
            f = t.detach().float()
            finite = f[torch.isfinite(f)]
            logger.info(
                f"STATS {label}: shape={tuple(f.shape)} finite_abs_mean={finite.abs().mean():.4g} "
                f"finite_max={finite.abs().max():.4g} nan_frac={torch.isnan(f).float().mean():.3f} "
                f"inf_frac={torch.isinf(f).float().mean():.3f} zero_frac={(f == 0).float().mean():.3f}"
            )

        _stats("v_embeds", v_embeds)
        _stats("a_embeds", a_embeds)
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
                # Softening blunts an IN-PLACE pin's wall against its two-sided neighbours. An
                # append-token keyframe never pins the grid — the frame stays free and converges onto a
                # HELD anchor — so softening only corrupts the reference itself: the anchor is denoised
                # by (1 - strength), and at the coarse 720p S1 (0.25) it lands three-quarters noise, so
                # every frame that converges on it decodes to mush. Anchors stay hard.
                if kf_fix and lat_idx != 0 and not append_token:
                    s1s = min(s1s, _keyframe_s1_strength(s1_height))  # soften non-frame-0 pin; s2 re-locks
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
            # Keyframe worker: the trace bakes a fixed anchor count, so a job with fewer non-frame-0
            # keyframes pads up to it with duplicates of its last anchor (same frame + latent). A
            # duplicate anchor only reinforces its own keyframe, so it stays inert while matching shape.
            _baked = len(self._kf_trace_anchors())
            if _baked:

                def _pad_anchors(conds):
                    nz = [c for c in conds if c[0] != 0]
                    return conds + [nz[-1]] * (_baked - len(nz)) if 0 < len(nz) < _baked else conds

                s1_conds, full_conds = _pad_anchors(s1_conds), _pad_anchors(full_conds)
            s1_image_conds, full_image_conds = s1_conds, full_conds

        # ----- IC-LoRA reference sheet: loop -> encode per stage -> appended clean block -----
        # The still is looped to a static reference video and VAE-encoded through the multi-frame
        # reference encoders (the existing self.vae_encoder is single-frame image-i2v), producing the
        # half-res (s1) and full-res (s2) reference latents appended clean in _denoise_no_guidance.
        ref_latent_s1 = ref_latent_full = None
        ref_strength = 1.0
        if reference_video is not None:
            assert self.vae_encoder is not None, "checkpoint has no VAE encoder; cannot run IC-LoRA reference"
            ref_path, ref_strength = reference_video
            self._build_ref_vae_encoders(ref_pixel_frames, height, width)
            t0 = time.time()
            # Memoized like _i2v_cond_cache, and for the same reason: the VAE encoder is eager, and an
            # encode issued AFTER a denoise trace replay deadlocks the device in the encoder's
            # gathered read-back (synchronize_device times out; chips drop off the PCIe bus). A
            # long-lived worker replays traces on its first gen, so an un-cached re-encode kills its
            # second reference job. The encode is deterministic in (path, R, resolution).
            key = (self._sheet_digest(ref_path), ref_pixel_frames, height, width)
            cached = self._ref_latent_cache.get(key)
            if cached is not None:
                ref_latent_s1, ref_latent_full = cached
            else:
                ref_clip_s1 = self._load_reference_video(ref_path, s1_height, s1_width, ref_pixel_frames)
                ref_clip_full = self._load_reference_video(ref_path, height, width, ref_pixel_frames)
                ref_latent_s1 = self.encode_image(ref_clip_s1, encoder=self.vae_ref_encoder_s1)
                ref_latent_full = self.encode_image(ref_clip_full, encoder=self.vae_ref_encoder_full)
                self._ref_latent_cache[key] = (ref_latent_s1, ref_latent_full)
            timings.append(("Reference encode", time.time() - t0))
            logger.info(
                f"IC-LoRA reference {ref_path}: {ref_pixel_frames} looped px frames -> "
                f"{tuple(ref_latent_full.shape)} latent (strength={ref_strength}) "
                f"{'(cached)' if cached is not None else ''}in {time.time() - t0:.1f}s"
            )

        # IC-LoRA runs on a wider (target+reference) sequence, so it captures / replays its OWN trace
        # family (s1_ref/s2_ref) and never collides with the narrower i2v/t2v traces (which bake the
        # target-only addresses). trace_key stays s1/s2 for every non-reference gen (byte-identical).
        has_ref = reference_video is not None
        s1_trace_key = "s1_ref" if has_ref else "s1"
        s2_trace_key = "s2_ref" if has_ref else "s2"

        t0 = time.time()
        self._prepare_transformer(0)
        if self.dynamic_load:
            logger.info(f"Transformer prepare: {time.time() - t0:.1f}s")

        logger.info(
            f"Stage 1: {s1_height}x{s1_width}, {len(s1_sigmas) - 1} steps"
            + (" [interior-keyframe aligned]" if s1_sigmas is not DISTILLED_SIGMA_VALUES else "")
        )
        t0 = time.time()
        # Reference gens keep the host round-trip: their combined (target+reference) rows are sliced on the
        # host and the device transition models the target grid only.
        device_resident = self._device_resident(images) and not has_ref
        s1_traced = self._traced and "s1" not in eager_stages
        s2_traced = self._traced and "s2" not in eager_stages
        s1_video, s1_audio = self._denoise_no_guidance(
            v_embeds,
            a_embeds,
            num_frames=num_frames,
            height=s1_height,
            width=s1_width,
            sigma_values=s1_sigmas,
            seed=seed,
            image_conds=s1_image_conds,
            ref_latent=ref_latent_s1,
            ref_strength=ref_strength,
            traced=s1_traced,
            trace_key=s1_trace_key,
            profile_drain=True,
            return_device_video=device_resident,
            device_prompts=self._device_prompt_handoff,
        )
        self._sync_for_stage_timing(device_resident)  # device-resident S1 replays do not block
        t_stage1 = time.time() - t0
        if not device_resident:
            _stats("s1_video", s1_video)
        timings.append(("Stage 1 denoise", t_stage1))
        logger.info(f"Stage 1 denoise: {t_stage1:.1f}s")
        _h0 = time.perf_counter()
        self._log_latent_stats("s1", s1_video, s1_audio)
        _h1 = time.perf_counter()

        latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
        s1_h, s1_w = s1_height // SPATIAL_COMPRESSION, s1_width // SPATIAL_COMPRESSION
        upsampled_flat = upsampled_dev = None
        t0 = time.time()
        self._ensure_upsampler_frames(num_frames)  # tail-pad upsamples one extra latent frame
        self._prepare_upsampler()
        _h2 = time.perf_counter()
        if device_resident:
            upsampled_dev = self._run_stage_transition(
                s1_video,
                num_frames=num_frames,
                s1_height=s1_height,
                s1_width=s1_width,
                height=height,
                width=width,
                seed=seed,
                traced=self._traced and "s2" not in eager_stages,
            )
            if upsampled_dev is None:  # host round-trip for this gen
                device_resident = False
                s1_video = LTXTransformerModel.device_to_host(
                    s1_video,
                    ccl_manager=self.ccl_manager,
                    parallel_config=self.parallel_config,
                    sp_already_gathered=False,
                    tp_already_gathered=True,
                ).squeeze(0)[:, : latent_frames * s1_h * s1_w, :]
        if not device_resident:
            s1_spatial = s1_video.reshape(1, latent_frames, s1_h, s1_w, 128).permute(0, 4, 1, 2, 3)
            upsampled = upsample_latent(self.upsampler, s1_spatial, *self._vae_per_channel_stats())
            _stats("upsampled", upsampled)
            upsampled_flat = upsampled.permute(0, 2, 3, 4, 1).reshape(
                1, latent_frames * (height // SPATIAL_COMPRESSION) * (width // SPATIAL_COMPRESSION), 128
            )
        self._sync_for_stage_timing(device_resident)  # the eager transition is async; charge it here, not to stage 2
        t_upsample = time.time() - t0
        timings.append(("Latent upsample", t_upsample))
        logger.info(f"Latent upsample: {t_upsample:.1f}s")
        _h3 = time.perf_counter()
        # Host wall only: a device-resident transition is enqueued here and its device time lands in
        # stage 2 unless LTX_TIME_STAGES syncs above.
        logger.info(
            f"S1->S2 hand-off: {(_h3 - _h0) * 1000:.1f} ms [stats {(_h1 - _h0) * 1000:.1f}, "
            f"upsampler prep {(_h2 - _h1) * 1000:.1f}, upsample {(_h3 - _h2) * 1000:.1f}"
            f"{' (device-resident)' if device_resident else ''}]"
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
            initial_video_latent_device=upsampled_dev,
            initial_audio_latent=s1_audio.unsqueeze(0) if s1_audio.dim() == 2 else s1_audio,
            device_prompts=self._device_prompt_handoff,
            reuse_prompt=s1_traced and s2_traced and os.environ.get("LTX_S2_PROMPT_REUSE", "1") == "1",
            image_conds=full_image_conds,
            ref_latent=ref_latent_full,
            ref_strength=ref_strength,
            traced=s2_traced,
            trace_key=s2_trace_key,
            profile_drain=True,
        )
        t_stage2 = time.time() - t0
        _stats("s2_video", s2_video)
        timings.append(("Stage 2 denoise", t_stage2))
        logger.info(f"Stage 2 denoise: {t_stage2:.1f}s")
        # Last point before the VAE: if this latent is white or partly unwritten, the
        # decode cannot recover and the served MP4 will be noise.
        self._log_latent_stats("s2", s2_video, s2_audio, warn=True)

        dump_prefix = os.environ.get("LTX_DUMP_LATENTS", "")
        if dump_prefix:
            # The denoised S2 latents are the last point where a transformer-only change is still
            # isolated — VAE/vocoder decode below would fold their own error in on top. One file per
            # generate() call (traced runs generate twice: capture, then steady-state replay), so an
            # A/B compares like with like instead of racing on one path.
            global _LATENT_DUMP_N  # noqa: PLW0603
            path = f"{dump_prefix}.gen{_LATENT_DUMP_N}.pt"
            _LATENT_DUMP_N += 1
            # Provenance travels inside the artifact: the flags are read back off the module that
            # actually ran, not off the environment. An A/B is only worth its weakest claim about
            # which code produced each side, and an env echo in a log does not survive the tree
            # being rebuilt underneath it.
            from ...models.transformers.ltx import transformer_ltx as _tx

            flags = {
                "LTX_FOLD_GATED_RESIDUAL": _tx.LTX_FOLD_GATED_RESIDUAL,
                "LTX_PROBE_ADDCMUL_SPLIT": _tx.LTX_PROBE_ADDCMUL_SPLIT,
                "seed": seed,
            }
            torch.save(
                {
                    "video": s2_video.detach().cpu().float(),
                    "audio": s2_audio.detach().cpu().float(),
                    "flags": flags,
                },
                path,
            )
            logger.info(
                f"LTX_DUMP_LATENTS: wrote video {tuple(s2_video.shape)} + audio "
                f"{tuple(s2_audio.shape)} to {path} with {flags}"
            )

        if os.environ.get("LTX_PROFILE_DENOISE_ONLY", "0") in ("1", "true", "True"):
            # Profiling-only: stop before VAE/audio so the device reservation is spent on the traced
            # denoise, which is the only thing STEP_MS scores.
            self.last_timings = list(timings)
            for _label, _secs in timings:
                if "denoise" in _label.lower():
                    walltime.record("gen", _label, _secs)
            logger.info(f"LTX_PROFILE_DENOISE_ONLY=1: denoise-only total {sum(s for _, s in timings):.1f}s")
            return None, None

        t0 = time.time()
        self._ensure_vae_decoder_frames(num_frames)  # tail-pad decodes one extra latent frame
        self._prepare_vae()
        if self.dynamic_load:
            logger.info(f"VAE prepare: {time.time() - t0:.1f}s")

        latent_h, latent_w = height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION
        # LTX_YUV_EXPORT routes the mp4 path through the on-device YUV 4:2:0 fast gather
        # Default: convert RGB -> yuv420p uint8 on device and hand libx264 native frames. The mp4 is yuv420p
        # either way, so fidelity is unchanged; what moves is ~1.8 GB of bf16 planes per 6 s clip that no
        # longer cross PCIe (yuv420p is 1.5 B/px vs 6 B/px) and the host-side float->uint8 conversion.
        # LTX_YUV_EXPORT=0 restores the float readback + host conversion.
        # DiffVAE without device boundaries returns host pixels and takes the float path.
        yuv_export = (
            output_path is not None
            and os.environ.get("LTX_YUV_EXPORT", "1") != "0"
            and getattr(self.vae_decoder, "supports_yuv", True)
        )
        # export_video_audio needs float [-1,1]; the frame-return path uses the requested output_type.
        # The raw-frame dump reads RGB float, so it forces the float readback.
        dump_path = os.environ.get("LTX_DUMP_FRAMES", "").strip()
        yuv_export = yuv_export and not dump_path
        decode_type = ("yuv" if yuv_export else "float") if output_path is not None else output_type
        # Roots recorded before this call belong to warm-up or an earlier gen; keep only what THIS
        # decode records so the perf table's breakdown never describes a different pass.
        roots_before = timing_tree.root_count()
        t0 = time.time()
        video_pixels = self.decode_latents(s2_video, latent_frames, latent_h, latent_w, output_type=decode_type)
        if num_frames_out != num_frames:  # drop the tail-pad frame(s): (T,H*3/2,W) yuv planes or (B,3,F,H,W) float
            video_pixels = video_pixels[:num_frames_out] if yuv_export else video_pixels[:, :, :num_frames_out]
        t_vae_decode = time.time() - t0
        _stats("video_pixels", video_pixels)
        timings.append(("VAE decode", t_vae_decode))
        new_roots = timing_tree.roots()[roots_before:]
        self.last_decode_tree = new_roots[-1] if new_roots else None
        logger.info(f"VAE decode (forward): {t_vae_decode:.1f}s — {tuple(video_pixels.shape)}")

        # LTX_DUMP_FRAMES writes the exact (F,H,W,C) uint8 tensor the H.264 export would quantize.
        # A pixel-level A/B read back from the mp4 measures the codec as much as the model; this
        # lands the frames losslessly so the comparison sees only the model. Out-of-place: float()
        # is a no-op view when video_pixels is already float32, so an in-place rescale here would
        # corrupt the tensor the export below still needs.
        if dump_path:
            frames_u8 = ((video_pixels[0].float() + 1.0) * 127.5).clamp(0.0, 255.0)
            torch.save(frames_u8.to(torch.uint8).permute(1, 2, 3, 0).contiguous().cpu(), dump_path)
            logger.info(f"LTX_DUMP_FRAMES: wrote raw frames to {dump_path}")

        # The vocoder decodes s2_audio, a latent distinct from the s2_video that produced video_pixels
        # above, so LTX_VIDEO_ONLY cannot perturb a single pixel — it only drops the eager vocoder
        # decode, which is trace-hidden in production but costs minutes untraced.
        video_only = os.environ.get("LTX_VIDEO_ONLY", "0") in ("1", "true", "True")
        # The host libx264 encode of the video track (~0.7 s at 1080p/145f) runs on a worker thread
        # under the device audio decode; the mp4 is byte-identical to the serial export.
        # LTX_ASYNC_EXPORT=0 restores the serial export.
        video_export = None
        if yuv_export and os.environ.get("LTX_ASYNC_EXPORT", "1") != "0":
            rate = None if video_only else self.tt_vocoder_with_bwe.output_sampling_rate
            video_export = YuvVideoExport(video_pixels, output_path, fps=fps, audio_sampling_rate=rate)
        t0 = time.time()
        try:
            if video_only:
                audio_obj = None
                logger.info("LTX_VIDEO_ONLY=1: skipping audio decode (video pixels unaffected)")
            else:
                audio_obj = self.decode_audio(s2_audio, num_frames_out, fps=fps)  # trim padded waveform to length
        except BaseException:
            if video_export is not None:
                try:
                    video_export.finish(None)
                except Exception:
                    pass  # surface the decode error, not the aborted export's
            raise
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
        if video_export is not None:
            video_export.finish(audio_obj)
        elif yuv_export:
            export_video_audio_yuv(video_pixels, output_path, fps=fps, audio=audio_obj)
        else:
            export_video_audio(video_pixels, output_path, fps=fps, audio=audio_obj)
        logger.info(f"Video export: {time.time() - t0:.1f}s")
        logger.info(f"Total (compute): {sum(s for _, s in timings):.1f}s | Output: {output_path}")
        # Every trace this pipeline takes is captured by now, so the encoder can start tracing: its
        # capture is last and nothing left will reclaim its activation region. Capturing it here,
        # after this request's output is written, keeps it off the next request's path.
        if self._traced and not self.dynamic_load:
            self.gemma_encoder_pair.open_trace_gate()
            self.gemma_encoder_pair.capture_trace()
        return output_path

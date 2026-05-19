# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ACE-Step v1.5 model: text encoder → DiT (TTNN) → VAE decoder.

This module wraps the full inference pipeline into a single class so that
callers only need to supply a text prompt and get a waveform tensor back.

Neural inference (Qwen3 embeddings, instrumental condition, DiT, Oobleck VAE) runs
on Tenstorrent via TTNN. Encoder SDPA masks are uploaded once per prompt via
:class:`AceStepV15TTNNPipeline.build_encoder_attention_mask_b1qk_optional` (no
per-step mask rebuild). Context latents from the silence template are cached on
device at init. DiT latents feed the VAE without a host round trip when using the
default path. Only Hugging Face **tokenization** (string → token ids) and
checkpoint **loading** (host files → device weights) use the CPU; optional
waveform peak-normalization can return a TTNN tensor via *return_waveform_ttnn*.

The standalone function :func:`run_ttnn_denoise_loop` is exported for reuse by
other scripts (e.g. the prompt-to-wav demo). For PyTorch VAE decode (e.g.
``--torch-vae`` in the demo), import :func:`decode_with_vae` from
``torch_ref.e2e_model``.

Performance profiling (Tracy / device profiler):

    TTNN_OP_PROFILER=1 is set when running ``python -m tracy -p -r -v -m pytest ...``
    (see ``tools/tracy``). Signposts label stages in the Tracy timeline and op CSV.

    For kernel-level CSV rows from the device, also set ``TT_METAL_DEVICE_PROFILER=1``.
    After the denoising loop and again after VAE decode, ``ttnn.synchronize_device`` and
    ``ttnn.ReadDeviceProfiler`` run when either env var is set so buffered device profiler data is
    flushed at stage boundaries (DiT vs VAE show up cleanly in reports).

Optional: set ``ACE_STEP_TRACY_EACH_DENOISE_STEP=1`` to emit one Tracy signpost per Euler step
(use shorter ``infer_steps`` if the report becomes too dense).

Optional: set ``ACE_STEP_USE_TRACE=1`` to wrap the per-step DiT body
(``pipe.forward_with_temb_tp``) in a captured TTNN trace + 2CQ replay. The wrapping covers
patch_embed + DiT core + output_head — the steady-state bulk of each Euler step. The pre-DiT
prep (``fp32_tile_to_row_bf16``, CFG batch-duplicate), the post-DiT CFG split, the
APG/ADG guidance, and the Euler subtract stay eager because they depend on per-step Python
scalars (``t_curr``, ``euler_dt``) that would otherwise be baked into the captured graph. The
trace is captured once per :class:`AceStepE2EModel` instance after two eager warmup steps in
the first ``generate(...)`` call, then replayed for every remaining step in that generate and
all subsequent generates of the same shape. Requires the device fixture in
``perf/conftest.py`` (2 CQs + 128 MB trace region, both enabled by default).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch

import ttnn

from .condition_encoder import TtAceStepInstrumentalConditionEncoder
from .dit_sampling_ttnn import (
    TtnnMomentumBufferApg,
    adg_guidance_velocity_ttnn,
    apg_guidance_velocity_ttnn,
    bf16_row_from_numpy_bc,
    concat_duplicate_batch,
    euler_subtract_v_dt,
    fp32_tile_to_row_bf16,
    slice_batch_btc,
    typecast_bf16_any_to_fp32_tile,
)
from .full_pipeline import AceStepV15TTNNPipeline
from .oobleck_vae_decoder import TtOobleckVaeDecoder
from .qwen3_embedding_ace_step import AceStepQwen3Encoder as TtQwen3EmbeddingEncoder


def _ace_step_prof_signpost(header: str, message: Optional[str] = None) -> None:
    """Emit a Tracy signpost when the repo ``tracy`` package is available (editable install)."""
    try:
        from tracy import signpost as _signpost  # type: ignore[import-untyped]
    except ImportError:
        return
    try:
        if message is None:
            _signpost(header)
        else:
            _signpost(header, message)
    except Exception:
        pass


def _ace_step_flush_device_profiler(device) -> None:
    """Sync and drain device profiler buffers when Tracy / device profiling is enabled.

    Also no-op when ``ACE_STEP_USE_TRACE=1`` so the per-step flush in
    :func:`run_ttnn_denoise_loop` doesn't try to call ``ttnn.synchronize_device`` between trace
    executions. The per-RISC profiler ring buffer fills up under tracing anyway (replays don't
    flush mid-graph), so the profiler and the e2e trace are mutually exclusive — Tracy users
    should run without ``ACE_STEP_USE_TRACE``.
    """
    if os.environ.get("TTNN_OP_PROFILER") != "1" and os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return
    if os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes"):
        return
    try:
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass


@dataclass
class E2EConfig:
    """Configuration for the end-to-end pipeline."""

    checkpoint_safetensors_path: str
    vae_dir: str
    text_model_dir: str
    silence_latent_path: str

    duration_sec: float = 10.0
    infer_steps: int = 50
    shift: float = 1.0
    guidance_scale: float = 7.0
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    use_adg: bool = True
    seed: int = 0
    sample_rate: int = 48000
    qwen_safetensors_path: Optional[str] = None
    vae_chunk_latents: int = 32
    vae_overlap_latents: int = 4


def _build_t_schedule(
    *,
    shift: float,
    infer_steps: int,
) -> List[float]:
    """Build the diffusion timestep schedule."""
    t = [float(i) / float(infer_steps) for i in range(infer_steps, -1, -1)]
    if shift != 1.0:
        s = float(shift)
        t = [s * x / (1.0 + (s - 1.0) * x) for x in t]
    return t


# ---------------------------------------------------------------------------
# Standalone helpers – importable by demo scripts to avoid code duplication.
# ---------------------------------------------------------------------------


def to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a contiguous float32 numpy array on CPU."""
    return t.detach().to(dtype=torch.float32).cpu().contiguous().numpy()


def _e2e_trace_enabled() -> bool:
    """Return True when ``ACE_STEP_USE_TRACE`` opts the E2E denoise loop into trace+2CQ."""
    return os.environ.get("ACE_STEP_USE_TRACE", "").lower() in ("1", "true", "yes")


class _E2EDenoiseTrace:
    """Persistent TTNN trace handle for the per-step DiT body inside :func:`run_ttnn_denoise_loop`.

    Wraps ``pipe.forward_with_temb_tp`` (patch_embed + DiT core + output_head) in a single
    captured TTNN trace with persistent input/output buffers. Captured once on the first
    :meth:`AceStepE2EModel.generate` call after two eager warmup Euler steps, then replayed
    for every step thereafter. Across generates of the same prompt-shape, per-prompt inputs
    (encoder hidden states, context latents, optional cross-attention SDPA mask) are streamed
    into the same persistent buffers via :meth:`prime_per_prompt`, and per-step inputs
    (current ``xt``, ``temb``, ``timestep_proj``) are streamed in via :meth:`replay`.

    Trace covers only the DiT body; the pre-DiT BF16 row build + CFG concat and the post-DiT
    slice + APG/ADG guidance + Euler subtract remain eager because they consume per-step
    Python scalars (``t_curr``, ``euler_dt``) that would bake into the captured graph.

    Buffer ownership: all persistent tensors below are owned by this object; the surrounding
    ``run_ttnn_denoise_loop`` MUST NOT call ``ttnn.deallocate`` on them. Call :meth:`release`
    to free them and the trace id (e.g. when ``AceStepE2EModel`` is destroyed or when the
    denoise shape changes and a re-capture is needed).
    """

    __slots__ = (
        "tid",
        "xt_buf",
        "temb_buf",
        "tp_buf",
        "enc_buf",
        "ctx_buf",
        "mask_buf",
        "acoustic_buf",
        "frames",
        "pipe_batch",
        "c_lat",
        "do_cfg",
        "output_addr",
    )

    def __init__(self) -> None:
        self.tid = None
        self.xt_buf: Optional[ttnn.Tensor] = None
        self.temb_buf: Optional[ttnn.Tensor] = None
        self.tp_buf: Optional[ttnn.Tensor] = None
        self.enc_buf: Optional[ttnn.Tensor] = None
        self.ctx_buf: Optional[ttnn.Tensor] = None
        self.mask_buf: Optional[ttnn.Tensor] = None
        self.acoustic_buf: Optional[ttnn.Tensor] = None
        self.frames: Optional[int] = None
        self.pipe_batch: Optional[int] = None
        self.c_lat: Optional[int] = None
        self.do_cfg: Optional[bool] = None
        self.output_addr: Optional[int] = None

    def is_ready(self) -> bool:
        """True when a trace id is currently installed (post-capture, pre-release)."""
        return self.tid is not None

    def has_buffers(self) -> bool:
        """True when persistent input/output buffers exist (independent of trace id state).

        After :meth:`release_trace_only`, ``has_buffers()`` stays True so the next denoise loop
        can call :meth:`recapture` without re-cloning every input.
        """
        return all(
            getattr(self, attr) is not None
            for attr in ("xt_buf", "temb_buf", "tp_buf", "enc_buf", "ctx_buf", "acoustic_buf")
        )

    def matches_shape(self, *, frames: int, pipe_batch: int, c_lat: int, do_cfg: bool) -> bool:
        """Return True iff persistent buffers were sized for the given denoise loop shape.

        Checks the cached ``(frames, pipe_batch, c_lat, do_cfg)`` recorded in :meth:`capture`,
        independent of whether the trace id is currently installed — used by ``generate()`` to
        decide whether to re-prime the existing buffers or :meth:`release` and rebuild from
        scratch when shape/CFG/mask-presence changes.
        """
        return (
            self.has_buffers()
            and self.frames == int(frames)
            and self.pipe_batch == int(pipe_batch)
            and self.c_lat == int(c_lat)
            and self.do_cfg == bool(do_cfg)
        )

    def prime_per_prompt(
        self,
        *,
        enc_tt_pipe: ttnn.Tensor,
        ctx_tt_pipe: ttnn.Tensor,
        mask_tt: Optional[ttnn.Tensor],
    ) -> None:
        """Copy fresh per-prompt inputs into the persistent buffers (call once per ``generate``).

        Source tensors must match the buffer shapes/dtypes captured in :meth:`capture`. They
        are read-only from this method's perspective; the caller can deallocate them after
        the call (the trace replays from the persistent copies, not the originals).

        Works as long as persistent buffers exist — the trace id may already be released
        (between generate calls); the next ``run_ttnn_denoise_loop`` will :meth:`recapture`
        before any replay.
        """
        if not self.has_buffers():
            raise RuntimeError("prime_per_prompt() called before capture() (no persistent buffers)")
        ttnn.copy(enc_tt_pipe, self.enc_buf)
        ttnn.copy(ctx_tt_pipe, self.ctx_buf)
        if self.mask_buf is not None:
            if mask_tt is None:
                raise RuntimeError(
                    "Trace was captured with an encoder SDPA mask but prime_per_prompt() got mask_tt=None. "
                    "All keys would now be valid, but the captured graph still references the mask buffer."
                )
            ttnn.copy(mask_tt, self.mask_buf)
        elif mask_tt is not None:
            raise RuntimeError(
                "Trace was captured without an encoder SDPA mask (all keys valid) but prime_per_prompt() "
                "got a non-None mask_tt for the new prompt. The captured graph does not reference any "
                "mask buffer; call release() and let the next generate() re-capture for this prompt."
            )

    def capture(
        self,
        *,
        pipe,
        device,
        xt_pipe_in: ttnn.Tensor,
        temb: ttnn.Tensor,
        tp: ttnn.Tensor,
        enc_tt_pipe: ttnn.Tensor,
        ctx_tt_pipe: ttnn.Tensor,
        encoder_attention_mask_b1qk: Optional[ttnn.Tensor],
        frames: int,
        pipe_batch: int,
        c_lat: int,
        do_cfg: bool,
    ) -> ttnn.Tensor:
        """Allocate persistent buffers, capture the DiT body trace, return ``acoustic_buf``.

        Steps:
            1. Clone every input tensor into a fresh persistent device buffer (``ttnn.clone``).
            2. Run one eager forward through those buffers to make sure every program-cache
               entry the trace will reference is already device-resident.
            3. ``begin_trace_capture`` / ``forward_with_temb_tp`` / ``end_trace_capture`` on CQ 0.
            4. Record the output buffer address; assert stability across subsequent
               :meth:`replay` calls via the trace.

        Inputs are NOT deallocated here — the caller owns them. The persistent clones live on
        ``self`` until :meth:`release`.
        """
        if self.is_ready():
            raise RuntimeError("capture() called on an already-captured trace; call release() first.")

        self.xt_buf = ttnn.clone(xt_pipe_in)
        self.temb_buf = ttnn.clone(temb)
        self.tp_buf = ttnn.clone(tp)
        self.enc_buf = ttnn.clone(enc_tt_pipe)
        self.ctx_buf = ttnn.clone(ctx_tt_pipe)
        self.mask_buf = ttnn.clone(encoder_attention_mask_b1qk) if encoder_attention_mask_b1qk is not None else None

        warm_out = pipe.forward_with_temb_tp(
            xt_bt64=self.xt_buf,
            context_latents_bt128=self.ctx_buf,
            encoder_hidden_states_btd=self.enc_buf,
            temb_bd=self.temb_buf,
            timestep_proj_b6d=self.tp_buf,
            attention_mask_1d_bt=None,
            encoder_attention_mask_1d_bk=None,
            encoder_attention_mask_b1qk=self.mask_buf,
        )
        ttnn.synchronize_device(device)
        try:
            ttnn.deallocate(warm_out)
        except Exception:
            pass

        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.acoustic_buf = pipe.forward_with_temb_tp(
            xt_bt64=self.xt_buf,
            context_latents_bt128=self.ctx_buf,
            encoder_hidden_states_btd=self.enc_buf,
            temb_bd=self.temb_buf,
            timestep_proj_b6d=self.tp_buf,
            attention_mask_1d_bt=None,
            encoder_attention_mask_1d_bk=None,
            encoder_attention_mask_b1qk=self.mask_buf,
        )
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        ttnn.synchronize_device(device)

        self.frames = int(frames)
        self.pipe_batch = int(pipe_batch)
        self.c_lat = int(c_lat)
        self.do_cfg = bool(do_cfg)
        try:
            self.output_addr = int(self.acoustic_buf.buffer_address())
        except Exception:
            self.output_addr = None
        return self.acoustic_buf

    def replay(
        self,
        *,
        device,
        xt_pipe_in: ttnn.Tensor,
        temb: ttnn.Tensor,
        tp: ttnn.Tensor,
        op_event,
    ) -> tuple[ttnn.Tensor, object]:
        """Stream per-step inputs onto CQ 1, run ``execute_trace`` on CQ 0, return ``(acoustic_buf, op_event)``.

        Caller passes the previous ``op_event`` so the host-side writes wait for the prior
        trace execution to finish before clobbering the persistent input buffers (otherwise we
        would race the trace's reads). Caller owns ``xt_pipe_in``; this method does NOT free it.
        """
        if not self.is_ready():
            raise RuntimeError("replay() called before capture()")
        ttnn.wait_for_event(1, op_event)
        ttnn.copy(xt_pipe_in, self.xt_buf)
        ttnn.copy(temb, self.temb_buf)
        ttnn.copy(tp, self.tp_buf)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        ttnn.execute_trace(device, self.tid, cq_id=0, blocking=False)
        op_event = ttnn.record_event(device, 0)
        return self.acoustic_buf, op_event

    def release_trace_only(self, device) -> None:
        """Release the captured trace id but keep every persistent buffer alive.

        Call at the END of each denoise loop so the Metal allocator goes back to
        :func:`mark_allocations_safe` mode before the surrounding ``generate()`` decodes the
        VAE / runs the next prompt's encoders (both allocate fresh buffers). Without this, the
        allocator stays in :func:`mark_allocations_unsafe` mode across generates and the trace
        will silently corrupt eager allocations that happen to land in trace-reserved memory
        regions (Whisper demo hits the same trap — see
        ``models/demos/audio/whisper/tt/whisper_generator.py::_release_decode_traces_for_safe_alloc``).

        The persistent buffers (``xt_buf`` / ``temb_buf`` / ``tp_buf`` / ``enc_buf`` / ``ctx_buf``
        / ``mask_buf`` / ``acoustic_buf``) survive — :meth:`recapture` reuses them.
        """
        if self.tid is not None:
            try:
                ttnn.release_trace(device, self.tid)
            except Exception:
                pass
            self.tid = None
        self.output_addr = None

    def recapture(
        self,
        *,
        pipe,
        device,
    ) -> ttnn.Tensor:
        """Re-capture the DiT body trace against the existing persistent buffers.

        Used at the start of every denoise loop after the first one: previous loop released the
        trace id via :meth:`release_trace_only` so eager VAE / encoder allocations stayed safe;
        now we re-arm the trace before stepping the new denoise loop. The persistent buffers
        already hold this prompt's contents (caller ran :meth:`prime_per_prompt` + step 0's
        ``xt``/``temb``/``tp`` will be streamed in via :meth:`replay` like every other step), so
        no re-clone is needed.

        The warm forward is still run because, although the program cache is hot from the
        previous loop's replays, some kernel paths may have device-resident scratch space
        torn down by the previous :func:`release_trace` (mirrors the warmup pass in
        ``test_pipe_body_trace_2cq.py``).
        """
        if not self.has_buffers():
            raise RuntimeError("recapture() called without persistent buffers; use capture() first")
        if self.is_ready():
            raise RuntimeError(
                "recapture() called while a trace id is still installed; call release_trace_only() first"
            )

        warm_out = pipe.forward_with_temb_tp(
            xt_bt64=self.xt_buf,
            context_latents_bt128=self.ctx_buf,
            encoder_hidden_states_btd=self.enc_buf,
            temb_bd=self.temb_buf,
            timestep_proj_b6d=self.tp_buf,
            attention_mask_1d_bt=None,
            encoder_attention_mask_1d_bk=None,
            encoder_attention_mask_b1qk=self.mask_buf,
        )
        ttnn.synchronize_device(device)
        try:
            ttnn.deallocate(warm_out)
        except Exception:
            pass

        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        # Re-bind ``acoustic_buf`` to the freshly captured trace's output — the previous output
        # buffer was implicitly released along with the trace id, so the address may differ.
        self.acoustic_buf = pipe.forward_with_temb_tp(
            xt_bt64=self.xt_buf,
            context_latents_bt128=self.ctx_buf,
            encoder_hidden_states_btd=self.enc_buf,
            temb_bd=self.temb_buf,
            timestep_proj_b6d=self.tp_buf,
            attention_mask_1d_bt=None,
            encoder_attention_mask_1d_bk=None,
            encoder_attention_mask_b1qk=self.mask_buf,
        )
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        ttnn.synchronize_device(device)
        try:
            self.output_addr = int(self.acoustic_buf.buffer_address())
        except Exception:
            self.output_addr = None
        return self.acoustic_buf

    def release(self, device) -> None:
        """Free the captured trace + every persistent buffer; safe to call repeatedly."""
        if self.tid is not None:
            try:
                ttnn.release_trace(device, self.tid)
            except Exception:
                pass
            self.tid = None
        for attr in ("xt_buf", "temb_buf", "tp_buf", "enc_buf", "ctx_buf", "mask_buf", "acoustic_buf"):
            buf = getattr(self, attr, None)
            if buf is not None:
                try:
                    ttnn.deallocate(buf)
                except Exception:
                    pass
                setattr(self, attr, None)
        self.frames = None
        self.pipe_batch = None
        self.c_lat = None
        self.do_cfg = None
        self.output_addr = None


def run_ttnn_denoise_loop(
    pipe: AceStepV15TTNNPipeline,
    device: ttnn.Device,
    act_dtype: ttnn.DataType,
    mem: ttnn.MemoryConfig,
    t_schedule: List[float],
    frames: int,
    enc_hs: Optional[torch.Tensor] = None,
    enc_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ctx_lat: Optional[torch.Tensor] = None,
    null_emb: Optional[torch.Tensor] = None,
    do_cfg: bool = False,
    seed: int = 0,
    *,
    use_adg: bool = False,
    guidance_scale: float = 7.0,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    progress_fn: Optional[Callable[[int, int, float, float], None]] = None,
    enc_tt_pipe: Optional[ttnn.Tensor] = None,
    ctx_tt_pipe: Optional[ttnn.Tensor] = None,
    return_device_latents: bool = False,
    encoder_attention_mask_b1qk: Optional[ttnn.Tensor] = None,
    deallocate_ctx_latents: bool = True,
    deallocate_encoder_mask: bool = True,
    trace_state: Optional[_E2EDenoiseTrace] = None,
    temb_per_step: Optional[list] = None,
    tp_per_step: Optional[list] = None,
) -> Union[torch.Tensor, ttnn.Tensor]:
    """Run the TTNN DiT denoising loop (latents stay on device; Euler + APG/ADG).

    Matches the demo path from ``dit_sampling_ttnn``: ``ttnn.randn`` init, row-major BF16 DiT inputs,
    FLOAT32 TILE latent state, and ``attention_mask_1d_bt=None`` (optional NumPy 1D encoder
    mask or pre-uploaded *encoder_attention_mask_b1qk*).

    Args:
        pipe: TTNN pipeline instance.
        device: TTNN device handle.
        act_dtype: Reserved for API compatibility (DiT path uses BF16 row tensors from NumPy).
        mem: TTNN memory config (e.g. ``DRAM_MEMORY_CONFIG``).
        t_schedule: Descending timestep floats.
        frames: Temporal frame count.
        enc_hs: Encoder hidden states ``[B, S, D]`` (host path only).
        enc_mask: Encoder attention mask ``[B, S]`` (float/bool); host NumPy or Torch tensor.
            Omit when *encoder_attention_mask_b1qk* is set.
        ctx_lat: Context latents ``[B, T, 128]`` (host path only).
        null_emb: Null-condition embedding for CFG on the host path only
            (broadcastable to *enc_hs*).
        do_cfg: Whether classifier-free guidance is active (batch doubles).
        seed: RNG seed for ``ttnn.randn`` initial noise.
        use_adg: If True and *do_cfg*, apply TTNN ADG in the CFG interval; else APG.
        guidance_scale: CFG strength when *do_cfg*.
        cfg_interval_start: CFG applies for ``t`` in
            ``[cfg_interval_start, cfg_interval_end]`` (ACE-Step semantics).
        cfg_interval_end: See *cfg_interval_start*.
        progress_fn: ``(step_idx, num_steps, t_curr, dt)`` after each Euler step.
        enc_tt_pipe: Optional pre-built DiT encoder hidden states on device
            (e.g. CFG batch already concatenated). When set, *ctx_tt_pipe* must
            be set; *enc_hs*, *ctx_lat*, and *null_emb* are ignored.
        ctx_tt_pipe: Pre-built context latents on device (CFG batch when *do_cfg*).
        return_device_latents: When True, return on-device TILE latents without ``ttnn.to_torch``;
            caller deallocates. Encoder/context pipe tensors (and optional *encoder_attention_mask_b1qk*)
            are deallocated in this function.
        encoder_attention_mask_b1qk: Pre-built cross-attention SDPA mask (see
            :meth:`AceStepV15TTNNPipeline.build_encoder_attention_mask_b1qk_optional`).
            When set, ``pipe.forward`` does not take *encoder_attention_mask_1d_bk*.
        deallocate_ctx_latents: If False, skip ``ttnn.deallocate(ctx_tt_pipe)`` at the end (for a
            shared cached context tensor on non-CFG runs).
        deallocate_encoder_mask: If False, skip ``ttnn.deallocate(encoder_attention_mask_b1qk)``
            at the end. Set to False when the mask came from
            :meth:`AceStepV15TTNNPipeline.build_encoder_attention_mask_b1qk_optional`, which
            now caches the mask per-prompt in ``self._enc_mask_cache`` and owns the tensor.
        trace_state: Optional :class:`_E2EDenoiseTrace` handle. When provided and not yet
            captured, two eager warmup steps run first, then the DiT body is captured into a
            TTNN trace and used for every remaining step. When provided and already captured
            (subsequent ``generate`` calls), every step is replayed via the trace. Set
            ``ACE_STEP_USE_TRACE=1`` to enable from :class:`AceStepE2EModel`. When ``None``
            (the default), the loop runs fully eagerly — identical to the pre-trace behavior.
        temb_per_step: Optional pre-computed list of per-step ``temb_bd`` device tensors. When
            provided, the loop skips the in-call ``compute_temb_tp`` precompute and does NOT
            deallocate the tensors on exit — caller owns them. Pair with *tp_per_step*. Both
            must have length ``len(t_schedule)``. ``AceStepE2EModel`` precomputes these in
            ``__init__`` so the same (t_schedule, do_cfg) shape doesn't pay ~10 ops × N steps
            of host-driven dispatch on every ``generate``.
        tp_per_step: Companion to *temb_per_step* — per-step ``timestep_proj_b6d`` device
            tensors (ROW_MAJOR layout, as :meth:`AceStepV15TTNNPipeline.compute_temb_tp` returns).

    Returns:
        Denoised latents ``[B, frames, 64]``: CPU float32 :class:`torch.Tensor`, or an on-device
        :class:`ttnn.Tensor` when *return_device_latents* is True.
    """
    _ = act_dtype  # DiT uses BF16 row-major staging from NumPy, not this dtype.
    num_steps = len(t_schedule)
    if num_steps < 1:
        raise ValueError("t_schedule must be non-empty")

    frames_i = int(frames)
    c_lat = 64
    cfg_lo = float(cfg_interval_start)
    cfg_hi = float(cfg_interval_end)
    gs = float(guidance_scale)

    # ``ttnn.rand`` is uniform in [from,to]; ACE-Step uses Gaussian latents like ``torch.randn``.
    if not hasattr(ttnn, "randn"):
        raise RuntimeError("This path needs ``ttnn.randn`` (Gaussian) for latent init; ``ttnn.rand`` is uniform-only.")
    xt_tt = ttnn.randn(
        (1, frames_i, c_lat),
        device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mem,
        seed=int(np.uint32(int(seed))),
    )

    if enc_mask is None and encoder_attention_mask_b1qk is None:
        raise ValueError("Provide enc_mask or encoder_attention_mask_b1qk.")

    encoder_attn_1d_bk_np: Optional[np.ndarray] = None
    if encoder_attention_mask_b1qk is None:
        assert enc_mask is not None
        if isinstance(enc_mask, np.ndarray):
            encoder_keep_np_single = np.asarray(enc_mask, dtype=np.float32)
        else:
            encoder_keep_np_single = np.asarray(enc_mask.detach().cpu().numpy(), dtype=np.float32)
        if encoder_keep_np_single.ndim != 2:
            raise ValueError(f"encoder_attention_mask must be rank-2 [B,S], got {encoder_keep_np_single.shape}")
        encoder_keep_np_single = (encoder_keep_np_single > np.float32(0.0)).astype(np.bool_)
        encoder_attn_1d_bk_np = (
            np.concatenate([encoder_keep_np_single, encoder_keep_np_single], axis=0)
            if do_cfg
            else encoder_keep_np_single
        )

    prebuilt = enc_tt_pipe is not None
    if prebuilt:
        if ctx_tt_pipe is None:
            raise ValueError("ctx_tt_pipe is required when enc_tt_pipe is set.")
    elif enc_hs is None or ctx_lat is None:
        raise ValueError("Host denoise path requires enc_hs and ctx_lat (or pass enc_tt_pipe and ctx_tt_pipe).")
    elif do_cfg and null_emb is None:
        raise ValueError("Host CFG path requires null_emb.")

    if not prebuilt:
        if do_cfg:
            assert enc_hs is not None and null_emb is not None and ctx_lat is not None
            enc_tt_pipe = bf16_row_from_numpy_bc(
                np.concatenate(
                    [to_numpy_f32(enc_hs), to_numpy_f32(null_emb.expand_as(enc_hs))],
                    axis=0,
                ),
                device=device,
                dram=mem,
            )
            ctx_row_one = bf16_row_from_numpy_bc(to_numpy_f32(ctx_lat), device=device, dram=mem)
            ctx_tt_pipe = concat_duplicate_batch(ctx_row_one)
            try:
                ttnn.deallocate(ctx_row_one)
            except Exception:
                pass
        else:
            assert enc_hs is not None and ctx_lat is not None
            enc_tt_pipe = bf16_row_from_numpy_bc(to_numpy_f32(enc_hs), device=device, dram=mem)
            ctx_tt_pipe = bf16_row_from_numpy_bc(to_numpy_f32(ctx_lat), device=device, dram=mem)

    momentum_ttnn = TtnnMomentumBufferApg() if do_cfg and not use_adg else None
    _trace_each_step = os.environ.get("ACE_STEP_TRACY_EACH_DENOISE_STEP", "").lower() in ("1", "true", "yes")
    # Flush the device-profiler marker buffer every N denoise steps to avoid the per-RISC 12000-entry
    # ring buffer filling up across the loop (which silently drops markers, then trips
    # ``Device data missing`` asserts in ``tools/tracy/process_ops_logs.py``). No-op when neither
    # ``TT_METAL_DEVICE_PROFILER`` nor ``TTNN_OP_PROFILER`` is set, so production runs are unaffected.
    try:
        _flush_every = int(os.environ.get("ACE_STEP_PROFILER_FLUSH_EVERY", "1"))
    except ValueError:
        _flush_every = 1
    if _flush_every < 0:
        _flush_every = 0

    # Per-step (temb, timestep_proj) device tensors.
    #
    # ``pipe.compute_temb_tp`` runs the time-embed MLP (~10 ops per step) on the host-control side
    # and returns persistent device tensors. The denoise body below calls
    # ``pipe.forward_with_temb_tp(... temb=temb_per_step[i], tp=tp_per_step[i] ...)`` so the
    # per-step time-embed compute is paid once at start of ``generate`` instead of every Euler step.
    # This is also the precondition for wrapping the DiT body in a TTNN trace: a single capture can
    # serve all N steps when the only per-step difference is the (temb, tp) device-tensor identities,
    # which the orchestrator can streams via ``ttnn.copy`` onto persistent buffers before each
    # ``execute_trace``.
    #
    # When the caller (``AceStepE2EModel``) supplied ``temb_per_step`` / ``tp_per_step`` kwargs
    # the lists are owned by the model (precomputed in ``__init__`` for the configured
    # ``(infer_steps, do_cfg)`` shape) — we use them as-is and do not deallocate on exit. Otherwise
    # we precompute on-the-fly and free at end of this call (legacy demo path).
    pipe_batch = 2 if do_cfg else 1
    _temb_steps_owned = False
    if temb_per_step is None or tp_per_step is None:
        temb_per_step = []
        tp_per_step = []
        for _idx in range(num_steps):
            _temb, _tp = pipe.compute_temb_tp(int(_idx), target_batch=pipe_batch)
            temb_per_step.append(_temb)
            tp_per_step.append(_tp)
        _temb_steps_owned = True
    else:
        if len(temb_per_step) != num_steps or len(tp_per_step) != num_steps:
            raise ValueError(
                f"temb_per_step / tp_per_step length mismatch with num_steps={num_steps} "
                f"(got {len(temb_per_step)} / {len(tp_per_step)})."
            )

    # Trace control flow:
    # - ``trace_state is None``: pure eager path (legacy behavior, ``ACE_STEP_USE_TRACE`` off).
    # - ``trace_state.is_ready()``: replay path — every step uses ``execute_trace`` against the
    #   persistent buffers. ``ctx_tt_pipe`` / ``enc_tt_pipe`` / ``encoder_attention_mask_b1qk``
    #   passed in by the caller are not used by the trace (caller already primed contents into
    #   the persistent buffers via :meth:`_E2EDenoiseTrace.prime_per_prompt`); the loop still
    #   deallocates them on exit per the usual ownership contract.
    # - ``trace_state`` provided, buffers exist, trace id released: ``recapture`` immediately at
    #   the top of the loop — reuses the persistent buffers (caller already primed them) and
    #   re-arms the trace id without paying for two eager warmups again.
    # - ``trace_state`` provided, no buffers: first-ever capture path — two eager warmup steps
    #   prime the program cache, then step 2 clones inputs into fresh persistent buffers and
    #   captures the trace; subsequent steps replay.
    if trace_state is not None and trace_state.has_buffers() and not trace_state.is_ready():
        # Second+ generate of the same shape — persistent buffers carry over from previous
        # ``run_ttnn_denoise_loop`` but the trace id was released so eager VAE / next-prompt
        # encoder allocations stayed safe. Re-arm now using the primed buffers.
        trace_state.recapture(pipe=pipe, device=device)

    _capture_after_step = 1  # first eager step is 0, second is 1; capture before step 2.
    _trace_op_event = None  # set lazily on first replay or capture.

    def _post_dit_eager(
        *, step_idx: int, t_curr_f: float, euler_dt: float, acoustic, xt_pipe_in, acoustic_is_persistent: bool
    ) -> None:
        """Common slice + APG/ADG guidance + Euler subtract; consumes ``acoustic`` then advances ``xt_tt``."""
        nonlocal xt_tt
        if do_cfg:
            apply_cfg_now = cfg_lo <= t_curr_f <= cfg_hi
            vpc_rm = slice_batch_btc(acoustic, 0, 1, frames_i, c_lat)
            vpu_rm = slice_batch_btc(acoustic, 1, 2, frames_i, c_lat)
            if apply_cfg_now:
                if use_adg:
                    vt_tt = adg_guidance_velocity_ttnn(
                        xt_tt,
                        vpc_rm,
                        vpu_rm,
                        float(t_curr_f),
                        gs,
                        device=device,
                        dram=mem,
                    )
                else:
                    vt_tt = apg_guidance_velocity_ttnn(
                        vpc_rm,
                        vpu_rm,
                        gs,
                        momentum_buffer=momentum_ttnn,
                        dims=[1],
                        dram=mem,
                    )
            else:
                try:
                    ttnn.deallocate(vpu_rm)
                except Exception:
                    pass
                vt_tt = typecast_bf16_any_to_fp32_tile(vpc_rm, dram=mem)
        else:
            vt_tt = typecast_bf16_any_to_fp32_tile(acoustic, dram=mem)

        try:
            ttnn.deallocate(xt_pipe_in)
        except Exception:
            pass
        # When ``acoustic`` is the trace's persistent output buffer, the replay overwrites it on
        # the next step — we MUST NOT free it here or the captured graph would write into freed
        # memory. The eager-path output is owned by this iterate and is safe to free.
        if not acoustic_is_persistent:
            try:
                ttnn.deallocate(acoustic)
            except Exception:
                pass

        xt_old = xt_tt
        xt_tt = euler_subtract_v_dt(xt=xt_tt, vt=vt_tt, dt=float(euler_dt), dram=mem)
        try:
            ttnn.deallocate(vt_tt)
        except Exception:
            pass
        try:
            ttnn.deallocate(xt_old)
        except Exception:
            pass

        if progress_fn is not None:
            progress_fn(step_idx, num_steps, t_curr_f, float(euler_dt))

    def _diffusion_iterate(*, step_idx: int, t_curr_f: float, euler_dt: float) -> None:
        xt_row = fp32_tile_to_row_bf16(xt_tt, dram=mem)
        if do_cfg:
            xt_pipe_in = concat_duplicate_batch(xt_row)
            try:
                ttnn.deallocate(xt_row)
            except Exception:
                pass
        else:
            xt_pipe_in = xt_row

        acoustic = pipe.forward_with_temb_tp(
            xt_bt64=xt_pipe_in,
            context_latents_bt128=ctx_tt_pipe,
            encoder_hidden_states_btd=enc_tt_pipe,
            temb_bd=temb_per_step[int(step_idx)],
            timestep_proj_b6d=tp_per_step[int(step_idx)],
            attention_mask_1d_bt=None,
            encoder_attention_mask_1d_bk=None if encoder_attention_mask_b1qk is not None else encoder_attn_1d_bk_np,
            encoder_attention_mask_b1qk=encoder_attention_mask_b1qk,
        )

        _post_dit_eager(
            step_idx=step_idx,
            t_curr_f=t_curr_f,
            euler_dt=euler_dt,
            acoustic=acoustic,
            xt_pipe_in=xt_pipe_in,
            acoustic_is_persistent=False,
        )

    def _diffusion_iterate_capture(*, step_idx: int, t_curr_f: float, euler_dt: float) -> None:
        """First-call capture path: clone current inputs into trace_state buffers, capture, use the captured output for this step."""
        assert trace_state is not None
        xt_row = fp32_tile_to_row_bf16(xt_tt, dram=mem)
        if do_cfg:
            xt_pipe_in = concat_duplicate_batch(xt_row)
            try:
                ttnn.deallocate(xt_row)
            except Exception:
                pass
        else:
            xt_pipe_in = xt_row

        acoustic = trace_state.capture(
            pipe=pipe,
            device=device,
            xt_pipe_in=xt_pipe_in,
            temb=temb_per_step[int(step_idx)],
            tp=tp_per_step[int(step_idx)],
            enc_tt_pipe=enc_tt_pipe,
            ctx_tt_pipe=ctx_tt_pipe,
            encoder_attention_mask_b1qk=encoder_attention_mask_b1qk,
            frames=frames_i,
            pipe_batch=pipe_batch,
            c_lat=c_lat,
            do_cfg=do_cfg,
        )

        _post_dit_eager(
            step_idx=step_idx,
            t_curr_f=t_curr_f,
            euler_dt=euler_dt,
            acoustic=acoustic,
            xt_pipe_in=xt_pipe_in,
            acoustic_is_persistent=True,
        )

    def _diffusion_iterate_traced(*, step_idx: int, t_curr_f: float, euler_dt: float) -> None:
        """Replay path: stream new (xt, temb, tp) into persistent buffers, execute trace, finish eagerly."""
        assert trace_state is not None and trace_state.is_ready()
        nonlocal _trace_op_event
        xt_row = fp32_tile_to_row_bf16(xt_tt, dram=mem)
        if do_cfg:
            xt_pipe_in = concat_duplicate_batch(xt_row)
            try:
                ttnn.deallocate(xt_row)
            except Exception:
                pass
        else:
            xt_pipe_in = xt_row

        if _trace_op_event is None:
            # First replay of this generate: nothing in flight, so synthesize a "completed" event by
            # recording on CQ 0 right away. ``ttnn.record_event`` returns immediately and a wait on it
            # is a no-op once dispatch has caught up.
            _trace_op_event = ttnn.record_event(device, 0)
        acoustic, _trace_op_event = trace_state.replay(
            device=device,
            xt_pipe_in=xt_pipe_in,
            temb=temb_per_step[int(step_idx)],
            tp=tp_per_step[int(step_idx)],
            op_event=_trace_op_event,
        )

        _post_dit_eager(
            step_idx=step_idx,
            t_curr_f=t_curr_f,
            euler_dt=euler_dt,
            acoustic=acoustic,
            xt_pipe_in=xt_pipe_in,
            acoustic_is_persistent=True,
        )

    def _iterate(*, step_idx: int, t_curr_f: float, euler_dt: float) -> None:
        if trace_state is None:
            _diffusion_iterate(step_idx=step_idx, t_curr_f=t_curr_f, euler_dt=euler_dt)
            return
        if trace_state.is_ready():
            _diffusion_iterate_traced(step_idx=step_idx, t_curr_f=t_curr_f, euler_dt=euler_dt)
            return
        if step_idx == _capture_after_step + 1 and num_steps > _capture_after_step + 1:
            _diffusion_iterate_capture(step_idx=step_idx, t_curr_f=t_curr_f, euler_dt=euler_dt)
            return
        _diffusion_iterate(step_idx=step_idx, t_curr_f=t_curr_f, euler_dt=euler_dt)

    for step_idx in range(num_steps - 1):
        if _trace_each_step:
            _ace_step_prof_signpost("Denoise Step", f"step {step_idx}/{num_steps}")
        t_curr_f = float(t_schedule[step_idx])
        t_next_f = float(t_schedule[step_idx + 1])
        dt = t_curr_f - t_next_f
        _iterate(step_idx=step_idx, t_curr_f=t_curr_f, euler_dt=dt)
        if _flush_every and ((step_idx + 1) % _flush_every) == 0:
            _ace_step_flush_device_profiler(device)

    if _trace_each_step:
        _ace_step_prof_signpost("Denoise Step", f"step {num_steps - 1}/{num_steps}")
    t_curr_final = float(t_schedule[-1])
    _iterate(step_idx=num_steps - 1, t_curr_f=t_curr_final, euler_dt=t_curr_final)

    # Make sure all enqueued ops (trace replays included) complete before we deallocate inputs.
    if trace_state is not None and trace_state.is_ready():
        ttnn.synchronize_device(device)
        if trace_state.output_addr is not None and trace_state.acoustic_buf is not None:
            try:
                cur_addr = int(trace_state.acoustic_buf.buffer_address())
            except Exception:
                cur_addr = trace_state.output_addr
            assert cur_addr == trace_state.output_addr, (
                f"Trace output buffer moved across executes: {trace_state.output_addr} -> {cur_addr}. "
                "An allocation inside the captured graph re-ran non-deterministically."
            )
        # Release the trace id BEFORE returning so the VAE decode (and the next prompt's text /
        # condition encoders) allocate in :func:`mark_allocations_safe` mode. Keeping the trace
        # alive across ``generate()`` calls is what produced the
        # "Allocating device buffers is unsafe due to the existence of an active trace" warning
        # on the first generate and caused the second generate's VAE to hang on corrupted
        # device buffers. Persistent buffers stay alive — the next loop reuses them via
        # :meth:`_E2EDenoiseTrace.recapture`.
        trace_state.release_trace_only(device)

        # ``xt_tt`` was allocated by the eager post-DiT path while the allocator was in
        # ``mark_allocations_unsafe`` mode (its address may sit inside the trace region). After
        # the release above we're back in safe mode, so re-home ``xt_tt`` into a tracked buffer
        # before handing it to the VAE — otherwise downstream VAE allocations could collide and
        # corrupt the final latents (manifests as a hang in the second generate's VAE).
        try:
            xt_tt_safe = ttnn.clone(xt_tt)
            try:
                ttnn.deallocate(xt_tt)
            except Exception:
                pass
            xt_tt = xt_tt_safe
        except Exception:
            # ``ttnn.clone`` may fail if it doesn't accept this dtype/layout; in that case keep
            # the unsafe ``xt_tt`` — we've already released the trace so the risk window is
            # smaller than the alternative of crashing here.
            pass

    _ace_step_flush_device_profiler(device)

    try:
        ttnn.deallocate(enc_tt_pipe)
        if deallocate_ctx_latents:
            ttnn.deallocate(ctx_tt_pipe)
        if encoder_attention_mask_b1qk is not None and deallocate_encoder_mask:
            ttnn.deallocate(encoder_attention_mask_b1qk)
    except Exception:
        pass
    # Free the precomputed per-step time embeddings only when this call owned them. When the
    # caller (``AceStepE2EModel`` with ``temb_per_step`` / ``tp_per_step`` kwargs) owns the
    # tensors, they outlive the call and are reused across every ``generate()``.
    if _temb_steps_owned:
        for _t in temb_per_step:
            try:
                ttnn.deallocate(_t)
            except Exception:
                pass
        for _t in tp_per_step:
            try:
                ttnn.deallocate(_t)
            except Exception:
                pass
    if momentum_ttnn is not None:
        momentum_ttnn.reset()

    if return_device_latents:
        return xt_tt

    # Single device→Torch copy of latents for host VAE or other CPU consumers.
    pred_latents = ttnn.to_torch(xt_tt, dtype=torch.float32).contiguous()
    try:
        ttnn.deallocate(xt_tt)
    except Exception:
        pass
    return pred_latents


class AceStepE2EModel:
    """End-to-end ACE-Step v1.5: text → DiT (TTNN) → VAE → waveform.

    Stages:
        1. Text encoding (TTNN Qwen3-Embedding encoder; Hugging Face tokenizer → NumPy only)
        2. Instrumental condition (TTNN ``TtAceStepInstrumentalConditionEncoder``, same as
           ``run_prompt_to_wav.py`` with ``--ttnn-condition-embedding``)
        3. TTNN DiT denoising loop
        4. VAE decode (TTNN Oobleck)
    """

    def __init__(
        self,
        config: E2EConfig,
        device: ttnn.Device,
    ) -> None:
        self.config = config
        self.device = device
        if hasattr(device, "enable_program_cache"):
            device.enable_program_cache()

        self.act_dtype = getattr(ttnn, "bfloat16")
        self.mem = getattr(ttnn, "DRAM_MEMORY_CONFIG")

        self._tokenizer = None
        self._qwen: Optional[TtQwen3EmbeddingEncoder] = None
        self._condition_encoder: Optional[TtAceStepInstrumentalConditionEncoder] = None
        self._tt_vae: Optional[TtOobleckVaeDecoder] = None
        self._ctx_bt128_cached: Optional[ttnn.Tensor] = None
        # ``ACE_STEP_USE_TRACE=1`` opts the DiT body (``pipe.forward_with_temb_tp``) into a
        # captured TTNN trace + 2CQ replay. Captured lazily on the first ``generate()`` after two
        # eager warmup steps; reused for every subsequent step + every subsequent generate of the
        # same shape via ``_E2EDenoiseTrace.prime_per_prompt`` / ``.replay``. ``None`` keeps the
        # existing fully-eager path.
        self._trace_state: Optional[_E2EDenoiseTrace] = _E2EDenoiseTrace() if _e2e_trace_enabled() else None

        self._load_silence_latent()

        self.t_schedule = _build_t_schedule(
            shift=config.shift,
            infer_steps=config.infer_steps,
        )
        self.timesteps_host = np.asarray(self.t_schedule + [0.0], dtype=np.float32)
        self.frames = int(round(config.duration_sec * 25.0))

        self.pipe = AceStepV15TTNNPipeline(
            device=device,
            checkpoint_safetensors_path=config.checkpoint_safetensors_path,
            timesteps_host=self.timesteps_host,
            expected_input_length=self.frames,
        )

        self._init_condition_encoder()
        self._init_qwen_encoder()
        self._init_ttnn_vae()
        self._ctx_bt128_cached = self._ctx_latents_ttnn()

        # Precompute per-step (temb, timestep_proj) device tensors once for the configured
        # (t_schedule, do_cfg) shape. ``run_ttnn_denoise_loop`` then skips its own per-call
        # precompute when these are passed in — saves ~10 device dispatches × len(t_schedule)
        # every ``generate()``, and makes the trace's per-step
        # ``ttnn.copy(temb_per_step[i], temb_buf)`` source a stable model-owned tensor instead of
        # a fresh one per generate.
        #
        # Note: ``len(self.t_schedule) == self.config.infer_steps + 1`` because the schedule is
        # the descending boundary list (``range(infer_steps, -1, -1)``), and the loop indexes
        # step 0 .. len(t_schedule) - 1 (the final step uses ``t_curr = t_schedule[-1] = 0.0``).
        _pipe_batch = 2 if (self.config.guidance_scale > 1.0 + 1e-6) else 1
        self._temb_per_step: list[ttnn.Tensor] = []
        self._tp_per_step: list[ttnn.Tensor] = []
        for _idx in range(len(self.t_schedule)):
            _t, _tp = self.pipe.compute_temb_tp(int(_idx), target_batch=_pipe_batch)
            self._temb_per_step.append(_t)
            self._tp_per_step.append(_tp)

        # Per-prompt cache: skips ``encode_text`` + ``condition_encoder.forward`` for repeat
        # prompts. Key includes ``duration_sec`` because the Qwen text prompt embeds it in the
        # metas dict (so different durations produce different token streams + embeddings).
        # Cached values are device tensors owned by this model; the cache evicts FIFO at
        # ``_prompt_cache_max`` entries to bound device memory.
        self._prompt_cache: "OrderedDict[tuple[str, float], tuple[ttnn.Tensor, np.ndarray]]" = OrderedDict()
        self._prompt_cache_max = int(os.environ.get("ACE_STEP_PROMPT_CACHE_MAX", "8"))
        # Drain any device-profiler markers accumulated by weight uploads / conv-weight prep / JIT
        # during init. Without this, the 12000-marker per-RISC ring buffer is already full before the
        # first ``generate()`` call gets a chance to flush — markers from the first few DiT layers
        # then get silently dropped, and ``cpp_device_perf_report.csv`` ends up missing op rows that
        # ``tools/tracy/process_ops_logs.py`` later asserts on. No-op when device profiling is off.
        _ace_step_flush_device_profiler(self.device)

    def _init_qwen_encoder(self) -> None:
        qwen_st = self.config.qwen_safetensors_path
        if qwen_st is None:
            qwen_st = str(Path(self.config.text_model_dir) / "model.safetensors")
        if not Path(qwen_st).is_file():
            raise FileNotFoundError(f"Missing Qwen embedding weights at {qwen_st}")
        self._qwen = TtQwen3EmbeddingEncoder(
            device=self.device,
            hf_model_dir=str(self.config.text_model_dir),
            qwen_safetensors_path=qwen_st,
        )

    def _init_ttnn_vae(self) -> None:
        vdir = Path(self.config.vae_dir)
        if not (vdir / "config.json").is_file():
            raise FileNotFoundError(f"TTNN VAE expects a Hugging Face-style folder with config.json at {vdir}.")
        self._tt_vae = TtOobleckVaeDecoder.from_hf_vae_dir(
            str(vdir),
            device=self.device,
            latent_frames=int(self.frames),
            batch_size=1,
            activation_dtype=self.act_dtype,
            weights_dtype=self.act_dtype,
        )

    def _init_condition_encoder(self) -> None:
        if not Path(self.config.checkpoint_safetensors_path).is_file():
            raise FileNotFoundError(
                f"TTNN condition encoder needs DiT weights at {self.config.checkpoint_safetensors_path}."
            )
        self._condition_encoder = TtAceStepInstrumentalConditionEncoder(
            device=self.device,
            checkpoint_safetensors_path=self.config.checkpoint_safetensors_path,
            dtype=self.act_dtype,
        )

    def _load_silence_latent(self) -> None:
        silence = torch.load(self.config.silence_latent_path, map_location="cpu").to(torch.float32)
        if silence.ndim != 3:
            raise RuntimeError(f"Unexpected silence_latent rank: {tuple(silence.shape)}")
        if int(silence.shape[-1]) == 64:
            pass
        elif int(silence.shape[1]) == 64:
            silence = silence.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected silence_latent shape: {tuple(silence.shape)}")
        self._silence_np = silence.contiguous().numpy()

    def encode_text(self, prompt: str) -> tuple[ttnn.Tensor, np.ndarray]:
        """Encode a text prompt with TTNN Qwen (HF tokenizer → NumPy tokens only).

        Returns:
            (``text_hs_tt`` ``[1,1,S,D]`` on device, ``attention_mask`` ``[1,S]`` float in ``{0,1}``)
        """
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.text_model_dir)
        dit_instruction = "Fill the audio semantic mask based on the given conditions:"
        metas = {"caption": prompt, "duration": self.config.duration_sec, "language": "en"}
        text_prompt = f"""# Instruction
{dit_instruction}

# Caption
{prompt}

# Metas
{metas}<|endoftext|>
"""
        tokens = self._tokenizer(text_prompt, padding="max_length", truncation=True, max_length=256)
        input_ids_np = np.asarray(tokens["input_ids"], dtype=np.uint32).reshape(1, -1)
        attn_mask_np = np.asarray(tokens["attention_mask"], dtype=np.float32).reshape(1, -1)
        if self._qwen is None:
            raise RuntimeError("Qwen TTNN encoder was not initialized.")
        text_hs_tt = self._qwen.forward(input_ids_np, attn_mask_np)
        return text_hs_tt, attn_mask_np

    def _ctx_latents_ttnn(self) -> ttnn.Tensor:
        """Silence-derived src latents + chunk mask on device (``[1,T,128]``)."""
        frames = int(self.frames)
        src = np.asarray(self._silence_np[:, :frames, :], dtype=np.float32)
        if src.shape[1] < frames:
            rep = (frames + src.shape[1] - 1) // int(src.shape[1])
            src = np.tile(src, (1, rep, 1))[:, :frames, :]
        chunk_np = np.ones((1, frames, 64), dtype=np.float32)
        src_latents_tt = ttnn.as_tensor(
            np.ascontiguousarray(src),
            device=self.device,
            dtype=self.act_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        chunk_masks_tt = ttnn.as_tensor(
            chunk_np,
            device=self.device,
            dtype=self.act_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        ctx_tt_one = ttnn.concat([src_latents_tt, chunk_masks_tt], dim=-1)
        try:
            ttnn.deallocate(src_latents_tt)
            ttnn.deallocate(chunk_masks_tt)
        except Exception:
            pass
        return ctx_tt_one

    def decode_vae(
        self,
        pred_latents: Union[torch.Tensor, np.ndarray, ttnn.Tensor],
        *,
        return_waveform_ttnn: bool = False,
    ) -> Union[torch.Tensor, ttnn.Tensor]:
        """Decode latents to waveform via the TTNN Oobleck VAE (tiled along time if needed).

        Args:
            pred_latents: ``[1, frames, 64]`` on host (NumPy or Torch) or already on device as TTNN
                (same as ``run_prompt_to_wav``: DiT TILE float32 latents need no re-upload).
            return_waveform_ttnn: If True, return ``wav_tt`` on device (no peak normalization).

        Returns:
            waveform [1, channels, samples] normalized to [-1, 1] (default), or raw ``ttnn`` audio
            when *return_waveform_ttnn* is True.
        """
        if self._tt_vae is None:
            raise RuntimeError("TTNN VAE was not initialized.")
        owns_lat_tt = True
        if isinstance(pred_latents, ttnn.Tensor):
            lat_tt = pred_latents
            owns_lat_tt = False
        elif hasattr(pred_latents, "detach"):
            lat_np = pred_latents.detach().float().cpu().contiguous().numpy()
            lat_tt = ttnn.as_tensor(
                np.asarray(lat_np, dtype=np.float32),
                device=self.device,
                dtype=self.act_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=self.mem,
            )
        else:
            lat_tt = ttnn.as_tensor(
                np.asarray(pred_latents, dtype=np.float32),
                device=self.device,
                dtype=self.act_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=self.mem,
            )
        chunk = int(os.environ.get("ACE_STEP_VAE_CHUNK_LATENTS", str(self.config.vae_chunk_latents)))
        overlap = int(os.environ.get("ACE_STEP_VAE_OVERLAP_LATENTS", str(self.config.vae_overlap_latents)))
        wav_tt = self._tt_vae.decode_tiled(lat_tt, chunk_size=chunk, overlap=overlap)
        try:
            if owns_lat_tt:
                ttnn.deallocate(lat_tt)
        except Exception:
            pass
        if return_waveform_ttnn:
            return wav_tt
        wav_bt_c = ttnn.to_torch(wav_tt, dtype=torch.float32).contiguous().numpy()
        wav_bct = np.ascontiguousarray(np.swapaxes(wav_bt_c, 1, 2))
        peak = np.maximum(np.amax(np.abs(wav_bct), axis=(1, 2), keepdims=True), 1e-8)
        wav_np = np.clip(wav_bct / peak, -1.0, 1.0)
        try:
            ttnn.deallocate(wav_tt)
        except Exception:
            pass
        return torch.from_numpy(wav_np)

    def generate(
        self,
        prompt: str,
        *,
        return_waveform_ttnn: bool = False,
    ) -> Union[torch.Tensor, ttnn.Tensor]:
        """Full end-to-end: text prompt → waveform tensor.

        Args:
            prompt: text description of the music.
            return_waveform_ttnn: If True, return raw TTNN waveform from the VAE (no host peak norm).

        Returns:
            waveform [1, channels, samples] normalized to [-1, 1] by default, or device tensor
            when *return_waveform_ttnn* is True.
        """
        if self._condition_encoder is None:
            raise RuntimeError("TTNN condition encoder was not initialized.")
        if self._ctx_bt128_cached is None:
            raise RuntimeError("Cached context latents were not initialized.")
        do_cfg = self.config.guidance_scale > 1.0 + 1e-6

        # Start each generate() with an empty device-profiler ring so the per-layer flush downstream
        # has full 12000-marker headroom; no-op when device profiling is off.
        _ace_step_flush_device_profiler(self.device)

        # Prompt-output cache: identical (prompt, duration_sec) → reuse the cached
        # ``enc_hs_tt_one`` + ``enc_mask_np`` so we skip both the Qwen3 text encoder forward
        # (~30 ms) and the instrumental condition encoder forward (~10 ms) on repeat calls.
        # The cache is FIFO-evicted at ``self._prompt_cache_max`` entries; cached tensors are
        # model-owned and deallocated only on eviction.
        cache_key = (prompt, float(self.config.duration_sec))
        if cache_key in self._prompt_cache:
            enc_hs_tt_one, enc_mask_np = self._prompt_cache[cache_key]
            # ``null_condition_emb`` is a persistent attribute of the condition encoder built once
            # at construction, so we can fetch it without re-running ``forward``.
            null_emb_tt = self._condition_encoder.null_condition_emb
            # Move this entry to the most-recently-used end so it's the last to be evicted.
            self._prompt_cache.move_to_end(cache_key)
            _ace_step_prof_signpost("ACE-Step E2E", "Start text encoding (cache hit)")
            _ace_step_prof_signpost("ACE-Step E2E", "Start condition encoding (cache hit)")
        else:
            _ace_step_prof_signpost("ACE-Step E2E", "Start text encoding")
            text_hs_tt, attn_mask_np = self.encode_text(prompt)
            _ace_step_flush_device_profiler(self.device)

            _ace_step_prof_signpost("ACE-Step E2E", "Start condition encoding")
            enc_hs_tt_one, enc_mask_np, null_emb_tt = self._condition_encoder.forward(text_hs_tt, attn_mask_np)
            _ace_step_flush_device_profiler(self.device)
            try:
                ttnn.deallocate(text_hs_tt)
            except Exception:
                pass

            # Insert into cache, evicting the oldest entry if full. We only deallocate
            # ``enc_hs_tt_one`` of the evicted entry — ``enc_mask_np`` is a host NumPy array and
            # ``null_emb_tt`` was the shared condition-encoder buffer (not stored per-entry).
            if self._prompt_cache_max > 0:
                if len(self._prompt_cache) >= self._prompt_cache_max:
                    _evicted_key, (_evicted_enc, _evicted_mask) = self._prompt_cache.popitem(last=False)
                    try:
                        ttnn.deallocate(_evicted_enc)
                    except Exception:
                        pass
                self._prompt_cache[cache_key] = (enc_hs_tt_one, enc_mask_np)

        ctx_tt_one = self._ctx_bt128_cached
        enc_tt_pipe: ttnn.Tensor
        ctx_tt_pipe: ttnn.Tensor

        # ``enc_hs_tt_one`` is owned by ``self._prompt_cache`` and must outlive this generate.
        # In both CFG and non-CFG paths, ``enc_tt_pipe`` is built as a fresh allocation (concat
        # for CFG, clone for non-CFG) so the loop's terminal ``ttnn.deallocate(enc_tt_pipe)``
        # frees the disposable working buffer and never touches the cached source.
        if do_cfg:
            d_enc = int(enc_hs_tt_one.shape[-1])
            s_enc = int(enc_hs_tt_one.shape[1])
            # ``null_emb_tt`` is the persistent ``self._condition_encoder.null_condition_emb``
            # initialized once at construction. ``ttnn.reshape`` returns a view in this build, so
            # deallocating ``null_4d`` previously freed the cached buffer and broke subsequent
            # ``generate`` calls with "Tensor is not allocated". Clone first so we own a temp buffer
            # and can free it safely.
            if not hasattr(ttnn, "clone"):
                raise RuntimeError(
                    "ttnn.clone is required to safely copy null_condition_emb before reshape/deallocate. "
                    "This TTNN build does not expose ttnn.clone."
                )
            null_owned = ttnn.clone(null_emb_tt)
            null_4d = ttnn.reshape(null_owned, (1, 1, 1, d_enc))
            null_rep_4d = ttnn.repeat(null_4d, (1, 1, s_enc, 1))
            null_rep = ttnn.reshape(null_rep_4d, (1, s_enc, d_enc))
            # ``ttnn.concat`` allocates a new tensor; the cached ``enc_hs_tt_one`` is only READ.
            enc_tt_pipe = ttnn.concat([enc_hs_tt_one, null_rep], dim=0)
            ctx_tt_pipe = concat_duplicate_batch(ctx_tt_one)
            try:
                # ``null_4d`` is a view of ``null_owned`` and ``null_rep`` is a view of ``null_rep_4d``;
                # deallocating the owners is sufficient (and avoids double-free on the views).
                ttnn.deallocate(null_rep_4d)
                ttnn.deallocate(null_owned)
            except Exception:
                pass
        else:
            # Non-CFG: ``enc_tt_pipe`` would alias the cache directly, and the loop's terminal
            # ``ttnn.deallocate(enc_tt_pipe)`` would free the cached buffer. Clone it into a
            # disposable working buffer so the cache survives every generate.
            enc_tt_pipe = ttnn.clone(enc_hs_tt_one)
            ctx_tt_pipe = ctx_tt_one

        enc_row = np.asarray(enc_mask_np, dtype=np.float32).reshape(1, -1)
        enc_attn_1d_bk = np.concatenate([enc_row, enc_row], axis=0) if do_cfg else enc_row

        _ace_step_prof_signpost("ACE-Step E2E", "Start DiT preparation (SDPA mask)")
        b_mask = 2 if do_cfg else 1
        xt_dummy = ttnn.zeros(
            (b_mask, int(self.frames), 64),
            device=self.device,
            dtype=self.act_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.mem,
        )
        ctx_for_mask = concat_duplicate_batch(ctx_tt_one) if do_cfg else ctx_tt_one
        mask_tt = self.pipe.build_encoder_attention_mask_b1qk_optional(
            xt_bt64=xt_dummy,
            context_latents_bt128=ctx_for_mask,
            encoder_hidden_states_btd=enc_tt_pipe,
            encoder_attention_mask_1d_bk=enc_attn_1d_bk,
        )
        try:
            ttnn.deallocate(xt_dummy)
            if do_cfg:
                ttnn.deallocate(ctx_for_mask)
        except Exception:
            pass

        # Refresh persistent trace buffers from this prompt's freshly-encoded enc/ctx/mask before
        # the loop replays the captured DiT body. ``prime_per_prompt`` runs device→device copies
        # only — no host transfers — and validates that the new prompt produced a mask in the
        # same presence regime as the one captured (all-keys-valid vs. some-masked-out). On any
        # mismatch (shape OR mask presence) we release + re-allocate on the next loop call.
        # Check ``has_buffers()`` rather than ``is_ready()``: the trace id is released at the
        # END of every denoise loop (so VAE / next-prompt encoders allocate safely), but the
        # persistent buffers carry over. The loop will :meth:`recapture` at its top.
        if self._trace_state is not None and self._trace_state.has_buffers():
            shape_ok = self._trace_state.matches_shape(
                frames=self.frames, pipe_batch=(2 if do_cfg else 1), c_lat=64, do_cfg=do_cfg
            )
            mask_presence_ok = (self._trace_state.mask_buf is not None) == (mask_tt is not None)
            if not (shape_ok and mask_presence_ok):
                self._trace_state.release(self.device)
            else:
                self._trace_state.prime_per_prompt(
                    enc_tt_pipe=enc_tt_pipe,
                    ctx_tt_pipe=ctx_tt_pipe,
                    mask_tt=mask_tt,
                )

        loop_kw = dict(
            pipe=self.pipe,
            device=self.device,
            act_dtype=self.act_dtype,
            mem=self.mem,
            t_schedule=self.t_schedule,
            frames=self.frames,
            enc_hs=None,
            ctx_lat=None,
            null_emb=None,
            do_cfg=do_cfg,
            seed=self.config.seed,
            use_adg=self.config.use_adg,
            guidance_scale=float(self.config.guidance_scale),
            cfg_interval_start=float(self.config.cfg_interval_start),
            cfg_interval_end=float(self.config.cfg_interval_end),
            enc_tt_pipe=enc_tt_pipe,
            ctx_tt_pipe=ctx_tt_pipe,
            return_device_latents=True,
            encoder_attention_mask_b1qk=mask_tt,
            deallocate_ctx_latents=do_cfg,
            # `mask_tt` comes from `pipe.build_encoder_attention_mask_b1qk_optional`, which now
            # caches the per-prompt mask in `self.pipe._enc_mask_cache`. Don't let the denoise
            # loop deallocate it on cleanup — the cache owns the device tensor across generates.
            deallocate_encoder_mask=False,
            trace_state=self._trace_state,
            # Pass our model-owned per-step (temb, timestep_proj) lists so the loop skips its
            # per-call precompute and does NOT deallocate them on exit.
            temb_per_step=self._temb_per_step,
            tp_per_step=self._tp_per_step,
        )
        if mask_tt is None:
            loop_kw["enc_mask"] = enc_mask_np
        else:
            loop_kw["enc_mask"] = None

        _ace_step_prof_signpost("ACE-Step E2E", "Start denoising loop")
        pred_latents = run_ttnn_denoise_loop(**loop_kw)
        _ace_step_flush_device_profiler(self.device)
        try:
            _ace_step_prof_signpost("ACE-Step E2E", "Start VAE decode")
            wav = self.decode_vae(pred_latents, return_waveform_ttnn=return_waveform_ttnn)
            _ace_step_flush_device_profiler(self.device)
            _ace_step_prof_signpost("ACE-Step E2E", "Generation complete")
        finally:
            try:
                ttnn.deallocate(pred_latents)
            except Exception:
                pass
        return wav

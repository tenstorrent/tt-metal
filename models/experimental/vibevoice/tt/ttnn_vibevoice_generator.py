# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice Generator — TTNN port of generate() from modeling_vibevoice_inference.py.

Pipeline (aligned with reference):
  1. Prefill: processor speech_tensors/masks → acoustic encode → scatter into inputs_embeds
  2. AR loop: greedy decode with valid-token constraint
  3. On speech_diffusion_id: CFG diffusion → decode → semantic encode → connector sum → next embed
"""

import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import ttnn

# Optional env-gated diagnostics for generate():
#   VV_PROFILE=1 — device-synced timing breakdown per phase
#   VV_DEBUG=1   — per-AR-step token + phase logs (also set by demo_ttnn.py --debug)
#   VV_PROFILE_PREFILL=1 — Tracy signposts ``start``/``stop`` around LM prefill
#     (``_lm_prefill`` only). Use with ``python -m tracy …`` then
#     ``tt-perf-report <csv> --start-signpost start --end-signpost stop``.
#   VV_PROFILE_PREFILL_EXIT=1 — return from generate() right after LM prefill (no AR).


def _vv_profile_enabled() -> bool:
    return os.environ.get("VV_PROFILE", "0") == "1"


def _vv_profile_prefill_enabled() -> bool:
    return os.environ.get("VV_PROFILE_PREFILL", "0") == "1"


def _vv_debug_enabled() -> bool:
    return os.environ.get("VV_DEBUG", "0") == "1"


def _vv_debug(msg: str) -> None:
    if _vv_debug_enabled():
        print(f"[VV_DEBUG] {msg}", flush=True)


class _Profiler:
    def __init__(self, device, enabled: Optional[bool] = None):
        self.device = device
        self.enabled = _vv_profile_enabled() if enabled is None else enabled
        self.totals: dict = {}
        self.counts: dict = {}

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        ttnn.synchronize_device(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ttnn.synchronize_device(self.device)
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def report(self) -> None:
        if not self.enabled or not self.totals:
            return
        total = sum(self.totals.values())
        print("\n[VV_PROFILE] ===== generate() timing breakdown (device-synced) =====", flush=True)
        for name in sorted(self.totals, key=lambda k: -self.totals[k]):
            t = self.totals[name]
            c = self.counts[name]
            print(
                f"[VV_PROFILE]   {name:30s} {t:9.3f}s  ({100 * t / total:5.1f}%)  "
                f"n={c:5d}  avg={1000 * t / max(c, 1):8.2f}ms",
                flush=True,
            )
        print(f"[VV_PROFILE]   {'TOTAL (profiled wall)':30s} {total:9.3f}s", flush=True)


from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import (
    TTVibeVoiceLM,
    KVCache,
    create_kv_cache,
)
from models.experimental.vibevoice.tt.ttnn_speech_connector import TTSpeechConnector
from models.experimental.vibevoice.tt.ttnn_diffusion_head import TTDiffusionHead
from models.experimental.vibevoice.tt.ttnn_dpm_scheduler import (
    TTDPMSolverMultistepScheduler,
    sample_speech_latents,
)
from models.experimental.vibevoice.tt.reference_lm_runner import ReferenceLMRunner


@dataclass
class TTVibeVoiceOutput:
    sequences: torch.Tensor  # [B, S] full token ids (prefill + generated)
    speech_outputs: List[torch.Tensor]  # concatenated waveforms per batch row
    prefill_wall_s: float = 0.0  # wall time covering embed-build + LM prefill forward
    decode_wall_s: float = 0.0  # wall time covering the full AR decode loop (fallback for non-traced runs)
    # Steady-state fused-frame decode timing (apples-to-apples with tt_transformers/llama demos):
    # time+count of trace-REPLAY frames only — warmup and capture frames are not timed.  Zero when
    # the fused-frame trace is not used (then decode_wall_s is the reported figure).
    steady_decode_s: float = 0.0
    steady_decode_frames: int = 0


def _greedy_argmax(logits: ttnn.Tensor, use_fp32: bool = False) -> int:
    """Greedy argmax on last-position logits."""
    if use_fp32:
        last = ttnn.to_torch(logits).to(torch.float32)[0, 0, -1, :]
        return int(last.argmax().item())
    idx = ttnn.argmax(logits, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return int(ttnn.to_torch(idx).reshape(-1)[-1].item())


def _apply_token_constraint(
    logits: ttnn.Tensor,
    valid_token_ids: List[int],
    device,
) -> ttnn.Tensor:
    """Mask logits so only valid_token_ids are selectable."""
    vocab_size = logits.shape[-1]
    mask = torch.full((1, 1, 1, vocab_size), float("-inf"), dtype=torch.bfloat16)
    mask[:, :, :, valid_token_ids] = 0.0
    mask_tt = ttnn.as_tensor(
        mask,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.add(logits, mask_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _embeds_to_host_2d(inputs_embeds: ttnn.Tensor) -> torch.Tensor:
    """[1, 1, S, H] device tensor → [S, H] float32 on host."""
    return ttnn.to_torch(inputs_embeds).to(torch.float32).squeeze(0).squeeze(0)


def _host_2d_to_embeds(embeds_2d: torch.Tensor, device, dtype: torch.dtype = torch.bfloat16) -> ttnn.Tensor:
    """[S, H] or [1, H] host → [1, 1, S, H] on device."""
    if embeds_2d.dim() == 1:
        embeds_2d = embeds_2d.unsqueeze(0)
    host = embeds_2d.unsqueeze(0).unsqueeze(0).to(dtype)
    ttnn_dtype = ttnn.float32 if dtype == torch.float32 else ttnn.bfloat16
    return ttnn.as_tensor(
        host,
        device=device,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _condition_from_hidden(last_hidden: ttnn.Tensor) -> ttnn.Tensor:
    """last_hidden [1,1,S,H] → condition [1,1,1,H] at last position."""
    h = last_hidden.shape[2] - 1
    return ttnn.slice(
        last_hidden,
        [0, 0, h, 0],
        [1, 1, h + 1, last_hidden.shape[-1]],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class _LoopBreaker:
    """Detect a sustained acoustic repeat-loop in the streamed audio and break it by
    re-drawing that frame's diffusion init noise (+ clamp the clipping the escape causes).

    Long-context bf16 numerical drift in the ~40k-step AR feedback loop can tip the
    trajectory into a degenerate attractor where the same short phrase is re-synthesised
    for minutes (the fp32 reference, per-step error ~1e-7, never does this).  The loop is
    NOT a raw-waveform repeat — each frame is acoustically distinct — but the CONTENT
    recurs, so we detect it from a per-frame log-spectral envelope: a high correlation at
    a phrase-scale lag (>= ``_MINLAG`` frames), sustained across a trailing window.

    Thresholds are calibrated on the eager 89-min render (``4p_climate_100min``): clean
    multi-speaker speech peaks at ~0.32 window-fraction (self-similarity sits at lag 1,
    excluded), the min-78..83 loop holds ~0.37-0.5 at lag ~15 (the 2 s phrase).  With
    ``_ON=0.40`` the detector never fires on clean audio, so on a clean render it is a
    no-op and output stays byte-identical.

    Intervention: on the frames of a detected loop, re-draw that frame's diffusion init
    noise from a frame-local RNG.  (Perturbing the loop-carried *hidden* directly was tried
    and was worse — the kick compounds through the on-device hidden-carry and drove the
    trajectory into worse loops + energy collapse; the noise re-draw is the gentler,
    better-behaved lever.)  It does not fully prevent the loop but shortens it and recovers
    ~4 min earlier.  The forced escape can briefly over-drive the diffusion into clipping,
    so the emitted audio is clamped to [-1, 1] during the loop episode + a recovery tail
    (never on clean audio, whose legitimate peaks can exceed 1.0).  Frame-local RNG →
    deterministic; the global draw order is untouched, so non-loop frames are unchanged.
    """

    _NB = 32  # log-spectral-envelope bins
    _MINLAG, _LMAX = 6, 24  # phrase-scale lag search (0.8 s .. 3.2 s @ 7.5 fps)
    _RMS_FLOOR, _TAU = 0.02, 0.92
    _W, _ON, _OFF = 90, 0.40, 0.18  # trailing window (frames); on/off fraction (hysteresis)
    _CLAMP_HOLD = 450  # clamp this many frames (~60 s) after an episode (recovery tail)

    def __init__(self, seed: int = 0x100B):
        self._seed = seed
        self._win = None
        self._edges = None
        self._vecs: deque = deque(maxlen=self._LMAX)  # recent normalised envelopes (past frames)
        self._rmss: deque = deque(maxlen=self._LMAX)
        self._flags: deque = deque(maxlen=self._W)  # recent repeat-ish flags (0/1)
        self._flagsum = 0
        self.active = False
        self._clamp_hold = 0

    def _feat(self, chunk_1d: torch.Tensor):
        x = chunk_1d.detach().to(torch.float32).numpy()
        if self._win is None or len(self._win) != len(x):
            self._win = np.hanning(len(x))
            nfreq = len(x) // 2 + 1
            self._edges = np.logspace(np.log10(4), np.log10(nfreq - 1), self._NB + 1).astype(int)
        rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
        mag = np.abs(np.fft.rfft(x * self._win))
        env = np.array([np.log1p(mag[self._edges[b] : self._edges[b + 1] + 1].mean()) for b in range(self._NB)])
        env -= env.mean()
        env /= np.sqrt((env**2).sum()) + 1e-9
        return env, rms

    def update(self, chunk_1d: torch.Tensor) -> None:
        """Feed one diffusion frame's audio; refresh ``active`` for the NEXT frame."""
        v, rms = self._feat(chunk_1d)
        repeatish = False
        if rms >= self._RMS_FLOOR and len(self._vecs) >= self._MINLAG:
            best = 0.0
            for lag in range(self._MINLAG, min(self._LMAX, len(self._vecs)) + 1):
                if self._rmss[-lag] >= self._RMS_FLOOR:
                    best = max(best, float(v @ self._vecs[-lag]))
            repeatish = best >= self._TAU
        self._vecs.append(v)
        self._rmss.append(rms)
        flag = 1 if repeatish else 0
        if len(self._flags) == self._W:
            self._flagsum -= self._flags[0]
        self._flags.append(flag)
        self._flagsum += flag
        frac = self._flagsum / len(self._flags)
        if not self.active and frac >= self._ON:
            self.active = True
        elif self.active and frac <= self._OFF:
            self.active = False
        if self.active:
            self._clamp_hold = self._CLAMP_HOLD
        elif self._clamp_hold > 0:
            self._clamp_hold -= 1

    def clamp_now(self) -> bool:
        """Clamp emitted audio during an episode + its recovery tail (never on clean audio)."""
        return self.active or self._clamp_hold > 0

    def perturb(self, noise_2x: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """Fresh diffusion-init noise from a frame-local RNG (reproducible; leaves the global
        draw order / other frames' noise untouched, so non-loop frames stay unchanged)."""
        g = torch.Generator().manual_seed(self._seed + int(frame_idx))
        return torch.randn(*noise_2x.shape, generator=g, dtype=torch.float32).to(noise_2x.dtype)


class TTVibeVoiceGenerator:
    """Full VibeVoice generation pipeline using TT modules."""

    def __init__(
        self,
        lm_tt: TTVibeVoiceLM,
        acoustic_connector: TTSpeechConnector,
        semantic_connector: TTSpeechConnector,
        diffusion_head: TTDiffusionHead,
        acoustic_tokenizer,
        semantic_tokenizer,
        scheduler: TTDPMSolverMultistepScheduler,
        device,
        speech_start_id: int,
        speech_end_id: int,
        speech_diffusion_id: int,
        eos_token_id: int,
        bos_token_id: Optional[int] = None,
        cfg_scale: float = 1.3,
        num_diffusion_steps: int = 10,
        max_new_tokens: Optional[int] = None,
        max_length_times: float = 2.0,
        speech_scaling_factor: Optional[float] = None,
        speech_bias_factor: Optional[float] = None,
        acoustic_fix_std: float = 0.5,
        acoustic_encode_chunk_samples: int = 3200,
        ref_inference=None,
    ):
        self.lm = lm_tt
        self.acoustic_conn = acoustic_connector
        self.semantic_conn = semantic_connector
        self.diffusion_head = diffusion_head
        self.acoustic_tok = acoustic_tokenizer
        self.semantic_tok = semantic_tokenizer
        self.scheduler = scheduler
        self.device = device
        self.ref_inference = ref_inference
        self._ref_acoustic_cache = None
        self._ref_semantic_cache = None
        self._ref_lm: Optional[ReferenceLMRunner] = None
        if ref_inference is not None:
            ref_inference.set_ddpm_inference_steps(num_diffusion_steps)
            self._ref_lm = ReferenceLMRunner(ref_inference, device)

        self.speech_start_id = speech_start_id
        self.speech_end_id = speech_end_id
        self.speech_diffusion_id = speech_diffusion_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.cfg_scale = cfg_scale
        self.num_diffusion_steps = num_diffusion_steps
        self.max_new_tokens = max_new_tokens
        self.max_length_times = max_length_times
        self.speech_scaling_factor = speech_scaling_factor
        self.speech_bias_factor = speech_bias_factor
        self.acoustic_fix_std = acoustic_fix_std
        self.acoustic_encode_chunk_samples = acoustic_encode_chunk_samples

        self.valid_token_ids = [
            speech_start_id,
            speech_end_id,
            speech_diffusion_id,
            eos_token_id,
        ]
        if bos_token_id is not None:
            self.valid_token_ids.append(bos_token_id)
        # Cached device-side logit mask for the valid-token constraint (built once,
        # reused every AR step — avoids a full-vocab host alloc + H2D upload per step).
        self._token_mask_tt: Optional[ttnn.Tensor] = None

        # Optional WHOLE-SEGMENT fused trace (opt-in via VV_TRACE_SEGMENT=1, set by demo --trace):
        # the true llama shape — a fully device-driven fused frame (the whole steady-state
        # speech-diffusion frame — neg-LM → diffusion → post-diffusion → pos-LM — as ONE graph),
        # so there are NO per-frame host RoPE/position writes and NO capture-poison re-run:
        # positions self-advance via ttnn.plus_one INSIDE the trace, RoPE rows are gathered on
        # device (bf16) from the device position, the neg embed is a per-frame input (a segment's
        # first frame decodes embed(speech_start) at neg_pos 0 — its neg-LM IS the negative
        # prefill — then embed(speech_diffusion)), and the pos hidden is loop-carried on device.
        # Lifecycle per segment: warmup -> throwaway capture -> reset (rewind positions, re-seed
        # hidden, zero the conv streaming caches IN PLACE) -> pure replay.  The trace is released
        # at each speech_start (so the boundary's eager LM decodes cannot corrupt a live capture)
        # and recaptured per segment; a single-segment generation captures once.  Validated:
        # tests/perf/{dev_rope_plusone,trace_fused_frame_llama,trace_segment_run}.py (PCC 1.0 vs
        # the eager dev-rope path).  bf16 RoPE makes it ~0.9999 vs the fp32 reference — the same
        # accepted precision as the bf16 SDPA-decode.
        self._trace_segment = os.environ.get("VV_TRACE_SEGMENT", "0") == "1"
        self._sf_tid = None
        self._sf_warm = 0
        self._sf_hidden_buf: Optional[ttnn.Tensor] = None  # loop-carried cond_pos source
        self._sf_hidden_seed: Optional[ttnn.Tensor] = None  # segment-start hidden ([1,1,1,H], last pos)
        self._sf_neg_embed: Optional[ttnn.Tensor] = None  # per-frame neg embed input buffer
        self._sf_neg_start: Optional[ttnn.Tensor] = None  # const embed(speech_start_id)
        self._sf_neg_diff: Optional[ttnn.Tensor] = None  # const embed(speech_diffusion_id)
        self._sf_pos_pos: Optional[ttnn.Tensor] = None
        self._sf_neg_pos: Optional[ttnn.Tensor] = None
        self._sf_noise: Optional[ttnn.Tensor] = None
        self._sf_t_tensors: Optional[list] = None
        self._sf_audio_out: Optional[ttnn.Tensor] = None
        self._sf_logits_out: Optional[ttnn.Tensor] = None
        # Constrained-decode (split-capture path): subset lm_head + in-trace argmax → local index.
        self._sf_tok_out: Optional[ttnn.Tensor] = None
        self._sf_valid_ids_sorted: Optional[List[int]] = None
        self._sf_lm_head_valid: Optional[ttnn.Tensor] = None
        # fp32 RoPE (VV_FP32_ROPE=1, default on): host-write the exact fp32 cos/sin rows per frame
        # into persistent buffers so the traced decode matches the EAGER fp32-rope path (which slices
        # the fp32 _cos_tt/_sin_tt table).  Off (=0) keeps the bf16 on-device embedding gather (A/B).
        self._sf_fp32_rope = os.environ.get("VV_FP32_ROPE", "1") == "1"
        self._sf_cos_pos: Optional[ttnn.Tensor] = None
        self._sf_sin_pos: Optional[ttnn.Tensor] = None
        self._sf_cos_neg: Optional[ttnn.Tensor] = None
        self._sf_sin_neg: Optional[ttnn.Tensor] = None
        self._sf_pos_pos_host = 0  # host mirror of the device _sf_pos_pos (for fp32 rope row select)
        self._sf_neg_pos_host = 0
        # Split-frame capture (VV_CAP_SPLIT=1, default): capture the steady speech-diffusion frame as
        # THREE separate traces — neg-LM | diffusion+post | pos-LM — instead of ONE monolithic capture.
        # Co-capturing the LM together with diffusion+post in a single trace causes a buffer-scheduling
        # aliasing whose replay diverges from eager at ~frame 177 (a tiny bf16 delta) and amplifies
        # chaotically into an unintelligible (but RMS-flat) long-form render; separate traces are
        # bit-identical to eager.  Set VV_CAP_SPLIT=0 to fall back to the monolithic capture (repro the
        # bug).  Address-stable hand-off buffers between the three traces: _sf_neg_hidden, _sf_fused_out.
        self._sf_cap_split = os.environ.get("VV_CAP_SPLIT", "1") == "1"
        self._sf_negtrace_tid = None
        self._sf_dptrace_tid = None
        self._sf_postrace_tid = None
        self._sf_neg_hidden: Optional[ttnn.Tensor] = None  # neg-LM last_hidden (neg-trace -> diff-trace)
        self._sf_fused_out: Optional[ttnn.Tensor] = None  # post-diffusion embed (diff-trace -> pos-LM-trace)
        # CFG batch-2 LM fusion (VV_CFG_BATCH2=1): fold the neg-LM + pos-LM into ONE batch-2 decode
        # forward that reads each layer's weights ONCE for both CFG rows (weight-DRAM-bound at M=1).
        # Software-pipelined: each frame's batched forward computes pos-LM(k) [row0, → cond_pos(k+1)]
        # and neg-LM(k+1) [row1, → cond_neg(k+1)]; the diffusion runs FIRST from cond buffers the
        # PREVIOUS frame's forward wrote.  A once-per-segment eager boot seeds neg-LM(0).  Proven
        # byte-identical per row to the two B=1 forwards (test_cfg_batch2_byteident.py) => Tier-0.
        # Requires cap-split token semantics (in-trace constrained argmax).
        self._sf_cfg_b2 = self._sf_cap_split and os.environ.get("VV_CFG_BATCH2", "1") == "1"
        self._sf_dp2trace_tid = None
        self._sf_lm2trace_tid = None
        # Diagnostic (VV_TRACE_NOCAPTURE=1): run the frame graph EAGERLY (no ttnn capture/replay) — also
        # clean (it exonerates the graph ops), but slower (no replay).  Used to isolate the bug above.
        self._sf_nocapture = os.environ.get("VV_TRACE_NOCAPTURE", "0") == "1"
        self._sf_nocap_started = False
        # Long-form energy stabilizer (VV_AUDIO_LIMIT=<R>, default 0=off): the bf16 AR feedback lets the
        # positive diffusion condition drift while the negative stays anchored; CFG amplifies the growing
        # divergence, so the acoustic-decode gain (== audio loudness) runs away into over-drive/clipping
        # once the (prefill-sized) stability window is exceeded.  When set, each EMITTED audio frame is
        # soft-limited to RMS <= R and peak <= VV_AUDIO_PEAK (see _emit_limit); the token/latent
        # trajectory is untouched, so content is preserved and only the audible energy is bounded.
        self._audio_limit_T = float(os.environ.get("VV_AUDIO_LIMIT", "0"))  # per-frame RMS ceiling
        self._AUDIO_PEAK = float(os.environ.get("VV_AUDIO_PEAK", "0.95"))  # per-frame peak ceiling (anti-clip)

    _SF_WARMUP = 2

    def _token_label(self, token_id: int) -> str:
        labels = {
            self.speech_start_id: "speech_start",
            self.speech_end_id: "speech_end",
            self.speech_diffusion_id: "speech_diffusion",
            self.eos_token_id: "eos",
        }
        if self.bos_token_id is not None:
            labels[self.bos_token_id] = "bos"
        return labels.get(token_id, f"id={token_id}")

    def _token_constraint_mask(self, vocab_size: int) -> ttnn.Tensor:
        if self._token_mask_tt is None:
            mask = torch.full((1, 1, 1, vocab_size), float("-inf"), dtype=torch.bfloat16)
            mask[:, :, :, self.valid_token_ids] = 0.0
            self._token_mask_tt = ttnn.as_tensor(
                mask,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._token_mask_tt

    def _reset_ref_tokenizer_caches(self):
        from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache

        self._ref_acoustic_cache = VibeVoiceTokenizerStreamingCache()
        self._ref_semantic_cache = VibeVoiceTokenizerStreamingCache()

    def _hidden_to_condition_torch(self, hidden_tt: ttnn.Tensor) -> torch.Tensor:
        """Extract last-position condition [1, H] float32 on CPU."""
        h = ttnn.to_torch(hidden_tt).to(torch.float32)
        if h.dim() == 4:
            return h[0, 0, -1, :].unsqueeze(0)
        if h.dim() == 3:
            return h[0, -1, :].unsqueeze(0)
        return h[-1, :].unsqueeze(0)

    def _audio_row_to_tt(self, wav_1d: torch.Tensor) -> ttnn.Tensor:
        """1D waveform [T] → [1, 1, 1, T] on device."""
        audio = wav_1d.to(torch.bfloat16).view(1, 1, 1, -1)
        return ttnn.as_tensor(
            audio,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _trim_trailing_zeros(wav_1d: torch.Tensor) -> torch.Tensor:
        """Drop processor padding so padded voice rows are not fully encoded on device."""
        if wav_1d.numel() == 0:
            return wav_1d
        nz = wav_1d != 0
        if not nz.any():
            return wav_1d[:0]
        last = int(nz.nonzero(as_tuple=True)[0][-1].item()) + 1
        return wav_1d[:last]

    def _latents_from_encode_output(self, lat_tt: ttnn.Tensor) -> torch.Tensor:
        """Device encode output → [T_enc, vae_dim] float32 on host."""
        out = ttnn.to_torch(lat_tt).to(torch.float32).squeeze(0).squeeze(0)
        if out.dim() == 1:
            return out.unsqueeze(0)
        return out

    def _encode_acoustic_latents(self, wav_1d: torch.Tensor) -> torch.Tensor:
        """Encode audio → [T_enc, vae_dim] float32 on host (with fix-std sampling).

        Long voice prompts are encoded in streaming chunks (one latent frame per chunk)
        so conv L1 circular buffers stay within device limits.
        """
        wav = self._trim_trailing_zeros(wav_1d)
        total_samples = wav.numel()
        chunk = self.acoustic_encode_chunk_samples

        if total_samples == 0:
            return torch.zeros(0, 0)

        if total_samples <= chunk:
            self.acoustic_tok._encoder_tt.reset_cache()
            lat_tt = self.acoustic_tok.encode(
                self._audio_row_to_tt(wav),
                use_cache=False,
                is_final_chunk=True,
            )
            lat = self._latents_from_encode_output(lat_tt)
        else:
            self.acoustic_tok._encoder_tt.reset_cache()
            frames: List[torch.Tensor] = []
            pos = 0
            while pos < total_samples:
                n = min(chunk, total_samples - pos)
                chunk_wav = wav[pos : pos + n]
                is_final = pos + n >= total_samples
                if chunk_wav.numel() < chunk:
                    # conv2d caches prepared weights per input width; keep chunks fixed-size.
                    chunk_wav = torch.nn.functional.pad(chunk_wav, (0, chunk - chunk_wav.numel()))
                lat_tt = self.acoustic_tok.encode(
                    self._audio_row_to_tt(chunk_wav),
                    use_cache=True,
                    is_final_chunk=is_final,
                )
                out = self._latents_from_encode_output(lat_tt)
                frames.append(out[-1:])
                pos += n
            lat = torch.cat(frames, dim=0)

        if self.acoustic_fix_std:
            lat = lat + self.acoustic_fix_std * torch.randn_like(lat)
        return lat

    def _compute_scale_bias(self, latents_list: List[torch.Tensor], speech_masks: torch.Tensor):
        """Match reference: scale=1/std(masked), bias=-mean(masked) on stacked latents."""
        parts = []
        for i in range(speech_masks.shape[0]):
            n = int(speech_masks[i].sum().item())
            if n > 0:
                parts.append(latents_list[i][:n].reshape(-1, latents_list[i].shape[-1]))
        if not parts:
            return 1.0, 0.0
        flat = torch.cat(parts, dim=0).flatten()
        return (1.0 / flat.std()).item(), (-flat.mean()).item()

    def _process_speech_prefill(
        self,
        speech_tensors: torch.Tensor,
        speech_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return speech_embeds [N_slots, hidden] for scatter into prefill (host float32)."""
        scale = self.speech_scaling_factor
        bias = self.speech_bias_factor
        latents_per_row = []
        for i in range(speech_tensors.shape[0]):
            latents_per_row.append(self._encode_acoustic_latents(speech_tensors[i]))

        if scale is None or bias is None:
            scale, bias = self._compute_scale_bias(latents_per_row, speech_masks)
            self.speech_scaling_factor = scale
            self.speech_bias_factor = bias

        speech_embeds_parts = []
        for i in range(speech_tensors.shape[0]):
            n = int(speech_masks[i].sum().item())
            feats = (latents_per_row[i][:n] + bias) * scale
            feats_tt = ttnn.as_tensor(
                feats.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            conn_out = self.acoustic_conn(feats_tt)
            conn_torch = ttnn.to_torch(conn_out).to(torch.float32)
            if conn_torch.dim() == 4:
                conn_torch = conn_torch.squeeze(0).squeeze(0)
            elif conn_torch.dim() == 3:
                conn_torch = conn_torch.squeeze(0)
            conn_torch = conn_torch[:n]
            speech_embeds_parts.append(conn_torch)

        return torch.cat(speech_embeds_parts, dim=0)

    def _build_prefill_embeds(
        self,
        input_ids: torch.Tensor,
        speech_tensors: Optional[torch.Tensor],
        speech_masks: Optional[torch.Tensor],
        speech_input_mask: Optional[torch.Tensor],
        prefill_speech_embeds: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Text embeds with speech slots scattered (reference forward prefill)."""
        if self._ref_lm is not None:
            cpu_embeds = self._ref_lm.build_prefill_embeds(input_ids, speech_input_mask, prefill_speech_embeds)
            return ttnn.as_tensor(
                cpu_embeds.unsqueeze(1).to(torch.float32),
                device=self.device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        inputs_embeds = self.lm._embed(input_ids)
        if speech_input_mask is None:
            return inputs_embeds

        embed_2d = _embeds_to_host_2d(inputs_embeds)
        if prefill_speech_embeds is not None:
            speech_embeds = prefill_speech_embeds.to(torch.float32)
        elif speech_tensors is not None and speech_masks is not None:
            speech_embeds = self._process_speech_prefill(speech_tensors, speech_masks)
        else:
            return inputs_embeds
        mask = speech_input_mask[0].cpu().bool()
        n_slots = int(mask.sum().item())
        embed_2d[mask[: embed_2d.shape[0]]] = speech_embeds[:n_slots].to(embed_2d.dtype)
        return _host_2d_to_embeds(embed_2d, self.device, dtype=torch.float32)

    def _run_speech_diffusion(
        self,
        condition: ttnn.Tensor,
        neg_condition: ttnn.Tensor,
        latent_size: int = 64,
        noise_2x: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> ttnn.Tensor:
        if self.ref_inference is not None:
            pos = self._hidden_to_condition_torch(condition)
            neg = self._hidden_to_condition_torch(neg_condition)
            with torch.no_grad():
                latent = self.ref_inference.sample_speech_tokens(pos, neg, cfg_scale=self.cfg_scale)
            return ttnn.as_tensor(
                latent.view(1, 1, 1, latent_size).to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Initial diffusion noise: 2×latent_size values matching the reference's
        # torch.randn(2, vae_dim) (it cats pos+neg into batch=2, draws one noise per
        # entry, then uses speech[:1]).  Normally pre-drawn once before the AR loop and
        # passed in via ``noise_2x`` (keeps the global RNG aligned with the reference);
        # falls back to drawing here when not supplied.
        # IMPORTANT: draw in float32 (the reference dtype) then cast to bfloat16 —
        # torch.randn(dtype=bfloat16) produces *different* values than randn(float32)
        # for the same seed, which would feed the diffusion completely different noise.
        if noise_2x is None:
            noise_2x = torch.randn(2, 1, 1, latent_size, dtype=torch.float32, generator=rng).to(torch.bfloat16)
        noise = noise_2x[:1]
        initial_latent = ttnn.as_tensor(
            noise,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return sample_speech_latents(
            self.diffusion_head,
            condition,
            neg_condition,
            self.scheduler,
            initial_latent,
            cfg_scale=self.cfg_scale,
            num_steps=self.num_diffusion_steps,
            head_runner=None,
        )

    def _reset_segment_frame_trace(self) -> None:
        """Release the whole-segment fused trace at a segment boundary.  The boundary's eager LM
        decodes (speech_end/speech_start) allocate DRAM; a live capture would be corrupted once
        re-executed (coexistence hazard), so drop the capture here and let the next segment's first
        frame re-warm + recapture.  The persistent I/O buffers and KV caches are address-stable and
        kept; the conv streaming caches are zeroed IN PLACE by the runner's frame-0 reset (not freed
        here, which would move their addresses out from under the recaptured trace)."""
        if self._sf_tid is not None:
            ttnn.release_trace(self.device, self._sf_tid)
        self._sf_tid = None
        self._sf_warm = 0
        # Split-capture (VV_CAP_SPLIT): release all three per-frame traces too, so the next
        # segment's frame 0 re-warms + recaptures them (same coexistence-hazard reasoning).
        for _attr in (
            "_sf_negtrace_tid",
            "_sf_dptrace_tid",
            "_sf_postrace_tid",
            "_sf_dp2trace_tid",
            "_sf_lm2trace_tid",
        ):
            _t = getattr(self, _attr, None)
            if _t is not None:
                ttnn.release_trace(self.device, _t)
            setattr(self, _attr, None)

    def _sf_write_int(self, buf: ttnn.Tensor, val: int) -> None:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([val], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            buf,
        )

    def _sf_write_rope(self, cos_buf: ttnn.Tensor, sin_buf: ttnn.Tensor, pos: int) -> None:
        """Host-write the exact fp32 RoPE cos/sin row for `pos` into a persistent [1,1,1,hd] buffer
        (host->device copy, no device alloc — same numerics as the eager sliced-fp32-table path)."""
        hd = self.lm.cfg.head_dim
        cos = torch.from_numpy(self.lm._cos_np[pos : pos + 1]).to(torch.float32).reshape(1, 1, 1, hd)
        sin = torch.from_numpy(self.lm._sin_np[pos : pos + 1]).to(torch.float32).reshape(1, 1, 1, hd)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(cos, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), cos_buf)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(sin, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), sin_buf)

    def _sf_set_inputs(self, seg_frame_idx: int, start_pos: int, noise_2x) -> None:
        """Per-frame non-allocating writes into the persistent trace buffers.  A segment's first
        frame (seg_frame_idx==0) rewinds the device positions, re-seeds the loop-carried hidden from
        the (already-sliced) segment-start hidden, and selects embed(speech_start) — so its neg-LM at
        neg_pos 0 IS the negative prefill; later frames select embed(speech_diffusion) and let the
        positions self-advance (ttnn.plus_one) on device.  All writes here are host->device or
        device->device copies into fixed-address buffers (no allocation), so they are safe to run
        while the fused trace is live."""
        if seg_frame_idx == 0:
            self._sf_write_int(self._sf_pos_pos, start_pos)
            self._sf_write_int(self._sf_neg_pos, 0)
            self._sf_pos_pos_host = start_pos
            self._sf_neg_pos_host = 0
            ttnn.copy(input_a=self._sf_hidden_seed, input_b=self._sf_hidden_buf)  # device->device seed
            ttnn.copy(input_a=self._sf_neg_start, input_b=self._sf_neg_embed)
        else:
            self._sf_pos_pos_host += 1  # mirror the on-device plus_one from the prior frame
            self._sf_neg_pos_host += 1
            ttnn.copy(input_a=self._sf_neg_diff, input_b=self._sf_neg_embed)
        if self._sf_fp32_rope:
            # fp32 rope rows for the current positions (device positions self-advance for KV/sdpa).
            self._sf_write_rope(self._sf_cos_pos, self._sf_sin_pos, self._sf_pos_pos_host)
            self._sf_write_rope(self._sf_cos_neg, self._sf_sin_neg, self._sf_neg_pos_host)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(noise_2x[:1].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            self._sf_noise,
        )

    def _sf_set_inputs_b2(self, seg_frame_idx: int, start_pos: int, noise_2x) -> None:
        """Per-frame input writes for the CFG batch-2 path.  Sets ONLY the lm2/dp2 inputs — the
        neg row's embed (speech_diffusion) and neg RoPE are managed by _sf_boot at frame 0.  Frame 0
        rewinds device positions + reseeds the loop-carried hidden; the batched forward's pos row
        reads pos_pos (@pos_pos_host) and the neg row reads neg_pos (one AHEAD, set by the boot)."""
        if seg_frame_idx == 0:
            self._sf_write_int(self._sf_pos_pos, start_pos)
            self._sf_write_int(self._sf_neg_pos, 0)
            self._sf_pos_pos_host = start_pos
            self._sf_neg_pos_host = 0  # boot advances the device tensor + this mirror to 1
            ttnn.copy(input_a=self._sf_hidden_seed, input_b=self._sf_hidden_buf)  # cond_pos(0) seed
            self._sf_write_rope(self._sf_cos_pos, self._sf_sin_pos, self._sf_pos_pos_host)
        else:
            self._sf_pos_pos_host += 1  # mirror the on-device plus_one from the prior frame's lm2
            self._sf_neg_pos_host += 1
            self._sf_write_rope(self._sf_cos_pos, self._sf_sin_pos, self._sf_pos_pos_host)
            self._sf_write_rope(self._sf_cos_neg, self._sf_sin_neg, self._sf_neg_pos_host)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(noise_2x[:1].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            self._sf_noise,
        )

    def _run_segment_frame_cfg_b2(self, seg_frame_idx, start_pos, noise_2x, kv_pos, kv_neg):
        """CFG batch-2 fused speech-diffusion frame.  Two captured traces per frame:
            _dp2trace : diffusion (cond_pos/cond_neg from persistent buffers) + post → audio, fused
            _lm2trace : ONE batch-2 LM decode = pos-LM(k) [row0] + neg-LM(k+1) [row1], reading each
                        layer's weights once; writes hidden_buf (cond_pos(k+1)) + neg_hidden
                        (cond_neg(k+1)); constrained argmax on row0 → token(k)
        plus a once-per-segment eager _boot (neg-LM(0), the negative prefill) seeding neg_hidden for
        frame 0.  Byte-identical per row to the split neg/pos B=1 forwards (Tier-0)."""
        dev = self.device
        lm = self.lm

        def _boot():
            # Eager B=1 negative-prefill: neg-LM on speech_start @ neg_pos 0 → _sf_neg_hidden.  Runs
            # once per segment while no trace is live (the frame-0 boundary), then switches the neg
            # row's embed/RoPE to the steady speech_diffusion @ neg_pos 1 for lm2.
            self._sf_write_rope(self._sf_cos_neg, self._sf_sin_neg, 0)
            ttnn.copy(input_a=self._sf_neg_start, input_b=self._sf_neg_embed)
            _, nh = lm.forward_decode_traced_embeds(
                self._sf_neg_embed,
                self._sf_cos_neg,
                self._sf_sin_neg,
                self._sf_neg_pos,
                kv_neg,
                return_last_hidden=True,
                need_logits=False,
            )
            if self._sf_neg_hidden is None:
                self._sf_neg_hidden = ttnn.clone(nh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                ttnn.copy(input_a=nh, input_b=self._sf_neg_hidden)
            ttnn.plus_one(self._sf_neg_pos)  # device neg_pos → 1
            self._sf_neg_pos_host = 1
            self._sf_write_rope(self._sf_cos_neg, self._sf_sin_neg, self._sf_neg_pos_host)
            ttnn.copy(input_a=self._sf_neg_diff, input_b=self._sf_neg_embed)  # steady embed for lm2

        def _dp2trace():
            cond_pos = _condition_from_hidden(self._sf_hidden_buf)
            cond_neg = _condition_from_hidden(self._sf_neg_hidden)
            latent = sample_speech_latents(
                self.diffusion_head,
                cond_pos,
                cond_neg,
                self.scheduler,
                self._sf_noise,
                cfg_scale=self.cfg_scale,
                num_steps=self.num_diffusion_steps,
                head_runner=None,
                t_tensors=self._sf_t_tensors,
            )
            fu, au = self._run_post_pipeline(latent)
            if self._sf_fused_out is None:
                self._sf_fused_out = ttnn.clone(fu, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                ttnn.copy(input_a=fu, input_b=self._sf_fused_out)
            return au

        def _lm2trace():
            H = lm.cfg.hidden_size
            pos_in = self._sf_fused_out
            if pos_in.dtype != self._sf_neg_embed.dtype:
                pos_in = ttnn.typecast(pos_in, self._sf_neg_embed.dtype)
            emb_b2 = ttnn.concat([pos_in, self._sf_neg_embed], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            logits0, hidden_b2 = lm.forward_decode_traced_embeds_b2(
                emb_b2,
                [(self._sf_cos_pos, self._sf_sin_pos), (self._sf_cos_neg, self._sf_sin_neg)],
                [self._sf_pos_pos, self._sf_neg_pos],
                [kv_pos, kv_neg],
                lm_head_w=self._sf_lm_head_valid,
            )
            h0 = ttnn.slice(hidden_b2, [0, 0, 0, 0], [1, 1, 1, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            h1 = ttnn.slice(hidden_b2, [1, 0, 0, 0], [2, 1, 1, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.copy(input_a=h0, input_b=self._sf_hidden_buf)  # cond_pos(k+1)
            ttnn.copy(input_a=h1, input_b=self._sf_neg_hidden)  # cond_neg(k+1)
            ttnn.plus_one(self._sf_pos_pos)
            ttnn.plus_one(self._sf_neg_pos)
            return ttnn.argmax(logits0, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self._sf_lm2trace_tid is None:
            # First frame-0 after a (re)capture: warmup (eager), capture dp2+lm2 (boot stays eager),
            # reset, then the real frame-0 replay — all internal, so warmup/capture frames are
            # discarded and never emitted.
            for _ in range(self._SF_WARMUP):
                self._sf_set_inputs_b2(0, start_pos, noise_2x)
                _boot()
                _dp2trace()
                _lm2trace()
            self._sf_set_inputs_b2(0, start_pos, noise_2x)
            _boot()  # seed neg_hidden for the captured dp2
            tb = ttnn.begin_trace_capture(dev, cq_id=0)
            self._sf_audio_out = _dp2trace()
            ttnn.end_trace_capture(dev, tb, cq_id=0)
            tc = ttnn.begin_trace_capture(dev, cq_id=0)
            self._sf_tok_out = _lm2trace()
            ttnn.end_trace_capture(dev, tc, cq_id=0)
            self._sf_dp2trace_tid, self._sf_lm2trace_tid = tb, tc
            # RESET for the real frame 0: rewind positions/hidden, zero conv, re-run boot.
            self._sf_set_inputs_b2(0, start_pos, noise_2x)
            self._sf_zero_conv()
            _boot()
            ttnn.execute_trace(dev, self._sf_dp2trace_tid, cq_id=0, blocking=False)
            ttnn.execute_trace(dev, self._sf_lm2trace_tid, cq_id=0, blocking=False)
            _vv_debug("segment_frame(cfg_b2): captured + reset")
            return self._sf_audio_out, self._sf_tok_out

        if seg_frame_idx == 0:
            self._sf_set_inputs_b2(0, start_pos, noise_2x)
            self._sf_zero_conv()
            _boot()
        else:
            self._sf_set_inputs_b2(1, start_pos, noise_2x)
        ttnn.execute_trace(dev, self._sf_dp2trace_tid, cq_id=0, blocking=False)
        ttnn.execute_trace(dev, self._sf_lm2trace_tid, cq_id=0, blocking=False)
        return self._sf_audio_out, self._sf_tok_out

    def _sf_zero_conv(self) -> None:
        """Zero the acoustic/semantic conv streaming caches IN PLACE (stable addresses) — the
        segment-boundary reset performed while the fused trace is live."""
        self.acoustic_tok.reset_decode_cache_inplace()
        self.semantic_tok.reset_cache_inplace()

    def _run_segment_frame_traced(self, seg_frame_idx, step_hidden, start_pos, noise_2x, kv_pos, kv_neg):
        """One speech-diffusion frame as ONE device-driven trace (Option 1, llama shape), replayed
        for the WHOLE segment.  Returns (audio_chunk, logits).  Frame graph:
            cond_pos = condition(hidden_buf);  neg_hidden = LM_dev_rope(neg_embed @ neg_pos, kv_neg)
            latent = DPM_loop(cond_pos, condition(neg_hidden), noise);  fused, audio = post(latent)
            logits, new_hidden = LM_dev_rope(fused @ pos_pos, kv_pos);  copy(new_hidden -> hidden_buf)
            plus_one(pos_pos); plus_one(neg_pos)
        On the first frame-0 after a (re)capture the runner warms up (eager — compiles + allocates
        the conv caches), does a throwaway capture, then RESETS (rewind positions, re-seed hidden +
        speech_start embed, zero conv caches in place) and replays — all internal, so the caller
        sees only the real frame's output and there is no capture-poison re-run.  RoPE is gathered
        on device (bf16) from the device position, so no per-frame host RoPE/position write."""
        lm = self.lm
        dev = self.device

        if self._sf_hidden_buf is None:
            H = lm.cfg.hidden_size

            def _z(shape, dt, lay):
                return ttnn.zeros(shape, dtype=dt, layout=lay, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            self._sf_hidden_buf = _z([1, 1, 1, H], ttnn.float32, ttnn.TILE_LAYOUT)
            self._sf_hidden_seed = _z([1, 1, 1, H], ttnn.float32, ttnn.TILE_LAYOUT)
            self._sf_neg_start = lm._embed(torch.tensor([[self.speech_start_id]], dtype=torch.long))
            self._sf_neg_diff = lm._embed(torch.tensor([[self.speech_diffusion_id]], dtype=torch.long))
            self._sf_neg_embed = _z([1, 1, 1, H], self._sf_neg_start.dtype, ttnn.TILE_LAYOUT)
            self._sf_pos_pos = _z([1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            self._sf_neg_pos = _z([1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            _hd = lm.cfg.head_dim
            self._sf_cos_pos = _z([1, 1, 1, _hd], ttnn.float32, ttnn.TILE_LAYOUT)
            self._sf_sin_pos = _z([1, 1, 1, _hd], ttnn.float32, ttnn.TILE_LAYOUT)
            self._sf_cos_neg = _z([1, 1, 1, _hd], ttnn.float32, ttnn.TILE_LAYOUT)
            self._sf_sin_neg = _z([1, 1, 1, _hd], ttnn.float32, ttnn.TILE_LAYOUT)
            # Constrained-decode lm_head subset (sorted valid ids → argmax tie-break parity with the
            # full-vocab masked argmax).  Pos-LM projects only these columns + in-trace argmax.
            self._sf_valid_ids_sorted = sorted(self.valid_token_ids)
            self._sf_lm_head_valid = lm.build_lm_head_subset(self._sf_valid_ids_sorted)
            self._sf_noise = _z([1, 1, 1, 64], ttnn.bfloat16, ttnn.TILE_LAYOUT)
            self.scheduler.set_timesteps(self.num_diffusion_steps)
            self._sf_t_tensors = [
                ttnn.full(
                    (2, 1, 1, 1),
                    float(t),
                    dtype=ttnn.bfloat16,
                    device=dev,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for t in self.scheduler.timesteps
            ]

        if seg_frame_idx == 0:
            # Capture the segment-start condition source ([1,1,1,H], last position of the
            # speech_start decode / prefill hidden) into a persistent buffer.  seg_frame_idx==0 is
            # only ever reached right after a speech_start released the trace (or at the very
            # start), so NO trace is live here — this is the one place the reducing slice may
            # allocate.  The reset then re-seeds from _sf_hidden_seed with an alloc-free copy.
            ttnn.copy(input_a=_condition_from_hidden(step_hidden), input_b=self._sf_hidden_seed)

        if self._sf_cfg_b2:
            return self._run_segment_frame_cfg_b2(seg_frame_idx, start_pos, noise_2x, kv_pos, kv_neg)

        self._sf_set_inputs(seg_frame_idx, start_pos, noise_2x)

        def _frame():
            cond_pos = _condition_from_hidden(self._sf_hidden_buf)
            if self._sf_fp32_rope:
                _, neg_hidden = lm.forward_decode_traced_embeds(
                    self._sf_neg_embed,
                    self._sf_cos_neg,
                    self._sf_sin_neg,
                    self._sf_neg_pos,
                    kv_neg,
                    return_last_hidden=True,
                )
            else:
                _, neg_hidden = lm.forward_decode_traced_embeds_dev_rope(
                    self._sf_neg_embed, self._sf_neg_pos, kv_neg, return_last_hidden=True
                )
            cond_neg = _condition_from_hidden(neg_hidden)
            latent = sample_speech_latents(
                self.diffusion_head,
                cond_pos,
                cond_neg,
                self.scheduler,
                self._sf_noise,
                cfg_scale=self.cfg_scale,
                num_steps=self.num_diffusion_steps,
                head_runner=None,
                t_tensors=self._sf_t_tensors,
            )
            fused, audio = self._run_post_pipeline(latent)
            if self._sf_fp32_rope:
                logits, new_hidden = lm.forward_decode_traced_embeds(
                    fused, self._sf_cos_pos, self._sf_sin_pos, self._sf_pos_pos, kv_pos, return_last_hidden=True
                )
            else:
                logits, new_hidden = lm.forward_decode_traced_embeds_dev_rope(
                    fused, self._sf_pos_pos, kv_pos, return_last_hidden=True
                )
            ttnn.copy(input_a=new_hidden, input_b=self._sf_hidden_buf)  # loop-carry on device
            ttnn.plus_one(self._sf_pos_pos)
            ttnn.plus_one(self._sf_neg_pos)
            return audio, logits

        if self._sf_cap_split:
            # THREE separate captured traces (neg-LM | diffusion+post | pos-LM) — the fix.
            def _negtrace():
                _, nh = lm.forward_decode_traced_embeds(
                    self._sf_neg_embed,
                    self._sf_cos_neg,
                    self._sf_sin_neg,
                    self._sf_neg_pos,
                    kv_neg,
                    return_last_hidden=True,
                    need_logits=False,  # neg logits are discarded — skip the full lm_head (bit-exact)
                )
                if self._sf_neg_hidden is None:
                    self._sf_neg_hidden = ttnn.clone(
                        nh, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )  # persistent alloc (eager warmup)
                else:
                    ttnn.copy(input_a=nh, input_b=self._sf_neg_hidden)
                ttnn.plus_one(self._sf_neg_pos)

            def _dptrace():
                cond_pos = _condition_from_hidden(self._sf_hidden_buf)
                cond_neg = _condition_from_hidden(self._sf_neg_hidden)
                latent = sample_speech_latents(
                    self.diffusion_head,
                    cond_pos,
                    cond_neg,
                    self.scheduler,
                    self._sf_noise,
                    cfg_scale=self.cfg_scale,
                    num_steps=self.num_diffusion_steps,
                    head_runner=None,
                    t_tensors=self._sf_t_tensors,
                )
                fu, au = self._run_post_pipeline(latent)
                if self._sf_fused_out is None:
                    self._sf_fused_out = ttnn.clone(
                        fu, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )  # persistent alloc (eager warmup)
                else:
                    ttnn.copy(input_a=fu, input_b=self._sf_fused_out)
                return au

            def _postrace():
                lg, nh = lm.forward_decode_traced_embeds(
                    self._sf_fused_out,
                    self._sf_cos_pos,
                    self._sf_sin_pos,
                    self._sf_pos_pos,
                    kv_pos,
                    return_last_hidden=True,
                    lm_head_w=self._sf_lm_head_valid,  # constrained-decode: project only selectable tokens
                )
                ttnn.copy(input_a=nh, input_b=self._sf_hidden_buf)  # loop-carry for next frame
                ttnn.plus_one(self._sf_pos_pos)
                # In-trace constrained argmax over the N selectable-token logits → LOCAL index
                # (== full-vocab masked argmax; caller maps local -> token id).
                return ttnn.argmax(lg, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            if self._sf_postrace_tid is None:
                for _ in range(self._SF_WARMUP):
                    _negtrace()
                    _dptrace()
                    _postrace()
                ta = ttnn.begin_trace_capture(dev, cq_id=0)
                _negtrace()
                ttnn.end_trace_capture(dev, ta, cq_id=0)
                tb = ttnn.begin_trace_capture(dev, cq_id=0)
                self._sf_audio_out = _dptrace()
                ttnn.end_trace_capture(dev, tb, cq_id=0)
                tc = ttnn.begin_trace_capture(dev, cq_id=0)
                self._sf_tok_out = _postrace()  # constrained argmax LOCAL index (persistent)
                ttnn.end_trace_capture(dev, tc, cq_id=0)
                self._sf_negtrace_tid, self._sf_dptrace_tid, self._sf_postrace_tid = ta, tb, tc
                self._sf_set_inputs(0, start_pos, noise_2x)  # reset positions/hidden/embed/rope
                self._sf_zero_conv()
            elif seg_frame_idx == 0:
                self._sf_zero_conv()
            ttnn.execute_trace(dev, self._sf_negtrace_tid, cq_id=0, blocking=False)
            ttnn.execute_trace(dev, self._sf_dptrace_tid, cq_id=0, blocking=False)
            ttnn.execute_trace(dev, self._sf_postrace_tid, cq_id=0, blocking=False)
            return self._sf_audio_out, self._sf_tok_out

        if self._sf_nocapture:
            # EAGER (no capture/replay): set_inputs already rewound frame-0 state above.
            if not self._sf_nocap_started:
                _frame()  # throwaway: allocate conv caches (positions/hidden reset below)
                self._sf_set_inputs(0, start_pos, noise_2x)
                self._sf_zero_conv()
                self._sf_nocap_started = True
                return _frame()
            if seg_frame_idx == 0:
                self._sf_zero_conv()
            return _frame()

        if self._sf_tid is None:
            # First frame-0 after a (re)capture: warmup (eager, compiles + allocates conv caches),
            # throwaway capture, reset, then the real replay — all internal, so warmup/capture
            # frames are discarded and never emitted (no capture-poison re-run needed).
            for _ in range(self._SF_WARMUP):
                _frame()
            tid = ttnn.begin_trace_capture(dev, cq_id=0)
            self._sf_audio_out, self._sf_logits_out = _frame()
            ttnn.end_trace_capture(dev, tid, cq_id=0)
            self._sf_tid = tid
            # RESET: rewind positions, re-seed hidden + speech_start embed, and zero the (now
            # allocated) conv caches in place — undoing the warmup/capture frames.  KV self-heals
            # by forward overwrite on replay.
            self._sf_set_inputs(0, start_pos, noise_2x)
            self._sf_zero_conv()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            _vv_debug("segment_frame: captured + reset")
            return self._sf_audio_out, self._sf_logits_out

        if seg_frame_idx == 0:
            self._sf_zero_conv()  # subsequent-segment reset (positions/hidden already rewound above)
        ttnn.execute_trace(dev, self._sf_tid, cq_id=0, blocking=False)
        return self._sf_audio_out, self._sf_logits_out

    def _post_diffusion_embeds(self, speech_latent: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Diffusion latent → (fused next-step embed, current audio chunk)."""
        if self.ref_inference is not None:
            return self._post_diffusion_embeds_ref(speech_latent)
        return self._post_diffusion_embeds_tt(speech_latent)

    def _post_diffusion_embeds_ref(self, speech_latent: ttnn.Tensor) -> Tuple[ttnn.Tensor, torch.Tensor]:
        m = self.ref_inference.model
        scale = self.speech_scaling_factor or m.speech_scaling_factor.item()
        bias = self.speech_bias_factor or m.speech_bias_factor.item()
        lat = ttnn.to_torch(speech_latent).to(torch.float32).reshape(1, -1)
        speech_latent_ref = lat.unsqueeze(1)
        scaled = speech_latent_ref / scale - bias
        sample_idx = torch.tensor([0])

        with torch.no_grad():
            audio_chunk = m.acoustic_tokenizer.decode(
                scaled,
                cache=self._ref_acoustic_cache,
                sample_indices=sample_idx,
                use_cache=True,
            )
            semantic_features = m.semantic_tokenizer.encode(
                audio_chunk,
                cache=self._ref_semantic_cache,
                sample_indices=sample_idx,
                use_cache=True,
            ).mean
            fused = m.acoustic_connector(speech_latent_ref) + m.semantic_connector(semantic_features)

        fused_tt = ttnn.as_tensor(
            fused.to(torch.float32).unsqueeze(1),
            device=self.device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return fused_tt, audio_chunk.reshape(-1)

    def _post_diffusion_embeds_tt(self, speech_latent: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """On-device streaming decode/encode/fusion (eager)."""
        return self._run_post_pipeline(speech_latent)

    def _run_post_pipeline(self, latent: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """The post-diffusion op graph: inverse-norm → acoustic decode → semantic encode
        → connectors → fused embed.  Reads ``latent`` for both the decode and acoustic
        connector (so a single persistent input tensor suffices under trace)."""
        scale = self.speech_scaling_factor or 1.0
        bias = self.speech_bias_factor or 0.0

        # Inverse-normalise the current latent frame to the acoustic VAE space, fully on device (no host round-trip).
        # scale/bias are Python floats, so this is scaled = latent * (1/scale) - bias.
        lat_f32 = ttnn.typecast(latent, ttnn.float32)
        scaled_f32 = ttnn.subtract(
            ttnn.mul(lat_f32, 1.0 / scale, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scaled_tt = ttnn.to_layout(ttnn.typecast(scaled_f32, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)

        # Streaming decode (current frame, cached causal context) → audio chunk.
        audio_chunk = self.acoustic_tok.decode(scaled_tt, use_cache=True)  # [1, 1, 1, T_audio]
        # Streaming semantic encode → this frame's semantic feature [1, 1, 1, vae_dim].
        sem_tt = self.semantic_tok.forward(audio_chunk, use_cache=True)
        t_enc = sem_tt.shape[2]
        semantic_last_tt = ttnn.slice(
            sem_tt,
            [0, 0, t_enc - 1, 0],
            [1, 1, t_enc, sem_tt.shape[-1]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        acoustic_embed = self.acoustic_conn(latent)
        semantic_embed = self.semantic_conn(semantic_last_tt)
        fused = ttnn.add(acoustic_embed, semantic_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return fused, audio_chunk

    def _emit_limit(self, audio_1d: torch.Tensor) -> torch.Tensor:
        """Long-form energy stabilizer (VV_AUDIO_LIMIT=<R>, default 0=off): soft-limit ONE emitted
        audio frame so RMS <= R (healthy loudness) AND peak <= VV_AUDIO_PEAK (no clipping), via a
        single per-frame gain = min(1, R/rms, P/peak).  Applied to the EMITTED audio ONLY — the
        semantic-encode feedback runs on the un-limited frame, so the token/latent trajectory is
        byte-identical to the un-limited render (content preserved); only the audio the listener hears
        is bounded.  (Limiting the feedback instead starves the LM and collapses the content early.)
        Frames already inside the band get gain 1 and pass through untouched, so natural loud/quiet
        dynamics are preserved; only the bf16-drift over-drive/clipping is pulled back."""
        rms = float(audio_1d.pow(2).mean().sqrt())
        peak = float(audio_1d.abs().max())
        gain = min(
            1.0,
            self._audio_limit_T / rms if rms > 1e-9 else 1.0,
            self._AUDIO_PEAK / peak if peak > 1e-9 else 1.0,
        )
        return audio_1d * gain if gain < 1.0 else audio_1d

    def _reset_neg_cache(self, kv_cache_neg: KVCache):
        """Negative prefill: single speech_start token."""
        if self._ref_lm is not None:
            return 1, self._ref_lm.reset_neg(self.speech_start_id)
        neg_ids = torch.tensor([[self.speech_start_id]], dtype=torch.long)
        neg_embeds = self.lm._embed(neg_ids)
        _, neg_hidden = self.lm.forward(neg_embeds, start_pos=0, kv_cache=kv_cache_neg, return_last_hidden=True)
        return 1, neg_hidden

    def _lm_prefill(
        self,
        inputs_embeds: ttnn.Tensor,
        kv_cache: KVCache,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        if self._ref_lm is not None:
            cpu = ttnn.to_torch(inputs_embeds).to(torch.float32).squeeze(1)
            return self._ref_lm.prefill(cpu)
        return self.lm.prefill_embeds(inputs_embeds, kv_cache=kv_cache, return_last_hidden=True)

    def _lm_step(
        self,
        inputs_embeds: ttnn.Tensor,
        start_pos: int,
        kv_cache: KVCache,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        if self._ref_lm is not None:
            cpu = ttnn.to_torch(inputs_embeds).to(torch.float32).squeeze(1)
            return self._ref_lm.step_embeds(cpu)
        logits, last_hidden = self.lm.forward(
            inputs_embeds,
            start_pos=start_pos,
            kv_cache=kv_cache,
            return_last_hidden=True,
        )
        return logits, last_hidden

    def _lm_decode_token(
        self,
        token_id: int,
        start_pos: int,
        kv_cache: KVCache,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        if self._ref_lm is not None:
            return self._ref_lm.step_token(token_id)
        token_ids = torch.tensor([[token_id]], dtype=torch.long)
        return self.lm.decode_step(token_ids, start_pos, kv_cache, return_last_hidden=True)

    def _neg_lm_step(self, token_id: int, neg_pos: int, kv_cache_neg: KVCache) -> ttnn.Tensor:
        if self._ref_lm is not None:
            return self._ref_lm.neg_step_token(token_id)
        neg_ids = torch.tensor([[token_id]], dtype=torch.long)
        _, neg_hidden = self.lm.decode_step(neg_ids, neg_pos, kv_cache_neg, return_last_hidden=True)
        return neg_hidden

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.Tensor] = None,
        speech_masks: Optional[torch.Tensor] = None,
        speech_input_mask: Optional[torch.Tensor] = None,
        prefill_speech_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        forced_token_ids: Optional[torch.Tensor] = None,
        rng: Optional[torch.Generator] = None,
    ) -> TTVibeVoiceOutput:
        """Run VibeVoice TTS generation aligned with reference generate().

        Pass ``forced_token_ids`` (1-D post-prefill token ids from reference generate)
        to replay the reference AR sequence on TT diffusion/decode — same duration and
        frame count as HuggingFace.
        """
        device = self.device
        cfg = self.lm.cfg

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        prof = _Profiler(device)

        # Op-level speech-frame profiling (VV_PROFILE_SPEECH_FRAME=<n>, 0=off): wrap the n-th
        # eager diffusion frame (neg-LM → diffusion → post → pos-LM → argmax) in Tracy
        # ``start``/``stop`` signposts so ``tt-perf-report --start-signpost start --end-signpost
        # stop`` isolates ONE warm frame.  VV_PROFILE_SPEECH_FRAME_EXIT=1 returns right after.
        # Env-gated + eager-path only (VV_TRACE_SEGMENT=0), so the shipping trace path is untouched.
        _profile_sf = int(os.environ.get("VV_PROFILE_SPEECH_FRAME", "0"))
        _profile_sf_exit = os.environ.get("VV_PROFILE_SPEECH_FRAME_EXIT", "0") == "1"

        _vv_debug(
            f"generate() start: input_ids={tuple(input_ids.shape)} "
            f"voice_cloning={speech_tensors is not None} "
            f"max_new_tokens={max_new_tokens} cfg_scale={self.cfg_scale} "
            f"diffusion_steps={self.num_diffusion_steps}"
        )

        _t_prefill_start = time.perf_counter()
        with prof.section("prefill_build_embeds (voice-clone encode)"):
            inputs_embeds = self._build_prefill_embeds(
                input_ids,
                speech_tensors,
                speech_masks,
                speech_input_mask,
                prefill_speech_embeds=prefill_speech_embeds,
            )
        prefill_len = inputs_embeds.shape[2]
        _vv_debug(
            f"prefill embeds built: seq_len={prefill_len} "
            f"speech_slots={int(speech_input_mask[0].sum().item()) if speech_input_mask is not None else 0} "
            f"scale={self.speech_scaling_factor} bias={self.speech_bias_factor}"
        )

        # Determine the max number of AR steps up front — it sizes the fixed KV cache.
        initial_length = input_ids.shape[-1]
        initial_len = int(attention_mask.sum(dim=-1)[0].item())
        forced_tokens: Optional[List[int]] = None
        if forced_token_ids is not None:
            forced_tokens = forced_token_ids.reshape(-1).tolist()
            if not forced_tokens:
                raise ValueError("forced_token_ids must be non-empty")
            max_steps = len(forced_tokens)
        elif max_new_tokens is not None:
            max_steps = max_new_tokens
        else:
            max_steps = min(
                cfg.max_position_embeddings - initial_length,
                int(self.max_length_times * initial_len),
            )

        # Preallocate fixed-size KV caches (TT LM path only).  Positive cache holds
        # prefill + all generated tokens; negative cache is reset per speech segment
        # (reused buffer), so it only needs to span one segment ≤ max_steps.
        if self._ref_lm is None:
            kv_cache_pos = self.lm.alloc_kv_cache(prefill_len + max_steps + 8)
            kv_cache_neg = self.lm.alloc_kv_cache(max_steps + 8)
        else:
            kv_cache_pos = create_kv_cache(cfg.num_hidden_layers)
            kv_cache_neg = create_kv_cache(cfg.num_hidden_layers)

        _profile_prefill = _vv_profile_prefill_enabled() and self._ref_lm is None
        if _profile_prefill:
            import tracy

            ttnn.synchronize_device(device)
            tracy.signpost("start")
            _vv_debug(f"Tracy signpost start: LM prefill seq_len={prefill_len}")
        with prof.section("lm_prefill"):
            logits_pos, prefill_hidden = self._lm_prefill(inputs_embeds, kv_cache_pos)
        if _profile_prefill:
            import tracy

            ttnn.synchronize_device(device)
            tracy.signpost("stop")
            _vv_debug(f"Tracy signpost stop: LM prefill seq_len={prefill_len}")
        _t_prefill_end = time.perf_counter()
        _vv_debug(f"lm_prefill done: kv_cache_pos size={prefill_len + max_steps + 8}")

        if _profile_prefill and os.environ.get("VV_PROFILE_PREFILL_EXIT", "0") == "1":
            _vv_debug("VV_PROFILE_PREFILL_EXIT=1 — ending generate after LM prefill")
            prof.report()
            return TTVibeVoiceOutput(
                sequences=input_ids.clone(),
                speech_outputs=[],
                prefill_wall_s=_t_prefill_end - _t_prefill_start,
                decode_wall_s=0.0,
            )

        neg_pos, neg_start_hidden = self._reset_neg_cache(kv_cache_neg)
        neg_prev_diffusion_token: Optional[int] = None  # delayed token for negative CFG

        sequences = input_ids.clone()
        # On-device streaming: each diffusion step decodes its audio chunk via the
        # acoustic decoder's causal cache; we accumulate the chunks to form the
        # final waveform (identical structure to the reference streaming decode).
        audio_chunks: List[torch.Tensor] = []
        # Optional disk-streaming (VV_STREAM_AUDIO=<path>): append each frame's fp32 samples to a raw
        # file (flushed → survives SIGKILL) instead of accumulating in host RAM.  Bounds host memory
        # on very long renders AND preserves partial audio if the process dies mid-run.  Numerically
        # identical (storage only).  Read back at the end (happy path) or offline from the .f32 file.
        _stream_path = os.environ.get("VV_STREAM_AUDIO", "")
        _stream_fh = open(_stream_path, "wb") if _stream_path else None

        def _emit_audio(chunk_1d: torch.Tensor) -> None:
            if _stream_fh is not None:
                chunk_1d.contiguous().numpy().tofile(_stream_fh)
                _stream_fh.flush()
            else:
                audio_chunks.append(chunk_1d)

        pending_embeds: Optional[ttnn.Tensor] = None

        # Fresh tokenizer streaming caches for this generation.
        self.acoustic_tok.reset_decode_cache()
        self.semantic_tok.reset_cache()
        if self.ref_inference is not None:
            self._reset_ref_tokenizer_caches()

        # On-device argmax (ttnn.argmax) — numerically identical to host fp32 argmax
        # (bf16→fp32 upcast is monotonic) and avoids copying the full vocab row.
        use_fp32_argmax = False
        forced_idx = 0
        if forced_tokens is not None:
            next_token = forced_tokens[0]
            forced_idx = 1
        else:
            next_token = _greedy_argmax(logits_pos, use_fp32=use_fp32_argmax)
        step_hidden = prefill_hidden
        _vv_debug(f"AR loop: max_steps={max_steps} first_token={next_token} ({self._token_label(next_token)})")

        # Pre-draw all diffusion init noise here — after the voice-encode RNG draws in
        # prefill, before the AR loop — hoisting the per-frame torch.randn out of the
        # loop.  torch.randn(N, ...) yields the same values, in order, as N sequential
        # per-frame draws, so this is bit-identical and keeps the global RNG aligned
        # with the reference.  Sized to max_steps (the upper bound on diffusion frames);
        # only the first #diffusion-frames rows are consumed.
        diffusion_noise: Optional[torch.Tensor] = None
        if self.ref_inference is None:
            diffusion_noise = torch.randn(max_steps, 2, 1, 1, 64, dtype=torch.float32, generator=rng).to(torch.bfloat16)

        # Anti-repetition loop-break (VV_LOOPBREAK, default on): watch the streamed audio for a
        # sustained content-repeat loop (long-context bf16-drift failure); on the loop frames re-draw
        # that frame's diffusion init noise to break out, and clamp the emitted audio to [-1, 1] over
        # the episode.  A no-op on clean audio (never fires) — see _LoopBreaker.
        loopbreaker = _LoopBreaker() if os.environ.get("VV_LOOPBREAK", "1") != "0" else None

        diffusion_frames = 0
        # Steady-state decode timing (cf. tt_transformers/llama demos): time ONLY the fused-frame
        # trace-replay frames — warmup and capture frames are not timed.
        _steady_decode_s = 0.0
        _steady_decode_frames = 0
        _t_decode_start = time.perf_counter()
        for step in range(max_steps):
            current_token = next_token
            sequences = torch.cat(
                [sequences, torch.tensor([[current_token]], dtype=torch.long)],
                dim=-1,
            )
            _vv_debug(f"step {step + 1}/{max_steps}: emit {self._token_label(current_token)}")

            if self._trace_segment and forced_tokens is None and current_token == self.speech_diffusion_id:
                # WHOLE-SEGMENT fused trace (llama shape): every speech-diffusion frame — INCLUDING
                # a segment's first frame (which folds the negative prefill) — replays one
                # device-driven capture.  step_hidden (the speech_start / prior-token pos-LM hidden)
                # seeds frame 0; the pos hidden is then loop-carried on device, positions
                # self-advance, RoPE is gathered on device.  Time ONLY steady replay frames
                # (a segment's first frame recaptures and is not timed).
                seg_frame_idx = 0 if neg_prev_diffusion_token is None else 1
                _sf_replay = self._sf_tid is not None
                _frame_t0 = time.perf_counter() if _sf_replay else None
                diffusion_frames += 1
                noise_2x = diffusion_noise[diffusion_frames - 1] if diffusion_noise is not None else None
                # Loop-break: on the frames of a detected loop, re-draw the diffusion init noise.
                if loopbreaker is not None and loopbreaker.active and noise_2x is not None:
                    noise_2x = loopbreaker.perturb(noise_2x, diffusion_frames)
                start_pos = prefill_len + step
                with prof.section("segment_frame"):
                    audio_chunk, _tok_or_logits = self._run_segment_frame_traced(
                        seg_frame_idx, step_hidden, start_pos, noise_2x, kv_cache_pos, kv_cache_neg
                    )
                neg_prev_diffusion_token = current_token
                _frame_audio = ttnn.to_torch(audio_chunk).to(torch.float32).reshape(-1)  # syncs frame
                if loopbreaker is not None:
                    loopbreaker.update(_frame_audio)  # detect on the raw (unclamped) audio
                    if loopbreaker.clamp_now():
                        _frame_audio = _frame_audio.clamp(-1.0, 1.0)
                if self._audio_limit_T > 0.0:
                    _frame_audio = self._emit_limit(_frame_audio)  # emit-only energy stabilizer
                _emit_audio(_frame_audio)
                if self._sf_cap_split:
                    # Split-capture folds the constrained argmax into the trace → LOCAL index.
                    with prof.section("argmax"):
                        _local_idx = int(ttnn.to_torch(_tok_or_logits).reshape(-1)[-1].item())  # syncs frame
                        next_token = self._sf_valid_ids_sorted[_local_idx]
                else:
                    with prof.section("token_constraint"):
                        logits = ttnn.add(
                            _tok_or_logits,
                            self._token_constraint_mask(_tok_or_logits.shape[-1]),
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    with prof.section("argmax"):
                        next_token = _greedy_argmax(logits, use_fp32=use_fp32_argmax)  # syncs frame (D2H)
                if _sf_replay:
                    _steady_decode_s += time.perf_counter() - _frame_t0
                    _steady_decode_frames += 1
                continue

            if current_token == self.speech_diffusion_id:
                diffusion_frames += 1
                if _profile_sf and diffusion_frames == _profile_sf:
                    import tracy

                    ttnn.synchronize_device(device)
                    tracy.signpost("start")
                    _vv_debug(f"Tracy signpost start: eager speech frame {diffusion_frames}")
                cond_pos = _condition_from_hidden(step_hidden)
                # Negative CFG: reference processes the PREVIOUS speech_diffusion_id
                # at each step (the current one is appended to negative_input_ids
                # AFTER the negative forward).  For the first diffusion step in a
                # segment, the reference runs the negative model on speech_start_id
                # alone — we captured that hidden in neg_start_hidden.
                with prof.section("neg_lm_step"):
                    if neg_prev_diffusion_token is None:
                        neg_hidden = neg_start_hidden
                    else:
                        neg_hidden = self._neg_lm_step(neg_prev_diffusion_token, neg_pos, kv_cache_neg)
                        neg_pos += 1
                    neg_prev_diffusion_token = current_token
                    cond_neg = _condition_from_hidden(neg_hidden)

                with prof.section("diffusion (CFG x num_steps)"):
                    noise_2x = diffusion_noise[diffusion_frames - 1] if diffusion_noise is not None else None
                    if loopbreaker is not None and loopbreaker.active and noise_2x is not None:
                        noise_2x = loopbreaker.perturb(noise_2x, diffusion_frames)
                    speech_latent = self._run_speech_diffusion(
                        cond_pos, cond_neg, latent_size=64, noise_2x=noise_2x, rng=rng
                    )

                # On-device streaming: fused next-step embed + this frame's audio chunk.
                with prof.section("post_diffusion (decode+sem_enc+conn)"):
                    pending_embeds, audio_chunk = self._post_diffusion_embeds(speech_latent)
                with prof.section("audio_chunk -> host"):
                    _chunk = (
                        audio_chunk.to(torch.float32).reshape(-1)
                        if isinstance(audio_chunk, torch.Tensor)
                        else ttnn.to_torch(audio_chunk).to(torch.float32).reshape(-1)
                    )
                    if loopbreaker is not None:
                        loopbreaker.update(_chunk)  # detect on the raw (unclamped) audio
                        if loopbreaker.clamp_now():
                            _chunk = _chunk.clamp(-1.0, 1.0)
                    if self._audio_limit_T > 0.0:
                        _chunk = self._emit_limit(_chunk)  # emit-only energy stabilizer
                    _emit_audio(_chunk)
                chunk_samples = _chunk.numel()
                _vv_debug(
                    f"  diffusion frame {diffusion_frames}: audio_chunk={chunk_samples} samples "
                    f"({chunk_samples / 24000:.3f}s)"
                )

            if current_token == self.eos_token_id:
                _vv_debug(f"EOS at step {step + 1}")
                break

            start_pos = prefill_len + step
            with prof.section("pos_lm_step"):
                if pending_embeds is not None:
                    logits, step_hidden = self._lm_step(pending_embeds, start_pos, kv_cache_pos)
                    pending_embeds = None
                else:
                    logits, step_hidden = self._lm_decode_token(current_token, start_pos, kv_cache_pos)

            if current_token == self.speech_start_id:
                if self._trace_segment:
                    # Whole-segment fused trace: release the capture so the boundary's eager LM
                    # decodes can't corrupt it, then let the next diffusion frame (frame 0) rewind
                    # positions, re-seed hidden, fold the negative prefill and zero the conv caches
                    # IN PLACE.  Do NOT free/realloc the conv or neg-KV caches here — that would
                    # move address-stable state out from under the recaptured trace.
                    _vv_debug("  new speech segment: release segment trace (recapture next frame)")
                    self._reset_segment_frame_trace()
                    neg_prev_diffusion_token = None
                else:
                    _vv_debug("  new speech segment: reset neg-CFG cache + acoustic/semantic streaming caches")
                    neg_pos, neg_start_hidden = self._reset_neg_cache(kv_cache_neg)
                    neg_prev_diffusion_token = None
                    self.acoustic_tok.reset_decode_cache()
                    self.semantic_tok.reset_cache()
                    if self.ref_inference is not None:
                        self._reset_ref_tokenizer_caches()

            with prof.section("token_constraint"):
                logits = ttnn.add(
                    logits,
                    self._token_constraint_mask(logits.shape[-1]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            with prof.section("argmax"):
                if forced_tokens is not None:
                    next_token = forced_tokens[forced_idx] if forced_idx < len(forced_tokens) else self.eos_token_id
                    forced_idx += 1
                else:
                    next_token = _greedy_argmax(logits, use_fp32=use_fp32_argmax)

            if _profile_sf and diffusion_frames == _profile_sf and current_token == self.speech_diffusion_id:
                import tracy

                ttnn.synchronize_device(device)
                tracy.signpost("stop")
                _vv_debug(f"Tracy signpost stop: eager speech frame {diffusion_frames}")
                if _profile_sf_exit:
                    _vv_debug("VV_PROFILE_SPEECH_FRAME_EXIT=1 — ending generate after profiled frame")
                    break

        _t_decode_end = time.perf_counter()
        # The per-step streaming decode already produced each frame's audio chunk
        # (with full causal context via the decoder cache); concatenate for the
        # final waveform — no separate batch decode needed.
        if _stream_fh is not None:
            _stream_fh.flush()
            _stream_fh.close()
            import numpy as _np

            speech_waveform = torch.from_numpy(_np.fromfile(_stream_path, dtype=_np.float32).copy())
        elif audio_chunks:
            speech_waveform = torch.cat(audio_chunks, dim=0)
        else:
            speech_waveform = torch.zeros(0)

        ar_tokens = sequences.shape[1] - input_ids.shape[1]
        _vv_debug(
            f"generate() done: ar_tokens={ar_tokens} diffusion_frames={diffusion_frames} "
            f"audio_samples={speech_waveform.numel()} ({speech_waveform.numel() / 24000:.2f}s)"
        )
        prof.report()

        return TTVibeVoiceOutput(
            sequences=sequences,
            speech_outputs=[speech_waveform],
            prefill_wall_s=_t_prefill_end - _t_prefill_start,
            decode_wall_s=_t_decode_end - _t_decode_start,
            steady_decode_s=_steady_decode_s,
            steady_decode_frames=_steady_decode_frames,
        )

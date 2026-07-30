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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import ttnn

# Optional env-gated diagnostics for generate():
#   VV_PROFILE=1 — device-synced timing breakdown per phase
#   VV_DEBUG=1   — per-AR-step token + phase logs (also set by demo/demo.py --debug)
#   VV_PROFILE_PREFILL=1 — Tracy signposts ``start``/``stop`` around LM prefill
#     (``_lm_prefill`` only). Use with ``python -m tracy …`` then
#     ``tt-perf-report <csv> --start-signpost start --end-signpost stop``.
#   VV_PROFILE_PREFILL_EXIT=1 — return from generate() right after LM prefill (no AR).
#   VV_PROFILE_DIFFUSION=<n> — Tracy start/stop around the n-th eager diffusion call
#     (``_run_speech_diffusion`` / sample_speech_latents only; VV_TRACE_SEGMENT=0).
#   VV_PROFILE_DIFFUSION_EXIT=1 — return from generate() right after that diffusion call.


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
from models.experimental.vibevoice.reference.lm_runner import ReferenceLMRunner


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

        # WHOLE-SEGMENT fused trace (VV_TRACE_SEGMENT=1). Demo / ISL sweep enable this by
        # default and open the device with a ~1.4 GB trace region + 2 CQs; other entry points
        # stay eager unless they set the env and reserve the region.  The true llama shape —
        # a fully device-driven fused frame (the whole steady-state
        # speech-diffusion frame — neg-LM → diffusion → post-diffusion → pos-LM — as ONE graph),
        # so there are NO per-frame host RoPE/position writes and NO capture-poison re-run:
        # positions self-advance via ttnn.plus_one INSIDE the trace, RoPE rows are gathered on
        # device (bf16) from the device position, the neg embed is a per-frame input (a segment's
        # first frame decodes embed(speech_start) at neg_pos 0 — its neg-LM IS the negative
        # prefill — then embed(speech_diffusion)), and the pos hidden is loop-carried on device.
        # Lifecycle per segment: warmup -> throwaway capture -> reset (rewind positions, re-seed
        # hidden, zero the conv streaming caches IN PLACE) -> pure replay.  The trace is released
        # at each speech_start (so the boundary's eager LM decodes cannot corrupt a live capture)
        # and recaptured per segment; a single-segment generation captures once.  Replay is PCC 1.0
        # against the eager device-RoPE path; its bf16 RoPE puts it at ~0.9999 vs the fp32
        # reference, the same accepted precision as the bf16 SDPA-decode.
        self._trace_segment = os.environ.get("VV_TRACE_SEGMENT", "0") == "1"
        # Voice-clone acoustic-encode chunk trace (see _ensure_encode_trace): (trace_id, out_tensor)
        # for the steady chunk and for the row's final chunk, plus their shared input buffer.  Lives
        # only for the duration of the prefill encode.
        self._enc_step: Optional[Tuple[int, ttnn.Tensor]] = None
        self._enc_final: Optional[Tuple[int, ttnn.Tensor]] = None
        self._enc_in: Optional[ttnn.Tensor] = None
        # Device-side encode audio (VV_DEV_ENC_AUDIO=1, default on): upload a voice row ONCE as a
        # [n_chunks, chunk] table and gather the chunk row inside the capture from a self-advancing
        # device index — 4 host uploads for the climate prompts instead of ~663.  =0 keeps the
        # per-chunk host write (A/B + rollback).
        self._enc_dev_audio = os.environ.get("VV_DEV_ENC_AUDIO", "1") == "1"
        self._enc_table: Optional[ttnn.Tensor] = None
        self._enc_idx: Optional[ttnn.Tensor] = None
        self._sf_tid = None
        self._sf_warm = 0
        self._sf_hidden_buf: Optional[ttnn.Tensor] = None  # loop-carried cond_pos source
        self._sf_hidden_seed: Optional[ttnn.Tensor] = None  # segment-start hidden ([1,1,1,H], last pos)
        self._sf_neg_embed: Optional[ttnn.Tensor] = None  # neg-LM input = prev frame's inputs_embeds
        self._sf_neg_start: Optional[ttnn.Tensor] = None  # const embed(speech_start_id)
        self._sf_pos_pos: Optional[ttnn.Tensor] = None
        self._sf_neg_pos: Optional[ttnn.Tensor] = None
        self._sf_noise: Optional[ttnn.Tensor] = None
        # Device-side diffusion noise (VV_DEV_NOISE=1, default on): the whole run's pre-drawn noise
        # is uploaded once as a [max_steps, 64] table and the frame's row is gathered INSIDE the
        # capture from a device index that self-advances — so a replayed frame does no host work for
        # noise at all.  =0 restores the per-frame host write (A/B + rollback).
        self._sf_dev_noise = os.environ.get("VV_DEV_NOISE", "1") == "1"
        # VV_TTNN_RANDN=1 draws that table with ttnn.randn ON DEVICE instead of torch.randn on host.
        # EXPERIMENTAL, default off: ttnn.randn is a different generator (device Box-Muller over
        # per-core PRNGs), so it does NOT reproduce torch's values for a seed — the diffusion init
        # noise changes and every rendered sample differs from the torch reference.  Measured vs
        # torch.randn: tails match (P(|z|>3) 0.00265 vs 0.00270 at n=1M), but small draws are
        # under-dispersed and slightly negative-biased (a [400,64] table came out std 0.938-1.002
        # across seeds, mean -0.004..-0.040, vs torch's steady std~1.000).  Judge by listening.
        self._sf_ttnn_randn = os.environ.get("VV_TTNN_RANDN", "0") == "1"
        self._ttnn_randn_draws = 0  # draw counter, so successive device draws use distinct seeds
        self._sf_noise_table: Optional[ttnn.Tensor] = None
        self._sf_noise_idx: Optional[ttnn.Tensor] = None
        self._sf_t_tensors: Optional[list] = None
        # Schedule-constant timestep embeddings (embed_timestep(t) per DPM step).  Built once with
        # `_sf_t_tensors` and reused every frame — byte-identical to per-step embed inside the head.
        self._sf_t_embs: Optional[list] = None
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
        # bit-identical to eager.  VV_CAP_SPLIT=0 falls back to the monolithic capture and reproduces
        # that divergence.  _sf_neg_hidden and _sf_fused_out are the address-stable hand-off buffers
        # between the three traces.
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
        # PREVIOUS frame's forward wrote.  A once-per-segment eager boot seeds neg-LM(0).  Each row
        # is byte-identical to its B=1 forward.  Requires cap-split token semantics (in-trace
        # constrained argmax).
        self._sf_cfg_b2 = self._sf_cap_split and os.environ.get("VV_CFG_BATCH2", "1") == "1"
        # Fused frame output (VV_FUSE_FRAME_OUT=1, default on): append the constrained-argmax index
        # to this frame's audio inside _lm2trace, so ONE D2H returns both.  The audio and the token
        # are complete at the same point in the queue (dp2 is enqueued before lm2), and reading the
        # 4-byte token separately costs ~0.06 ms of per-call overhead on every frame.  =0 keeps the
        # two reads (A/B + rollback).
        self._sf_out_fused = self._sf_cfg_b2 and os.environ.get("VV_FUSE_FRAME_OUT", "1") == "1"
        self._sf_dp2trace_tid = None
        self._sf_lm2trace_tid = None
        # Diagnostic (VV_TRACE_NOCAPTURE=1): run the frame graph eagerly, with no ttnn
        # capture/replay.  Also bit-clean but slower, which isolates capture aliasing from the
        # graph ops themselves.
        self._sf_nocapture = os.environ.get("VV_TRACE_NOCAPTURE", "0") == "1"
        self._sf_nocap_started = False
        # Diagnostic (VV_LOG_TRAJ=<csv path>, default off): per-frame loop-state trace — the
        # loop-carried hidden's rms/absmax alongside the emitted audio's rms/peak, keyed by both
        # frame index and absolute position.  Used to tell an absolute-position degradation apart
        # from cumulative AR-feedback drift; the frame audio is already synced here, so this adds
        # only one small [1,1,1,H] D2H read per frame.
        self._traj_path = os.environ.get("VV_LOG_TRAJ", "")
        self._traj_fh = None

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
        from models.experimental.vibevoice.reference.modular.modular_vibevoice_tokenizer import (
            VibeVoiceTokenizerStreamingCache,
        )

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

    def _ensure_encode_trace(self, chunk: int, n_rows: int = 0) -> bool:
        """Capture the streaming acoustic-encode chunk graph once; replayed for every chunk.

        The voice-clone encode dispatches the whole conv encoder per 3200-sample chunk (~663
        chunks for the 4 climate prompts), and measured it is entirely HOST-bound: ~38.6 ms of
        op dispatch per chunk against an idle device.  Capturing the graph once and replaying it
        drops that to the device floor (10.9 ms/chunk measured, vs 11.1 ms achieved).

        Trace-safe by the same properties the fused-frame decode trace relies on: TTConv1d holds
        its streaming cache in a fixed-address buffer updated in place (``ttnn.copy``), and the
        prepared conv weights are cached per input geometry — one geometry here, since every
        chunk is `chunk` samples wide.  Two graphs are captured because ``is_final_chunk`` adds a
        ceil-alignment right-pad and so changes conv widths; a row's last chunk replays that one.

        Warm-up runs both variants eagerly first (allocate caches, prepare weights, fill the
        program cache) so capture records no allocation.  Warm-up and capture leave the streaming
        caches dirty; the caller zeroes them in place before the first real chunk, which restores
        exactly the fresh ``ttnn.zeros`` state the eager path starts from.  Verified bit-exact vs
        the eager encode (maxabsdiff 0.0) on the climate voice prompts.

        Returns False when no trace region is reserved (``--no-trace``), leaving the eager path.
        """
        if self._enc_step is not None:
            return True
        if not self._trace_segment:
            return False
        dev = self.device
        self._enc_in = ttnn.from_torch(
            torch.zeros(1, 1, 1, chunk, dtype=torch.bfloat16),
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self._enc_dev_audio and n_rows:
            # Row table + index, allocated BEFORE the capture so their addresses are stable for it.
            # Sized to the longest row (rows are processor-padded to a common length), with a floor
            # of the warmup count so the eager warmup gathers stay in bounds.
            self._enc_table = ttnn.zeros(
                [max(n_rows, 4), chunk],
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._enc_idx = ttnn.zeros(
                [1],
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        self.acoustic_tok._encoder_tt.reset_cache()
        for _final in (False, False, True, True):
            self.acoustic_tok.encode(self._enc_input(chunk), use_cache=True, is_final_chunk=_final)
        ts = ttnn.begin_trace_capture(dev, cq_id=0)
        out_s = self.acoustic_tok.encode(self._enc_input(chunk), use_cache=True, is_final_chunk=False)
        ttnn.end_trace_capture(dev, ts, cq_id=0)
        tf = ttnn.begin_trace_capture(dev, cq_id=0)
        out_f = self.acoustic_tok.encode(self._enc_input(chunk), use_cache=True, is_final_chunk=True)
        ttnn.end_trace_capture(dev, tf, cq_id=0)
        self._enc_step, self._enc_final = (ts, out_s), (tf, out_f)
        _vv_debug(f"acoustic encode: captured chunk trace (chunk={chunk})")
        return True

    def _enc_input(self, chunk: int) -> ttnn.Tensor:
        """The chunk graph's audio input: this chunk's row gathered on device from the pre-uploaded
        row table, with the index self-advancing so a replay needs no host write.  Falls back to the
        host-written buffer when VV_DEV_ENC_AUDIO=0.  Bit-identical either way — the table holds the
        same bf16 samples, zero-padded exactly as the per-chunk path padded its tail, and
        ``ttnn.embedding`` is pure data movement (verified byte-equal at every row)."""
        if self._enc_table is None:
            return self._enc_in
        idx = ttnn.reshape(ttnn.typecast(self._enc_idx, ttnn.uint32), [1, 1])
        row = ttnn.embedding(idx, self._enc_table, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.plus_one(self._enc_idx)
        return ttnn.reshape(row, [1, 1, 1, chunk])

    def _enc_upload_row(self, wav: torch.Tensor, chunk: int, n_chunks: int) -> None:
        """One H2D for a whole voice row, replacing one per chunk, and rewind the gather index.

        Written into the persistent table address (so the capture stays valid across rows).  The row
        is zero-padded to the table's full height; rows past this voice's ``n_chunks`` are never
        gathered."""
        rows = torch.nn.functional.pad(wav, (0, n_chunks * chunk - wav.numel())).reshape(n_chunks, chunk)
        table_rows = self._enc_table.shape[0]
        if n_chunks < table_rows:
            rows = torch.nn.functional.pad(rows, (0, 0, 0, table_rows - n_chunks))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(rows, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), self._enc_table
        )
        self._sf_write_int(self._enc_idx, 0)

    def _release_encode_trace(self) -> None:
        """Drop the encode traces.  Called once the voice prompts are encoded, so the eager LM
        prefill that follows — and later the fused-frame captures — never allocate against a live
        capture (the coexistence hazard `_reset_segment_frame_trace` guards for the decode path)."""
        if self._enc_step is None:
            return
        ttnn.release_trace(self.device, self._enc_step[0])
        ttnn.release_trace(self.device, self._enc_final[0])
        self._enc_step = self._enc_final = None
        self._enc_in = self._enc_table = self._enc_idx = None
        # Free the fixed-address streaming caches too: nothing references their addresses now, and
        # the decode path allocates its own.
        self.acoustic_tok._encoder_tt.reset_cache()

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

        # One bf16 cast for the whole row instead of one per chunk.  The cast is elementwise, so
        # slicing it and zero-padding it below are bit-identical to casting each chunk (0.0 is
        # exact in bf16); the trim above still runs on the fp32 samples.
        wav = wav.to(torch.bfloat16)

        if total_samples <= chunk:
            self._release_encode_trace()  # reset_cache() below would move the captured addresses
            self.acoustic_tok._encoder_tt.reset_cache()
            lat_tt = self.acoustic_tok.encode(
                self._audio_row_to_tt(wav),
                use_cache=False,
                is_final_chunk=True,
            )
            lat = self._latents_from_encode_output(lat_tt)
        else:
            n_chunks = -(-total_samples // chunk)
            # Size the table from the UNTRIMMED length: rows are processor-padded to a common
            # length, so the first row's bound covers every later row (the capture pins the address).
            traced = self._ensure_encode_trace(chunk, -(-wav_1d.numel() // chunk))
            if traced:
                # Zero the streaming caches IN PLACE (the fresh-alloc state is ttnn.zeros, so this
                # is exactly equivalent) — reset_cache() would free them out from under the trace.
                self.acoustic_tok._encoder_tt.reset_cache_inplace()
                if self._enc_table is not None:
                    self._enc_upload_row(wav, chunk, n_chunks)
            else:
                self.acoustic_tok._encoder_tt.reset_cache()
            frames: List[torch.Tensor] = []
            pos = 0
            while pos < total_samples:
                n = min(chunk, total_samples - pos)
                is_final = pos + n >= total_samples
                if self._enc_table is None or not traced:
                    chunk_wav = wav[pos : pos + n]
                    if chunk_wav.numel() < chunk:
                        # conv2d caches prepared weights per input width; keep chunks fixed-size.
                        chunk_wav = torch.nn.functional.pad(chunk_wav, (0, chunk - chunk_wav.numel()))
                if traced:
                    if self._enc_table is None:
                        ttnn.copy_host_to_device_tensor(
                            ttnn.from_torch(
                                chunk_wav.view(1, 1, 1, -1),
                                dtype=ttnn.bfloat16,
                                layout=ttnn.ROW_MAJOR_LAYOUT,
                            ),
                            self._enc_in,
                        )
                    # else: the replay gathers its own row (index self-advances on device)
                    tid, lat_tt = self._enc_final if is_final else self._enc_step
                    ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
                else:
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
            if self._sf_ttnn_randn:
                # VV_TTNN_RANDN=1: draw the fix-std jitter on device.  ``lat`` is already a host
                # tensor here (the encode output is D2H'd per chunk), so this adds one D2H per
                # voice row — the point is to move the RANDOMNESS off torch, not to save time.
                jitter = ttnn.to_torch(
                    ttnn.randn(
                        list(lat.shape),
                        device=self.device,
                        dtype=ttnn.float32,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        seed=self._ttnn_randn_seed(),
                    )
                ).to(lat.dtype)
            else:
                jitter = torch.randn_like(lat)
            lat = lat + self.acoustic_fix_std * jitter
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
    ) -> ttnn.Tensor:
        """Return speech embeds [1, 1, N_slots, hidden] float32 ON DEVICE for the prefill scatter."""
        scale = self.speech_scaling_factor
        bias = self.speech_bias_factor
        latents_per_row = []
        try:
            for i in range(speech_tensors.shape[0]):
                latents_per_row.append(self._encode_acoustic_latents(speech_tensors[i]))
        finally:
            # Everything after this — the connector, the LM prefill, the fused-frame captures —
            # allocates eagerly, so the encode capture must not still be live.
            self._release_encode_trace()

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
            # Stay on device: fp32 here is the same exact widening the host path did via
            # to_torch().to(float32), and the prefill embed tensor is fp32.
            conn_out = ttnn.typecast(self.acoustic_conn(feats_tt), ttnn.float32)
            if conn_out.shape[2] != n:
                conn_out = ttnn.slice(
                    conn_out, [0, 0, 0, 0], [1, 1, n, conn_out.shape[-1]], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            speech_embeds_parts.append(conn_out)

        if len(speech_embeds_parts) == 1:
            return speech_embeds_parts[0]
        return ttnn.concat(speech_embeds_parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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

        if prefill_speech_embeds is not None:
            speech_dev = _host_2d_to_embeds(prefill_speech_embeds.to(torch.float32), self.device, dtype=torch.float32)
        elif speech_tensors is not None and speech_masks is not None:
            speech_dev = self._process_speech_prefill(speech_tensors, speech_masks)
        else:
            return inputs_embeds
        return self._scatter_speech_embeds(inputs_embeds, speech_dev, speech_input_mask[0].cpu().bool())

    def _scatter_speech_embeds(
        self,
        inputs_embeds: ttnn.Tensor,
        speech_dev: ttnn.Tensor,
        mask: torch.Tensor,
    ) -> ttnn.Tensor:
        """Splice the voice embeds into the text embeds' speech slots, entirely on device.

        Replaces a host round trip that pulled the whole [S, hidden] prefill embed down as fp32
        (~141 MB D2H), did a boolean-mask assign, and pushed it back (~141 MB H2D) — measured
        0.39 s.  The speech slots are contiguous runs (one per speaker: the mask for the 4-speaker
        climate prompt is 9 runs total), so the scatter is just alternating slices of the text and
        speech tensors concatenated back together — pure data movement, no arithmetic.

        Bit-exact vs the host path: ``inputs_embeds`` is bf16 and the widening to fp32 is exact,
        the speech side is already fp32, and slice/concat move values unchanged.  Tile alignment
        is not required — the run boundaries are arbitrary and TILE-layout concat handles them.
        """
        S, hidden = inputs_embeds.shape[2], inputs_embeds.shape[-1]
        m = mask[:S].to(torch.int8)
        # Run-length decompose the mask: boundaries are where it flips.
        bounds = [0] + ((m[1:] != m[:-1]).nonzero().reshape(-1) + 1).tolist() + [S]
        made_f32 = inputs_embeds.dtype != ttnn.float32
        text_f32 = ttnn.typecast(inputs_embeds, ttnn.float32) if made_f32 else inputs_embeds

        parts, spos = [], 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if int(m[a]) == 0:
                parts.append(
                    ttnn.slice(text_f32, [0, 0, a, 0], [1, 1, b, hidden], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                )
            else:
                parts.append(
                    ttnn.slice(
                        speech_dev,
                        [0, 0, spos, 0],
                        [1, 1, spos + (b - a), hidden],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                spos += b - a
        if made_f32:
            ttnn.deallocate(text_f32)  # the slices above are copies; the 141 MB source is dead
        if len(parts) == 1:
            return parts[0]
        return ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _ensure_diffusion_t_embs(self) -> list:
        """Build schedule-constant DPM timestep tensors + embeddings once (eager + traced)."""
        if self._sf_t_embs is not None:
            return self._sf_t_embs
        dev = self.device
        self.scheduler.set_timesteps(self.num_diffusion_steps)
        if self._sf_t_tensors is None:
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
        self._sf_t_embs = [self.diffusion_head.embed_timestep(t) for t in self._sf_t_tensors]
        return self._sf_t_embs

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
        t_embs = self._ensure_diffusion_t_embs()
        return sample_speech_latents(
            self.diffusion_head,
            condition,
            neg_condition,
            self.scheduler,
            initial_latent,
            cfg_scale=self.cfg_scale,
            num_steps=self.num_diffusion_steps,
            head_runner=None,
            t_tensors=self._sf_t_tensors,
            t_embs=t_embs,
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

    def _sf_noise_row(self) -> ttnn.Tensor:
        """Gather this frame's diffusion init-noise row on device and advance the index.

        Called from INSIDE the frame capture, so on replay the noise costs no host work: the whole
        run's noise is pre-drawn on host and uploaded once (``_sf_upload_noise_table``), and the row
        index self-advances here exactly like the device positions do.  The index is rewound by
        ``_sf_set_inputs_b2(0, ...)``, which runs before every warmup / capture / segment-start
        frame, so it stays in lockstep with the AR loop's ``diffusion_frames``.

        Bit-identical to the per-frame host write it replaces: the same pre-drawn bf16 values, and
        ``ttnn.embedding`` on a bf16 table is pure data movement (verified equal at every index).
        ``embedding`` requires a uint32 [1, 1] index, hence the typecast+reshape (as in the LM's
        on-device RoPE row gather)."""
        idx = ttnn.reshape(ttnn.typecast(self._sf_noise_idx, ttnn.uint32), [1, 1])
        row = ttnn.embedding(idx, self._sf_noise_table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.plus_one(self._sf_noise_idx)
        return ttnn.reshape(row, [1, 1, 1, row.shape[-1]])

    def _ttnn_randn_seed(self) -> int:
        """Deterministic seed for the next VV_TTNN_RANDN device draw.

        Derived by READING the run's torch seed (``--seed``, set via torch.manual_seed) rather than
        drawing from it, so no host RNG is consumed and the torch stream is left untouched; the
        counter makes successive draws differ (each voice row's jitter, then the noise table)."""
        self._ttnn_randn_draws += 1
        return (int(torch.initial_seed()) + self._ttnn_randn_draws) & 0x7FFFFFFF

    def _sf_randn_noise_table(self, max_steps: int, seed: int) -> bool:
        """VV_TTNN_RANDN=1: draw the run's diffusion init noise ON DEVICE as the [max_steps, 64]
        bf16 gather table, replacing the host torch.randn + H2D upload.

        Generated directly at the table's final shape and ROW_MAJOR layout (what ttnn.embedding
        consumes) — no reshape, so there is no tiled-reshape hazard.  Returns False when the gather
        table is not the active noise path (eager / --no-trace / VV_DEV_NOISE=0), leaving the caller
        on the host draw."""
        if not (self._sf_dev_noise and self._trace_segment and self._sf_cfg_b2):
            return False
        self._sf_noise_table = ttnn.randn(
            [max_steps, 64],
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            seed=seed,
        )
        return True

    def _sf_upload_noise_table(self, diffusion_noise: torch.Tensor) -> None:
        """Upload the run's pre-drawn diffusion noise as a [max_steps, 64] bf16 gather table.

        ``diffusion_noise`` is [max_steps, 2, 1, 1, 64]; only row 0 of the CFG pair is consumed per
        frame (``noise_2x[:1]``), so the table holds exactly the values the host used to upload."""
        if not (self._sf_dev_noise and self._trace_segment and self._sf_cfg_b2) or diffusion_noise is None:
            return  # only the cfg_b2 frame graph gathers it; other paths keep the host write
        rows = diffusion_noise[:, 0].reshape(diffusion_noise.shape[0], -1)
        self._sf_noise_table = ttnn.from_torch(
            rows,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _sf_set_inputs(self, seg_frame_idx: int, start_pos: int, noise_2x) -> None:
        """Per-frame non-allocating writes into the persistent trace buffers.  A segment's first
        frame (seg_frame_idx==0) rewinds the device positions, re-seeds the loop-carried hidden from
        the (already-sliced) segment-start hidden, and selects embed(speech_start) — so its neg-LM at
        neg_pos 0 IS the negative prefill; later frames read the PREVIOUS frame's fused embed, which
        the frame graph itself wrote into _sf_neg_embed (see _dptrace/_frame), and let the positions
        self-advance (ttnn.plus_one) on device.  All writes here are host->device or device->device
        copies into fixed-address buffers (no allocation), so they are safe to run while the fused
        trace is live."""
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
        if self._sf_fp32_rope and not self.lm._fused_rope:
            # fp32 rope rows for the current positions (device positions self-advance for KV/sdpa).
            self._sf_write_rope(self._sf_cos_pos, self._sf_sin_pos, self._sf_pos_pos_host)
            self._sf_write_rope(self._sf_cos_neg, self._sf_sin_neg, self._sf_neg_pos_host)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(noise_2x[:1].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            self._sf_noise,
        )

    def _log_traj(self, frame_idx: int, abs_pos: int, audio_1d: torch.Tensor) -> None:
        """Append one row of loop-state diagnostics to VV_LOG_TRAJ (see __init__).

        Also records the CFG contrast cos(cond_pos, cond_neg).  The diffusion samples
        ``neg + cfg*(pos - neg)``, so if the loop-fed positive condition drifts onto the
        feedback-free negative one the guidance term vanishes and the head emits the
        negative's prediction (silence) — logged to test that against the measured latch.
        """
        if self._traj_fh is None:
            # VV_LOG_TRAJ names the csv outright, so there is no base to pin it under.
            # Absolutized and extension-checked inline, so the path reaching ``open`` is a
            # normalized .csv artifact path.
            traj_path = os.path.abspath(str(self._traj_path))
            if not traj_path.endswith(".csv"):
                raise ValueError(f"refusing output path {traj_path!r}: expected a .csv file")
            self._traj_fh = open(traj_path, "w")
            self._traj_fh.write(
                "frame,abs_pos,hidden_rms,hidden_absmax,audio_rms,audio_peak,neg_rms,cos_pos_neg,posneg_dist\n"
            )
        h = ttnn.to_torch(self._sf_hidden_buf).to(torch.float32).reshape(-1)
        if self._sf_neg_hidden is not None:
            n = ttnn.to_torch(self._sf_neg_hidden).to(torch.float32).reshape(-1)
            n = n[-h.numel() :] if n.numel() >= h.numel() else n
            cos = float(torch.nn.functional.cosine_similarity(h, n, dim=0))
            neg_rms, dist = float(n.pow(2).mean().sqrt()), float((h - n).pow(2).mean().sqrt())
        else:
            cos = neg_rms = dist = float("nan")
        self._traj_fh.write(
            f"{frame_idx},{abs_pos},"
            f"{float(h.pow(2).mean().sqrt()):.6e},{float(h.abs().max()):.6e},"
            f"{float(audio_1d.pow(2).mean().sqrt()):.6e},{float(audio_1d.abs().max()):.6e},"
            f"{neg_rms:.6e},{cos:.6f},{dist:.6e}\n"
        )
        if frame_idx % 64 == 0:
            self._traj_fh.flush()

    def _sf_set_inputs_b2(self, seg_frame_idx: int, start_pos: int, noise_2x, noise_idx: int = 0) -> None:
        """Per-frame input writes for the CFG batch-2 path.  Sets ONLY the lm2/dp2 inputs — the
        neg row's embed (speech_diffusion) and neg RoPE are managed by _sf_boot at frame 0.  Frame 0
        rewinds device positions + reseeds the loop-carried hidden; the batched forward's pos row
        reads pos_pos (@pos_pos_host) and the neg row reads neg_pos (one AHEAD, set by the boot)."""
        if seg_frame_idx == 0:
            self._sf_write_int(self._sf_pos_pos, start_pos)
            self._sf_write_int(self._sf_neg_pos, 0)
            ttnn.copy(input_a=self._sf_hidden_seed, input_b=self._sf_hidden_buf)  # cond_pos(0) seed
            if not self.lm._fused_rope:
                # The host position mirrors exist ONLY to index the fp32 RoPE tables.  On the fused
                # path the rows are gathered on device from the device positions, so there is nothing
                # to mirror — the device counters are the only positions that matter.
                self._sf_pos_pos_host = start_pos
                self._sf_neg_pos_host = 0  # boot advances the device tensor + this mirror to 1
                self._sf_write_rope(self._sf_cos_pos, self._sf_sin_pos, self._sf_pos_pos_host)
            if self._sf_noise_table is not None:
                # Rewind the device noise index.  Frame 0 is the one place this runs, and it runs
                # before every warmup / capture / reset replay, so the in-trace plus_one stays in
                # lockstep with the AR loop's frame counter.
                self._sf_write_int(self._sf_noise_idx, noise_idx)
        elif not self.lm._fused_rope:
            self._sf_pos_pos_host += 1  # mirror the on-device plus_one from the prior frame's lm2
            self._sf_neg_pos_host += 1
            self._sf_write_rope(self._sf_cos_pos, self._sf_sin_pos, self._sf_pos_pos_host)
            self._sf_write_rope(self._sf_cos_neg, self._sf_sin_neg, self._sf_neg_pos_host)
        if self._sf_noise_table is None:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(noise_2x[:1].to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                self._sf_noise,
            )

    def _run_segment_frame_cfg_b2(self, seg_frame_idx, start_pos, noise_2x, kv_pos, kv_neg, noise_idx=0):
        """CFG batch-2 fused speech-diffusion frame.  Two captured traces per frame:
            _dp2trace : diffusion (cond_pos/cond_neg from persistent buffers) + post → audio, fused
            _lm2trace : ONE batch-2 LM decode = pos-LM(k) [row0] + neg-LM(k+1) [row1], reading each
                        layer's weights once; writes hidden_buf (cond_pos(k+1)) + neg_hidden
                        (cond_neg(k+1)); constrained argmax on row0 → token(k)
        plus a once-per-segment eager _boot (neg-LM(0), the negative prefill) seeding neg_hidden for
        frame 0.  Byte-identical per row to the split neg/pos B=1 forwards."""
        dev = self.device
        lm = self.lm

        def _boot():
            # Eager B=1 negative-prefill: neg-LM on speech_start @ neg_pos 0 → _sf_neg_hidden.  Runs
            # once per segment while no trace is live (the frame-0 boundary), then switches the neg
            # row's embed/RoPE to the steady speech_diffusion @ neg_pos 1 for lm2.
            if not self.lm._fused_rope:
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
            if not self.lm._fused_rope:
                self._sf_neg_pos_host = 1
                self._sf_write_rope(self._sf_cos_neg, self._sf_sin_neg, self._sf_neg_pos_host)

        def _dp2trace():
            cond_pos = _condition_from_hidden(self._sf_hidden_buf)
            cond_neg = _condition_from_hidden(self._sf_neg_hidden)
            # Noise: gathered on device from the pre-uploaded table (index self-advances in-capture),
            # or read from the host-written buffer when VV_DEV_NOISE=0.
            noise_in = self._sf_noise_row() if self._sf_noise_table is not None else self._sf_noise
            latent = sample_speech_latents(
                self.diffusion_head,
                cond_pos,
                cond_neg,
                self.scheduler,
                noise_in,
                cfg_scale=self.cfg_scale,
                num_steps=self.num_diffusion_steps,
                head_runner=None,
                t_tensors=self._sf_t_tensors,
                t_embs=self._sf_t_embs,
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
            # BOTH CFG rows consume the SAME inputs_embeds (this frame's fused acoustic+semantic
            # embed) — the reference's negative branch overrides input_ids with the positive
            # branch's inputs_embeds and differs only in attention context (no text prefill) and
            # position.  Feeding the neg row a constant embed(speech_diffusion) instead leaves it
            # blind to the audio feedback, so CFG stops cancelling the shared state component and
            # extrapolates it by cfg_scale (loop gain > 1 → long-form energy runaway / latch).
            emb_b2 = ttnn.concat([pos_in, pos_in], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
            tok = ttnn.argmax(logits0, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # 1 elem, LOCAL idx
            if not self._sf_out_fused:
                return tok
            # Append the token index to this frame's audio (written by the dp2 replay that ran just
            # before this trace) so the host reads one tensor instead of two.  Casting the index to
            # the audio dtype is exact: it indexes _sf_valid_ids_sorted, i.e. the handful of control
            # tokens, and every small integer is exactly representable in bf16/fp32.
            tok_cast = ttnn.typecast(ttnn.reshape(tok, [1, 1, 1, 1]), self._sf_audio_out.dtype)
            return ttnn.concat([self._sf_audio_out, tok_cast], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self._sf_lm2trace_tid is None:
            # First frame-0 after a (re)capture: warmup (eager), capture dp2+lm2 (boot stays eager),
            # reset, then the real frame-0 replay — all internal, so warmup/capture frames are
            # discarded and never emitted.
            for _ in range(self._SF_WARMUP):
                self._sf_set_inputs_b2(0, start_pos, noise_2x, noise_idx)
                _boot()
                # Keep the warmup's audio handle: _lm2trace reads it when the frame output is fused
                # (the capture below re-points it at the captured dp2's output before lm2 records).
                self._sf_audio_out = _dp2trace()
                _lm2trace()
            self._sf_set_inputs_b2(0, start_pos, noise_2x, noise_idx)
            _boot()  # seed neg_hidden for the captured dp2
            tb = ttnn.begin_trace_capture(dev, cq_id=0)
            self._sf_audio_out = _dp2trace()
            ttnn.end_trace_capture(dev, tb, cq_id=0)
            tc = ttnn.begin_trace_capture(dev, cq_id=0)
            self._sf_tok_out = _lm2trace()
            ttnn.end_trace_capture(dev, tc, cq_id=0)
            self._sf_dp2trace_tid, self._sf_lm2trace_tid = tb, tc
            # RESET for the real frame 0: rewind positions/hidden, zero conv, re-run boot.
            self._sf_set_inputs_b2(0, start_pos, noise_2x, noise_idx)
            self._sf_zero_conv()
            _boot()
            ttnn.execute_trace(dev, self._sf_dp2trace_tid, cq_id=0, blocking=False)
            ttnn.execute_trace(dev, self._sf_lm2trace_tid, cq_id=0, blocking=False)
            _vv_debug("segment_frame(cfg_b2): captured + reset")
            return self._sf_audio_out, self._sf_tok_out

        if seg_frame_idx == 0:
            self._sf_set_inputs_b2(0, start_pos, noise_2x, noise_idx)
            self._sf_zero_conv()
            _boot()
        else:
            self._sf_set_inputs_b2(1, start_pos, noise_2x, noise_idx)
        ttnn.execute_trace(dev, self._sf_dp2trace_tid, cq_id=0, blocking=False)
        ttnn.execute_trace(dev, self._sf_lm2trace_tid, cq_id=0, blocking=False)
        return self._sf_audio_out, self._sf_tok_out

    def _sf_zero_conv(self) -> None:
        """Zero the acoustic/semantic conv streaming caches IN PLACE (stable addresses) — the
        segment-boundary reset performed while the fused trace is live."""
        self.acoustic_tok.reset_decode_cache_inplace()
        self.semantic_tok.reset_cache_inplace()

    def _run_segment_frame_traced(self, seg_frame_idx, step_hidden, start_pos, noise_2x, kv_pos, kv_neg, noise_idx=0):
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
            self._sf_noise_idx = _z([1], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)  # device noise row index
            # Schedule-constant t scalars + embeddings (outside capture; reused every frame).
            self._ensure_diffusion_t_embs()

        if seg_frame_idx == 0:
            # Capture the segment-start condition source ([1,1,1,H], last position of the
            # speech_start decode / prefill hidden) into a persistent buffer.  seg_frame_idx==0 is
            # only ever reached right after a speech_start released the trace (or at the very
            # start), so NO trace is live here — this is the one place the reducing slice may
            # allocate.  The reset then re-seeds from _sf_hidden_seed with an alloc-free copy.
            ttnn.copy(input_a=_condition_from_hidden(step_hidden), input_b=self._sf_hidden_seed)

        if self._sf_cfg_b2:
            return self._run_segment_frame_cfg_b2(seg_frame_idx, start_pos, noise_2x, kv_pos, kv_neg, noise_idx)

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
                t_embs=self._sf_t_embs,
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
            # Same inputs_embeds for the next frame's neg-LM as for the pos-LM (see _dptrace).
            ttnn.copy(input_a=fused, input_b=self._sf_neg_embed)
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
                    t_embs=self._sf_t_embs,
                )
                fu, au = self._run_post_pipeline(latent)
                if self._sf_fused_out is None:
                    self._sf_fused_out = ttnn.clone(
                        fu, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )  # persistent alloc (eager warmup)
                else:
                    ttnn.copy(input_a=fu, input_b=self._sf_fused_out)
                # Hand this frame's fused embed to the NEXT frame's neg-LM: the reference feeds the
                # negative branch the same inputs_embeds as the positive one (it only differs in
                # attention context + position), and _negtrace runs before _dptrace, so the buffer
                # still holds frame k-1's embed when the neg-LM reads it.
                ttnn.copy(input_a=fu, input_b=self._sf_neg_embed)
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

    def _neg_lm_step(self, inputs_embeds: ttnn.Tensor, neg_pos: int, kv_cache_neg: KVCache) -> ttnn.Tensor:
        """One negative-CFG decode step on the SAME inputs_embeds the positive branch consumed.

        The reference's negative branch overrides ``input_ids`` with the positive branch's
        ``inputs_embeds`` (the previous frame's fused acoustic+semantic embed) and differs only in
        attention context (no text prefill) and position.
        """
        if self._ref_lm is not None:
            cpu = ttnn.to_torch(inputs_embeds).to(torch.float32).squeeze(1)
            return self._ref_lm.neg_step_embeds(cpu)
        _, neg_hidden = self.lm.forward(
            inputs_embeds, start_pos=neg_pos, kv_cache=kv_cache_neg, return_last_hidden=True
        )
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
        # Diffusion-only window (VV_PROFILE_DIFFUSION=<n>): signposts around `_run_speech_diffusion`
        # only (CFG×num_steps head + scheduler).  Eager-path only; EXIT returns after the call.
        _profile_diff = int(os.environ.get("VV_PROFILE_DIFFUSION", "0"))
        _profile_diff_exit = os.environ.get("VV_PROFILE_DIFFUSION_EXIT", "0") == "1"

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
        neg_prev_diffusion_token: Optional[int] = None  # segment-first-frame flag (traced path)
        # Previous frame's inputs_embeds — what the reference's negative branch consumes (eager path).
        neg_prev_embeds: Optional[ttnn.Tensor] = None

        # Generated token ids collected as a host list (O(1) append) and concatenated to input_ids
        # once after the loop — avoids the per-frame torch.cat that reallocated an O(seq_len) tensor
        # every AR step (device-idle host bookkeeping in the steady loop).
        _gen_tokens: List[int] = []
        # On-device streaming: each diffusion step decodes its audio chunk via the
        # acoustic decoder's causal cache; we accumulate the chunks to form the
        # final waveform (identical structure to the reference streaming decode).
        audio_chunks: List[torch.Tensor] = []
        # Optional disk-streaming (VV_STREAM_AUDIO=<path>): append each frame's fp32 samples to a raw
        # file (flushed → survives SIGKILL) instead of accumulating in host RAM.  Bounds host memory
        # on very long renders AND preserves partial audio if the process dies mid-run.  Numerically
        # identical (storage only).  Read back at the end (happy path) or offline from the .f32 file.
        # Absolutized inline so the path reaching ``open`` is a normalized artifact path.
        # No extension check here: the chunked demo path appends ``.part<N>`` to the base
        # name (see demo.py), so this writer has no single expected suffix.
        _stream_path = os.environ.get("VV_STREAM_AUDIO", "")
        if _stream_path:
            _stream_path = os.path.abspath(_stream_path)
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
            # VV_TTNN_RANDN=1 draws the table on device instead (different values for the same seed
            # — see _sf_randn_noise_table).  _ttnn_randn_seed() READS the run seed rather than
            # drawing from it, so the torch stream is untouched on this path.
            _dev_seed = self._ttnn_randn_seed() if self._sf_ttnn_randn else 0
            if self._sf_ttnn_randn and self._sf_randn_noise_table(max_steps, _dev_seed):
                _vv_debug(f"diffusion noise: ttnn.randn on device, seed={_dev_seed} (NOT torch-reference values)")
            else:
                diffusion_noise = torch.randn(max_steps, 2, 1, 1, 64, dtype=torch.float32, generator=rng).to(
                    torch.bfloat16
                )
                # Upload it once as a gather table so the traced frame picks its row on device.
                self._sf_upload_noise_table(diffusion_noise)

        diffusion_frames = 0
        # Steady-state decode timing (cf. tt_transformers/llama demos): time ONLY the fused-frame
        # trace-replay frames — warmup and capture frames are not timed.
        _steady_decode_s = 0.0
        _steady_decode_frames = 0
        _t_decode_start = time.perf_counter()
        for step in range(max_steps):
            current_token = next_token
            _gen_tokens.append(current_token)
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
                # With the device noise table the traced frame gathers its own row, so only the row
                # INDEX is needed here — don't slice the host block for a value nobody reads.
                noise_2x = (
                    None
                    if self._sf_noise_table is not None or diffusion_noise is None
                    else diffusion_noise[diffusion_frames - 1]
                )
                start_pos = prefill_len + step
                with prof.section("segment_frame"):
                    audio_chunk, _tok_or_logits = self._run_segment_frame_traced(
                        seg_frame_idx,
                        step_hidden,
                        start_pos,
                        noise_2x,
                        kv_cache_pos,
                        kv_cache_neg,
                        noise_idx=diffusion_frames - 1,
                    )
                neg_prev_diffusion_token = current_token
                _local_idx = None
                if self._sf_out_fused:
                    # ONE D2H returns [audio ..., token_idx] (see _lm2trace).
                    with prof.section("argmax"):
                        _out = ttnn.to_torch(_tok_or_logits).reshape(-1)  # syncs frame
                    _frame_audio = _out[:-1].to(torch.float32)
                    _local_idx = int(_out[-1].item())
                else:
                    _frame_audio = ttnn.to_torch(audio_chunk).to(torch.float32).reshape(-1)  # syncs frame
                if self._traj_path:
                    self._log_traj(diffusion_frames, start_pos, _frame_audio)
                _emit_audio(_frame_audio)
                if _local_idx is not None:
                    next_token = self._sf_valid_ids_sorted[_local_idx]
                elif self._sf_cap_split:
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
                # Negative CFG: the reference feeds the negative branch the SAME inputs_embeds as
                # the positive branch — the PREVIOUS frame's fused acoustic+semantic embed — and
                # differs from it only in attention context (no text prefill) and position; the
                # token ids in negative_input_ids are never embedded (input_ids is overridden with
                # inputs_embeds).  For a segment's first diffusion step that embed is
                # embed(speech_start), whose negative hidden we captured in neg_start_hidden.
                with prof.section("neg_lm_step"):
                    if neg_prev_embeds is None:
                        neg_hidden = neg_start_hidden
                    else:
                        neg_hidden = self._neg_lm_step(neg_prev_embeds, neg_pos, kv_cache_neg)
                        neg_pos += 1
                    cond_neg = _condition_from_hidden(neg_hidden)

                with prof.section("diffusion (CFG x num_steps)"):
                    noise_2x = diffusion_noise[diffusion_frames - 1] if diffusion_noise is not None else None
                    if _profile_diff and diffusion_frames == _profile_diff:
                        import tracy

                        ttnn.synchronize_device(device)
                        tracy.signpost("start")
                        _vv_debug(f"Tracy signpost start: eager diffusion {_profile_diff}")
                    speech_latent = self._run_speech_diffusion(
                        cond_pos, cond_neg, latent_size=64, noise_2x=noise_2x, rng=rng
                    )
                    if _profile_diff and diffusion_frames == _profile_diff:
                        import tracy

                        ttnn.synchronize_device(device)
                        tracy.signpost("stop")
                        _vv_debug(f"Tracy signpost stop: eager diffusion {_profile_diff}")
                        if _profile_diff_exit:
                            _vv_debug("VV_PROFILE_DIFFUSION_EXIT=1 — ending generate after profiled diffusion")
                            break

                # On-device streaming: fused next-step embed + this frame's audio chunk.
                with prof.section("post_diffusion (decode+sem_enc+conn)"):
                    pending_embeds, audio_chunk = self._post_diffusion_embeds(speech_latent)
                    # Both CFG branches consume this embed next frame (pos now, neg one frame later).
                    neg_prev_embeds = pending_embeds
                with prof.section("audio_chunk -> host"):
                    _chunk = (
                        audio_chunk.to(torch.float32).reshape(-1)
                        if isinstance(audio_chunk, torch.Tensor)
                        else ttnn.to_torch(audio_chunk).to(torch.float32).reshape(-1)
                    )
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
                    neg_prev_embeds = None
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

        sequences = (
            torch.cat([input_ids, torch.tensor([_gen_tokens], dtype=torch.long)], dim=-1)
            if _gen_tokens
            else input_ids.clone()
        )
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

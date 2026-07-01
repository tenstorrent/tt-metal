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
#   VV_DEBUG=1   — per-AR-step token + phase logs (also set by demo_ttnn.py --debug)


def _vv_profile_enabled() -> bool:
    return os.environ.get("VV_PROFILE", "0") == "1"


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
    decode_wall_s: float = 0.0  # wall time covering the full AR decode loop


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

        # Optional ttnn trace of the post-diffusion block (opt-in via VV_TRACE_POSTDIFF=1,
        # set by demo_ttnn.py --trace).  Per speech segment: 1 eager frame allocates the
        # streaming caches at fixed addresses, the next frame is captured, the rest replay
        # with a single dispatch.  Invalidated + re-captured on each segment reset.
        # NOTE: not yet bit-exact vs eager (~0.985 PCC) — the in-place streaming-cache
        # update has a write-after-read that trace replay does not order. For eval only.
        self._trace_postdiff = os.environ.get("VV_TRACE_POSTDIFF", "0") == "1"
        self._pd_tid = None
        self._pd_lat_in: Optional[ttnn.Tensor] = None
        self._pd_fused_out: Optional[ttnn.Tensor] = None
        self._pd_audio_out: Optional[ttnn.Tensor] = None
        self._pd_eager_frames = 0

        # Optional ttnn trace of the diffusion-head forward (opt-in via VV_TRACE_DIFFUSION=1,
        # set by demo_ttnn.py --trace).  The head is stateless (no streaming caches) and has a
        # fixed CFG-batched shape, so a single capture replays for every step of every frame —
        # no per-segment reset, and bit-exact vs eager (the WAR hazard that limits the
        # post-diffusion trace is absent here).  scheduler.step stays host-side: its per-step
        # coefficients are Python floats that would bake into the trace.
        self._trace_diffusion = os.environ.get("VV_TRACE_DIFFUSION", "0") == "1"
        self._diff_tid = None
        self._diff_sample_in: Optional[ttnn.Tensor] = None
        self._diff_t_in: Optional[ttnn.Tensor] = None
        self._diff_cond_in: Optional[ttnn.Tensor] = None
        self._diff_eps_out: Optional[ttnn.Tensor] = None
        self._diff_eager_steps = 0

        # Optional ttnn trace of the positive 28-layer LM decode step (opt-in via
        # VV_TRACE_LM=1, set by demo_ttnn.py --trace).  This is the dominant per-token
        # cost.  It is driven entirely by device tensors (KV write position + SDPA read
        # bound via cur_pos, RoPE via a host-written row), so one capture replays for the
        # whole generation (positive KV cache persists — no per-segment reset), and it is
        # bit-exact vs eager (validated to 64 chunks).  See TTVibeVoiceLM.forward_decode_
        # traced_embeds.  Only the fused post-diffusion embed step is traced; the rare
        # token-input steps (_lm_decode_token) stay eager.
        self._trace_lm = os.environ.get("VV_TRACE_LM", "0") == "1"
        self._lm_tid = None
        self._lm_emb_in: Optional[ttnn.Tensor] = None
        self._lm_cos_in: Optional[ttnn.Tensor] = None
        self._lm_sin_in: Optional[ttnn.Tensor] = None
        self._lm_pos_in: Optional[ttnn.Tensor] = None
        self._lm_logits_out: Optional[ttnn.Tensor] = None
        self._lm_hidden_out: Optional[ttnn.Tensor] = None
        self._lm_warmup = 0

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
            head_runner=self._diff_head_runner if self._trace_diffusion else None,
        )

    def _diff_head_runner(
        self, noisy_images: ttnn.Tensor, timesteps: ttnn.Tensor, condition: ttnn.Tensor
    ) -> ttnn.Tensor:
        """Trace/replay the (stateless, fixed-shape) diffusion-head forward.

        Drop-in for ``diffusion_head(...)`` inside the DPM loop.  The three inputs are
        copied into persistent fixed-address buffers before each replay; the persistent
        output is fully consumed (CFG split/combine + scheduler.step) within the step
        before the next replay overwrites it.

        Lifecycle (mirrors the post-diffusion trace so the two coexist safely): the head
        runs eager for a whole first frame (``num_diffusion_steps`` calls), then captures
        on the next call, then replays.  The full-frame warm-up matters because the
        post-diffusion streaming caches are allocated lazily during frame 0's post-diffusion
        block — capturing before that allocation, or letting it happen while the trace is
        live, corrupts the trace.  ``_reset_diffusion_trace`` drops the capture at each
        segment boundary (where those caches are reset + reallocated) so it re-warms there."""
        dev = self.device
        if self._diff_sample_in is None:
            self._diff_sample_in = ttnn.zeros(
                list(noisy_images.shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._diff_t_in = ttnn.zeros(
                list(timesteps.shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._diff_cond_in = ttnn.zeros(
                list(condition.shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.copy(input_a=noisy_images, input_b=self._diff_sample_in)
        ttnn.copy(input_a=timesteps, input_b=self._diff_t_in)
        ttnn.copy(input_a=condition, input_b=self._diff_cond_in)

        if self._diff_tid is None:
            if self._diff_eager_steps < self.num_diffusion_steps:
                self._diff_eager_steps += 1
                return self.diffusion_head(self._diff_sample_in, self._diff_t_in, self._diff_cond_in)
            self._diff_tid = ttnn.begin_trace_capture(dev, cq_id=0)
            self._diff_eps_out = self.diffusion_head(self._diff_sample_in, self._diff_t_in, self._diff_cond_in)
            ttnn.end_trace_capture(dev, self._diff_tid, cq_id=0)
            _vv_debug("diffusion_head: trace captured")
            return self._diff_eps_out
        ttnn.execute_trace(dev, self._diff_tid, cq_id=0, blocking=False)
        return self._diff_eps_out

    def _reset_diffusion_trace(self) -> None:
        """Invalidate the diffusion-head trace at a segment boundary.  The head is stateless,
        but the post-diffusion streaming caches are reset + reallocated here; allocating DRAM
        while any trace is live corrupts it, so release the head trace and re-warm/recapture
        on the next segment (the persistent I/O buffers are kept — only the capture is dropped)."""
        if self._diff_tid is not None:
            ttnn.release_trace(self.device, self._diff_tid)
        self._diff_tid = None
        self._diff_eager_steps = 0

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
        """On-device streaming decode/encode/fusion (eager, or ttnn-traced when enabled)."""
        if not self._trace_postdiff:
            return self._run_post_pipeline(speech_latent)
        return self._post_diffusion_embeds_tt_traced(speech_latent)

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

    def _post_diffusion_embeds_tt_traced(self, speech_latent: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Trace/replay the post-diffusion block.  Per segment: frame 0 runs eager (so the
        streaming caches are allocated at fixed addresses — a cache ``ttnn.zeros`` captured
        inside the trace would re-zero on every replay); frame 1 is captured; frames 2+ replay."""
        dev = self.device
        # Persistent input latent (fixed address) the captured graph reads each frame.
        if self._pd_lat_in is None:
            self._pd_lat_in = ttnn.zeros(
                list(speech_latent.shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.copy(input_a=speech_latent, input_b=self._pd_lat_in)

        if self._pd_tid is None:
            if self._pd_eager_frames == 0:
                self._pd_eager_frames = 1
                return self._run_post_pipeline(self._pd_lat_in)
            self._pd_tid = ttnn.begin_trace_capture(dev, cq_id=0)
            self._pd_fused_out, self._pd_audio_out = self._run_post_pipeline(self._pd_lat_in)
            ttnn.end_trace_capture(dev, self._pd_tid, cq_id=0)
            _vv_debug("post_diffusion: trace captured")
            return self._pd_fused_out, self._pd_audio_out
        # Replay: reads _pd_lat_in, advances the streaming caches, writes _pd_*_out in place.
        # Both outputs are consumed within this AR iteration (fused → LM step; audio → host)
        # before the next replay overwrites them.
        ttnn.execute_trace(dev, self._pd_tid, cq_id=0, blocking=False)
        return self._pd_fused_out, self._pd_audio_out

    def _reset_postdiff_trace(self) -> None:
        """Invalidate the post-diffusion trace at a segment boundary (the streaming caches
        are about to be reset/reallocated, so the captured addresses go stale)."""
        if self._pd_tid is not None:
            ttnn.release_trace(self.device, self._pd_tid)
        self._pd_tid = None
        self._pd_eager_frames = 0

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
        # Trace the 28-layer decode (opt-in).  Requires a 1024-aligned KV cache so the
        # fused SDPA-decode k_chunk is 512 (the validated regime); alloc_kv_cache always
        # satisfies this.  The positive cache persists for the whole generation, so a
        # single capture replays for every embed step — no per-segment reset.
        if self._trace_lm and kv_cache.max_seq % 1024 == 0:
            return self._lm_step_traced(inputs_embeds, start_pos, kv_cache)
        logits, last_hidden = self.lm.forward(
            inputs_embeds,
            start_pos=start_pos,
            kv_cache=kv_cache,
            return_last_hidden=True,
        )
        return logits, last_hidden

    _LM_TRACE_WARMUP = 2

    def _set_lm_trace_inputs(self, inputs_embeds: ttnn.Tensor, start_pos: int) -> None:
        """Copy this step's inputs into the persistent (fixed-address) trace buffers."""
        lm = self.lm
        hd = lm.cfg.head_dim
        if self._lm_emb_in is None:
            self._lm_emb_in = ttnn.zeros(
                list(inputs_embeds.shape),
                dtype=inputs_embeds.dtype,
                layout=inputs_embeds.layout,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._lm_cos_in = ttnn.zeros(
                [1, 1, 1, hd],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._lm_sin_in = ttnn.zeros(
                [1, 1, 1, hd],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._lm_pos_in = ttnn.zeros(
                [1],
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ttnn.copy(input_a=inputs_embeds, input_b=self._lm_emb_in)  # device->device, fixed address
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.from_numpy(lm._cos_np[start_pos]).reshape(1, 1, 1, hd).to(torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            ),
            self._lm_cos_in,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.from_numpy(lm._sin_np[start_pos]).reshape(1, 1, 1, hd).to(torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            ),
            self._lm_sin_in,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([start_pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            self._lm_pos_in,
        )

    def _lm_step_traced(
        self,
        inputs_embeds: ttnn.Tensor,
        start_pos: int,
        kv_cache: KVCache,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        lm = self.lm
        dev = self.device
        self._set_lm_trace_inputs(inputs_embeds, start_pos)

        def _run():
            return lm.forward_decode_traced_embeds(
                self._lm_emb_in,
                self._lm_cos_in,
                self._lm_sin_in,
                self._lm_pos_in,
                kv_cache,
                return_last_hidden=True,
            )

        if self._lm_tid is None:
            if self._lm_warmup < self._LM_TRACE_WARMUP:
                self._lm_warmup += 1
                return _run()  # eager warm-up (compiles programs, writes real KV)
            # Capture once.  Capture records ops without computing, so the KV write at this
            # position is garbage; re-run this step eagerly afterward to fix it before any
            # replay attends over it.  The captured output handles are what replays fill.
            self._lm_tid = ttnn.begin_trace_capture(dev, cq_id=0)
            self._lm_logits_out, self._lm_hidden_out = _run()
            ttnn.end_trace_capture(dev, self._lm_tid, cq_id=0)
            _vv_debug("lm_step: trace captured")
            return _run()  # capture-poison fix (returns this step's correct logits/hidden)

        ttnn.execute_trace(dev, self._lm_tid, cq_id=0, blocking=False)
        return self._lm_logits_out, self._lm_hidden_out

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

        with prof.section("lm_prefill"):
            logits_pos, prefill_hidden = self._lm_prefill(inputs_embeds, kv_cache_pos)
        _t_prefill_end = time.perf_counter()
        _vv_debug(f"lm_prefill done: kv_cache_pos size={prefill_len + max_steps + 8}")

        neg_pos, neg_start_hidden = self._reset_neg_cache(kv_cache_neg)
        neg_prev_diffusion_token: Optional[int] = None  # delayed token for negative CFG

        sequences = input_ids.clone()
        # On-device streaming: each diffusion step decodes its audio chunk via the
        # acoustic decoder's causal cache; we accumulate the chunks to form the
        # final waveform (identical structure to the reference streaming decode).
        audio_chunks: List[torch.Tensor] = []
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

        diffusion_frames = 0
        _t_decode_start = time.perf_counter()
        for step in range(max_steps):
            current_token = next_token
            sequences = torch.cat(
                [sequences, torch.tensor([[current_token]], dtype=torch.long)],
                dim=-1,
            )
            _vv_debug(f"step {step + 1}/{max_steps}: emit {self._token_label(current_token)}")

            if current_token == self.speech_diffusion_id:
                diffusion_frames += 1
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
                    speech_latent = self._run_speech_diffusion(
                        cond_pos, cond_neg, latent_size=64, noise_2x=noise_2x, rng=rng
                    )

                # On-device streaming: fused next-step embed + this frame's audio chunk.
                with prof.section("post_diffusion (decode+sem_enc+conn)"):
                    pending_embeds, audio_chunk = self._post_diffusion_embeds(speech_latent)
                with prof.section("audio_chunk -> host"):
                    if isinstance(audio_chunk, torch.Tensor):
                        audio_chunks.append(audio_chunk.to(torch.float32).reshape(-1))
                    else:
                        audio_chunks.append(ttnn.to_torch(audio_chunk).to(torch.float32).reshape(-1))
                chunk_samples = audio_chunks[-1].numel()
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
                _vv_debug("  new speech segment: reset neg-CFG cache + acoustic/semantic streaming caches")
                neg_pos, neg_start_hidden = self._reset_neg_cache(kv_cache_neg)
                neg_prev_diffusion_token = None
                # Release both traces before touching the streaming caches: the caches are
                # freed + reallocated here, and a live trace referencing them (or any DRAM
                # alloc while a trace is live) would be corrupted.
                self._reset_postdiff_trace()
                self._reset_diffusion_trace()
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

        _t_decode_end = time.perf_counter()
        # The per-step streaming decode already produced each frame's audio chunk
        # (with full causal context via the decoder cache); concatenate for the
        # final waveform — no separate batch decode needed.
        if audio_chunks:
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
        )

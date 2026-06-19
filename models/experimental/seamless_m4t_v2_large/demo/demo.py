# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Seamless M4T v2 — TTNN demo of all five inference tasks.

Tasks demonstrated (every one runs through ``TTSeamlessM4Tv2Model.generate``):

  1. **T2TT** Text-to-Text Translation        (English text → Hindi text)
  2. **T2ST** Text-to-Speech Translation      (English text → Hindi speech)
  3. **S2TT** Speech-to-Text Translation      (English speech → Hindi text)
  4. **S2ST** Speech-to-Speech Translation    (English speech → Spanish speech)
  5. **ASR**  Automatic Speech Recognition    (English speech → English text)

Each task opens its own mesh device, runs untimed warmup ``generate()`` calls, then times a
steady-state iteration — so reported runtimes are not affected by prior tasks' program-cache
clears or L1 pressure. Speech-input tasks (3–5) use a fixed English preamble WAV; task 2 still
writes its Hindi speech output locally.

Output audio is written next to this file:

  * ``outputs/t2st_hindi_speech.wav``   — task 2 output
  * ``outputs/s2st_spanish_speech.wav`` — task 4 output
  * ``outputs/preamble10.wav``          — downloaded input for tasks 3–5

Run from repo root:

  python models/experimental/seamless_m4t_v2_large/demo/demo.py

Optional: cap T2TT/T2ST source text to the first *N* sentences (utterance-scale inputs):

  SEAMLESS_DEMO_MAX_SENTENCES=2 python models/experimental/seamless_m4t_v2_large/demo/demo.py

  python models/experimental/seamless_m4t_v2_large/demo/demo.py

Optional: ``SEAMLESS_M4T_V2_WEIGHTS=/path/to/seamless-m4t-v2-large`` if not using the default tree.
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
import urllib.error
import urllib.request
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import torch
import ttnn
from transformers import AutoProcessor, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.common import (
    hf_aligned_generation_kwargs,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import _requires_bh_qb
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import (
    SeamlessGenerateTimings,
    TTSeamlessM4Tv2GenerationOutput,
    TTSeamlessM4Tv2GreedySearchOutput,
    TTSeamlessM4Tv2Model,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
T2ST_WAV = OUTPUT_DIR / "t2st_hindi_speech.wav"
S2ST_WAV = OUTPUT_DIR / "s2st_spanish_speech.wav"
PREAMBLE_WAV_URL = "https://www.cs.kzoo.edu/cs107/MediaSources/preamble10.wav"
PREAMBLE_WAV = OUTPUT_DIR / "preamble10.wav"
_MIN_DEMO_WAV_BYTES = 1024
_MIN_DEMO_WAV_DURATION_S = 0.5

# Untimed warmups before timed runs; min() over measure_iters drops host jitter.
_DEMO_WARMUP_ITERS = 1
_DEMO_SPEECH_WARMUP_ITERS = 2
_DEMO_MEASURE_ITERS = 1
_DEMO_SPEECH_MEASURE_ITERS = 2

_DEFAULT_SRC_TEXT = (
    "going along slushy country roads and speaking to damp audiences in draughty schoolrooms "
    "day after day for a fortnight he'll have to put in an appearance at some place of worship "
    "on sunday morning and he can come to us immediately afterwards"
)


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries (``. ``, ``? ``, ``! ``); keeps trailing punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _demo_src_text() -> str:
    """Return demo source text, optionally truncated to the first N sentences.

    Set ``SEAMLESS_DEMO_MAX_SENTENCES=N`` (positive int) to limit utterance length for T2TT/T2ST.
    """
    raw = os.environ.get("SEAMLESS_DEMO_MAX_SENTENCES", "").strip()
    if not raw:
        return _DEFAULT_SRC_TEXT
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_SRC_TEXT
    if n < 1:
        return _DEFAULT_SRC_TEXT
    sents = _split_sentences(_DEFAULT_SRC_TEXT)
    return " ".join(sents[:n]) if sents else _DEFAULT_SRC_TEXT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weights_dir() -> Path:
    env = os.environ.get("SEAMLESS_M4T_V2_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return ensure_seamless_m4t_v2_large_weights()


def torch_ids_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def torch_feats_to_ttnn(device: ttnn.Device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16).cpu().contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def make_tt_model(device: ttnn.Device, model: torch.nn.Module, cfg, t2u_cfg) -> TTSeamlessM4Tv2Model:
    params = create_seamless_m4t_v2_model_parameters(model, device=device)
    return TTSeamlessM4Tv2Model(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        pad_token_id=cfg.pad_token_id,
        decoder_start_token_id=cfg.decoder_start_token_id,
        vocab_size=cfg.vocab_size,
        adaptor_kernel_size=cfg.adaptor_kernel_size,
        adaptor_stride=cfg.adaptor_stride,
        t2u_eos_token_id=cfg.t2u_eos_token_id,
        t2u_pad_token_id=t2u_cfg.pad_token_id,
        vocoder_offset=cfg.vocoder_offset,
        t2u_layer_norm_eps=t2u_cfg.layer_norm_eps,
        t2u_encoder_layers=t2u_cfg.encoder_layers,
        t2u_encoder_attention_heads=t2u_cfg.encoder_attention_heads,
        t2u_decoder_layers=t2u_cfg.decoder_layers,
        t2u_decoder_attention_heads=t2u_cfg.decoder_attention_heads,
        variance_predictor_embed_dim=t2u_cfg.variance_predictor_embed_dim,
        variance_predictor_hidden_dim=t2u_cfg.variance_predictor_hidden_dim,
        variance_predictor_kernel_size=t2u_cfg.variance_predictor_kernel_size,
        vocoder_config=cfg,
        generation_config=model.generation_config,
        hf_config=cfg,
    )


def _waveform_to_mono_fp32(waveform_tt: ttnn.Tensor, lengths_tt: ttnn.Tensor) -> np.ndarray:
    """Read a TT vocoder waveform back to host as a 1-D fp32 numpy array, trimmed to valid length.

    TT vocoder output shape: ``[B, T_max, 1]`` (right-padded with zeros to the batch max). The valid
    sample count per row is in ``lengths_tt`` — we trim to that to drop trailing silence padding.
    """
    arr = to_torch_replicated_first_shard(waveform_tt).float().squeeze().cpu().numpy()
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    valid_len = int(to_torch_replicated_first_shard(lengths_tt).long().reshape(-1)[0].item())
    if 0 < valid_len <= arr.size:
        arr = arr[:valid_len]
    return arr


def _save_wav(path: Path, waveform_np: np.ndarray, sample_rate: int) -> None:
    """Save a mono fp32 waveform to ``path`` as a 16-bit PCM WAV (stdlib ``wave`` only)."""
    arr = np.clip(waveform_np, -1.0, 1.0)
    pcm16 = (arr * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


def _load_mono_wav(path: Path) -> np.ndarray:
    """Load a mono fp32 waveform from a PCM WAV (inverse of :func:`_save_wav`)."""
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sw} in {path}")
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    return pcm.astype(np.float32)


def _load_mono_wav_resampled(path: Path, target_rate: int) -> tuple[np.ndarray, int]:
    """Load mono fp32 audio from ``path``, resampling to ``target_rate`` Hz if needed."""
    with wave.open(str(path), "rb") as wf:
        src_rate = int(wf.getframerate())
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sw} in {path}")
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    pcm = pcm.astype(np.float32)
    if src_rate != int(target_rate):
        n_out = int(round(pcm.size * float(target_rate) / float(src_rate)))
        x_old = np.linspace(0.0, 1.0, pcm.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
    return pcm, int(target_rate)


def _format_demo_wav_summary(data: bytes) -> str:
    """Human-readable WAV metadata for logging after a successful download."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        nframes = wf.getnframes()
    duration = nframes / float(rate) if rate > 0 else 0.0
    pcm_bits = 8 * sw
    return f"{len(data)} bytes, {nch} ch, {pcm_bits}-bit PCM @ {rate} Hz, {duration:.2f}s"


def _validate_demo_wav_bytes(data: bytes) -> None:
    """Reject empty, truncated, or non-WAV downloads before they are cached."""
    if len(data) < _MIN_DEMO_WAV_BYTES:
        raise ValueError(f"WAV too small ({len(data)} bytes, min {_MIN_DEMO_WAV_BYTES})")
    if not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file")

    with wave.open(io.BytesIO(data), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        nframes = wf.getnframes()

    if nch < 1:
        raise ValueError(f"Invalid channel count {nch}")
    if sw not in (2, 4):
        raise ValueError(f"Unsupported sample width {sw} (expected 16- or 32-bit PCM)")
    if rate <= 0 or nframes <= 0:
        raise ValueError(f"Invalid WAV rate/frames: {rate} Hz, {nframes} frames")

    duration = nframes / float(rate)
    if duration < _MIN_DEMO_WAV_DURATION_S:
        raise ValueError(f"WAV too short ({duration:.2f}s, min {_MIN_DEMO_WAV_DURATION_S}s)")


def _validate_demo_wav_file(path: Path) -> None:
    _validate_demo_wav_bytes(path.read_bytes())


def ensure_demo_audio(
    url: str = PREAMBLE_WAV_URL,
    dest: Path = PREAMBLE_WAV,
) -> Path:
    """Download demo input audio to ``dest`` if missing or invalid; raise on failure."""
    dest = dest.expanduser().resolve()
    if dest.is_file() and dest.stat().st_size > 0:
        try:
            _validate_demo_wav_file(dest)
            return dest
        except ValueError:
            dest.unlink(missing_ok=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            status = getattr(resp, "status", None) or getattr(resp, "code", None)
            if status is not None and int(status) >= 400:
                raise RuntimeError(f"HTTP {status}")
            data = resp.read()
        if not data:
            raise RuntimeError("response body was empty")
        _validate_demo_wav_bytes(data)
        tmp.write_bytes(data)
        tmp.replace(dest)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download demo audio from {url}: {exc}") from exc

    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"Failed to download demo audio from {url}: file missing or empty")
    try:
        _validate_demo_wav_file(dest)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded demo audio failed validation: {exc}") from exc

    print(f"  Downloaded demo audio: {_format_demo_wav_summary(dest.read_bytes())} -> {dest}")
    return dest


_TT_ONLY_GEN_KEYS = frozenset({"use_kv_cache", "use_decode_trace", "use_2cq", "use_t2u_trace", "return_timings"})


def _hf_gen_kwargs(gen_common: dict) -> dict:
    """HF ``generate()`` kwargs — strip TT-only perf flags."""
    return {k: v for k, v in gen_common.items() if k not in _TT_ONLY_GEN_KEYS}


def _decode(tokenizer: Any, sequences_tt: ttnn.Tensor) -> str:
    """Read a TT decoder sequence back to host and decode to a single string (special tokens skipped)."""
    ids = to_torch_replicated_first_shard(sequences_tt).to(torch.int64).cpu()
    return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]


def _tt_row_length(t: ttnn.Tensor) -> int:
    """Logical length of a 1-D or ``[1, L]`` int sequence on device."""
    return int(to_torch_replicated_first_shard(t).long().reshape(-1).numel())


def _valid_unit_frames(unit_tt: ttnn.Tensor, *, pad_id: int) -> int:
    """Count non-pad unit ids in the vocoder input timeline."""
    u = to_torch_replicated_first_shard(unit_tt).long().reshape(-1)
    return int((u != int(pad_id)).sum().item())


def _print_header(idx: int, name: str, abbrev: str, src: str, tgt: str) -> None:
    print()
    print("=" * 78)
    print(f"  {idx}. {abbrev}  —  {name}  ({src} → {tgt})")
    print("=" * 78)


def _time_generate(device: ttnn.Device, generate_fn):
    """Time a single ``tt_model.generate(...)`` call with explicit synchronize before/after.

    Returns ``(output, elapsed_seconds)``. The ``synchronize_device`` calls bracket *only* the
    model runtime — input tensors must already be uploaded to device, and any host post-processing
    (token decode, waveform readback, WAV write) must happen *outside* this window so it's
    excluded from the throughput metric.
    """
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    out = generate_fn()
    ttnn.synchronize_device(device)
    return out, time.perf_counter() - t0


def _warmup_and_time(
    device: ttnn.Device,
    generate_fn,
    release_fn,
    *,
    warmup_iters: int = _DEMO_WARMUP_ITERS,
    measure_iters: int = _DEMO_MEASURE_ITERS,
    post_warmup_fn=None,
):
    """Optional untimed warmups, then timed runs; return last output and min elapsed seconds.

    ``post_warmup_fn(tt_model)`` runs after warmups (e.g. vocoder shape prewarm from cached scalars).
    """
    for _ in range(warmup_iters):
        warm_out = generate_fn()
        ttnn.synchronize_device(device)
        release_fn(warm_out)

    if post_warmup_fn is not None:
        post_warmup_fn()

    times = []
    best_out = None
    best_elapsed = float("inf")
    for _ in range(measure_iters):
        out, elapsed = _time_generate(device, generate_fn)
        times.append(elapsed)
        if elapsed < best_elapsed:
            if best_out is not None:
                release_fn(best_out)
            best_out = out
            best_elapsed = elapsed
        else:
            release_fn(out)
    return best_out, min(times) if times else 0.0


def _record_tt_perf(perf_tt: list, task: str, timings: SeamlessGenerateTimings) -> None:
    """Record TT-catalog-style phase metrics from ``generate(return_timings=True)``."""
    perf_tt.append((task, timings))
    steady = timings.steady_decode_ms_per_tok
    decode_tps = timings.decode_tok_s_u
    print(
        f"  {task} TT metrics: TTFT {timings.ttft_ms:.1f} ms, "
        f"encoder {timings.encoder_ms:.1f} ms, prefill {timings.prefill_ms:.1f} ms, "
        f"decode {decode_tps:.2f} t/s/u ({steady:.1f} ms/tok steady), "
        f"e2e {timings.e2e_ms:.1f} ms ({timings.output_tokens} out tokens)"
    )
    if timings.t2u_ms > 0 or timings.vocoder_ms > 0:
        rtf_str = f", RTF {timings.rtf:.2f}x" if timings.output_samples > 0 else ""
        char_prep = f", char prep {timings.t2u_char_prep_ms:.2f} ms" if timings.t2u_char_prep_ms > 0 else ""
        print(
            f"  {task} speech synth: T2U {timings.t2u_ms:.1f} ms "
            f"(dec hidden {timings.t2u_decoder_hidden_ms:.0f} ms, "
            f"forward {timings.t2u_forward_ms:.0f} ms{char_prep}), "
            f"vocoder {timings.vocoder_ms:.1f} ms, "
            f"TTFT audio {timings.ttft_audio_ms:.1f} ms{rtf_str}"
        )
    if timings.input_is_speech and timings.mel_frames > 0:
        print(f"  {task} input: {timings.mel_frames} mel frames")


def _print_tt_perf_summary(perf_tt: list) -> None:
    if not perf_tt:
        return
    print()
    print("-" * 78)
    print("  TT-aligned runtime summary (phase-separated; decode t/s/u = 1000 / steady ms/tok)")
    print("-" * 78)
    hdr = f"  {'Task':<6} {'TTFT':>8} {'Enc':>8} {'Pref':>8} " f"{'Dec t/s/u':>10} {'ms/tok':>8} {'E2E':>9} {'Out':>8}"
    print(hdr)
    for task_name, timings in perf_tt:
        out = f"{timings.output_tokens}tok"
        if timings.output_samples > 0:
            out = f"{timings.output_samples}smp"
        print(
            f"  {task_name:<6} {timings.ttft_ms:>7.1f}ms {timings.encoder_ms:>7.1f}ms "
            f"{timings.prefill_ms:>7.1f}ms {timings.decode_tok_s_u:>10.2f} "
            f"{timings.steady_decode_ms_per_tok:>7.1f}ms {timings.e2e_ms:>8.1f}ms {out:>8}"
        )
        if timings.output_samples > 0:
            char_prep = f" + char prep {timings.t2u_char_prep_ms:.1f} ms" if timings.t2u_char_prep_ms > 0 else ""
            print(
                f"         T2U {timings.t2u_ms:.0f} ms "
                f"(hidden {timings.t2u_decoder_hidden_ms:.0f} + "
                f"fwd {timings.t2u_forward_ms:.0f}{char_prep})  "
                f"vocoder {timings.vocoder_ms:.0f} ms  "
                f"RTF {timings.rtf:.2f}x"
            )
    print("-" * 78)
    print(
        "  decode t/s/u = 1000 / steady ms/tok (text-decoder steps 2+). "
        "TTFT includes encoder + prefill + first token. E2E includes T2U/vocoder on speech tasks."
    )


def _release_speech_out(o: TTSeamlessM4Tv2GenerationOutput) -> None:
    ttnn.deallocate(o.waveform)
    ttnn.deallocate(o.waveform_lengths)
    if getattr(o, "sequences", None) is not None:
        ttnn.deallocate(o.sequences)
    if getattr(o, "unit_sequences", None) is not None:
        ttnn.deallocate(o.unit_sequences)


@contextmanager
def _isolated_task_session(
    *,
    hf_model: torch.nn.Module,
    cfg,
    t2u_cfg,
    gen_common: dict,
    original_default: Any,
) -> Iterator[tuple[Any, TTSeamlessM4Tv2Model]]:
    """Open mesh device + TT model for one task; tear down on exit."""
    from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import open_seamless_mesh_device

    device, _mesh_shape = open_seamless_mesh_device(
        enable_decode_trace=bool(gen_common.get("use_decode_trace")),
        enable_2cq=bool(gen_common.get("use_2cq")),
    )
    ttnn.SetDefaultDevice(device)
    tt_model = make_tt_model(device, hf_model, cfg, t2u_cfg)
    try:
        yield device, tt_model
    finally:
        try:
            tt_model.release_generation_runtime()
        except Exception as exc:
            print(f"Warning: release_generation_runtime failed during teardown: {exc}", file=sys.stderr)
        if original_default is not None:
            ttnn.SetDefaultDevice(original_default)
        ttnn.close_mesh_device(device)


def _prewarm_speech_encoder(tt_model: TTSeamlessM4Tv2Model, mel_seq_len: int) -> None:
    """JIT-warm speech encoder for ``mel_seq_len`` (no program-cache clear afterwards)."""
    tt_model.prewarm_speech_encoder([int(mel_seq_len)])
    ttnn.synchronize_device(tt_model.device)


def _prewarm_vocoder_from_last_generate(tt_model: TTSeamlessM4Tv2Model) -> None:
    """JIT-warm vocoder conv weights and Metal programs for the last ``generate()`` speech path."""
    tt_model.prewarm_vocoder_programs()


def _process_jit_preflight(
    *,
    session_kw: dict,
    input_ids: torch.Tensor,
    input_text_attn: torch.Tensor,
    gen_common: dict,
) -> None:
    """One untimed T2TT generate on a throwaway device to warm the global JIT cache."""
    print("  Cold-start preflight: one untimed T2TT warmup on a throwaway device …")
    with _isolated_task_session(**session_kw) as (device, tt_model):
        ids_tt = torch_ids_to_ttnn(device, input_ids)
        attn_tt = torch_ids_to_ttnn(device, input_text_attn)
        out = tt_model.generate(
            input_ids=ids_tt,
            attention_mask=attn_tt,
            generate_speech=False,
            tgt_lang="hin",
            **gen_common,
        )
        ttnn.synchronize_device(device)
        ttnn.deallocate(out.sequences)
    print("  Cold-start preflight: done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    weights_dir = _weights_dir()
    path = os.fspath(weights_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

    torch.manual_seed(0)
    hf_model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = hf_model.t2u_model.config
    sample_rate = int(getattr(cfg, "sampling_rate", 16000))

    # ---- Single English prompt drives T2TT/T2ST (optional sentence cap via env) ----
    src_text = _demo_src_text()
    if src_text != _DEFAULT_SRC_TEXT:
        print(f"  Text input segmented: using first {os.environ.get('SEAMLESS_DEMO_MAX_SENTENCES')} sentence(s)")
    src_lang = "eng"
    tgt_translate = "hin"  # task 1, 2: translate eng → hin
    speech_src_lang = "eng"  # tasks 3–5: preamble WAV is English
    tgt_s2tt = "hin"  # task 3: English speech → Hindi text
    tgt_speech_other = "spa"  # task 4: English speech → Spanish speech
    tgt_asr = "eng"  # task 5: transcribe English speech → English text

    text_inputs = processor(text=src_text, src_lang=src_lang, return_tensors="pt")
    input_ids = text_inputs["input_ids"]
    input_text_attn = text_inputs["attention_mask"]

    use_decode_trace = True
    use_2cq = True
    gen_common = hf_aligned_generation_kwargs(
        hf_model.generation_config,
        use_kv_cache=True,
        use_decode_trace=use_decode_trace,
        use_2cq=use_2cq,
        use_t2u_trace=False,
        return_timings=True,
    )

    try:
        original_default = ttnn.GetDefaultDevice()
    except Exception:
        # Default device may be unset before the first demo task opens a mesh.
        original_default = None

    trace_info = "trace+2CQ" if (use_decode_trace and use_2cq) else ("trace" if use_decode_trace else "eager")
    print(f"  Demo mode: one mesh device open/close per task — decode: {trace_info}")
    print(
        f"  Warmup: text={_DEMO_WARMUP_ITERS}, speech={_DEMO_SPEECH_WARMUP_ITERS} untimed iter(s), "
        f"then {_DEMO_MEASURE_ITERS} timed text / {_DEMO_SPEECH_MEASURE_ITERS} timed speech "
        f"(report min elapsed + phase timings from fastest timed iter)"
    )
    print(
        f"  HF-aligned greedy: max_new_tokens={gen_common['max_new_tokens']} "
        f"(cap), eos_token_id={gen_common['eos_token_id']}, "
        f"repetition_penalty={gen_common['repetition_penalty']}, "
        f"decode=trace+2CQ+ttnn_argmax"
    )

    perf_tt_log: list = []
    session_kw = dict(
        hf_model=hf_model,
        cfg=cfg,
        t2u_cfg=t2u_cfg,
        gen_common=gen_common,
        original_default=original_default,
    )

    _process_jit_preflight(
        session_kw=session_kw,
        input_ids=input_ids,
        input_text_attn=input_text_attn,
        gen_common=gen_common,
    )

    preamble_path = ensure_demo_audio()
    preamble_wav, _ = _load_mono_wav_resampled(preamble_path, sample_rate)
    audio_inputs = processor(audios=preamble_wav, sampling_rate=sample_rate, return_tensors="pt")
    input_features = audio_inputs["input_features"]
    input_speech_attn = audio_inputs["attention_mask"]
    mel_frames = int(input_speech_attn.sum().item())
    print(
        f"  Speech-input tasks use: {preamble_path} "
        f"({preamble_wav.size} samples @ {sample_rate} Hz, "
        f"{preamble_wav.size / sample_rate:.2f}s, mel_frames={mel_frames})"
    )

    # =========================================================================
    # 1. T2TT — Text-to-Text Translation (English → Hindi)
    # =========================================================================
    _print_header(1, "Text-to-Text Translation", "T2TT", "eng", tgt_translate)
    print(f"  Input text  ({src_lang}): {src_text}")
    with _isolated_task_session(**session_kw) as (device, tt_model):
        ids_tt = torch_ids_to_ttnn(device, input_ids)
        attn_tt = torch_ids_to_ttnn(device, input_text_attn)
        t2tt_out, _ = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_ids=ids_tt,
                attention_mask=attn_tt,
                generate_speech=False,
                tgt_lang=tgt_translate,
                **gen_common,
            ),
            release_fn=lambda o: ttnn.deallocate(o.sequences),
        )
        if not isinstance(t2tt_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"T2TT expected TTSeamlessM4Tv2GreedySearchOutput, got {type(t2tt_out)}")
        if t2tt_out.timings is not None:
            _record_tt_perf(perf_tt_log, "T2TT", t2tt_out.timings)
        t2tt_text = _decode(tokenizer, t2tt_out.sequences)
        ttnn.deallocate(t2tt_out.sequences)
    print(f"  Output text ({tgt_translate}): {t2tt_text}")

    # =========================================================================
    # 2. T2ST — Text-to-Speech Translation (English text → Hindi speech)
    # =========================================================================
    _print_header(2, "Text-to-Speech Translation", "T2ST", "eng", tgt_translate)
    print(f"  Input text  ({src_lang}): {src_text}")
    with _isolated_task_session(**session_kw) as (device, tt_model):
        ids_tt = torch_ids_to_ttnn(device, input_ids)
        attn_tt = torch_ids_to_ttnn(device, input_text_attn)
        t2st_out, _ = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_ids=ids_tt,
                attention_mask=attn_tt,
                generate_speech=True,
                return_intermediate_token_ids=True,
                tgt_lang=tgt_translate,
                speaker_id=0,
                **gen_common,
            ),
            release_fn=_release_speech_out,
            warmup_iters=_DEMO_SPEECH_WARMUP_ITERS,
            measure_iters=_DEMO_SPEECH_MEASURE_ITERS,
            post_warmup_fn=lambda: _prewarm_vocoder_from_last_generate(tt_model),
        )
        if not isinstance(t2st_out, TTSeamlessM4Tv2GenerationOutput):
            raise TypeError(f"T2ST expected TTSeamlessM4Tv2GenerationOutput, got {type(t2st_out)}")
        if t2st_out.timings is not None:
            _record_tt_perf(perf_tt_log, "T2ST", t2st_out.timings)
        t2st_text = _decode(tokenizer, t2st_out.sequences)
        hindi_wav_np = _waveform_to_mono_fp32(t2st_out.waveform, t2st_out.waveform_lengths)
        t2u_pad = int(t2u_cfg.pad_token_id)
        t2st_text_tokens = _tt_row_length(t2st_out.sequences)
        t2st_n_units = _valid_unit_frames(t2st_out.unit_sequences, pad_id=t2u_pad)
        _release_speech_out(t2st_out)
    print(f"  Intermediate text ({tgt_translate}): {t2st_text}")
    _save_wav(T2ST_WAV, hindi_wav_np, sample_rate=sample_rate)
    print(
        f"  T2ST stats: text_tokens={t2st_text_tokens}, "
        f"unit_frames={t2st_n_units}, "
        f"audio={hindi_wav_np.size} samples ({hindi_wav_np.size / sample_rate:.2f}s)"
    )
    print(f"  Saved to: {T2ST_WAV}")

    # =========================================================================
    # 3. S2TT — Speech-to-Text Translation (English speech → Hindi text)
    # =========================================================================
    _print_header(3, "Speech-to-Text Translation", "S2TT", speech_src_lang, tgt_s2tt)
    print(f"  Input audio ({speech_src_lang}): {preamble_path} ({sample_rate} Hz)")
    with _isolated_task_session(**session_kw) as (device, tt_model):
        _prewarm_speech_encoder(tt_model, mel_frames)
        feats_tt = torch_feats_to_ttnn(device, input_features)
        attn_tt = torch_ids_to_ttnn(device, input_speech_attn)
        s2tt_out, _ = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_features=feats_tt,
                attention_mask=attn_tt,
                generate_speech=False,
                tgt_lang=tgt_s2tt,
                **gen_common,
            ),
            release_fn=lambda o: ttnn.deallocate(o.sequences),
            warmup_iters=_DEMO_SPEECH_WARMUP_ITERS,
        )
        if not isinstance(s2tt_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"S2TT expected TTSeamlessM4Tv2GreedySearchOutput, got {type(s2tt_out)}")
        if s2tt_out.timings is not None:
            _record_tt_perf(perf_tt_log, "S2TT", s2tt_out.timings)
        s2tt_text = _decode(tokenizer, s2tt_out.sequences)
        ttnn.deallocate(s2tt_out.sequences)
    print(f"  Output text ({tgt_s2tt}): {s2tt_text}")

    # =========================================================================
    # 4. S2ST — Speech-to-Speech Translation (English speech → Spanish speech)
    # =========================================================================
    _print_header(4, "Speech-to-Speech Translation", "S2ST", speech_src_lang, tgt_speech_other)
    print(f"  Input audio ({speech_src_lang}): {preamble_path} ({sample_rate} Hz)")
    with _isolated_task_session(**session_kw) as (device, tt_model):
        _prewarm_speech_encoder(tt_model, mel_frames)
        feats_tt = torch_feats_to_ttnn(device, input_features)
        attn_tt = torch_ids_to_ttnn(device, input_speech_attn)
        s2st_out, _ = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_features=feats_tt,
                attention_mask=attn_tt,
                generate_speech=True,
                return_intermediate_token_ids=True,
                tgt_lang=tgt_speech_other,
                speaker_id=0,
                **gen_common,
            ),
            release_fn=_release_speech_out,
            warmup_iters=_DEMO_SPEECH_WARMUP_ITERS,
            measure_iters=_DEMO_SPEECH_MEASURE_ITERS,
            post_warmup_fn=lambda: _prewarm_vocoder_from_last_generate(tt_model),
        )
        if not isinstance(s2st_out, TTSeamlessM4Tv2GenerationOutput):
            raise TypeError(f"S2ST expected TTSeamlessM4Tv2GenerationOutput, got {type(s2st_out)}")
        if s2st_out.timings is not None:
            _record_tt_perf(perf_tt_log, "S2ST", s2st_out.timings)
        s2st_text = _decode(tokenizer, s2st_out.sequences)
        spanish_wav_np = _waveform_to_mono_fp32(s2st_out.waveform, s2st_out.waveform_lengths)
        s2st_n_samples = spanish_wav_np.size
        _release_speech_out(s2st_out)
    print(f"  Intermediate text ({tgt_speech_other}): {s2st_text}")
    _save_wav(S2ST_WAV, spanish_wav_np, sample_rate=sample_rate)
    print(f"  Output audio ({tgt_speech_other}, {sample_rate} Hz, {s2st_n_samples} samples)")
    print(f"  Saved to: {S2ST_WAV}")

    # =========================================================================
    # 5. ASR — Automatic Speech Recognition (English speech → English text)
    # =========================================================================
    _print_header(5, "Automatic Speech Recognition", "ASR", speech_src_lang, tgt_asr)
    print(f"  Input audio ({speech_src_lang}): {preamble_path} ({sample_rate} Hz)")
    print("  Note: ASR transcribes the WAV (speech→text), not any intermediate text string.")
    with torch.no_grad():
        hf_asr_out = hf_model.generate(
            input_features=input_features.float(),
            attention_mask=input_speech_attn,
            generate_speech=False,
            tgt_lang=tgt_asr,
            **_hf_gen_kwargs(gen_common),
        )
    hf_asr_ids = (
        hf_asr_out.sequences[0].cpu().tolist() if hasattr(hf_asr_out, "sequences") else hf_asr_out[0].cpu().tolist()
    )
    print(
        f"  HF reference ({tgt_asr}, {len(hf_asr_ids)} tokens): "
        f"{tokenizer.batch_decode([hf_asr_ids], skip_special_tokens=True)[0]}"
    )
    with _isolated_task_session(**session_kw) as (device, tt_model):
        _prewarm_speech_encoder(tt_model, mel_frames)
        feats_tt = torch_feats_to_ttnn(device, input_features)
        attn_tt = torch_ids_to_ttnn(device, input_speech_attn)
        asr_out, _ = _warmup_and_time(
            device,
            lambda: tt_model.generate(
                input_features=feats_tt,
                attention_mask=attn_tt,
                generate_speech=False,
                tgt_lang=tgt_asr,
                **gen_common,
            ),
            release_fn=lambda o: ttnn.deallocate(o.sequences),
            warmup_iters=_DEMO_SPEECH_WARMUP_ITERS,
        )
        if not isinstance(asr_out, TTSeamlessM4Tv2GreedySearchOutput):
            raise TypeError(f"ASR expected TTSeamlessM4Tv2GreedySearchOutput, got {type(asr_out)}")
        tt_asr_ids = to_torch_replicated_first_shard(asr_out.sequences).long().reshape(-1).tolist()
        lcp = 0
        for a, b in zip(hf_asr_ids, tt_asr_ids):
            if a != b:
                break
            lcp += 1
        print(f"  HF/TT token prefix match: {lcp} (seed + {max(0, lcp - 2)} content tokens)")
        if asr_out.timings is not None:
            _record_tt_perf(perf_tt_log, "ASR", asr_out.timings)
        asr_text = _decode(tokenizer, asr_out.sequences)
        ttnn.deallocate(asr_out.sequences)
    print(f"  TT output ({tgt_asr}): {asr_text}")

    print()
    print("=" * 78)
    print("  ok — all five tasks completed")
    print("=" * 78)
    print(f"  Audio outputs saved under: {OUTPUT_DIR}")
    _print_tt_perf_summary(perf_tt_log)


@pytest.mark.timeout(7200)
@pytest.mark.skipif(
    _requires_bh_qb(),
    reason="requires exactly 4 devices (MeshShape(1, 4))",
)
def test_seamless_m4t_v2_demo():
    """CI smoke: all five tasks via ``main()`` (Blackhole demo pipeline entry point)."""
    main()


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS demo — **fully on TT, zero reference model**.

All neural-network forward passes (text, acoustic, audio tokenizer decode) run on the
Tenstorrent device via ``VoxtralTTSPipeline.forward_device_resident()``.  The only CPU work is:

- Mistral-common **tokenization** (equivalent to ``AutoTokenizer.encode()``)
- Voice-embedding file load (a single ``torch.load`` of a small ``.pt`` file)
- Per-step acoustic **code** accumulation for EOA / output (tiny host tensors)

Modes
-----
``text`` (default)
    ``text`` + ``voice`` in JSON → ``pipe.forward_device_resident()`` → ``.wav``.
    Alternatively use ``text_paragraphs``: a JSON array of short strings; the demo
    joins them with spaces into one prompt for the tokenizer (readable in the file).

``codes``
    Pre-computed ``[1,37,T]`` codes tensor → ``pipe.decode_waveform_from_codes_tt()``

``latents``
    Pre-computed ``[1,1,T,C]`` latent tensor → TT mel decode + pretransform

Run (from tt-metal repo root)::

    export VOXTRAL_TTS_MODEL=mistralai/Voxtral-4B-TTS-2603
    ./python_env/bin/python models/experimental/voxtraltts/demo/demo.py \\
        --prompts models/experimental/voxtraltts/demo/data/sample_prompts.json \\
        --output-dir models/experimental/voxtraltts/demo/data
    python models/experimental/voxtraltts/demo/demo.py --text "this is a test message for VoxtralTTS. What is the architecture of the voxtral tts and how does it work?"

"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from loguru import logger
from scipy.io import wavfile

import ttnn

from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL, load_voxtral_config
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_hf_aligned_optimizations


# ---------------------------------------------------------------------------
# A. Argument groups  (Llama DemoArgs pattern: model / tt / data)
# ---------------------------------------------------------------------------


@dataclass
class ModelArgs:
    model_name_or_path: str = DEFAULT_VOXTRAL_MODEL


@dataclass
class TTArgs:
    text_max_seq_len: int = 4096
    text_dtype: str = "bfloat16"
    acoustic_dtype: str = "bfloat16"
    tokenizer_dtype: str = "bfloat16"
    use_paged_kv_cache: bool = False
    paged_block_size: int = 32


@dataclass
class DataArgs:
    prompts_file: str = "models/experimental/voxtraltts/demo/data/sample_prompts.json"
    output_dir: str = "generated/voxtraltts_demo"
    mode: str = "text"
    # Upper bound on AR acoustic steps. The demo auto-raises this per-prompt when the
    # word count implies more tokens are needed (see _min_speech_tokens). Use a small
    # value like 64 for quick smoke tests; leave at 0 to always use the auto-estimate.
    max_speech_tokens: int = 5000
    seed: int = 0
    default_voice: str = "casual_female"
    warmup_iters: int = 0
    inline_texts: list[str] | None = None
    voice: str | None = None


@dataclass
class DemoArgs:
    model: ModelArgs
    tt: TTArgs
    data: DataArgs


def _ttnn_dtype(name: str) -> ttnn.DataType:
    return getattr(ttnn, name)


def _parse_demo_args(argv: list[str] | None = None) -> DemoArgs:
    p = argparse.ArgumentParser(description="Voxtral TTS fully-TT demo.")
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VOXTRAL_TTS_MODEL") or os.environ.get("HF_MODEL") or DEFAULT_VOXTRAL_MODEL,
    )
    p.add_argument("--prompts", type=str, default=DataArgs.prompts_file)
    p.add_argument(
        "--text",
        type=str,
        nargs="+",
        action="append",
        default=None,
        help="Inline text prompt (text mode). Quotes are optional; repeat --text for multiple prompts. Bypasses --prompts JSON.",
    )
    p.add_argument("--output-dir", type=str, default=DataArgs.output_dir)
    p.add_argument("--mode", type=str, choices=("text", "codes", "latents"), default="text")
    p.add_argument("--text-max-seq-len", type=int, default=4096)
    p.add_argument(
        "--max-speech-tokens",
        type=int,
        default=DataArgs.max_speech_tokens,
        help="Autoregressive acoustic steps (upper bound). The demo auto-raises this value "
        "when the word count requires more tokens (~8 tokens/word). Default 12000 covers "
        "~1500 words; use 0 or a small value for quick smoke tests.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-iters", type=int, default=0)
    p.add_argument("--default-voice", type=str, default="casual_male")
    p.add_argument(
        "--use-paged-kv-cache",
        action="store_true",
        default=False,
        help="Enable paged KV cache to bypass L1 CB size limit. Allows text_max_seq_len>4096 on Blackhole P150.",
    )
    p.add_argument(
        "--paged-block-size", type=int, default=32, help="KV block size for paged attention (multiple of 32)."
    )
    ns = p.parse_args(argv)
    inline_texts = [" ".join(parts).strip() for parts in ns.text] if ns.text else None
    if inline_texts and ns.mode != "text":
        p.error("--text is only valid with --mode text")
    if inline_texts and any(not text for text in inline_texts):
        p.error("--text requires a non-empty prompt")
    return DemoArgs(
        model=ModelArgs(model_name_or_path=ns.model),
        tt=TTArgs(
            text_max_seq_len=ns.text_max_seq_len,
            use_paged_kv_cache=ns.use_paged_kv_cache,
            paged_block_size=ns.paged_block_size,
        ),
        data=DataArgs(
            prompts_file=ns.prompts,
            output_dir=ns.output_dir,
            mode=ns.mode,
            max_speech_tokens=ns.max_speech_tokens,
            seed=ns.seed,
            default_voice=ns.default_voice,
            warmup_iters=ns.warmup_iters,
            inline_texts=inline_texts,
        ),
    )


# ---------------------------------------------------------------------------
# B. Device + pipeline init
# ---------------------------------------------------------------------------


def _open_device():
    from tests.scripts.common import get_updated_device_params
    from models.experimental.voxtraltts.demo.decode_trace_2cq import (
        decode_trace_enabled,
        num_command_queues_for_decode,
    )

    n_devices = ttnn.get_num_devices()
    force_single = os.environ.get("VOXTRAL_TTS_FORCE_SINGLE_DEVICE", "").lower() in ("1", "true", "yes", "on")
    params = {}
    # Traced decode needs a trace region; 2CQ needs a second command queue. Both default on
    # (see decode_trace_2cq). trace_region_size is env-tunable for the 26-layer decode graph.
    if decode_trace_enabled():
        params["trace_region_size"] = int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000)))
        params["num_command_queues"] = num_command_queues_for_decode()
    updated = get_updated_device_params(params)
    original = ttnn.GetDefaultDevice()

    if n_devices >= 4 and not force_single:
        # Blackhole QuietBox: use all four chips as a 1x4 tensor-parallel mesh.
        # Fabric must be configured before opening the mesh or TT CCL collectives can fail.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_shape = ttnn.MeshShape(1, 4)
        device_ids = [0, 1, 2, 3]
        logger.info("Opening Voxtral TTS on 1x4 BH QB mesh (physical_device_ids={})", device_ids)
    elif ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.TG:
        mesh_shape = ttnn.MeshShape(1, 1)
        device_ids = [4]
        logger.info("Opening Voxtral TTS on TG single-device mesh (physical_device_ids={})", device_ids)
    else:
        mesh_shape = ttnn.MeshShape(1, 1)
        device_ids = [0]
        logger.info("Opening Voxtral TTS on 1x1 mesh (physical_device_ids={})", device_ids)

    mesh = ttnn.open_mesh_device(mesh_shape, physical_device_ids=device_ids, **updated)
    ttnn.SetDefaultDevice(mesh)
    return mesh, original


def _check_seq_len_memory(text_max_seq_len: int, n_devices: int = 1) -> None:
    """Warn early if text_max_seq_len will likely cause OOM.

    KV cache is pre-allocated at model init regardless of actual input length.
    For TP=n_devices the text KV cache is sharded, so each chip holds 1/n_devices.
    Formula per chip: seq_len × 32 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes / n_devices.
    """
    n_devices = max(1, int(n_devices))
    kv_bytes = text_max_seq_len * 32 * 8 * 128 * 2 * 2 // n_devices
    kv_gb = kv_bytes / (1024**3)
    DEVICE_DRAM_GB = 32.0  # Blackhole — 32 GB LPDDR5 DRAM per chip
    WEIGHTS_GB = 14.5  # Conservative full-pipeline footprint; text weights are TP-sharded on QB.
    RUNTIME_HEADROOM_GB = 4.0
    usable_gb = DEVICE_DRAM_GB * 0.85
    estimated_total_gb = WEIGHTS_GB + kv_gb + RUNTIME_HEADROOM_GB
    available_gb = usable_gb - WEIGHTS_GB - RUNTIME_HEADROOM_GB

    logger.info(
        f"[memory] text_max_seq_len={text_max_seq_len}, n_devices={n_devices} → "
        f"per-chip KV cache = {kv_gb:.2f} GB "
        f"(Blackhole: {DEVICE_DRAM_GB:.0f} GB DRAM/chip, estimated per-chip footprint ~{estimated_total_gb:.2f} GB)"
    )
    if estimated_total_gb > usable_gb:
        raise MemoryError(
            f"\n{'='*70}\n"
            f"OOM RISK: text_max_seq_len={text_max_seq_len} requires {kv_gb:.2f} GB for the text KV cache.\n"
            f"Estimated total footprint is ~{estimated_total_gb:.2f} GB, above the safe {usable_gb:.2f} GB budget\n"
            f"for a {DEVICE_DRAM_GB:.0f} GB Blackhole P150 after model weights and runtime headroom.\n\n"
            f"This can reach 'Pipeline ready' and then be killed by the OS/runtime without a Python traceback.\n\n"
            f"Fix — reduce text_max_seq_len. Recommended values:\n"
            f"  ≤650  words / ≤4 min  →  text_max_seq_len=4096   (KV={4096*32*8*128*4/1024**3:.2f} GB)\n"
            f"  ≤1300 words / ≤8 min  →  text_max_seq_len=8192   (KV={8192*32*8*128*4/1024**3:.2f} GB)\n"
            f"  ≤2600 words / ≤17 min →  text_max_seq_len=16384  (KV={16384*32*8*128*4/1024**3:.2f} GB)\n"
            f"  ≤5200 words / ≤34 min →  text_max_seq_len=32768  (KV={32768*32*8*128*4/1024**3:.2f} GB)\n"
            f"{'='*70}"
        )
    elif kv_gb > available_gb * 0.6:
        logger.warning(
            f"[memory] KV cache ({kv_gb:.2f} GB) uses {kv_gb/available_gb*100:.0f}% of available DRAM — "
            f"consider reducing text_max_seq_len to avoid OOM."
        )


def _load_pipeline(mesh: ttnn.Device, args: DemoArgs) -> VoxtralTTSPipeline:
    _check_seq_len_memory(args.tt.text_max_seq_len, n_devices=mesh.get_num_devices())
    return VoxtralTTSPipeline.from_model_name(
        mesh,
        model_name_or_path=args.model.model_name_or_path,
        text_max_seq_len=args.tt.text_max_seq_len,
        text_dtype=_ttnn_dtype(args.tt.text_dtype),
        text_optimizations=voxtral_text_hf_aligned_optimizations,
        acoustic_dtype=_ttnn_dtype(args.tt.acoustic_dtype),
        tokenizer_dtype=_ttnn_dtype(args.tt.tokenizer_dtype),
        use_paged_kv_cache=args.tt.use_paged_kv_cache,
        paged_block_size=args.tt.paged_block_size,
    )


# ---------------------------------------------------------------------------
# C. JSON prompt loading
# ---------------------------------------------------------------------------


def load_prompt_items(path: str, default_voice: str) -> list[dict[str, Any]]:
    # Plain-text support: entire .txt file is treated as ONE prompt item.
    # All lines are joined with a space so paragraphs flow into a single string.
    if str(path).endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        text = " ".join(line.strip() for line in content.splitlines() if line.strip())
        if not text:
            raise ValueError(f"No text found in {path}")
        return [{"id": 0, "text": text, "voice": default_voice}]

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "items" in raw:
        raw = raw["items"]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected non-empty JSON list in {path}")
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw):
        if isinstance(row, str):
            out.append({"id": i, "text": row, "voice": default_voice})
        elif isinstance(row, dict):
            row = dict(row)
            row.setdefault("id", i)
            row.setdefault("voice", default_voice)
            # Readable multi-line prompts: optional ``text_paragraphs`` list → one ``text`` string (space-joined).
            if not (row.get("text") or "").strip():
                paras = row.get("text_paragraphs")
                if isinstance(paras, list) and paras:
                    row["text"] = " ".join(str(p).strip() for p in paras if str(p).strip())
            if not (row.get("text") or "").strip():
                raise ValueError(
                    f"Prompt entry {i} needs non-empty ``text`` or ``text_paragraphs`` (got keys={list(row.keys())})"
                )
            out.append(row)
        else:
            raise ValueError(f"Bad prompt entry {i}: {type(row)}")
    return out


# ---------------------------------------------------------------------------
# D + E. Inference helpers with TTFT / throughput logging
# ---------------------------------------------------------------------------


def _hf_bump_ratio(w: np.ndarray, sample_rate: int) -> float:
    """Energy ratio of the 5-7 kHz band to the 3-5 kHz band.

    A normal speech roll-off gives a ratio < 1 (HF decays).  A value > ~1.15 means
    the 5-7 kHz band carries *more* energy than the band below it — an aperiodic HF
    "grain" bump that some Voxtral voices (e.g. casual_male) exhibit and others
    (e.g. casual_female, ratio ~0.96) do not.
    """
    spec = np.abs(np.fft.rfft(w))
    freqs = np.linspace(0.0, sample_rate / 2.0, len(spec))
    e_35 = float(spec[(freqs >= 3000) & (freqs < 5000)].sum())
    e_57 = float(spec[(freqs >= 5000) & (freqs < 7000)].sum())
    return e_57 / max(e_35, 1e-9)


def _save_wav(path: Path, waveform_f32: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = waveform_f32.detach().float().cpu().numpy().reshape(-1)

    # Adaptive HF de-graining: only voices with an abnormal 5-7 kHz bump are filtered.
    # The 300-3400 Hz speech band is preserved 100% by the low-pass, so intelligibility
    # is unaffected; normal voices (ratio below the gate) are left bit-for-bit unchanged.
    # Tunable via env: VOXTRAL_DEGRAIN_GATE (ratio threshold), VOXTRAL_DEGRAIN_HZ (cutoff),
    # set VOXTRAL_DEGRAIN_HZ=0 to disable entirely.
    cutoff_hz = float(os.environ.get("VOXTRAL_DEGRAIN_HZ", "6000"))
    gate = float(os.environ.get("VOXTRAL_DEGRAIN_GATE", "1.15"))
    if cutoff_hz > 0 and _hf_bump_ratio(w, sample_rate) > gate:
        try:
            from scipy.signal import butter, sosfiltfilt

            sos = butter(6, cutoff_hz / (sample_rate / 2.0), "low", output="sos")
            w = sosfiltfilt(sos, w).astype(np.float32)
            logger.info(f"De-grained output (HF bump > {gate:.2f}) with {cutoff_hz:.0f} Hz low-pass")
        except ImportError:
            logger.warning("scipy unavailable; skipping HF de-graining")

    peak = float(np.abs(w).max())
    if peak > 0.95:
        w = w * (0.95 / peak)
    w = np.clip(w * 32767.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, w)


def _min_speech_tokens(text: str) -> int:
    """Lower-bound estimate of acoustic tokens needed for *text* with a 1.4× safety margin.

    Voxtral generates at ~12.5 acoustic tokens/second.  Typical narration speed is
    ~130 words/minute (2.17 words/second) → ~5.8 tokens/word.  Multiplied by 1.4 gives
    ~8 tokens/word.  The returned value is a FLOOR; callers should take
    ``max(user_supplied, _min_speech_tokens(text))``.
    """
    return max(4096, len(text.split()) * 8)


# Threshold above which text is split into chunks to prevent AR degeneration.
# The model's free-run PCC (~0.79) causes accumulated errors that collapse the
# semantic code distribution to a single repeated value after ~200 tokens (~20 words).
# Keeping each chunk ≤ _CHUNK_MAX_WORDS prevents that collapse.
_CHUNK_THRESHOLD_WORDS = 100  # threshold for text_max_seq_len > 4096 (long context)
_CHUNK_THRESHOLD_WORDS_SHORT = 200  # threshold for text_max_seq_len <= 4096 (short context)
_CHUNK_MAX_WORDS = 200


def _split_into_chunks(text: str, max_words: int = _CHUNK_MAX_WORDS) -> list[str]:
    """Split *text* into sentence-aligned chunks of at most *max_words* words each."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buf: list[str] = []
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if len(buf) + len(words) <= max_words:
            buf.extend(words)
        else:
            if buf:
                chunks.append(" ".join(buf))
            # If a single sentence exceeds the limit, hard-split it word-by-word.
            while len(words) > max_words:
                chunks.append(" ".join(words[:max_words]))
                words = words[max_words:]
            buf = words
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if c.strip()]


def _run_chunked_text_mode(
    pipe: "VoxtralTTSPipeline",
    text: str,
    voice: str,
    seed: int,
    sample_rate: int,
    out_path: Path,
) -> None:
    """Generate audio chunk-by-chunk to avoid AR degeneration on long texts.

    Each chunk is generated by an independent ``forward_device_resident`` call so the
    AR loop never accumulates enough errors to collapse the semantic-code distribution.
    The chunk waveforms are concatenated and peak-normalised before saving.
    """
    chunks = _split_into_chunks(text)
    logger.info(f"[chunked] {len(text.split())} words → {len(chunks)} chunks " f"(max {_CHUNK_MAX_WORDS} words each)")
    all_wavs: list[torch.Tensor] = []
    n_frames_total = 0
    first_frame_s: float | None = None  # TTFA = first audible frame of the whole run (first chunk)
    t_start = perf_counter()

    for i, chunk in enumerate(chunks):
        n_words = len(chunk.split())
        # Give each chunk 12 tokens/word headroom (vs 8 for full-text estimate) because
        # short prompts generate slower speech and may have longer pauses.
        chunk_max_tokens = max(600, n_words * 12)
        logger.info(f"[chunked] chunk {i + 1}/{len(chunks)}: {n_words} words | max_tokens={chunk_max_tokens}")

        out = pipe.forward_device_resident(
            text=chunk,
            voice=voice,
            max_tokens=chunk_max_tokens,
            seed=seed,
        )
        if first_frame_s is None and out.first_frame_s is not None:
            first_frame_s = out.first_frame_s
        n_frames_total += int(out.codes_b37t.shape[2])
        if out.waveform.numel() > 0:
            all_wavs.append(out.waveform.reshape(-1))
        if not out.hit_end_audio:
            logger.warning(
                f"[chunked] chunk {i + 1}: reached max_tokens={chunk_max_tokens} without END_AUDIO — "
                f"chunk may be cut off."
            )

    total_s = perf_counter() - t_start
    n_samples = sum(int(w.numel()) for w in all_wavs)
    audio_s = n_samples / sample_rate if sample_rate > 0 else None
    _log_perf(
        "chunked_generate",
        total_s=total_s,
        audio_s=audio_s,
        n_chars=len(text),
        n_frames=n_frames_total,
        first_frame_s=first_frame_s,
        bitrate_bps=_codec_bitrate_bps(pipe),
    )

    if not all_wavs:
        logger.error("[chunked] No audio generated from any chunk.")
        return

    combined = torch.cat(all_wavs, dim=0)
    _save_wav(out_path, combined, sample_rate)
    logger.info(
        f"Saved chunked waveform → {out_path}  "
        f"({combined.numel()} samples @ {sample_rate} Hz, {len(chunks)} chunks)"
    )


def _codec_bitrate_bps(pipe: "VoxtralTTSPipeline") -> float | None:
    """Discrete-codec bitrate (bits/s): bits-per-frame × frame-rate.

    ``bits/frame = log2(semantic_codebook_size) + n_acoustic_codebook · log2(acoustic_codebook_size)``;
    ``frame-rate = sampling_rate / downsample_factor``. This is a model constant (independent of the
    utterance — it assumes full codebook usage), reported for parity with neural-codec model cards.
    """
    try:
        am = pipe.config.audio_model_args
        sr = int(am.audio_encoding_args.sampling_rate)
        ds = int(pipe._downsample_factor)
        if sr <= 0 or ds <= 0:
            return None
        frame_rate = sr / ds
        bits_per_frame = math.log2(am.semantic_codebook_size) + am.n_acoustic_codebook * math.log2(
            am.acoustic_codebook_size
        )
        return bits_per_frame * frame_rate
    except Exception:
        return None


def _log_perf(
    label: str,
    *,
    total_s: float,
    audio_s: float | None = None,
    n_chars: int | None = None,
    n_frames: int | None = None,
    first_frame_s: float | None = None,
    bitrate_bps: float | None = None,
) -> None:
    """Log non-streaming perf metrics.

    Latency matches vLLM-Omni ``vllm_elapsed`` (end-to-end generation wall time).
    RTF uses the model-card convention: ``generation_time / audio_duration`` (lower = better;
    RTF < 1 means faster than realtime). vLLM end2end.py inverts this (audio/gen); we do not.
    First-audio latency (TTFA) is the wall time until the first acoustic frame is produced.
    Frame rate is generation throughput (acoustic frames decoded per wall-second).
    Bitrate is the discrete-codec rate (a model constant; see :func:`_codec_bitrate_bps`).
    """
    logger.info(f"[{label}] Latency: {total_s * 1000:.2f} ms  ({total_s:.3f} s)")
    if first_frame_s is not None and first_frame_s > 0:
        logger.info(f"[{label}] First-audio latency (TTFA): {first_frame_s * 1000:.2f} ms")
    if audio_s is not None and audio_s > 0 and total_s > 0:
        rtf = total_s / audio_s
        logger.info(
            f"[{label}] RTF: {rtf:.4f}  (lower=better; {audio_s / total_s:.2f}x realtime)  "
            f"[audio {audio_s:.2f}s / gen {total_s:.2f}s]"
        )
    if n_chars is not None and n_chars > 0 and total_s > 0:
        logger.info(f"[{label}] Throughput: {n_chars / total_s:.2f} char/s  ({n_chars} chars in {total_s:.3f} s)")
    if n_frames is not None and n_frames > 0 and total_s > 0:
        logger.info(f"[{label}] Frame rate: {n_frames / total_s:.2f} frame/s  ({n_frames} frames in {total_s:.3f} s)")
    if bitrate_bps is not None and bitrate_bps > 0:
        logger.info(f"[{label}] Bitrate: {bitrate_bps / 1000.0:.2f} kbps  (discrete codec; {bitrate_bps:.0f} bit/s)")


def run_text_mode(
    pipe: VoxtralTTSPipeline,
    text: str,
    voice: str,
    max_tokens: int,
    seed: int,
    sample_rate: int,
    out_path: Path,
    text_max_seq_len: int = 4096,
) -> None:
    """Full TT TTS (device-resident AR loop): text → acoustic codes → waveform.

    For texts longer than the chunk threshold words the input is split into
    sentence-aligned chunks and each chunk is generated independently.  This prevents the
    AR degeneration (semantic-code collapse) that occurs after ~200 acoustic tokens when
    running in free-run mode with PCC ≈ 0.79.

    Chunking thresholds:
      - text_max_seq_len > 4096  (long context / paged KV): chunk at > 35 words
      - text_max_seq_len <= 4096 (short context):           chunk at > 200 words
    """
    chunk_threshold = _CHUNK_THRESHOLD_WORDS if text_max_seq_len > 4096 else _CHUNK_THRESHOLD_WORDS_SHORT
    if len(text.split()) > chunk_threshold:
        _run_chunked_text_mode(pipe, text, voice, seed, sample_rate, out_path)
        return

    # Short text path — single forward pass.
    # Auto-raise max_tokens if the word count suggests more tokens are needed.
    min_needed = _min_speech_tokens(text)
    if max_tokens < min_needed:
        logger.warning(
            f"[tt_generate] max_tokens={max_tokens} is below the estimated minimum "
            f"({min_needed}) for {len(text.split())} words. Auto-raising to {min_needed}."
        )
        max_tokens = min_needed

    t0 = perf_counter()
    out = pipe.forward_device_resident(text=text, voice=voice, max_tokens=max_tokens, seed=seed)
    wav = out.waveform
    t1 = perf_counter()
    total_s = t1 - t0

    n_samples = int(wav.numel())
    audio_s = n_samples / sample_rate if sample_rate > 0 else None
    n_frames = int(out.codes_b37t.shape[2])
    _log_perf(
        "tt_generate",
        total_s=total_s,
        audio_s=audio_s,
        n_chars=len(text),
        n_frames=n_frames,
        first_frame_s=out.first_frame_s,
        bitrate_bps=_codec_bitrate_bps(pipe),
    )
    if out.shifted_codes_t37.numel() > 0:
        semantic = out.shifted_codes_t37[:, 0]
        logger.info(
            "[tt_generate] semantic shifted-code stats: "
            f"min={int(semantic.min().item())}, max={int(semantic.max().item())}, "
            f"unique={int(semantic.unique().numel())}, first10={semantic[:10].tolist()}"
        )
    if not out.hit_end_audio:
        logger.warning(
            f"[tt_generate] Reached max_speech_tokens={max_tokens} without END_AUDIO; output was truncated. "
            f"Suggested minimum for {len(text.split())} words: {_min_speech_tokens(text)} tokens."
        )

    codes_path = out_path.with_suffix(".codes.pt")
    torch.save(
        {
            "codes_b37t": out.codes_b37t.cpu(),
            "shifted_codes_t37": out.shifted_codes_t37.cpu(),
            "hit_end_audio": out.hit_end_audio,
            "downsample_factor": pipe._downsample_factor,
        },
        codes_path,
    )
    logger.info(f"Saved TT generated codes → {codes_path}")

    _save_wav(out_path, wav, sample_rate)
    logger.info(f"Saved TT waveform → {out_path}  ({n_samples} samples @ {sample_rate} Hz)")


def run_codes_mode(
    pipe: VoxtralTTSPipeline,
    codes_b37t: torch.Tensor,
    sample_rate: int,
    out_path: Path,
) -> None:
    T = int(codes_b37t.shape[2])
    t0 = perf_counter()
    wav = pipe.decode_waveform_from_codes_tt(codes_b37t.long())
    t1 = perf_counter()
    audio_s = int(wav.numel()) / sample_rate if sample_rate > 0 else None
    _log_perf("tt_codes_decode", total_s=t1 - t0, audio_s=audio_s)
    _save_wav(out_path, wav.squeeze(0).squeeze(0), sample_rate)
    logger.info(f"Saved → {out_path}")


def run_latents_mode(
    pipe: VoxtralTTSPipeline,
    latent: torch.Tensor,
    sample_rate: int,
    out_path: Path,
) -> None:
    if latent.dim() == 3:
        latent = latent.unsqueeze(0)
    if latent.dim() != 4 or int(latent.shape[1]) != 1:
        raise ValueError(f"Expected latent [1,1,T,C], got {tuple(latent.shape)}")
    latent = latent.to(dtype=torch.bfloat16).contiguous()
    t0 = perf_counter()
    lt = ttnn.from_torch(
        latent,
        device=pipe.mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mel = pipe.audio_tokenizer.decode_latent_to_mel_b1tc(lt)
    ttnn.deallocate(lt)
    wav_tt = pipe.audio_tokenizer.pretransform_decode_tt(mel)
    ttnn.deallocate(mel)
    wav = ttnn.to_torch(wav_tt).float()
    ttnn.deallocate(wav_tt)
    t1 = perf_counter()
    T = int(latent.shape[2])
    audio_s = int(wav.numel()) / sample_rate if sample_rate > 0 else None
    _log_perf("tt_latents_decode", total_s=t1 - t0, audio_s=audio_s)
    _save_wav(out_path, wav.squeeze(0).squeeze(0), sample_rate)
    logger.info(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main demo runner
# ---------------------------------------------------------------------------


def run_demo(args: DemoArgs) -> None:
    cfg = load_voxtral_config(args.model.model_name_or_path)
    sample_rate = int(cfg.audio_model_args.audio_encoding_args.sampling_rate)

    if args.data.inline_texts:
        voice = args.data.voice or args.data.default_voice
        items = [{"id": i, "text": t} for i, t in enumerate(args.data.inline_texts)]
    else:
        items = load_prompt_items(args.data.prompts_file, args.data.default_voice)
    out_dir = Path(args.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default the demo to traced decode on single-chip only. The current trace staging buffers
    # are single-chip/full-width; on TP=4 the normal decode path is the validated route.
    # Override explicitly with VOXTRAL_DECODE_TRACE=0/1.
    if "VOXTRAL_DECODE_TRACE" not in os.environ:
        os.environ["VOXTRAL_DECODE_TRACE"] = "0" if ttnn.get_num_devices() >= 4 else "1"

    mesh, original = _open_device()
    pipe: VoxtralTTSPipeline | None = None
    try:
        logger.info(f"Loading VoxtralTTSPipeline from {args.model.model_name_or_path!r} …")
        t0 = perf_counter()
        pipe = _load_pipeline(mesh, args)
        logger.info(f"Pipeline ready in {(perf_counter() - t0) * 1000:.1f} ms")

        for w in range(args.data.warmup_iters + 1):
            is_warmup = w < args.data.warmup_iters
            tag = "warmup" if is_warmup else "run"

            for item in items:
                pid = item.get("id", 0)

                if args.data.mode == "text":
                    text = item.get("text")
                    if not text:
                        raise ValueError("text mode requires a 'text' field in each JSON entry.")
                    voice = str(item.get("voice", args.data.default_voice))
                    out_path = out_dir / f"{tag}_item{pid}.wav"

                    if is_warmup:
                        # Short warmup — only 4 acoustic frames
                        pipe.forward_device_resident(text=text, voice=voice, max_tokens=4, seed=args.data.seed)
                        continue

                    run_text_mode(
                        pipe=pipe,
                        text=text,
                        voice=voice,
                        max_tokens=args.data.max_speech_tokens,
                        seed=args.data.seed,
                        sample_rate=sample_rate,
                        out_path=out_path,
                        text_max_seq_len=args.tt.text_max_seq_len,
                    )

                elif args.data.mode == "codes":
                    if is_warmup:
                        continue
                    raw = item.get("codes")
                    if raw is None:
                        cp = item.get("codes_path")
                        if not cp:
                            raise ValueError("codes mode needs 'codes' or 'codes_path'.")
                        try:
                            raw = torch.load(cp, map_location="cpu", weights_only=False)
                        except TypeError:
                            raw = torch.load(cp, map_location="cpu")
                    codes = torch.as_tensor(raw, dtype=torch.long)
                    if codes.dim() == 2:
                        codes = codes.unsqueeze(0).transpose(1, 2)
                    if codes.dim() != 3 or int(codes.shape[1]) != 37:
                        raise ValueError(f"Expected codes [1,37,T] or [T,37], got {tuple(codes.shape)}")
                    run_codes_mode(pipe, codes, sample_rate, out_dir / f"{tag}_item{pid}.wav")

                elif args.data.mode == "latents":
                    if is_warmup:
                        continue
                    lp = item.get("latent_path")
                    if not lp:
                        raise ValueError("latents mode needs 'latent_path'.")
                    try:
                        lat = torch.load(lp, map_location="cpu", weights_only=False)
                    except TypeError:
                        lat = torch.load(lp, map_location="cpu")
                    run_latents_mode(pipe, lat, sample_rate, out_dir / f"{tag}_item{pid}.wav")

    finally:
        if pipe is not None:
            pipe.cleanup_all()
        ttnn.SetDefaultDevice(original)
        ttnn.close_mesh_device(mesh)


def main(argv: list[str] | None = None) -> None:
    args = _parse_demo_args(argv)
    torch.manual_seed(args.data.seed)
    run_demo(args)


if __name__ == "__main__":
    main()

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

Audio tokenizer decode uses **dense ALiBi SDPA** by default (production-quality waveform).
Pass ``--native-sdpa`` for the faster native sliding-window path (perf-oriented; audible hiss).

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
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request
from models.experimental.voxtraltts.tests.common import close_voxtral_runtime_mesh, open_voxtral_runtime_mesh
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline
from models.experimental.voxtraltts.tt.voxtral_tt_args import (
    voxtral_text_default_optimizations,
    voxtral_text_hf_aligned_optimizations,
)
from models.experimental.voxtraltts.utils.audio_tokenizer_optimizations import (
    voxtral_audio_tokenizer_dense_mask_sdpa_optimizations,
    voxtral_audio_tokenizer_native_sdpa_optimizations,
)


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
    dense_alibi_sdpa: bool = True
    hf_aligned_text: bool = False


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
    warmup_iters: int = 1
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
    p.add_argument("--warmup-iters", type=int, default=1, help="Untimed warmup passes before the measured run.")
    p.add_argument("--default-voice", type=str, default="casual_male")
    p.add_argument(
        "--voice", type=str, default=None, help="Voice for inline --text prompts (overrides --default-voice)."
    )
    p.add_argument(
        "--use-paged-kv-cache",
        action="store_true",
        default=False,
        help="Enable paged KV cache to bypass L1 CB size limit. Allows text_max_seq_len>4096 on Blackhole P150.",
    )
    p.add_argument(
        "--paged-block-size", type=int, default=32, help="KV block size for paged attention (multiple of 32)."
    )
    p.add_argument(
        "--native-sdpa",
        action="store_true",
        default=False,
        help="Use native sliding-window SDPA for audio tokenizer decode (faster, audible hiss). "
        "Default is dense ALiBi SDPA (cleaner audio).",
    )
    p.add_argument(
        "--hf-aligned-text",
        action="store_true",
        default=False,
        help="Use HF-aligned text decode (slower, higher PCC). Default uses production perf optimizations.",
    )
    p.add_argument(
        "--no-decode-trace",
        action="store_true",
        default=False,
        help="Disable 2CQ input staging only (trace replay stays on for audio quality).",
    )
    p.add_argument(
        "--dense-alibi-sdpa",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    ns = p.parse_args(argv)
    inline_texts = [" ".join(parts).strip() for parts in ns.text] if ns.text else None
    if inline_texts and ns.mode != "text":
        p.error("--text is only valid with --mode text")
    if inline_texts and any(not text for text in inline_texts):
        p.error("--text requires a non-empty prompt")
    if ns.no_decode_trace:
        # Explicit user opt-out: short-chunk path. Mark it so the mesh-based default in
        # run_demo() does not re-enable trace (otherwise a stale env var is indistinguishable
        # from an explicit choice).
        os.environ["VOXTRAL_DECODE_TRACE"] = "0"
        os.environ["VOXTRAL_DECODE_TRACE_2CQ"] = "0"
        os.environ["VOXTRAL_DEMO_TRACE_EXPLICIT"] = "1"
    use_dense_alibi = (not ns.native_sdpa) or ns.dense_alibi_sdpa
    return DemoArgs(
        model=ModelArgs(model_name_or_path=ns.model),
        tt=TTArgs(
            text_max_seq_len=ns.text_max_seq_len,
            use_paged_kv_cache=ns.use_paged_kv_cache,
            paged_block_size=ns.paged_block_size,
            dense_alibi_sdpa=use_dense_alibi,
            hf_aligned_text=ns.hf_aligned_text,
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
            voice=ns.voice,
        ),
    )


# ---------------------------------------------------------------------------
# B. Device + pipeline init
# ---------------------------------------------------------------------------


def _open_device():
    from models.experimental.voxtraltts.demo.decode_trace_2cq import num_command_queues_for_decode

    # Production decode always trace-replays; trace region is required regardless of
    # VOXTRAL_DECODE_TRACE (that flag only toggles 2CQ overlap for perf experiments).
    params = {
        "trace_region_size": int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000))),
        "num_command_queues": num_command_queues_for_decode(),
    }
    runtime = open_voxtral_runtime_mesh(params)
    return runtime


def _check_seq_len_memory(text_max_seq_len: int) -> None:
    """Warn early if text_max_seq_len will likely cause OOM.

    KV cache is pre-allocated at model init regardless of actual input length.
    Formula: seq_len × 32 layers × 8 KV heads × 128 head_dim × 2 (K+V) × 2 bytes (bf16).
    """
    kv_bytes = text_max_seq_len * 32 * 8 * 128 * 2 * 2
    kv_gb = kv_bytes / (1024**3)
    DEVICE_DRAM_GB = 32.0  # Blackhole P150 (A84) — 32 GB LPDDR5 DRAM
    WEIGHTS_GB = 14.5  # Voxtral-4B-TTS: text+acoustic+tokenizer weights + driver/OS overhead
    RUNTIME_HEADROOM_GB = 4.0
    usable_gb = DEVICE_DRAM_GB * 0.85
    estimated_total_gb = WEIGHTS_GB + kv_gb + RUNTIME_HEADROOM_GB
    available_gb = usable_gb - WEIGHTS_GB - RUNTIME_HEADROOM_GB

    logger.info(
        f"[memory] text_max_seq_len={text_max_seq_len} → KV cache = {kv_gb:.2f} GB "
        f"(Blackhole P150: {DEVICE_DRAM_GB:.0f} GB DRAM, estimated total footprint ~{estimated_total_gb:.2f} GB)"
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
    _check_seq_len_memory(args.tt.text_max_seq_len)
    audio_tokenizer_optimizations = (
        voxtral_audio_tokenizer_dense_mask_sdpa_optimizations()
        if args.tt.dense_alibi_sdpa
        else voxtral_audio_tokenizer_native_sdpa_optimizations()
    )
    text_optimizations = (
        voxtral_text_hf_aligned_optimizations if args.tt.hf_aligned_text else voxtral_text_default_optimizations
    )
    return VoxtralTTSPipeline.from_model_name(
        mesh,
        model_name_or_path=args.model.model_name_or_path,
        text_max_seq_len=args.tt.text_max_seq_len,
        text_dtype=_ttnn_dtype(args.tt.text_dtype),
        text_optimizations=text_optimizations,
        acoustic_dtype=_ttnn_dtype(args.tt.acoustic_dtype),
        tokenizer_dtype=_ttnn_dtype(args.tt.tokenizer_dtype),
        audio_tokenizer_optimizations=audio_tokenizer_optimizations,
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


def _output_highpass_hz() -> float:
    """Sub-speech high-pass cutoff (Hz) for the final waveform; ``0`` disables.

    The Voxtral codec/vocoder emits a low-frequency rumble (~−23 dB, almost entirely
    below ~150 Hz) that is present even in the pure-torch CPU reference rollout — i.e.
    it is inherent to the model, not a TT/trace artifact. A gentle high-pass below the
    speech band removes the audible "feeble noise" in the gaps while leaving speech
    (male fundamental ≳85 Hz) and all device/trace numerics untouched.
    """
    try:
        return max(0.0, float(os.environ.get("VOXTRAL_OUTPUT_HPF_HZ", "80")))
    except ValueError:
        return 60.0


def _apply_output_highpass(w: np.ndarray, sample_rate: int) -> np.ndarray:
    fc = _output_highpass_hz()
    if fc <= 0.0 or w.size == 0 or fc >= sample_rate / 2:
        return w
    try:
        from scipy import signal as _sig
    except Exception:
        return w  # SciPy unavailable — skip cleanup rather than fail the demo
    sos = _sig.butter(2, fc, btype="highpass", fs=sample_rate, output="sos")
    return _sig.sosfilt(sos, w).astype(np.float32)


def _save_wav(path: Path, waveform_f32: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = waveform_f32.detach().float().cpu().numpy().reshape(-1)
    w = _apply_output_highpass(w, sample_rate)
    peak = float(np.abs(w).max())
    if peak > 0.95:
        w = w * (0.95 / peak)
    w = np.clip(w * 32767.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, w)


def _short_chunking_enabled() -> bool:
    """Short-chunk + crossfade path is for ``VOXTRAL_DECODE_TRACE=0`` only.

    Trace-enabled runs keep the original single-pass / large-chunk demo behavior (RTF + quality
    validated with trace replay). Production compute in ``voxtral_tts`` is unchanged either way.
    """
    from models.experimental.voxtraltts.demo.decode_trace_2cq import decode_trace_enabled

    return not decode_trace_enabled()


def _min_speech_tokens(text: str) -> int:
    n_words = len(text.split())
    estimate = _estimate_acoustic_frames(n_words)
    if _short_chunking_enabled():
        return max(64, estimate)
    # Trace path: headroom for END_AUDIO without forcing 4096 frames on short prompts
    # (4096 undecoded frames decode as noise when END_AUDIO is never hit → 1×4 whole-prompt-missing).
    return max(256, estimate * 4)


def _prompt_seq_len(text: str, voice: str, model_name_or_path: str) -> int:
    request = compose_speech_request(text, model_name_or_path, voice=voice)
    return len(request["prompt_token_ids"])


def _max_decode_tokens(prompt_seq_len: int, text_max_seq_len: int) -> int:
    if prompt_seq_len >= text_max_seq_len:
        raise ValueError(
            f"Prompt token length ({prompt_seq_len}) exceeds text KV cache "
            f"(text_max_seq_len={text_max_seq_len}). Shorten the prompt or raise --text-max-seq-len."
        )
    return text_max_seq_len - prompt_seq_len


def _resolve_max_speech_tokens(
    text: str,
    voice: str,
    model_name_or_path: str,
    max_tokens: int,
    text_max_seq_len: int,
    *,
    log_prefix: str = "tt_generate",
) -> int:
    if not _short_chunking_enabled():
        min_needed = _min_speech_tokens(text)
        if max_tokens < min_needed:
            logger.warning(
                f"[{log_prefix}] max_tokens={max_tokens} is below the estimated minimum "
                f"({min_needed}) for {len(text.split())} words. Auto-raising to {min_needed}."
            )
            return min_needed
        return max_tokens

    prompt_seq_len = _prompt_seq_len(text, voice, model_name_or_path)
    max_decode = _max_decode_tokens(prompt_seq_len, text_max_seq_len)
    min_needed = _min_speech_tokens(text)
    if max_tokens < min_needed:
        logger.warning(
            f"[{log_prefix}] max_tokens={max_tokens} below estimate {min_needed} for "
            f"{len(text.split())} words; raising to {min_needed}."
        )
        max_tokens = min_needed
    if max_tokens > max_decode:
        logger.warning(f"[{log_prefix}] Capping max_tokens {max_tokens} → {max_decode} (KV: prompt={prompt_seq_len}).")
        max_tokens = max_decode
    if max_tokens > _CHUNK_MAX_ACOUSTIC_FRAMES:
        logger.warning(
            f"[{log_prefix}] Capping max_tokens {max_tokens} → {_CHUNK_MAX_ACOUSTIC_FRAMES} "
            f"(AR collapses after ~200 frames / ~16 s)."
        )
        max_tokens = _CHUNK_MAX_ACOUSTIC_FRAMES
    return max_tokens


def _text_generation_passes(
    text: str,
    max_tokens: int,
    text_max_seq_len: int,
    *,
    voice: str,
    model_name_or_path: str,
) -> list[tuple[str, int]]:
    """Return ``(chunk_text, max_tokens)`` pairs using the same rules as ``run_text_mode``."""
    if not _needs_chunking(text, text_max_seq_len):
        return [
            (
                text,
                _resolve_max_speech_tokens(
                    text, voice, model_name_or_path, max_tokens, text_max_seq_len, log_prefix="tt_generate"
                ),
            )
        ]

    passes: list[tuple[str, int]] = []
    for chunk in _split_into_chunks(text):
        n_words = len(chunk.split())
        if _short_chunking_enabled():
            chunk_budget = _chunk_token_budget(n_words)
        else:
            chunk_budget = max(600, n_words * 12)
        passes.append(
            (
                chunk,
                _resolve_max_speech_tokens(
                    chunk,
                    voice,
                    model_name_or_path,
                    chunk_budget,
                    text_max_seq_len,
                    log_prefix="chunked",
                ),
            )
        )
    return passes


# Short-chunk path (VOXTRAL_DECODE_TRACE=0): small passes for AR stability + crossfade joins.
_ACOUSTIC_FRAME_RATE_HZ = 12.5
_CHUNK_MAX_ACOUSTIC_FRAMES = 100  # ~8 s per pass
_TOKENS_PER_WORD_ESTIMATE = 8
_CHUNK_THRESHOLD_WORDS = 25
_CHUNK_THRESHOLD_WORDS_SHORT = 8
_CHUNK_MAX_WORDS = 8
_CHUNK_CROSSFADE_MS = 40.0
_CHUNK_TAIL_RELEASE_MS = 40.0
_CHUNK_LEAD_TRIM_MS = 0.0
_CHUNK_END_FADE_MS = 30.0
# Trailing frames quieter than this fraction of the chunk's 90th-percentile frame RMS are treated
# as the post-speech hiss release and trimmed. ~0.15 sits well below soft speech (~0.25× ref) yet
# above the hiss floor (~0.10× ref) measured on the acoustic decoder output.
_CHUNK_TAIL_RMS_FRACTION = 0.15
_DEGENERACY_WINDOW_FRAMES = 20
_DEGENERACY_MIN_UNIQUE = 6

# Single-device (1×1) trace path: original large-chunk thresholds — typical demo prompts run as a
# single pass (the proven P150 behavior). MUST stay this way; 1×1 trace replay is bit-stable over a
# long pass, so chunking it only hurts quality/RTF.
_TRACE_CHUNK_THRESHOLD_WORDS = 100
_TRACE_CHUNK_THRESHOLD_WORDS_SHORT = 200
_TRACE_CHUNK_MAX_WORDS = 200

# Multi-device (TP / 1×4) trace path ONLY: sentence-aligned chunks. A single long pass drifts
# off-text on the tensor-parallel backbone and emits END_AUDIO early — the model drops whole
# trailing sentences (1×4 stops mid-prompt) while each sentence generated on its own completes
# correctly. Capping a pass at ~one sentence keeps the full prompt spoken at low RTF.
# ``_split_into_chunks`` groups whole sentences up to the cap, so single-sentence prompts still
# run as one pass. These thresholds never apply to 1×1 (see ``_is_multi_device_mesh``).
_TP_TRACE_CHUNK_THRESHOLD_WORDS = 50
_TP_TRACE_CHUNK_MAX_WORDS = 50


def _estimate_acoustic_frames(n_words: int) -> int:
    return max(0, n_words) * _TOKENS_PER_WORD_ESTIMATE


def _chunk_token_budget(n_words: int) -> int:
    if n_words <= 2:
        return max(48, _estimate_acoustic_frames(n_words))
    return _CHUNK_MAX_ACOUSTIC_FRAMES


def _is_multi_device_mesh() -> bool:
    """True only for multi-device compute meshes (e.g. 1×4 TP). 1×1 returns False so the
    sentence-chunking workaround for TP autoregressive drift never touches the single-device path."""
    from models.experimental.voxtraltts.tests.common import voxtral_requested_compute_mesh_shape

    return tuple(voxtral_requested_compute_mesh_shape()) != (1, 1)


def _tp_trace_chunk_max_words() -> int:
    """Multi-device (TP) sentence-chunk cap, overridable via ``VOXTRAL_TP_CHUNK_MAX_WORDS``.
    Set it very high (e.g. 10000) to force a single pass on 1×4 (no inter-chunk modulation), at
    the risk of the TP backbone dropping trailing sentences on long prompts."""
    try:
        v = int(os.environ.get("VOXTRAL_TP_CHUNK_MAX_WORDS", str(_TP_TRACE_CHUNK_MAX_WORDS)))
        return v if v > 0 else _TP_TRACE_CHUNK_MAX_WORDS
    except ValueError:
        return _TP_TRACE_CHUNK_MAX_WORDS


def _chunking_threshold_words(text_max_seq_len: int) -> int:
    if _short_chunking_enabled():
        return _CHUNK_THRESHOLD_WORDS if text_max_seq_len > 4096 else _CHUNK_THRESHOLD_WORDS_SHORT
    # Trace path: only multi-device (TP) gets sentence chunking; 1×1 keeps the single-pass thresholds.
    if _is_multi_device_mesh():
        return _tp_trace_chunk_max_words()
    return _TRACE_CHUNK_THRESHOLD_WORDS if text_max_seq_len > 4096 else _TRACE_CHUNK_THRESHOLD_WORDS_SHORT


def _chunk_max_words() -> int:
    if _short_chunking_enabled():
        return _CHUNK_MAX_WORDS
    return _tp_trace_chunk_max_words() if _is_multi_device_mesh() else _TRACE_CHUNK_MAX_WORDS


def _needs_chunking(text: str, text_max_seq_len: int) -> bool:
    n_words = len(text.split())
    if n_words > _chunking_threshold_words(text_max_seq_len):
        return True
    if _short_chunking_enabled():
        return _estimate_acoustic_frames(n_words) > _CHUNK_MAX_ACOUSTIC_FRAMES
    return False


def _find_degeneracy_cut_frame(shifted_codes_t37: torch.Tensor) -> int | None:
    if shifted_codes_t37.numel() == 0:
        return None
    sem = shifted_codes_t37[:, 0].detach().long().cpu()
    n = int(sem.numel())
    if n < _DEGENERACY_WINDOW_FRAMES:
        return None
    for end in range(_DEGENERACY_WINDOW_FRAMES, n + 1):
        window = sem[end - _DEGENERACY_WINDOW_FRAMES : end]
        if int(window.unique().numel()) <= _DEGENERACY_MIN_UNIQUE:
            return max(0, end - _DEGENERACY_WINDOW_FRAMES)
    return None


def _speech_activity_bounds(wav: torch.Tensor, *, peak_ratio: float = 0.02) -> tuple[int, int]:
    x = wav.detach().float().reshape(-1)
    n = int(x.numel())
    if n == 0:
        return 0, 0
    peak = float(x.abs().max())
    if peak <= 0:
        return 0, n
    idx = np.flatnonzero((x.abs().numpy() >= peak * peak_ratio))
    if idx.size == 0:
        return 0, n
    return int(idx[0]), int(idx[-1]) + 1


def _trim_trailing_hiss_frames(x: torch.Tensor, *, frame_samples: int) -> torch.Tensor:
    """Drop the contiguous run of low-energy trailing frames the model emits after the last
    phoneme (the post-speech / END_AUDIO release). These frames sit just above the 2%-peak
    speech-activity bound, so they survive normal trimming but render as background hiss when a
    chunk is concatenated mid-stream."""
    n = int(x.numel())
    if frame_samples <= 0 or n < frame_samples * 2:
        return x
    nf = n // frame_samples
    fr = x[: nf * frame_samples].reshape(nf, frame_samples).float()
    rms = torch.sqrt((fr**2).mean(dim=1))
    ref = float(torch.quantile(rms, 0.9))
    if ref <= 0.0:
        return x
    thr = ref * _CHUNK_TAIL_RMS_FRACTION
    last = nf - 1
    while last >= 0 and float(rms[last]) < thr:
        last -= 1
    if last < 0:
        return x
    keep = (last + 1) * frame_samples
    return x[:keep] if keep < n else x


def _fade_out(x: torch.Tensor, sample_rate: int, fade_ms: float = _CHUNK_END_FADE_MS) -> torch.Tensor:
    """Equal-power cosine fade over the last *fade_ms* so a chunk tail never clicks at the join."""
    n = int(x.numel())
    fn = min(n, int(sample_rate * fade_ms / 1000.0))
    if fn < 2:
        return x
    ramp = torch.cos(torch.linspace(0.0, float(np.pi) / 2.0, fn, dtype=torch.float32))
    y = x.clone()
    y[-fn:] = y[-fn:] * ramp
    return y


def _prepare_chunk_waveform(
    wav: torch.Tensor,
    sample_rate: int,
    *,
    is_first: bool,
    hit_end_audio: bool,
    n_acoustic_frames: int | None = None,
    downsample_factor: int | None = None,
    shifted_codes_t37: torch.Tensor | None = None,
) -> torch.Tensor:
    x = wav.detach().float().reshape(-1)
    if x.numel() == 0:
        return x
    if n_acoustic_frames is not None and downsample_factor is not None and n_acoustic_frames > 0:
        x = x[: n_acoustic_frames * downsample_factor]
    deg_cut = _find_degeneracy_cut_frame(shifted_codes_t37) if shifted_codes_t37 is not None else None
    if deg_cut is not None and downsample_factor is not None:
        deg_samples = deg_cut * downsample_factor
        if deg_samples < x.numel():
            logger.warning(
                f"[chunked] trimming {int(x.numel()) - deg_samples} samples before semantic collapse "
                f"(frame {deg_cut})"
            )
            x = x[:deg_samples]
    elif not hit_end_audio and n_acoustic_frames is not None and downsample_factor is not None:
        tail_frames = min(16, max(8, n_acoustic_frames // 6))
        tail_samples = tail_frames * downsample_factor
        if tail_samples < x.numel():
            logger.warning(
                f"[chunked] trimming {tail_samples} samples ({tail_frames} frames) — "
                "max_tokens reached without END_AUDIO"
            )
            x = x[: x.numel() - tail_samples]
    start, end = _speech_activity_bounds(x)
    if not is_first:
        start = min(start + int(sample_rate * _CHUNK_LEAD_TRIM_MS / 1000.0), int(x.numel()))
    x = x[start:] if start < int(x.numel()) else x
    # Drop the low-energy hiss tail the model renders after the last phoneme (around END_AUDIO):
    # ~0.3-0.5 s of frames whose RMS sits just above the 2%-peak speech bound, so they survive
    # _speech_activity_bounds and — spliced at every chunk join — read as "noise at the end of
    # each chunk". Then fade the (now speech-terminated) tail so the join never clicks.
    if downsample_factor is not None and downsample_factor > 0:
        x = _trim_trailing_hiss_frames(x, frame_samples=int(downsample_factor))
    else:
        x = x[: max(0, end - start)]
    return _fade_out(x, sample_rate)


def _level_chunks(parts: list[torch.Tensor], *, gain_clamp: float = 3.0) -> list[torch.Tensor]:
    """Scale each chunk toward the median speech RMS so independently-generated chunks share a
    consistent loudness. Cold-started chunks (especially a short trailing fragment) often render
    several dB louder/softer than the body; hard-concatenating them makes that level jump read as
    a burst of "background noise". Gain is clamped to ``[1/gain_clamp, gain_clamp]`` so we never
    amplify a quiet chunk's hiss aggressively."""
    rms = []
    for w in parts:
        if w.numel() == 0:
            rms.append(0.0)
            continue
        rms.append(float(torch.sqrt(torch.mean(w.float() ** 2)).item()))
    active = [r for r in rms if r > 1e-5]
    if len(active) < 2:
        return parts
    target = float(np.median(active))
    leveled: list[torch.Tensor] = []
    for w, r in zip(parts, rms):
        if r <= 1e-5:
            leveled.append(w)
            continue
        g = max(1.0 / gain_clamp, min(gain_clamp, target / r))
        leveled.append(w * g)
    return leveled


def _match_chunk_brightness(
    parts: list[torch.Tensor], sample_rate: int, *, tol: float = 0.10, max_blend: float = 0.5
) -> list[torch.Tensor]:
    """Nudge each chunk's spectral brightness toward the median so independently-generated chunks
    don't step in timbre at the joins (the residual "modulation" left after RMS leveling). A gentle
    one-pole low-pass is blended in proportion to how far a chunk's spectral centroid sits above the
    median, bounded by ``max_blend``. Chunks within ``tol`` of the median — and any chunk darker than
    the median — are left untouched (we never add brightness, which would amplify hiss)."""
    if len(parts) < 2:
        return parts
    try:
        from scipy.signal import lfilter
    except Exception:
        return parts

    def _centroid(w: torch.Tensor) -> float:
        if w.numel() < 8:
            return 0.0
        x = w.detach().float().numpy()
        mag = np.abs(np.fft.rfft(x))
        freq = np.fft.rfftfreq(x.size, 1.0 / sample_rate)
        return float((freq * mag).sum() / (mag.sum() + 1e-9))

    cents = [_centroid(w) for w in parts]
    valid = [c for c in cents if c > 0]
    if len(valid) < 2:
        return parts
    target = float(np.median(valid))
    if target <= 0:
        return parts
    out: list[torch.Tensor] = []
    for w, c in zip(parts, cents):
        if c <= target * (1.0 + tol) or w.numel() < 8:
            out.append(w)
            continue
        blend = min(max_blend, c / target - 1.0)
        a = float(np.exp(-2.0 * np.pi * target / sample_rate))  # one-pole LP at ~median centroid
        x = w.detach().float()
        lp = torch.from_numpy(lfilter([1.0 - a], [1.0, -a], x.numpy()).astype(np.float32))
        out.append((1.0 - blend) * x + blend * lp)
    return out


def _crossfade_concat(parts: list[torch.Tensor], sample_rate: int) -> torch.Tensor:
    if not parts:
        return torch.tensor([], dtype=torch.float32)
    if len(parts) == 1:
        return parts[0]
    fade_n = max(8, int(sample_rate * _CHUNK_CROSSFADE_MS / 1000.0))
    out = parts[0]
    for nxt in parts[1:]:
        if out.numel() == 0:
            out = nxt
            continue
        if nxt.numel() == 0:
            continue
        n = min(fade_n, int(out.numel()) // 2, int(nxt.numel()) // 2)
        if n < 8:
            out = torch.cat([out, nxt])
            continue
        # Equal-power crossfade; keep a short linear ramp at chunk edges so tail words are not
        # fully attenuated (cosine fade_out→0 was dropping last words like "Portuguese, Dutch").
        t = torch.linspace(0, 1, n, dtype=torch.float32)
        fade_out = 1.0 - t
        fade_in = t
        overlap = out[-n:] * fade_out + nxt[:n] * fade_in
        out = torch.cat([out[:-n], overlap, nxt[n:]])
    return out


def _take_word_chunk(words: list[str], max_words: int) -> tuple[list[str], list[str]]:
    """Take up to *max_words* from the front of *words*, preferring comma clause boundaries."""
    if len(words) <= max_words:
        return words, []
    split_at = max_words
    mid = max(3, max_words // 2)
    # Prefer a comma break in the upper half of the window (keeps list tails out of chunk ends).
    for j in range(mid, max_words + 1):
        if words[j - 1].rstrip().endswith(","):
            split_at = j
            break
    else:
        # Otherwise take the earliest comma in the lower half (short clausal prefix).
        for j in range(3, mid):
            if words[j - 1].rstrip().endswith(","):
                split_at = j
                break
    # Comma-dense run (language lists): if we'd end on list items, take the whole run in one chunk.
    head = words[:split_at]
    tail = words[split_at:]
    if tail and head:
        comma_head = sum(1 for w in head if w.rstrip().endswith(","))
        if comma_head >= 3 and len(head) + len(tail) <= max_words + 4:
            combined = words[: min(len(words), max_words + 4)]
            return combined, words[len(combined) :]
    return head, tail


def _split_into_chunks(text: str, max_words: int | None = None) -> list[str]:
    """Split *text* into sentence-aligned chunks of at most *max_words* words each."""
    if max_words is None:
        max_words = _chunk_max_words()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buf: list[str] = []
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        if len(buf) + len(words) <= max_words:
            buf.extend(words)
            continue
        if buf:
            chunks.append(" ".join(buf))
            buf = []
        while words:
            if _short_chunking_enabled():
                head, words = _take_word_chunk(words, max_words)
            else:
                if len(words) <= max_words:
                    head, words = words, []
                else:
                    head, words = words[:max_words], words[max_words:]
            if head:
                chunks.append(" ".join(head))
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
    text_max_seq_len: int = 4096,
) -> None:
    """Generate audio chunk-by-chunk: trim each chunk to speech + crossfade joins (both paths)."""
    short = _short_chunking_enabled()
    max_w = _chunk_max_words()
    chunks = _split_into_chunks(text)
    mode = "short-chunk" if short else "trace"
    logger.info(f"[chunked/{mode}] {len(text.split())} words → {len(chunks)} chunks (max {max_w} words)")
    prepared: list[torch.Tensor] = []
    n_frames_total = 0
    first_frame_s: float | None = None
    t_start = perf_counter()

    for i, chunk in enumerate(chunks):
        n_words = len(chunk.split())
        if short:
            chunk_budget = _chunk_token_budget(n_words)
        else:
            chunk_budget = max(600, n_words * 12)
        chunk_max_tokens = _resolve_max_speech_tokens(
            chunk,
            voice,
            pipe.model_name_or_path,
            chunk_budget,
            text_max_seq_len,
            log_prefix=f"chunked {i + 1}/{len(chunks)}",
        )
        logger.info(f"[chunked] chunk {i + 1}/{len(chunks)}: {n_words} words | max_tokens={chunk_max_tokens}")

        out = pipe.forward_device_resident(
            text=chunk,
            voice=voice,
            max_tokens=chunk_max_tokens,
            seed=seed,
        )
        if first_frame_s is None and out.first_frame_s is not None:
            first_frame_s = out.first_frame_s
        n_frames = int(out.codes_b37t.shape[2])
        n_frames_total += n_frames
        if out.waveform.numel() > 0:
            wav = out.waveform.reshape(-1)
            # Both short- and trace-chunk paths trim each chunk to its speech-active region and
            # crossfade the joins. Hard-concatenating raw chunk waveforms (the old trace path)
            # splices each chunk's trailing breath/hiss and level mismatches mid-stream → audible
            # background noise at the sentence boundaries; trimming + crossfade removes it.
            shifted = out.shifted_codes_t37.cpu() if out.shifted_codes_t37.numel() > 0 else None
            if shifted is not None and shifted.numel() > 0:
                sem = shifted[:, 0]
                logger.info(
                    f"[chunked] chunk {i + 1}/{len(chunks)} codes: frames={n_frames} "
                    f"unique={int(sem.unique().numel())} hit_end={out.hit_end_audio}"
                )
            trimmed = _prepare_chunk_waveform(
                wav,
                sample_rate,
                is_first=(i == 0),
                hit_end_audio=bool(out.hit_end_audio),
                n_acoustic_frames=n_frames,
                downsample_factor=int(pipe._downsample_factor),
                shifted_codes_t37=shifted,
            )
            if trimmed.numel() > 0:
                prepared.append(trimmed)
        if not out.hit_end_audio:
            logger.warning(f"[chunked] chunk {i + 1}: no END_AUDIO at max_tokens={chunk_max_tokens}")

    total_s = perf_counter() - t_start
    prepared = _match_chunk_brightness(prepared, sample_rate)
    prepared = _level_chunks(prepared)
    n_samples = sum(int(w.numel()) for w in prepared)
    combined = _crossfade_concat(prepared, sample_rate) if prepared else torch.tensor([], dtype=torch.float32)
    join_note = f"crossfade={_CHUNK_CROSSFADE_MS:.0f} ms, leveled+timbre-matched"

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

    if combined.numel() == 0:
        logger.error("[chunked] No audio generated.")
        return

    _save_wav(out_path, combined, sample_rate)
    logger.info(f"Saved chunked waveform → {out_path} ({combined.numel()} samples, {len(chunks)} chunks, {join_note})")


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

    Chunking (demo layer only — ``voxtral_tts`` trace replay is unchanged):

      Trace enabled (``VOXTRAL_DECODE_TRACE=1``, default — incl. multi-device 1×4):
        - Single pass for prompts up to ~one sentence (≤ ``_TRACE_CHUNK_MAX_WORDS``)
        - Otherwise sentence-aligned chunks, one single-pass each, hard-concat. A single long
          pass drifts off-text on the TP backbone and emits END_AUDIO early (drops trailing
          sentences); per-sentence passes keep the full prompt at low RTF.

      Non-trace (``VOXTRAL_DECODE_TRACE=0`` / ``--no-decode-trace``):
        - Short 8-word chunks, 100-frame cap, crossfade + edge trim (clarity fix)
    """
    passes = _text_generation_passes(
        text,
        max_tokens,
        text_max_seq_len,
        voice=voice,
        model_name_or_path=pipe.model_name_or_path,
    )
    if len(passes) > 1:
        _run_chunked_text_mode(pipe, text, voice, seed, sample_rate, out_path, text_max_seq_len)
        return

    chunk_text, max_tokens = passes[0]
    assert chunk_text == text

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
    from models.experimental.voxtraltts.utils.mesh import voxtral_to_torch_replicated

    wav = voxtral_to_torch_replicated(wav_tt).float()
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

    # Trace/chunking gated by compute mesh. ``VOXTRAL_DECODE_TRACE`` selects the demo chunking
    # strategy: 0 → short 8-word chunks (crossfade), 1 → single-pass / large-chunk.
    #   • 1×1 BH: short-chunk default (clean audio).
    #   • Multi-device (e.g. 1×4 TP): single-pass trace is the validated path — full prompt,
    #     RTF ≈ 0.7. Short chunking here is ~7× slower (RTF ≈ 4.8) and collapses some chunks to a
    #     few frames (dropped words / "missing prompt"), so it is NOT used unless the user opts in
    #     with --no-decode-trace. We therefore set the default authoritatively (overriding any stale
    #     VOXTRAL_DECODE_TRACE left in the environment) rather than via setdefault().
    from models.experimental.voxtraltts.tests.common import voxtral_requested_compute_mesh_shape

    _trace_explicit = os.environ.get("VOXTRAL_DEMO_TRACE_EXPLICIT") == "1"
    if voxtral_requested_compute_mesh_shape() == (1, 1):
        os.environ.setdefault("VOXTRAL_DECODE_TRACE", "0")
        os.environ.setdefault("VOXTRAL_DECODE_TRACE_2CQ", "0")
    elif not _trace_explicit:
        os.environ["VOXTRAL_DECODE_TRACE"] = "1"
        os.environ["VOXTRAL_DECODE_TRACE_2CQ"] = "1"
    from models.experimental.voxtraltts.demo.decode_trace_2cq import decode_trace_2cq_enabled, decode_trace_enabled

    if voxtral_requested_compute_mesh_shape() != (1, 1) and not decode_trace_enabled():
        logger.warning(
            "[demo] Multi-device mesh with trace disabled → slow short-chunk path "
            "(high RTF, may drop words on long prompts). Remove --no-decode-trace / "
            "unset VOXTRAL_DECODE_TRACE for the single-pass trace path."
        )
    logger.info(
        f"[demo] trace replay={'on' if decode_trace_enabled() else 'off'}, "
        f"2CQ={'on' if decode_trace_2cq_enabled() else 'off'}, "
        f"chunking={'short (non-trace)' if not decode_trace_enabled() else 'trace default'}"
    )

    runtime = _open_device()
    pipe: VoxtralTTSPipeline | None = None
    try:
        logger.info(f"Loading VoxtralTTSPipeline from {args.model.model_name_or_path!r} …")
        t0 = perf_counter()
        pipe = _load_pipeline(runtime.compute_device, args)
        logger.info(f"Pipeline ready in {(perf_counter() - t0) * 1000:.1f} ms")

        # skip warmup when trace disabled
        from models.experimental.voxtraltts.demo.decode_trace_2cq import decode_trace_enabled

        warmup_iters = args.data.warmup_iters if decode_trace_enabled() else 0

        for w in range(warmup_iters + 1):
            is_warmup = w < warmup_iters
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
                        # Warmup: same text/chunking/max_tokens as the measured run (untimed, no WAV).
                        passes = _text_generation_passes(
                            text,
                            args.data.max_speech_tokens,
                            args.tt.text_max_seq_len,
                            voice=voice,
                            model_name_or_path=pipe.model_name_or_path,
                        )
                        logger.info(
                            f"[warmup] {len(text.split())} words → {len(passes)} pass(es); "
                            f"max_tokens={[mt for _, mt in passes]}"
                        )
                        for pass_text, pass_max_tokens in passes:
                            pipe.forward_device_resident(
                                text=pass_text,
                                voice=voice,
                                max_tokens=pass_max_tokens,
                                seed=args.data.seed,
                            )
                        ttnn.synchronize_device(runtime.compute_device)
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
        close_voxtral_runtime_mesh(runtime)


def main(argv: list[str] | None = None) -> None:
    args = _parse_demo_args(argv)
    torch.manual_seed(args.data.seed)
    run_demo(args)


if __name__ == "__main__":
    main()

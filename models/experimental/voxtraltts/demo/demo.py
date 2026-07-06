# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Voxtral TTS demo CLI — text/codes/latents to WAV via ``VoxtralTTSPipeline`` on TTNN."""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from loguru import logger
from scipy.io import wavfile

import ttnn

from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_MODEL, load_voxtral_config
from models.experimental.voxtraltts.utils.common import (
    VOXTRAL_STANDARD_CHAR_TEXT,
    close_voxtral_runtime_mesh,
    open_voxtral_runtime_mesh,
)
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

# Shared standard prompt (same as PCC / perf tests in ``utils/common.py``).
DEMO_DEFAULT_TEXT = VOXTRAL_STANDARD_CHAR_TEXT
DEMO_DEFAULT_VOICE = "cheerful_female"
DEMO_DEFAULT_TEXT_MAX_SEQ_LEN = 65536
DEMO_DEFAULT_MAX_SPEECH_TOKENS = 0
DEMO_DEFAULT_OUTPUT_DIR = "models/experimental/voxtraltts/voxtraltts_demo_output"
# Single-pass word ceiling before sentence chunking (overridable via ``--single-pass-max-words``).
DEMO_DEFAULT_SINGLE_PASS_MAX_WORDS = 180


@dataclass
class ModelArgs:
    model_name_or_path: str = DEFAULT_VOXTRAL_MODEL


@dataclass
class TTArgs:
    text_max_seq_len: int = DEMO_DEFAULT_TEXT_MAX_SEQ_LEN
    text_dtype: str = "bfloat16"
    acoustic_dtype: str = "bfloat16"
    tokenizer_dtype: str = "bfloat16"
    use_paged_kv_cache: bool = True
    paged_block_size: int = 32
    dense_alibi_sdpa: bool = True
    hf_aligned_text: bool = False
    decode_trace: bool = True
    decode_trace_2cq: bool = True


@dataclass
class DataArgs:
    output_dir: str = DEMO_DEFAULT_OUTPUT_DIR
    mode: str = "text"
    # Upper bound on AR acoustic steps. Capped to ``text_max_seq_len − prompt_seq_len``.
    # Use ``0`` to consume the full decode budget (full-context runs).
    max_speech_tokens: int = DEMO_DEFAULT_MAX_SPEECH_TOKENS
    seed: int = 0
    default_voice: str = DEMO_DEFAULT_VOICE
    warmup_iters: int = 1
    inline_texts: list[str] | None = None
    voice: str | None = None
    codes_path: str | None = None
    latent_path: str | None = None
    # ``None`` → ``DEMO_DEFAULT_SINGLE_PASS_MAX_WORDS`` (180).
    single_pass_max_words: int | None = None


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
    p.add_argument(
        "--text",
        type=str,
        nargs="+",
        action="append",
        default=None,
        help="Inline text prompt (text mode). Quotes are optional; repeat --text for multiple prompts. "
        f"Default: shared standard prompt (``VOXTRAL_STANDARD_CHAR_TEXT``).",
    )
    p.add_argument("--output-dir", type=str, default=DEMO_DEFAULT_OUTPUT_DIR)
    p.add_argument("--mode", type=str, choices=("text", "codes", "latents"), default="text")
    p.add_argument("--text-max-seq-len", type=int, default=DEMO_DEFAULT_TEXT_MAX_SEQ_LEN)
    p.add_argument(
        "--max-speech-tokens",
        type=int,
        default=DataArgs.max_speech_tokens,
        help="Max autoregressive acoustic frames (model runs at frame_rate Hz from params.json). "
        "Capped to text_max_seq_len − prompt_seq_len. Use 0 for the full decode budget.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Untimed warmup passes before the measured run (compile + trace capture; default 1). "
        "Set 0 to skip (logged RTF then includes compile/capture). Ignored when trace is disabled.",
    )
    p.add_argument("--default-voice", type=str, default=DEMO_DEFAULT_VOICE)
    p.add_argument(
        "--voice", type=str, default=None, help="Voice for inline --text prompts (overrides --default-voice)."
    )
    p.add_argument(
        "--codes-path",
        "--codes",
        type=str,
        default=None,
        dest="codes_path",
        help="Path to a .pt file with pre-computed [1,37,T] or [T,37] codes (required for --mode codes). "
        "Also accepts dict checkpoints saved by text mode (uses ``codes_b37t``).",
    )
    p.add_argument(
        "--latent-path",
        type=str,
        default=None,
        help="Path to a .pt file with pre-computed latents (required for --mode latents).",
    )
    p.add_argument(
        "--no-paged-kv-cache",
        action="store_false",
        dest="use_paged_kv_cache",
        default=True,
        help="Disable paged KV attention (default: paged on for all sequence lengths).",
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
        help="Disable traced text-decode replay (slower direct forward per AR step).",
    )
    p.add_argument(
        "--no-decode-trace-2cq",
        action="store_true",
        default=False,
        help="Disable second command queue for overlapped decode input staging (default: on).",
    )
    p.add_argument(
        "--dense-alibi-sdpa",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--single-pass-max-words",
        type=int,
        default=None,
        help="Prompts with at most this many words run as one AR pass; longer prompts are split into "
        "sentence-aligned chunks and crossfaded. Default: "
        f"{DEMO_DEFAULT_SINGLE_PASS_MAX_WORDS} words.",
    )
    ns = p.parse_args(argv)
    inline_texts = [" ".join(parts).strip() for parts in ns.text] if ns.text else None
    if ns.mode == "text":
        if not inline_texts:
            inline_texts = [DEMO_DEFAULT_TEXT]
    elif ns.mode == "codes":
        if not ns.codes_path:
            p.error("--codes-path is required when --mode codes")
        if inline_texts:
            p.error("--text is only valid with --mode text")
    elif ns.mode == "latents":
        if not ns.latent_path:
            p.error("--latent-path is required when --mode latents")
        if inline_texts:
            p.error("--text is only valid with --mode text")
    if inline_texts and any(not text for text in inline_texts):
        p.error("--text requires a non-empty prompt")
    decode_trace = not ns.no_decode_trace
    decode_trace_2cq = not ns.no_decode_trace_2cq
    use_dense_alibi = (not ns.native_sdpa) or ns.dense_alibi_sdpa
    return DemoArgs(
        model=ModelArgs(model_name_or_path=ns.model),
        tt=TTArgs(
            text_max_seq_len=ns.text_max_seq_len,
            use_paged_kv_cache=ns.use_paged_kv_cache,
            paged_block_size=ns.paged_block_size,
            dense_alibi_sdpa=use_dense_alibi,
            hf_aligned_text=ns.hf_aligned_text,
            decode_trace=decode_trace,
            decode_trace_2cq=decode_trace_2cq,
        ),
        data=DataArgs(
            output_dir=ns.output_dir,
            mode=ns.mode,
            max_speech_tokens=ns.max_speech_tokens,
            seed=ns.seed,
            default_voice=ns.default_voice,
            warmup_iters=ns.warmup_iters,
            inline_texts=inline_texts,
            voice=ns.voice,
            codes_path=ns.codes_path,
            latent_path=ns.latent_path,
            single_pass_max_words=ns.single_pass_max_words,
        ),
    )


# ---------------------------------------------------------------------------
# B. Device + pipeline init
# ---------------------------------------------------------------------------


def _open_device():
    from models.experimental.voxtraltts.demo.decode_trace_2cq import num_command_queues_for_decode

    params = {
        "trace_region_size": int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000))),
        "num_command_queues": num_command_queues_for_decode(),
    }
    runtime = open_voxtral_runtime_mesh(params)
    return runtime


def _text_kv_cache_bytes_per_device(
    seq_len: int,
    *,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    num_devices: int,
    paged_block_size: int,
    kv_dtype_bytes: int = 2,
) -> int:
    """Per-device KV bytes for paged cache (matches ``Attention.init_kv_cache``)."""
    max_num_blocks = math.ceil(seq_len / paged_block_size)
    n_local_kv_heads = max(1, n_kv_heads // max(1, num_devices))
    per_layer = 2 * max_num_blocks * n_local_kv_heads * paged_block_size * head_dim * kv_dtype_bytes
    return n_layers * per_layer


def _mesh_device_dram_gb(mesh_device: ttnn.Device) -> float:
    dram_view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return (dram_view.total_bytes_per_bank * dram_view.num_banks) / (1024**3)


def _mesh_device_label(mesh_device: ttnn.Device) -> str:
    from models.experimental.voxtraltts.utils.mesh import voxtral_mesh_device_compute_shape

    mesh_env = os.environ.get("MESH_DEVICE", "").strip()
    rows, cols = voxtral_mesh_device_compute_shape()
    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    name = mesh_env or ttnn.get_arch_name()
    return f"{name} ({rows}×{cols}, {num_devices} device{'s' if num_devices != 1 else ''})"


def _estimate_run_peak_seq_len(args: DemoArgs, items: list[dict]) -> int | None:
    """Max ``prompt_seq_len + max_speech_tokens`` across text-mode passes; ``None`` otherwise."""
    if args.data.mode != "text":
        return None
    spmw = _resolve_single_pass_max_words(args.data.single_pass_max_words)
    peak = 0
    for item in items:
        text = item["text"]
        voice = str(item.get("voice", args.data.default_voice))
        for pass_text, pass_max_tokens in _text_generation_passes(
            text,
            args.data.max_speech_tokens,
            args.tt.text_max_seq_len,
            voice=voice,
            model_name_or_path=args.model.model_name_or_path,
            single_pass_max_words=spmw,
        ):
            prompt_len = _speech_prompt_seq_len(pass_text, voice, args.model.model_name_or_path)
            peak = max(peak, prompt_len + pass_max_tokens)
    return peak


def _check_seq_len_memory(
    mesh_device: ttnn.Device,
    args: DemoArgs,
    *,
    peak_seq_len: int | None,
) -> None:
    """Log memory for this run and fail early when the pre-allocated KV budget cannot fit.

    KV cache is pre-allocated at ``text_max_seq_len`` at pipeline init. Logs use the opened
    mesh device DRAM and, for text mode, the peak context length of this demo run.
    """
    cfg = load_voxtral_config(args.model.model_name_or_path)
    num_devices = int(mesh_device.get_num_devices()) if hasattr(mesh_device, "get_num_devices") else 1
    kv_kwargs = dict(
        n_layers=cfg.n_layers,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        num_devices=num_devices,
        paged_block_size=args.tt.paged_block_size,
    )
    budget_bytes = _text_kv_cache_bytes_per_device(args.tt.text_max_seq_len, **kv_kwargs)
    budget_gb = budget_bytes / (1024**3)
    peak_bytes = (
        _text_kv_cache_bytes_per_device(peak_seq_len, **kv_kwargs) if peak_seq_len is not None else budget_bytes
    )
    peak_gb = peak_bytes / (1024**3)

    device_dram_gb = _mesh_device_dram_gb(mesh_device)
    device_label = _mesh_device_label(mesh_device)
    WEIGHTS_GB = 14.5  # text + acoustic + tokenizer weights (approximate)
    RUNTIME_HEADROOM_GB = 4.0
    usable_gb = device_dram_gb * 0.85
    estimated_total_gb = WEIGHTS_GB + budget_gb + RUNTIME_HEADROOM_GB

    if peak_seq_len is not None and peak_seq_len < args.tt.text_max_seq_len:
        logger.info(
            f"[memory] device={device_label} DRAM={device_dram_gb:.1f} GB/device | "
            f"this run peak_seq_len={peak_seq_len} → KV ~{peak_gb:.2f} GB"
        )
        logger.info(
            f"[memory] pre-allocated KV budget: text_max_seq_len={args.tt.text_max_seq_len} → "
            f"{budget_gb:.2f} GB (estimated total footprint ~{estimated_total_gb:.2f} GB)"
        )
    else:
        logger.info(
            f"[memory] text_max_seq_len={args.tt.text_max_seq_len} → KV cache = {budget_gb:.2f} GB "
            f"({device_label}, {device_dram_gb:.1f} GB DRAM/device, "
            f"estimated total footprint ~{estimated_total_gb:.2f} GB)"
        )

    if estimated_total_gb > usable_gb:
        raise MemoryError(
            f"\n{'='*70}\n"
            f"OOM RISK: text_max_seq_len={args.tt.text_max_seq_len} pre-allocates {budget_gb:.2f} GB "
            f"for the text KV cache on each device.\n"
            f"Estimated total footprint is ~{estimated_total_gb:.2f} GB, above the safe {usable_gb:.2f} GB budget\n"
            f"for {device_label} ({device_dram_gb:.1f} GB DRAM/device) after model weights and runtime headroom.\n\n"
            f"This can reach 'Pipeline ready' and then be killed by the OS/runtime without a Python traceback.\n\n"
            f"Fix — reduce text_max_seq_len. Recommended values:\n"
            f"  ≤650  words / ≤4 min  →  text_max_seq_len=4096   "
            f"(KV={_text_kv_cache_bytes_per_device(4096, **kv_kwargs) / 1024**3:.2f} GB)\n"
            f"  ≤1300 words / ≤8 min  →  text_max_seq_len=8192   "
            f"(KV={_text_kv_cache_bytes_per_device(8192, **kv_kwargs) / 1024**3:.2f} GB)\n"
            f"  ≤2600 words / ≤17 min →  text_max_seq_len=16384  "
            f"(KV={_text_kv_cache_bytes_per_device(16384, **kv_kwargs) / 1024**3:.2f} GB)\n"
            f"  ≤5200 words / ≤34 min →  text_max_seq_len=32768  "
            f"(KV={_text_kv_cache_bytes_per_device(32768, **kv_kwargs) / 1024**3:.2f} GB)\n"
            f"{'='*70}"
        )


def _load_pipeline(mesh: ttnn.Device, args: DemoArgs) -> VoxtralTTSPipeline:
    from models.experimental.voxtraltts.utils.mesh import voxtral_mesh_device_compute_shape

    # QB2 (1×N): HF-aligned fp32 SDPA reduces text-hidden drift in the free-run AR loop so acoustic
    # reaches natural END_AUDIO. P150 (1×1) keeps the default perf profile unless --hf-aligned-text.
    multi_device = voxtral_mesh_device_compute_shape() != (1, 1)
    text_optimizations = (
        voxtral_text_hf_aligned_optimizations
        if args.tt.hf_aligned_text or multi_device
        else voxtral_text_default_optimizations
    )
    audio_tokenizer_optimizations = (
        voxtral_audio_tokenizer_dense_mask_sdpa_optimizations()
        if args.tt.dense_alibi_sdpa
        else voxtral_audio_tokenizer_native_sdpa_optimizations()
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


def _load_codes_checkpoint(path: str | Path) -> torch.Tensor:
    """Load ``[1,37,T]`` codes from a raw tensor checkpoint or a text-mode dict (``.codes.pt``)."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        if "codes_b37t" in raw:
            raw = raw["codes_b37t"]
        else:
            raise ValueError(f"Expected a tensor or dict with 'codes_b37t' in {path}, got keys: {sorted(raw.keys())}")
    codes = torch.as_tensor(raw, dtype=torch.long)
    if codes.dim() == 2:
        codes = codes.unsqueeze(0).transpose(1, 2)
    if codes.dim() != 3 or int(codes.shape[1]) != 37:
        raise ValueError(f"Expected codes [1,37,T] or [T,37], got {tuple(codes.shape)}")
    return codes


def _waveform_duration_stats(waveform_f32: torch.Tensor, sample_rate: int, *, rms_thresh: float = 0.02) -> dict:
    """Return file duration, codec-neutral active speech span, and peak amplitude."""
    w = waveform_f32.detach().float().cpu().numpy().reshape(-1)
    n_samples = int(w.shape[0])
    file_s = n_samples / sample_rate if sample_rate > 0 else 0.0
    peak = float(np.abs(w).max()) if n_samples else 0.0
    active_s = 0.0
    if n_samples > 0 and sample_rate > 0:
        win = max(1, int(0.05 * sample_rate))
        rms = [float(np.sqrt(np.mean(w[i : i + win] ** 2))) for i in range(0, max(1, n_samples - win), win)]
        if rms:
            active_idx = [i for i, v in enumerate(rms) if v > rms_thresh]
            if active_idx:
                active_s = (active_idx[-1] - active_idx[0] + 1) * (win / sample_rate)
    return {"file_s": file_s, "active_s": active_s, "peak": peak, "n_samples": n_samples}


def _log_waveform_duration(
    label: str,
    waveform_f32: torch.Tensor,
    sample_rate: int,
    *,
    n_frames: int | None = None,
    downsample_factor: int | None = None,
    model_frame_rate_hz: float | None = None,
) -> None:
    """Log WAV file duration vs codec-implied duration and audible active span."""
    stats = _waveform_duration_stats(waveform_f32, sample_rate)
    logger.info(
        f"[{label}] WAV file duration: {stats['file_s']:.3f} s "
        f"({stats['n_samples']} samples @ {sample_rate} Hz, peak={stats['peak']:.4f})"
    )
    if n_frames is not None and downsample_factor and sample_rate > 0:
        codec_samples = int(n_frames) * int(downsample_factor)
        codec_s = codec_samples / sample_rate
        logger.info(
            f"[{label}] Codec-implied duration: {codec_s:.3f} s "
            f"({n_frames} frames × {downsample_factor} samples/frame"
            + (f" @ {model_frame_rate_hz:.2f} Hz" if model_frame_rate_hz else "")
            + ")"
        )
        if abs(stats["file_s"] - codec_s) > 0.02:
            logger.warning(
                f"[{label}] WAV duration ({stats['file_s']:.3f} s) differs from codec-implied "
                f"({codec_s:.3f} s) by {abs(stats['file_s'] - codec_s):.3f} s"
            )
    if stats["active_s"] > 0:
        logger.info(
            f"[{label}] Active audio span (RMS>{0.02:.2f}): {stats['active_s']:.3f} s "
            f"of {stats['file_s']:.3f} s file"
        )
    if n_frames is not None and downsample_factor and sample_rate > 0:
        codec_s = int(n_frames) * int(downsample_factor) / sample_rate
        if abs(stats["file_s"] - codec_s) <= 0.02:
            logger.info(f"[{label}] Duration check: WAV matches codec-implied length ({codec_s:.3f} s)")


def _finalize_waveform_for_output(waveform_f32: torch.Tensor, sample_rate: int) -> np.ndarray:
    """HPF + peak normalize to 0.95 (matches saved ``.wav`` content)."""
    w = waveform_f32.detach().float().cpu().numpy().reshape(-1)
    w = _apply_output_highpass(w, sample_rate)
    peak = float(np.abs(w).max())
    if peak > 0.0:
        w = w * (0.95 / peak)
    return w


def _save_wav(path: Path, waveform_f32: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = _finalize_waveform_for_output(waveform_f32, sample_rate)
    w = np.clip(w * 32767.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(str(path), sample_rate, w)


def _speech_prompt_seq_len(text: str, voice: str, model_name_or_path: str) -> int:
    from models.experimental.voxtraltts.utils.common import speech_prompt_seq_len

    return speech_prompt_seq_len(text, model_name=model_name_or_path, voice=voice)


def _decode_token_budget(text: str, voice: str, model_name_or_path: str, text_max_seq_len: int) -> tuple[int, int]:
    """Return ``(prompt_seq_len, text_max_seq_len − prompt_seq_len)`` for one forward."""
    prompt_len = _speech_prompt_seq_len(text, voice, model_name_or_path)
    return prompt_len, max(0, text_max_seq_len - prompt_len)


def _model_frame_rate_hz(pipe: "VoxtralTTSPipeline") -> float:
    """Acoustic frame rate from model config (params.json ``frame_rate``, typically 12.5 Hz)."""
    return float(pipe.config.audio_model_args.audio_encoding_args.frame_rate)


def _resolve_max_speech_tokens(
    text: str,
    voice: str,
    model_name_or_path: str,
    max_tokens: int,
    text_max_seq_len: int,
    *,
    log_prefix: str = "tt_generate",
) -> int:
    prompt_len, budget = _decode_token_budget(text, voice, model_name_or_path, text_max_seq_len)
    if budget <= 0:
        raise ValueError(
            f"prompt_seq_len={prompt_len} exceeds text_max_seq_len={text_max_seq_len}; "
            "raise --text-max-seq-len or shorten the prompt."
        )
    requested = budget if max_tokens <= 0 else max_tokens
    resolved = min(requested, budget)
    if max_tokens > budget:
        logger.warning(
            f"[{log_prefix}] max_tokens={max_tokens} exceeds decode budget {budget} "
            f"(prompt_seq_len={prompt_len}); capping to {resolved}."
        )
    logger.info(f"[{log_prefix}] prompt_seq_len={prompt_len} decode_budget={budget} max_speech_tokens={resolved}")
    return resolved


# Sentence chunking (demo layer only): prompts longer than ``single_pass_max_words`` are split,
# generated chunk-by-chunk, trimmed, level-matched, and crossfaded (see ``--single-pass-max-words``).
_CHUNK_CROSSFADE_MS = 40.0
_CHUNK_LEAD_TRIM_MS = 0.0
_CHUNK_END_FADE_MS = 30.0
_CHUNK_TAIL_RMS_FRACTION = 0.15
_DEGENERACY_WINDOW_FRAMES = 20
_DEGENERACY_MIN_UNIQUE = 6


def _is_multi_device_mesh() -> bool:
    from models.experimental.voxtraltts.utils.mesh import voxtral_mesh_device_compute_shape

    return tuple(voxtral_mesh_device_compute_shape()) != (1, 1)


def _default_single_pass_max_words() -> int:
    return DEMO_DEFAULT_SINGLE_PASS_MAX_WORDS


def _resolve_single_pass_max_words(override: int | None) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError(f"--single-pass-max-words must be positive, got {override}")
        return override
    return _default_single_pass_max_words()


def _needs_chunking(text: str, single_pass_max_words: int) -> bool:
    return len(text.split()) > single_pass_max_words


def _split_into_chunks(text: str, max_words: int) -> list[str]:
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
            continue
        if buf:
            chunks.append(" ".join(buf))
            buf = []
        while words:
            if len(words) <= max_words:
                head, words = words, []
            else:
                head, words = words[:max_words], words[max_words:]
            if head:
                chunks.append(" ".join(head))
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if c.strip()]


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
    if downsample_factor is not None and downsample_factor > 0:
        x = _trim_trailing_hiss_frames(x, frame_samples=int(downsample_factor))
    else:
        x = x[: max(0, end - start)]
    return _fade_out(x, sample_rate)


def _level_chunks(parts: list[torch.Tensor], *, gain_clamp: float = 3.0) -> list[torch.Tensor]:
    rms = [float(torch.sqrt(torch.mean(w.float() ** 2)).item()) if w.numel() else 0.0 for w in parts]
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
    """Nudge each chunk's spectral brightness toward the median so joins don't step in timbre."""
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
        a = float(np.exp(-2.0 * np.pi * target / sample_rate))
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
        t = torch.linspace(0, 1, n, dtype=torch.float32)
        overlap = out[-n:] * (1.0 - t) + nxt[:n] * t
        out = torch.cat([out[:-n], overlap, nxt[n:]])
    return out


def _text_generation_passes(
    text: str,
    max_tokens: int,
    text_max_seq_len: int,
    *,
    voice: str,
    model_name_or_path: str,
    single_pass_max_words: int,
) -> list[tuple[str, int]]:
    """Return ``(chunk_text, max_tokens)`` pairs; sentence-chunk when over ``single_pass_max_words``."""
    if not _needs_chunking(text, single_pass_max_words):
        return [
            (
                text,
                _resolve_max_speech_tokens(
                    text, voice, model_name_or_path, max_tokens, text_max_seq_len, log_prefix="tt_generate"
                ),
            )
        ]

    passes: list[tuple[str, int]] = []
    for chunk in _split_into_chunks(text, single_pass_max_words):
        passes.append(
            (
                chunk,
                _resolve_max_speech_tokens(
                    chunk,
                    voice,
                    model_name_or_path,
                    max_tokens,
                    text_max_seq_len,
                    log_prefix="chunked",
                ),
            )
        )
    return passes


def _run_chunked_text_mode(
    pipe: VoxtralTTSPipeline,
    text: str,
    voice: str,
    seed: int,
    sample_rate: int,
    out_path: Path,
    text_max_seq_len: int,
    passes: list[tuple[str, int]],
    single_pass_max_words: int,
) -> None:
    logger.info(
        f"[chunked] {len(text.split())} words → {len(passes)} chunks "
        f"(max {single_pass_max_words} words/chunk, mesh={'1×4' if _is_multi_device_mesh() else '1×1'})"
    )
    prepared: list[torch.Tensor] = []
    n_frames_total = 0
    first_frame_s: float | None = None
    t_start = perf_counter()

    for i, (chunk, chunk_max_tokens) in enumerate(passes):
        n_words = len(chunk.split())
        logger.info(f"[chunked] chunk {i + 1}/{len(passes)}: {n_words} words | max_tokens={chunk_max_tokens}")
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
            shifted = out.shifted_codes_t37.cpu() if out.shifted_codes_t37.numel() > 0 else None
            trimmed = _prepare_chunk_waveform(
                out.waveform.reshape(-1),
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
    combined = _crossfade_concat(prepared, sample_rate) if prepared else torch.tensor([], dtype=torch.float32)
    audio_s = int(combined.numel()) / sample_rate if sample_rate > 0 else None
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
    logger.info(
        f"Saved chunked waveform → {out_path} ({combined.numel()} samples, {len(passes)} chunks, "
        f"crossfade={_CHUNK_CROSSFADE_MS:.0f} ms, leveled+timbre-matched)"
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
    if n_frames is not None and audio_s is not None and n_frames > 0 and audio_s > 0:
        logger.info(
            f"[{label}] Output acoustic rate: {n_frames / audio_s:.2f} frame/s "
            f"({n_frames} frames / {audio_s:.2f}s audio)"
        )
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
    text_max_seq_len: int = 65536,
    single_pass_max_words: int | None = None,
) -> None:
    """Full TT TTS: single AR pass up to ``single_pass_max_words``; sentence chunks + crossfade above."""
    spmw = _resolve_single_pass_max_words(single_pass_max_words)
    passes = _text_generation_passes(
        text,
        max_tokens,
        text_max_seq_len,
        voice=voice,
        model_name_or_path=pipe.model_name_or_path,
        single_pass_max_words=spmw,
    )
    if len(passes) > 1:
        _run_chunked_text_mode(pipe, text, voice, seed, sample_rate, out_path, text_max_seq_len, passes, spmw)
        return

    _, max_tokens = passes[0]

    t0 = perf_counter()
    out = pipe.forward_device_resident(text=text, voice=voice, max_tokens=max_tokens, seed=seed)
    wav = out.waveform
    t1 = perf_counter()
    total_s = t1 - t0

    n_samples = int(wav.numel())
    audio_s = n_samples / sample_rate if sample_rate > 0 else None
    n_frames = int(out.codes_b37t.shape[2])
    prompt_len, decode_budget = _decode_token_budget(text, voice, pipe.model_name_or_path, text_max_seq_len)
    model_fps = _model_frame_rate_hz(pipe)
    _log_perf(
        "tt_generate",
        total_s=total_s,
        audio_s=audio_s,
        n_chars=len(text),
        n_frames=n_frames,
        first_frame_s=out.first_frame_s,
        bitrate_bps=_codec_bitrate_bps(pipe),
    )
    _log_waveform_duration(
        "tt_generate",
        torch.from_numpy(_finalize_waveform_for_output(wav, sample_rate)),
        sample_rate,
        n_frames=n_frames,
        downsample_factor=pipe._downsample_factor,
        model_frame_rate_hz=model_fps,
    )
    logger.info(
        f"[tt_generate] context: prompt_seq_len={prompt_len} peak_seq_len={prompt_len + n_frames} "
        f"text_max_seq_len={text_max_seq_len} model_frame_rate={model_fps:.2f} Hz"
    )
    if out.shifted_codes_t37.numel() > 0:
        semantic = out.shifted_codes_t37[:, 0]
        logger.info(
            "[tt_generate] semantic shifted-code stats: "
            f"min={int(semantic.min().item())}, max={int(semantic.max().item())}, "
            f"unique={int(semantic.unique().numel())}, first10={semantic[:10].tolist()}"
        )
    logger.info(
        f"[tt_generate] generation stop: hit_end_audio={out.hit_end_audio} "
        f"frames={n_frames} peak_seq_len={prompt_len + n_frames}"
    )
    if out.hit_end_audio:
        logger.info(f"[tt_generate] Natural END_AUDIO reached after {n_frames} frames.")
    elif n_frames >= max_tokens:
        logger.warning(
            f"[tt_generate] Reached max_speech_tokens={max_tokens} without END_AUDIO; output was truncated. "
            f"Decode budget for this prompt is {decode_budget} (raise --max-speech-tokens or use 0 for full budget)."
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
    logger.info(f"Saved TT waveform → {out_path}")


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
    _log_waveform_duration(
        "tt_codes_decode",
        wav.squeeze(0).squeeze(0),
        sample_rate,
        n_frames=int(codes_b37t.shape[2]),
        downsample_factor=pipe._downsample_factor,
    )
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

    if args.data.mode == "text":
        voice = args.data.voice or args.data.default_voice
        assert args.data.inline_texts is not None
        items = [{"id": i, "text": t, "voice": voice} for i, t in enumerate(args.data.inline_texts)]
    elif args.data.mode == "codes":
        assert args.data.codes_path is not None
        items = [{"id": 0, "codes_path": args.data.codes_path}]
    else:
        assert args.data.latent_path is not None
        items = [{"id": 0, "latent_path": args.data.latent_path}]
    out_dir = Path(args.data.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from models.experimental.voxtraltts.demo.decode_trace_2cq import (
        configure_decode_trace,
        decode_trace_2cq_enabled,
        decode_trace_enabled,
    )
    from models.experimental.voxtraltts.utils.mesh import voxtral_mesh_device_compute_shape

    configure_decode_trace(decode_trace=args.tt.decode_trace, decode_trace_2cq=args.tt.decode_trace_2cq)

    if voxtral_mesh_device_compute_shape() != (1, 1) and not decode_trace_enabled():
        logger.info(
            "[demo] Multi-device mesh with trace disabled → slower direct forward per AR step. "
            "Use default trace for best RTF."
        )
    logger.info(
        f"[demo] trace replay={'on' if decode_trace_enabled() else 'off'}, "
        f"2CQ={'on' if decode_trace_2cq_enabled() else 'off'}"
    )
    if args.data.mode == "text":
        spmw = _resolve_single_pass_max_words(args.data.single_pass_max_words)
        src = "CLI" if args.data.single_pass_max_words is not None else "default"
        logger.info(f"[demo] single_pass_max_words={spmw} ({src}); chunk when prompt exceeds this")

    runtime = _open_device()
    pipe: VoxtralTTSPipeline | None = None
    try:
        peak_seq_len = _estimate_run_peak_seq_len(args, items)
        _check_seq_len_memory(runtime.compute_device, args, peak_seq_len=peak_seq_len)
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
                    text = item["text"]
                    voice = str(item.get("voice", args.data.default_voice))
                    out_path = out_dir / f"{tag}_item{pid}.wav"

                    if is_warmup:
                        # Warmup: same text/max_tokens as the measured run (untimed, no WAV).
                        spmw = _resolve_single_pass_max_words(args.data.single_pass_max_words)
                        passes = _text_generation_passes(
                            text,
                            args.data.max_speech_tokens,
                            args.tt.text_max_seq_len,
                            voice=voice,
                            model_name_or_path=pipe.model_name_or_path,
                            single_pass_max_words=spmw,
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
                        single_pass_max_words=args.data.single_pass_max_words,
                    )

                elif args.data.mode == "codes":
                    if is_warmup:
                        continue
                    cp = item.get("codes_path")
                    if not cp:
                        raise ValueError("codes mode requires --codes-path.")
                    codes = _load_codes_checkpoint(cp)
                    run_codes_mode(pipe, codes, sample_rate, out_dir / f"{tag}_item{pid}.wav")

                elif args.data.mode == "latents":
                    if is_warmup:
                        continue
                    lp = item.get("latent_path")
                    if not lp:
                        raise ValueError("latents mode requires --latent-path.")
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

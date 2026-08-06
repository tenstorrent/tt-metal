# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for demo / ISL-sweep wall-clock performance reporting."""

from __future__ import annotations

from typing import Any

# Audio codec constants (see README "Model description"): each AR frame is a fixed
# 3200-sample chunk of 24 kHz mono audio → 7.5 frames/s. Rendered audio duration is
# therefore ``ar_tokens / FRAME_RATE_HZ`` seconds.
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 3200
FRAME_RATE_HZ = SAMPLE_RATE / SAMPLES_PER_FRAME  # 7.5


def crop_processor_inputs_to_isl(inputs: dict, isl: int) -> dict:
    """Crop a processor batch to the first ``isl`` tokens (post-tokenization ISL).

    Crops ``input_ids`` / ``attention_mask`` / ``speech_input_mask``. Voice audio
    (``speech_tensors`` / ``speech_masks``) is left intact — acoustic encode still
    produces a full speech-embed bank; scatter uses only the slots that remain in
    the cropped ``speech_input_mask`` (see ``_build_prefill_embeds``).
    """
    if isl <= 0:
        raise ValueError(f"isl must be positive, got {isl}")
    seq_len = int(inputs["input_ids"].shape[-1])
    if isl > seq_len:
        raise ValueError(f"isl={isl} exceeds tokenized length {seq_len}")

    out = dict(inputs)
    for key in ("input_ids", "attention_mask", "speech_input_mask"):
        if key in out and out[key] is not None:
            out[key] = out[key][..., :isl].contiguous()
    return out


def summarize_generate_perf(
    *,
    prefill_len: int,
    ar_tokens: int,
    prefill_wall_s: float,
    decode_wall_s: float,
    generate_wall_s: float,
    steady_decode_s: float = 0.0,
    steady_decode_frames: int = 0,
) -> dict[str, Any]:
    """Build the standard wall-clock perf dict (demo meta + ISL sweep).

    TTFT equals prefill wall: the first AR token is taken from prefill logits, so
    time-to-first-token ends when LM prefill finishes.

    RTF (real-time factor) = processing time / rendered audio duration, where the
    rendered duration is ``ar_tokens / FRAME_RATE_HZ`` seconds. ``rtf`` is the
    end-to-end figure (full ``generate()`` wall over the audio produced); ``rtf_decode``
    isolates the sustained decode rate (``FRAME_RATE_HZ / decode_tok_s``). RTF < 1 means
    faster than real time; ``rtf_x`` = 1/rtf is the "× real time" headline.
    """
    prefill_tok_s = (prefill_len / prefill_wall_s) if prefill_wall_s > 0 else 0.0

    if steady_decode_frames > 0 and steady_decode_s > 0:
        decode_s = steady_decode_s
        decode_tok_s = steady_decode_frames / steady_decode_s
        ms_per_tok = (steady_decode_s * 1e3) / steady_decode_frames
        decode_mode = "steady_trace"
    else:
        decode_s = decode_wall_s
        decode_tok_s = (ar_tokens / decode_wall_s) if decode_wall_s > 0 else 0.0
        ms_per_tok = (decode_wall_s * 1e3 / ar_tokens) if ar_tokens > 0 else 0.0
        decode_mode = "eager_loop"

    audio_s = ar_tokens / FRAME_RATE_HZ
    rtf = (generate_wall_s / audio_s) if audio_s > 0 else 0.0
    rtf_decode = (FRAME_RATE_HZ / decode_tok_s) if decode_tok_s > 0 else 0.0
    rtf_x = (1.0 / rtf) if rtf > 0 else 0.0

    return {
        "prefill_tokens": int(prefill_len),
        "ar_tokens_generated": int(ar_tokens),
        "prefill_s": round(prefill_wall_s, 4),
        "prefill_tok_s": round(prefill_tok_s, 2),
        "ttft_s": round(prefill_wall_s, 4),
        "decode_s": round(decode_s, 4),
        "decode_tok_s": round(decode_tok_s, 2),
        "ms_per_tok_steady": round(ms_per_tok, 3),
        "e2e_s": round(generate_wall_s, 4),
        "audio_s": round(audio_s, 4),
        "rtf": round(rtf, 4),
        "rtf_x": round(rtf_x, 3),
        "rtf_decode": round(rtf_decode, 4),
        "decode_mode": decode_mode,
        "steady_decode_frames": int(steady_decode_frames),
    }


def format_perf_line(metrics: dict[str, Any], *, prefix: str = "") -> str:
    """One-line human-readable summary of ``summarize_generate_perf`` output."""
    p = f"{prefix}" if prefix else ""
    mode = metrics.get("decode_mode", "")
    steady = metrics.get("steady_decode_frames", 0)
    mode_s = f"  mode={mode} steady_fr={steady}" if mode else ""
    return (
        f"{p}prefill={metrics['prefill_s']:.3f}s ({metrics['prefill_tok_s']:.1f} tok/s)  "
        f"TTFT={metrics['ttft_s']:.3f}s  "
        f"decode={metrics['decode_tok_s']:.2f} tok/s ({metrics['ms_per_tok_steady']:.2f} ms/tok)  "
        f"e2e={metrics['e2e_s']:.3f}s  "
        f"audio={metrics.get('audio_s', 0.0):.2f}s  "
        f"RTF={metrics.get('rtf', 0.0):.4f} ({metrics.get('rtf_x', 0.0):.2f}x, decode {metrics.get('rtf_decode', 0.0):.4f})  "
        f"ar_tokens={metrics['ar_tokens_generated']}  "
        f"isl={metrics['prefill_tokens']}{mode_s}"
    )


def default_isl_sweep(max_tokens: int | None = None) -> list[int]:
    """Powers of two from 32 … 16384, then ``max_tokens`` if it is larger / not already listed.

    When ``max_tokens`` is set (tokenized prompt length), checkpoints above that length
    are dropped and the full length is appended last if missing.
    """
    isls: list[int] = []
    n = 32
    while n <= 16384:
        if max_tokens is not None and n > max_tokens:
            break
        isls.append(n)
        n *= 2
    if max_tokens is not None and max_tokens >= 32 and (not isls or isls[-1] != max_tokens):
        isls.append(int(max_tokens))
    return isls

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ICL voice-clone helpers: keep reference codec length below text length so
`text_lens > codec_lens` and per-step text conditioning works.

codec_embed length is 1 (codec_bos) + num_reference_codec_frames.
text_embed length is len(ref_text_ids) + len(target_text_ids) + 1 (tts_eos).
We require text_lens > codec_lens => num_ref_frames <= text_lens - 2.

For long outputs, we also cap reference length so enough *trailing* text embeddings
remain for decode: trailing_len = text_lens - (1 + num_ref_frames).
"""

from __future__ import annotations

import torch

# Minimum projected text tokens to add per decode step after the ICL prefix (when possible).
# Balance: higher => more trailing text during decode but shorter reference (weaker timbre).
# 32 was too aggressive for ~4s refs; 16 roughly doubles the old ~6-token tail when ref was full length.
_DEFAULT_MIN_TRAILING_TEXT_TOKENS = 16


def _reference_frame_cap_and_text_lens(
    tokenizer,
    ref_text: str,
    target_text: str,
    min_trailing_text_tokens: int,
) -> tuple[int, int]:
    ref_ids = tokenizer.encode(ref_text, add_special_tokens=False)
    tgt_ids = tokenizer.encode(target_text, add_special_tokens=False)
    text_lens = len(ref_ids) + len(tgt_ids) + 1

    upper_strict = text_lens - 2
    upper_trailing = text_lens - 1 - min_trailing_text_tokens
    if upper_trailing < 1:
        cap = upper_strict
    else:
        cap = min(upper_strict, upper_trailing)

    return max(1, cap), text_lens


def max_reference_codec_frames(
    tokenizer,
    ref_text: str,
    target_text: str,
    min_trailing_text_tokens: int = _DEFAULT_MIN_TRAILING_TEXT_TOKENS,
) -> int:
    """
    Maximum reference codec frame count so:
    - text_lens > codec_lens (strict ICL), and when possible
    - trailing text tokens after the codec-aligned prefix >= min_trailing_text_tokens.

    text_lens = len(ref_ids) + len(target_ids) + 1  (includes trailing tts_eos embed)
    codec_lens = 1 + num_ref_frames
    Trailing = text_lens - codec_lens = text_lens - 1 - num_ref_frames
    """
    return _reference_frame_cap_and_text_lens(tokenizer, ref_text, target_text, min_trailing_text_tokens)[0]


def trim_reference_for_icl_conditioning(
    ref_codes: torch.Tensor,
    audio_data: torch.Tensor,
    tokenizer,
    ref_text: str,
    target_text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    If reference is too long for the tokenized text, keep only the first N codec
    frames and proportionally trim the waveform (same alignment as the encoder).

    Args:
        ref_codes: [T, 16]
        audio_data: [num_samples] at 24 kHz mono float32
        tokenizer: HF tokenizer for Qwen3-TTS
        ref_text / target_text: same strings used for ICL

    Returns:
        (trimmed ref_codes, trimmed audio_data)
    """
    max_frames, text_lens = _reference_frame_cap_and_text_lens(
        tokenizer, ref_text, target_text, _DEFAULT_MIN_TRAILING_TEXT_TOKENS
    )
    n = int(ref_codes.shape[0])
    if n <= max_frames:
        return ref_codes, audio_data

    orig_audio_len = int(audio_data.shape[0])
    ref_codes = ref_codes[:max_frames].contiguous()
    n_samples = max(1, int(orig_audio_len * max_frames / n))
    n_samples = min(n_samples, orig_audio_len)
    audio_data = audio_data[:n_samples].contiguous()

    trailing = text_lens - (1 + max_frames)
    print(
        f"\n  Reference shortened for ICL: "
        f"{n} -> {max_frames} codec frames (~{n_samples / 24000:.2f}s audio), "
        f"~{trailing} trailing text tokens for decode"
    )

    return ref_codes, audio_data

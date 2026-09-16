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
    """No-op pass-through: match HF's `Qwen3TTSForConditionalGeneration.generate_icl_prompt`,
    which never trims `ref_code`. The previous heuristic ("ensure >=16 trailing text
    tokens") was our own invention and produces a leading reference echo at the
    start of generated audio — verified by running QwenLM/Qwen3-TTS's official
    `generate_voice_clone` on the same prompt + ref_audio (clean) vs ours (bleeds).
    HF's `generate_icl_prompt` simply pads text with `tts_pad_embed` when
    text_lens <= codec_lens, and otherwise emits trailing = text_lens - codec_lens
    even when that's small.
    """
    return ref_codes, audio_data

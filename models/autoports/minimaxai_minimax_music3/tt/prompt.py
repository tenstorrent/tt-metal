# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The caption cleaning, lyrics normalization and prompt template below are transcribed verbatim from
# diffusers ``src/diffusers/modular_pipelines/minimax_music3/encoders.py`` (``_clean_caption``,
# ``_normalize_lyrics``, ``MiniMaxMusic3TokenizeStep``), Apache-2.0, Copyright 2026 The MiniMax Team
# and The HuggingFace Team. They are part of the checkpoint contract: even whitespace-level changes
# to the assembled prompt change the generated audio, so nothing here may be "improved".
"""Prompt assembly of MiniMax-Music3: caption + lyrics -> conditional / unconditional token ids."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch

from models.autoports.minimaxai_minimax_music3.tt.constants import AUDIO_CFG_TOKEN_ID, MAX_PROMPT_TOKENS

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
CAPTION_START, CAPTION_END = "<|caption_start|>", "<|caption_end|>"
LYRICS_START, LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
AUDIO_START = "<|audio_start|>"

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def clean_caption(caption: str) -> str:
    """diffusers ``_clean_caption``: rewrite ``<|k v|>`` tags as ``k is v`` and strip the accepted markdown forms."""

    def _rewrite_special_tag(match: re.Match) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    """diffusers ``_normalize_lyrics``: keep only leading structure tags on a tag line, lower-case tags, prepend ``[start]``."""
    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def assemble_prompt(prompt: str, lyrics: str) -> str:
    """The exact string ``MiniMaxMusic3TokenizeStep`` tokenizes."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"`prompt` (the music description) must be a non-empty string, got {prompt!r}")
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise ValueError(f"`lyrics` must be a non-empty string, got {lyrics!r}")
    return (
        f"{IM_START}{CAPTION_START}{clean_caption(prompt)}{CAPTION_END}"
        f"{LYRICS_START}{normalize_lyrics(lyrics)}{LYRICS_END}{IM_END}{AUDIO_START}"
    )


class PromptEncoder:
    """``MiniMaxMusic3TokenizeStep``: ``(prompt, lyrics) -> text_ids [2, L]`` with the CFG row."""

    def __init__(self, tokenizer_dir: Optional[str] = None):
        from transformers import Qwen2Tokenizer

        if tokenizer_dir is None:
            from models.autoports.minimaxai_minimax_music3.reference.hf_llm import weights_dir

            tokenizer_dir = weights_dir() / "tokenizer"
        self.tokenizer_dir = Path(tokenizer_dir)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(str(self.tokenizer_dir))

    def encode(self, prompt: str, lyrics: str) -> torch.LongTensor:
        text = assemble_prompt(prompt, lyrics)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > MAX_PROMPT_TOKENS:
            raise ValueError(
                f"The assembled prompt has {input_ids.shape[1]} tokens; the maximum is {MAX_PROMPT_TOKENS}"
            )
        unconditional_ids = input_ids.clone()
        # Every token except the first (<|im_start|>) and the two trailing structure tokens becomes the CFG token.
        unconditional_ids[:, 1:-2] = AUDIO_CFG_TOKEN_ID
        return torch.cat((input_ids, unconditional_ids), dim=0)

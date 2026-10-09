"""Prompt contract: caption cleaning, lyrics normalization, the special-token template and the CFG (unconditional)
row. Exact port of encoders.py (MiniMaxMusic3TextEncoderStep); tokenization via `tokenizers` (tokenizer/tokenizer.json)
which stage 02 asserts equals transformers' Qwen2Tokenizer on the same strings."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence

import torch

from models.autoports.minimaxai_minimax_music3.config import (
    AUDIO_CFG_TOKEN_ID,
    AUDIO_START,
    CAPTION_END,
    CAPTION_START,
    IM_END,
    IM_START,
    LYRICS_END,
    LYRICS_START,
    MAX_PROMPT_TOKENS,
)

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def clean_caption(caption: str) -> str:
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
    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda m: f"[{m.group(1).lower()}]", text)
    return f"[start]\n{text}"


def prompt_text(caption: str, lyrics: str) -> str:
    return (
        f"{IM_START}{CAPTION_START}{clean_caption(caption)}{CAPTION_END}"
        f"{LYRICS_START}{normalize_lyrics(lyrics)}{LYRICS_END}{IM_END}{AUDIO_START}"
    )


class Music3Tokenizer:
    """Thin wrapper over the HF `tokenizers` fast tokenizer in tokenizer/tokenizer.json (no transformers needed)."""

    def __init__(self, snapshot):
        from tokenizers import Tokenizer

        self.tk = Tokenizer.from_file(str(Path(snapshot) / "tokenizer" / "tokenizer.json"))
        for tok, expect in ((AUDIO_START, 151669), ("<|audio_end|>", 151670)):
            got = self.tk.token_to_id(tok)
            assert got == expect, (tok, got, expect)

    def encode(self, text: str) -> List[int]:
        return self.tk.encode(text, add_special_tokens=True).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=False)


def build_text_ids(tok: Music3Tokenizer, caption: str, lyrics: str) -> torch.Tensor:
    """[2, S] int64: row 0 conditional prompt, row 1 its classifier-free counterpart (every token except the first
    and the two trailing structure tokens replaced by the audio-CFG token)."""
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("`caption` (the music description) must be a non-empty string")
    if not isinstance(lyrics, str) or not lyrics.strip():
        raise ValueError("`lyrics` must be a non-empty string")
    ids = torch.tensor([tok.encode(prompt_text(caption, lyrics))], dtype=torch.int64)
    if ids.shape[1] > MAX_PROMPT_TOKENS:
        raise ValueError(f"The assembled prompt has {ids.shape[1]} tokens; the maximum is {MAX_PROMPT_TOKENS}")
    uncond = ids.clone()
    uncond[:, 1:-2] = AUDIO_CFG_TOKEN_ID
    return torch.cat((ids, uncond), dim=0)

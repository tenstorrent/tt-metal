"""Prompt construction: text (+ optional reference codes) -> the (1 + num_codebooks, T) frame matrix the
slow tower consumes. Mirrors fish-speech's Conversation/ContentSequence for the TTS case exactly (see
tests/test_prompt.py, which asserts equality against the upstream implementation):

  <|im_start|>system\n{system parts}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<|voice|>

System parts without a reference: "convert the provided text to speech".
With reference(s): "convert the provided text to speech reference to the following:\n\nText:\n",
"<|speaker:0|>{ref text}" (joined by \n), "\n\nSpeech:\n", then the reference VQ codes as
<|semantic:k|> tokens with all codebook rows filled. Each TextPart is tokenized separately with
add_special_tokens=False and the results concatenated, as upstream does.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from models.autoports.fishaudio_s2_pro.config import (
    IM_END_ID,
    IM_END_TOKEN,
    IM_START_TOKEN,
    SEMANTIC_BEGIN_ID,
    SEMANTIC_END_ID,
    VOICE_TOKEN,
)


class S2Tokenizer:
    """Thin wrapper over the HF `tokenizers` fast tokenizer in tokenizer.json (no transformers needed)."""

    def __init__(self, snapshot):
        from tokenizers import Tokenizer

        self.tk = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
        self.semantic_begin_id = self.tk.token_to_id("<|semantic:0|>")
        self.semantic_end_id = self.tk.token_to_id("<|semantic:4095|>")
        self.im_end_id = self.tk.token_to_id(IM_END_TOKEN)
        assert (
            self.semantic_begin_id == SEMANTIC_BEGIN_ID
            and self.semantic_end_id == SEMANTIC_END_ID
            and self.im_end_id == IM_END_ID
        ), (self.semantic_begin_id, self.semantic_end_id, self.im_end_id)

    def encode(self, text: str) -> List[int]:
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=False)


def _tag_speakers(texts: Sequence[str]) -> List[str]:
    out = []
    for i, t in enumerate(texts):
        out.append(t if re.search(r"<\|speaker:\d+\|>", t) else f"<|speaker:{i}|>{t}")
    return out


def build_prompt(
    tok: S2Tokenizer,
    text: str,
    ref_codes: Optional[Sequence[torch.Tensor]] = None,
    ref_texts: Optional[Sequence[str]] = None,
    num_codebooks: int = 10,
) -> torch.Tensor:
    """Returns int64 (1 + num_codebooks, T). ref_codes: list of (num_codebooks, T_i) tensors."""
    parts: List[tuple] = []  # ("text", str) | ("vq", codes)

    def message(role: str, body: List[tuple], modality: Optional[str] = None, im_end: bool = True):
        parts.append(("text", f"{IM_START_TOKEN}{role}\n{VOICE_TOKEN if modality == 'voice' else ''}"))
        parts.extend(body)
        if im_end:
            parts.append(("text", IM_END_TOKEN + "\n"))

    if ref_codes:
        assert ref_texts and len(ref_texts) == len(ref_codes)
        sys_body = [
            ("text", "convert the provided text to speech reference to the following:\n\nText:\n"),
            ("text", "\n".join(_tag_speakers(ref_texts))),
            ("text", "\n\nSpeech:\n"),
            ("vq", torch.cat([c.to(torch.int64) for c in ref_codes], dim=1)),
        ]
    else:
        sys_body = [("text", "convert the provided text to speech")]
    message("system", sys_body)
    message("user", [("text", text)])
    message("assistant", [], modality="voice", im_end=False)

    ids: List[torch.Tensor] = []
    vq_rows: List[torch.Tensor] = []
    vq_mask: List[torch.Tensor] = []
    for kind, payload in parts:
        if kind == "text":
            t = torch.tensor(tok.encode(payload), dtype=torch.int64)
            ids.append(t)
            vq_mask.append(torch.zeros(len(t), dtype=torch.bool))
        else:
            codes = payload
            t = codes[0] + tok.semantic_begin_id
            ids.append(t)
            vq_mask.append(torch.ones(len(t), dtype=torch.bool))
            vq_rows.append(codes)
    tokens = torch.cat(ids)
    values = torch.zeros((num_codebooks + 1, len(tokens)), dtype=torch.int64)
    values[0] = tokens
    if vq_rows:
        values[1:, torch.cat(vq_mask)] = torch.cat(vq_rows, dim=1)
    return values


def split_text_by_speaker(text: str) -> List[str]:
    """Upstream: chunking only applies when the text carries <|speaker:N|> tags."""
    pieces = re.split(r"(<\|speaker:\d+\|>)", text)
    turns, cur = [], ""
    for p in pieces:
        if re.fullmatch(r"<\|speaker:\d+\|>", p):
            if cur.strip():
                turns.append(cur)
            cur = p
        else:
            cur += p
    if cur.strip() and re.match(r"<\|speaker:\d+\|>", cur):
        turns.append(cur)
    return turns if len(turns) > 0 and re.search(r"<\|speaker:\d+\|>", text) else []

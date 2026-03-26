# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizer for N150 PI0 demo.

Attempts to load the official Gemma tokenizer first (best accuracy),
falls back to word-based tokenizer if Gemma is unavailable.
"""

import torch
from typing import Tuple


class DemoTokenizer:
    """Auto-selecting tokenizer: Gemma first, word-based fallback."""

    def __init__(self):
        self._gemma = None
        self.name = "SimpleTokenizer"
        try:
            from transformers import AutoTokenizer
            self._gemma = AutoTokenizer.from_pretrained("google/gemma-2b")
            self.name = "Gemma (SentencePiece)"
        except Exception:
            pass

    def encode(self, text: str, max_length: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._gemma is not None:
            enc = self._gemma(text, max_length=max_length, padding="max_length",
                              truncation=True, return_tensors="pt")
            return enc["input_ids"], enc["attention_mask"].bool()

        VOCAB = {
            "<pad>": 0, "<bos>": 2, "<eos>": 3,
            "pick": 100, "place": 101, "grasp": 102, "push": 106,
            "lift": 108, "reach": 105, "move": 104,
            "cube": 200, "block": 201, "object": 202,
            "up": 300, "down": 301, "left": 302, "right": 303, "forward": 304,
            "the": 400, "a": 401, "to": 403, "and": 405, "red": 500,
        }
        tokens = [VOCAB["<bos>"]]
        for w in text.lower().split():
            tokens.append(VOCAB.get(w, hash(w) % 255000 + 1000))
        tokens.append(VOCAB["<eos>"])
        mask = [True] * len(tokens)
        pad = max_length - len(tokens)
        if pad > 0:
            tokens += [0] * pad
            mask += [False] * pad
        else:
            tokens, mask = tokens[:max_length], mask[:max_length]
        return (torch.tensor([tokens], dtype=torch.long),
                torch.tensor([mask], dtype=torch.bool))

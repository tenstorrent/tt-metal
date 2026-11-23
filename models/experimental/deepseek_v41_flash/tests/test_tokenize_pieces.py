# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The server's piecewise tokenization (``_tokenize``) must give exactly the ids of one tokenizer call.

The pieces are shrunk to a few dozen characters, so every text below is cut hundreds of times, next to every kind
of pre-token the cut could disturb: digit runs (grouped in threes from their start), CJK runs, whitespace runs,
newlines, punctuation, contractions and the chat template's special tokens.
"""

from __future__ import annotations

import random

import pytest

import models.experimental.deepseek_v4_flash.demo.server as server
from models.experimental.deepseek_v41_flash.demo import perf_client
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR
from models.experimental.deepseek_v4_flash.tt.weight_loader import resolve_snapshot_dir


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


pytestmark = pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")


def _texts() -> dict[str, str]:
    rng = random.Random(0)
    words = ["alpha", "Beta", "it's", "don't", "x", "I", "über", "naïve", "COVID-19", "e.g.", "U.S.A", "foo_bar"]
    odd = [" ", "  ", "\t", "\n", "\n\n", " \n ", "1234567", "3.14159", "中文字符", "かなカナ", "...", "!?", "$100", "\r\n"]
    mixed = "".join(rng.choice(words) + rng.choice([" ", " ", " ", *odd]) for _ in range(6000))
    code = "\n".join(
        f"    def f{i}(self, x{i}: int) -> int:  # comment {i}\n        return x{i} * {i * 7919} + self.y_{i}"
        for i in range(800)
    )
    return {
        "perf prompt": perf_client._make_prompt(20000, rng),
        "mixed": mixed,
        "code": code,
        "chat": "".join(f"<｜User｜>question {i} about things<｜Assistant｜></think>answer {i} is here" for i in range(2000)),
    }


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_snapshot())


@pytest.mark.parametrize("piece_chars", [37, 1000])
@pytest.mark.parametrize("name", list(_texts()))
def test_pieces_tokenize_like_the_whole(tokenizer, monkeypatch, name, piece_chars):
    text = _texts()[name]
    monkeypatch.setattr(server, "_TOKENIZE_PIECE_CHARS", piece_chars)
    want = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    assert server._tokenize(tokenizer, text) == want
    # It was really cut, not passed through whole.
    assert len(server._TOKENIZE_CUT.findall(text)) > 3


def test_short_text_is_one_call(tokenizer):
    assert server._tokenize(tokenizer, "hello world") == list(tokenizer("hello world", add_special_tokens=False)["input_ids"])

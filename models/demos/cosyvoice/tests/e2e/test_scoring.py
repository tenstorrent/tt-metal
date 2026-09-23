# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The WER/CER harness's text normalisation (`scripts/eval_wer_sim.py`), on the host."""
from __future__ import annotations

import importlib.util
import os

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "eval_wer_sim.py")


def _normalize():
    spec = importlib.util.spec_from_file_location("eval_wer_sim", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize


def test_english_normalises_to_words():
    """English is scored in words. The punctuation class once held `_-–`, a range from U+005F
    to U+2013 that takes in a-z, so every English reference normalised to no words at all and
    every English WER came out 0.00 whatever the audio said."""
    normalize = _normalize()
    assert normalize("The quick brown fox, jumps!", "en") == ["the", "quick", "brown", "fox", "jumps"]
    assert normalize("well-known — fact", "en") == ["well", "known", "fact"]


def test_cjk_normalises_to_characters():
    """CJK is scored in characters, punctuation and spaces removed. Kana and Hangul only in the
    non-Chinese cases: the Traditional-to-Simplified fold applies to every CER language."""
    normalize = _normalize()
    assert normalize("收到好友，从远方寄来。", "zh") == list("收到好友从远方寄来")
    assert normalize("ありがとう、ございます。", "ja") == list("ありがとうございます")
    assert normalize("안녕하세요, 여러분.", "ko") == list("안녕하세요여러분")

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The WER/CER harness's text normalisation (`scripts/eval_wer_sim.py`), on the host."""
from __future__ import annotations

from models.experimental.cosyvoice.scripts.eval_wer_sim import normalize


def test_english_normalises_to_words():
    """English is scored in words. A bare hyphen between `_` and `–` in the punctuation class
    is a range from U+005F to U+2013 that takes in a-z, which normalises every English
    reference to no words and every English WER to 0.00 whatever the audio says."""
    assert normalize("The quick brown fox, jumps!", "en") == ["the", "quick", "brown", "fox", "jumps"]
    assert normalize("well-known — fact", "en") == ["well", "known", "fact"]


def test_cjk_normalises_to_characters():
    """CJK is scored in characters, punctuation and spaces removed. Kana and Hangul only in the
    non-Chinese cases: the Traditional-to-Simplified fold applies to every CER language."""
    assert normalize("收到好友，从远方寄来。", "zh") == list("收到好友从远方寄来")
    assert normalize("ありがとう、ございます。", "ja") == list("ありがとうございます")
    assert normalize("안녕하세요, 여러분.", "ko") == list("안녕하세요여러분")

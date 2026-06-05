# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tokenizer-free task-quality metrics (no device, no weights).

These pin the metric behavior so the accuracy gates built on them (``test_task_accuracy.py``) are
trustworthy: identical text scores perfectly, unrelated text scores low, and ``script_fraction``
correctly distinguishes Devanagari from Latin — the wrong-language detector.
"""

from __future__ import annotations

import pytest

from models.experimental.seamless_m4t_v2_large.tests.accuracy.metrics import (
    corpus_cer,
    corpus_char_bleu,
    corpus_chrf,
    corpus_wer,
    script_fraction,
)

_HIN = "माया एक छोटे से शहर में रहती थी"
_HIN_NEAR = "माया एक छोटी सी शहर में रहती थी"  # one-word variant
_ENG = "Maya lived in a small town"


def test_identical_text_scores_perfect():
    assert corpus_chrf([_HIN], [_HIN]) == pytest.approx(100.0, abs=1e-6)
    assert corpus_char_bleu([_HIN], [_HIN]) == pytest.approx(100.0, abs=1e-6)
    assert corpus_wer([_ENG], [_ENG]) == 0.0
    assert corpus_cer([_HIN], [_HIN]) == 0.0


def test_unrelated_text_scores_low():
    # Different scripts share no character n-grams -> chrF ~ 0.
    assert corpus_chrf([_ENG], [_HIN]) < 5.0
    assert corpus_char_bleu([_ENG], [_HIN]) < 5.0


def test_near_match_scores_high_but_not_perfect():
    chrf = corpus_chrf([_HIN_NEAR], [_HIN])
    assert 60.0 < chrf < 100.0
    assert 0.0 < corpus_wer([_HIN_NEAR], [_HIN]) <= 1.0


def test_error_rates_bounds():
    # Completely wrong hypothesis -> high error; empty hyp -> all deletions (rate ~1).
    assert corpus_wer(["totally different words here now"], [_ENG]) > 0.5
    assert corpus_cer([""], [_HIN]) == pytest.approx(1.0, abs=1e-6)


def test_script_fraction_distinguishes_language():
    # The core wrong-language detector.
    assert script_fraction(_HIN, "deva") > 0.95
    assert script_fraction(_HIN, "latin") < 0.05
    assert script_fraction(_ENG, "latin") > 0.95
    assert script_fraction(_ENG, "deva") < 0.05
    # Mixed / numeric / empty.
    assert script_fraction("", "deva") == 0.0
    assert script_fraction("123 ... !!!", "deva") == 0.0


def test_score_corpus_shape():
    out = score_corpus_safe([_HIN], [_HIN])
    assert set(out) == {"bleu", "chrf", "backend"}
    assert out["chrf"] == pytest.approx(100.0, abs=1e-6)


def score_corpus_safe(hyps, refs):
    from models.experimental.seamless_m4t_v2_large.tests.accuracy.metrics import score_corpus

    return score_corpus(hyps, refs)

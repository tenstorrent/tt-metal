# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The WER scorer itself: the metric, the normaliser and language detection.

Host only -- no device, no checkpoint. A broken scorer reports a broken model, so the normaliser is
checked against non-Latin scripts and language detection is checked to read text rather than a
voice name.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_wer.py
"""

import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORER = os.path.join(HERE, "scripts", "score_quality_set_scipy.py")


@pytest.fixture(scope="module")
def sc():
    """The scoring script, imported by path: it is a script, and it is the one the report runs."""
    spec = importlib.util.spec_from_file_location("_scorer", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wer_returns_errors_and_reference_length(sc):
    """wer() returns a count and a denominator, not a rate, so folds can sum both."""
    errs, n = sc.wer("the quick brown fox", "the quick brown fox")
    assert (errs, n) == (0, 4)


def test_wer_counts_substitution_insertion_deletion(sc):
    assert sc.wer("a b c", "a x c")[0] == 1, "substitution"
    assert sc.wer("a b c", "a b x c")[0] == 1, "insertion"
    assert sc.wer("a b c", "a c")[0] == 1, "deletion"
    assert sc.wer("a b c", "x y z")[0] == 3, "all three substituted"


def test_wer_of_empty_hypothesis_is_every_word(sc):
    errs, n = sc.wer("one two three", "")
    assert (errs, n) == (3, 3)


def test_norm_strips_punctuation_and_collapses_whitespace(sc):
    assert sc.norm("  Hello,   WORLD!!  ") == "hello world"
    assert sc.norm("it's fine") == "it's fine" or sc.norm("it's fine") == "it s fine"


def test_norm_preserves_non_latin_scripts(sc):
    """The normaliser must preserve non-Latin scripts; erasing them scores perfect audio as 100%."""
    hindi = "नमस्ते दुनिया"
    arabic = "مرحبا بالعالم"
    for label, text in (("hindi", hindi), ("arabic", arabic)):
        out = sc.norm(text)
        assert out.strip(), f"{label} normalised to nothing: {out!r}"
        assert len(out.split()) == 2, f"{label} lost a word: {out!r}"
    # and identical text must score zero, not 100%
    assert sc.wer(hindi, hindi) == (0, 2)
    assert sc.wer(arabic, arabic) == (0, 2)


def test_detect_lang_identifies_non_latin_scripts(sc):
    assert sc.detect_lang("नमस्ते दुनिया") == "hindi"
    assert sc.detect_lang("مرحبا بالعالم") == "arabic"


def test_detect_lang_reads_text_not_the_voice_name(sc):
    """detect_lang reads text only, so a voice name cannot force the wrong language."""
    assert sc.detect_lang("Hello.") != "arabic"
    assert sc.detect_lang("ar_male says hello") != "arabic"


def test_detect_lang_uses_word_markers_for_latin_languages(sc):
    assert sc.detect_lang("bonjour tout le monde") == "french"
    assert sc.detect_lang("olá coração") == "portuguese"


def test_detect_lang_returns_none_for_unmarked_english(sc):
    """None means let the recogniser decide, which is the safe default."""
    assert sc.detect_lang("the quick brown fox") is None

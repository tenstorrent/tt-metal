# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The WER scorer itself. Nothing tested it before, which is the gap that matters here: a broken
scorer reports a broken model, and STATUS trap 14 records three harness bugs that each faked a bad
TTS result on perfect audio.

Host only -- no device, no checkpoint. The three trap regressions are the reason this file exists:

  - a naive `[^a-z0-9' ]` normaliser ERASES Hindi and Arabic and scores them 100% WER on perfect
    audio. `norm()` is category-based instead, and `test_norm_preserves_non_latin_scripts` pins it.
  - a voice-name prefix is not a language. Fixture case 4 is English "Hello." spoken by `ar_male`,
    and forcing Arabic decoding made Whisper hallucinate a filler and report 100% WER on one word.
    `detect_lang()` reads TEXT only, and `test_detect_lang_reads_text_not_the_voice_name` pins it.
  - Whisper's encoder is a fixed 30 s window, so a plain `generate` on a 37 s clip transcribes the
    first 30 s and charges the rest as deletions (~20% phantom WER). Chunking lives in
    `transcribe_one`, which needs the model; covered by the corpus run, not here.

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
    """The scoring script, imported by path -- it is a script, not a package module, and it is the
    thing the quality report actually runs. Importing the real one is the point: a copy of `wer()`
    in the test would pass while the script's own diverged."""
    spec = importlib.util.spec_from_file_location("_scorer", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wer_returns_errors_and_reference_length(sc):
    """`wer()` returns a COUNT and a denominator, not a rate. Folds across utterances sum both --
    averaging per-utterance rates would weight a 3-word prompt like a 300-word one."""
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
    """TRAP 14. A `[^a-z0-9' ]` normaliser erases these entirely, and a scorer comparing two empty
    strings reports 0 errors of 0 words -- or, folded against a non-empty reference, 100% WER on
    perfect audio."""
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
    """TRAP 14. Fixture case 4 is English "Hello." spoken by `ar_male`. Forcing Arabic decoding on
    it made Whisper hallucinate a filler and report 100% WER on one word. `detect_lang` takes text
    and nothing else, so the voice name structurally cannot reach it -- this pins that."""
    assert sc.detect_lang("Hello.") != "arabic"
    assert sc.detect_lang("ar_male says hello") != "arabic"


def test_detect_lang_uses_word_markers_for_latin_languages(sc):
    assert sc.detect_lang("bonjour tout le monde") == "french"
    assert sc.detect_lang("olá coração") == "portuguese"


def test_detect_lang_returns_none_for_unmarked_english(sc):
    """None means "let Whisper decide", which is the safe default -- forcing a language is what
    trap 14's third case punished."""
    assert sc.detect_lang("the quick brown fox") is None

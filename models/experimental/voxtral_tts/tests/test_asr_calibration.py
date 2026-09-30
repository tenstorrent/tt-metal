# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The WER recogniser, calibrated on known audio before it is trusted to gate anything.

Runs the gate's own `Asr` and `wer` on fp32-reference speech that must score near zero, on silence,
noise and cut clips that must score badly, and on a >30 s clip with long form on and (the control)
off. The clips are codes in asr_calibration_fixture.pt, decoded here by the fp32 codec; rebuild with
make_asr_calibration_fixture.py (bring-up tooling, see the README). The metric tests score typed text
only, so a recogniser that truncates long audio or a cleaner that erases a script would pass them.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_asr_calibration.py      # CPU
"""

import os

import pytest

torch = pytest.importorskip("torch")

from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT  # noqa: E402
from models.experimental.voxtral_tts.tests.test_wer_languages import (  # noqa: E402
    COLLAPSE,
    OUTPUT_SR,
    Asr,
    _words,
    wer,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "asr_calibration_fixture.pt")
SAMPLES_PER_FRAME = 1920

# Known-good bounds: each clip's own score plus one word, per clip since Whisper's error rate differs
# by language.
GOOD_MAX = {"en_medium": 0.05, "en_long": 0.02, "ar": 0.10, "hi": 0.15}
CUT_KEEP = 0.5  # keep this fraction of the audio for the cut-tail clips
CUT_MIN = 0.25  # half the audio gone must cost at least a quarter of the words
TAIL_WORDS = 3  # the last words of a sentence: present when whole, absent when cut
SCRIPT_RANGES = {"hi": ("ऀ", "ॿ"), "ar": ("؀", "ۿ")}

pytestmark = [
    pytest.mark.slow,
    pytest.mark.timeout(1800),
    pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}"),
]


def _load_fixture():
    return torch.load(FIXTURE)["clips"]


@pytest.fixture(scope="module")
def clips():
    """-> {key: (clip dict, waveform torch [N] @ 24 kHz)}, decoded by the fp32 reference codec."""
    from models.experimental.voxtral_tts.reference import voxtral_codec_ref as cref

    w = cref.load_codec_state()
    out = {}
    for key, c in _load_fixture().items():
        codes = cref.strip_offset_and_trim(c["frames"].long())
        out[key] = (c, cref.reference_decode(codes, w).reshape(-1).float())
    return out


@pytest.fixture(scope="module")
def asr():
    return Asr()


def _tail_present(text, hyp, n=TAIL_WORDS):
    """How many of the reference's last `n` words appear in the transcript's last 2n words."""
    ref, h = _words(text), _words(hyp)
    return sum(1 for w in ref[-n:] if w in h[-2 * n :])


def _noise(seconds, rms, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * OUTPUT_SR), generator=g) * rms


# ------------------------------------------------------------------------------ the fixture itself


@pytest.mark.parametrize("key", ["en_medium", "en_long", "hi", "ar"])
def test_fixture_has_the_clips_calibration_needs(key):
    """Every clip the checks rely on is present, and the long one really exceeds Whisper's 30 s
    window, or the long-form test would never reach the code it exists for."""
    c = _load_fixture().get(key)
    assert c is not None, f"{key} missing from {FIXTURE}; rebuild with make_asr_calibration_fixture.py"
    seconds = c["frames"].shape[0] * SAMPLES_PER_FRAME / OUTPUT_SR
    if key == "en_long":
        assert seconds > 30, f"en_long is {seconds:.1f} s -- it must exceed Whisper's 30 s window"
    assert c["lang"] in ("en", "hi", "ar")


# ------------------------------------------------------------------------------ known good


@pytest.mark.parametrize("key", ["en_medium", "hi", "ar"])
def test_known_good_scores_near_zero(clips, asr, key):
    c, wav = clips[key]
    hyp = asr(wav, c["lang"])
    w = wer(c["text"], hyp)
    print(f"\n  {key}: WER {w:.4f}\n    ref {c['text']}\n    hyp {hyp}")
    assert w <= GOOD_MAX[key], f"{key}: known-good clip scored {w:.4f} > {GOOD_MAX[key]}: {hyp!r}"


@pytest.mark.parametrize("key", ["hi", "ar"])
def test_non_latin_transcript_survives_the_cleaner(clips, asr, key):
    """A real transcript in the target script must come out of the cleaner with its words, in that
    script, not emptied to a free zero."""
    c, wav = clips[key]
    hyp = asr(wav, c["lang"])
    words = _words(hyp)
    lo, hi = SCRIPT_RANGES[key]
    letters = [ch for ch in "".join(words) if ch.isalpha()]
    in_script = sum(1 for ch in letters if lo <= ch <= hi)
    print(f"\n  {key}: {len(words)} words after cleaning, {in_script}/{len(letters)} letters in script")
    assert len(words) >= len(_words(c["text"])) // 2, f"{key}: cleaner left {len(words)} words: {words}"
    assert (
        letters and in_script / len(letters) > 0.9
    ), f"{key}: transcript is not in the expected script ({in_script}/{len(letters)}): {hyp!r}"


# ------------------------------------------------------------------------------ known bad


def test_silence_reads_as_collapse(asr):
    w = wer(_load_fixture()["en_medium"]["text"], hyp := asr(torch.zeros(8 * OUTPUT_SR), "en"))
    print(f"\n  silence: WER {w:.4f}  hyp {hyp!r}")
    assert w >= COLLAPSE, f"silence scored {w:.4f}, under the collapse line {COLLAPSE}: {hyp!r}"


def test_noise_reads_as_collapse(clips, asr):
    c, wav = clips["en_medium"]
    n = _noise(wav.shape[0] / OUTPUT_SR, float(wav.pow(2).mean().sqrt()))
    w = wer(c["text"], hyp := asr(n, "en"))
    print(f"\n  noise at speech RMS: WER {w:.4f}  hyp {hyp!r}")
    assert w >= COLLAPSE, f"white noise scored {w:.4f}, under the collapse line {COLLAPSE}: {hyp!r}"


@pytest.mark.parametrize("key", ["en_medium", "hi", "ar"])
def test_cut_tail_shows_the_missing_words(clips, asr, key):
    """Half the audio gone must cost words, and the words it costs must be the tail's."""
    c, wav = clips[key]
    cut = wav[: int(wav.shape[0] * CUT_KEEP)]
    whole, part = asr(wav, c["lang"]), asr(cut, c["lang"])
    w_whole, w_cut = wer(c["text"], whole), wer(c["text"], part)
    print(f"\n  {key}: whole {w_whole:.4f}  cut to {CUT_KEEP:.0%} {w_cut:.4f}\n    hyp {part}")
    assert w_cut >= CUT_MIN, f"{key}: cutting half the audio only cost {w_cut:.4f}: {part!r}"
    assert w_cut > w_whole + CUT_MIN, f"{key}: cut {w_cut:.4f} is not clearly worse than whole {w_whole:.4f}"
    assert _tail_present(c["text"], part) == 0, f"{key}: the cut clip still has its last words: {part!r}"
    assert _words(part)[:1] == _words(c["text"])[:1], f"{key}: the kept head is not the sentence's start"


# ------------------------------------------------------------------------------ long form


def test_long_clip_transcribes_to_the_end(clips, asr):
    c, wav = clips["en_long"]
    seconds = wav.shape[0] / OUTPUT_SR
    hyp = asr(wav, "en")
    w = wer(c["text"], hyp)
    print(
        f"\n  en_long ({seconds:.1f} s, {len(_words(c['text']))} words): WER {w:.4f}, "
        f"last {TAIL_WORDS} words present {_tail_present(c['text'], hyp)}"
    )
    assert seconds > 30
    assert w <= GOOD_MAX["en_long"], f"long known-good clip scored {w:.4f} > {GOOD_MAX['en_long']}"
    assert _tail_present(c["text"], hyp) == TAIL_WORDS, f"the transcript stops short: ...{hyp[-120:]!r}"


def test_long_clip_truncates_without_long_form(clips, asr):
    """The control: the same clip with long form OFF loses its tail, so the duration branch is what
    makes the long test pass."""
    c, wav = clips["en_long"]
    hyp = asr(wav, "en", long_form=False)
    w = wer(c["text"], hyp)
    print(f"\n  en_long, long form OFF: WER {w:.4f}, tail present {_tail_present(c['text'], hyp)}")
    assert _tail_present(c["text"], hyp) == 0, "long form OFF still reached the end -- no truncation"
    assert w >= 0.10, f"truncation cost only {w:.4f}; expected the post-30 s words as deletions"

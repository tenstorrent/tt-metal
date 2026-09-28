# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The WER recogniser, calibrated on known audio before it is trusted to gate anything.

test_wer.py and test_wer_languages.py's metric tests prove the SCORING on typed text. Neither feeds
real audio through Whisper, so two failures this branch has actually had would pass both:

  * the 30 s truncation (BUG-13) -- Whisper cuts audio at 30 s unless asked for long form, and that
    made word-perfect long utterances score 0.245 and look like the model losing the thread;
  * the script eraser -- a cleaner that drops Devanagari and Arabic scores blank against blank as
    perfect. Its fix is tested on typed text, never on a transcript Whisper actually produced.

So this runs the gate's own recogniser (`Asr`, whisper-large-v3) and scorer (`wer`) on audio whose
right answer is known:

  known good   fp32-reference speech that must score near zero, in English, Hindi and Arabic
  known bad    silence and noise, which must read as collapse; and clips with the tail cut off,
               which must show exactly the missing words
  long form    a 39 s clip that must transcribe to its last word -- and, as a control, must NOT when
               long form is switched off, so the branch that fixes BUG-13 is proven load-bearing

The known-good clips are the fp32 CPU REFERENCE's output (scripts/make_asr_calibration_fixture.py),
stored as codes and decoded here by the fp32 codec: the instrument is calibrated on audio that does
not depend on the device it will later judge. Whisper is greedy and on CPU, so every score here is
deterministic and the bounds are set from measured values with margin, not guessed.

The quality report's own scorer (scripts/score_quality_set_scipy.py, whisper-base.en with its own
long-form chunking) gates wer_longform on English, so it gets the same English checks.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_asr_calibration.py      # ~5 min, CPU
"""

import importlib.util
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
SCORER = os.path.join(os.path.dirname(HERE), "scripts", "score_quality_set_scipy.py")
SAMPLES_PER_FRAME = 1920

# Bounds, from the measured scores of these exact clips (deterministic -- see the module docstring).
# A known-good clip's bound is its MEASURED score plus one word of headroom -- per clip, because
# Whisper's error rate is not language independent (the gate's ceilings are per language for the
# same reason). Measured 2026-09-28, whisper-large-v3, greedy, CPU:
#   en_medium 0.0000 (0 of 22)   en_long 0.0000 (0 of 115, 39.4 s)
#   ar 0.0500 (1 of 20: ثلاث -> ثلاثة, a grammatical variant)
#   hi 0.1111 (3 of 27: नक्शे -> नकशे drops the virama, a spelling variant; खोज -> खोच and
#                चाहता -> चाता are recognition slips on clean fp32-reference speech -- Whisper is
#                weakest on Hindi, and the gate's own hi/medium ceiling is 0.16)
GOOD_MAX = {"en_medium": 0.05, "en_long": 0.02, "ar": 0.10, "hi": 0.15}
CUT_KEEP = 0.5               # keep this fraction of the audio for the cut-tail clips
CUT_MIN = 0.25               # half the audio gone must cost at least a quarter of the words
TAIL_WORDS = 3               # the last words of a sentence: present when whole, absent when cut
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
    return sum(1 for w in ref[-n:] if w in h[-2 * n:])


def _noise(seconds, rms, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(int(seconds * OUTPUT_SR), generator=g) * rms


# ------------------------------------------------------------------------------ the fixture itself

@pytest.mark.parametrize("key", ["en_medium", "en_long", "hi", "ar"])
def test_fixture_has_the_clips_calibration_needs(key):
    """Coverage asserted, not assumed: every clip the checks below rely on is present, and the long
    one really is past Whisper's 30 s window -- a shorter "long" clip would pass the long-form test
    without ever reaching the code it exists for."""
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
    """The failure this guards scored blank against blank as perfect. A real transcript in the
    target script must come out of the cleaner with its words, in that script."""
    c, wav = clips[key]
    hyp = asr(wav, c["lang"])
    words = _words(hyp)
    lo, hi = SCRIPT_RANGES[key]
    letters = [ch for ch in "".join(words) if ch.isalpha()]
    in_script = sum(1 for ch in letters if lo <= ch <= hi)
    print(f"\n  {key}: {len(words)} words after cleaning, {in_script}/{len(letters)} letters in script")
    assert len(words) >= len(_words(c["text"])) // 2, f"{key}: cleaner left {len(words)} words: {words}"
    assert letters and in_script / len(letters) > 0.9, (
        f"{key}: transcript is not in the expected script ({in_script}/{len(letters)}): {hyp!r}")


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
    """Half the audio gone must cost words, and the words it costs must be the TAIL's -- which is
    also a known-bad clip in each non-Latin script, where an eraser would score it 0."""
    c, wav = clips[key]
    cut = wav[: int(wav.shape[0] * CUT_KEEP)]
    whole, part = asr(wav, c["lang"]), asr(cut, c["lang"])
    w_whole, w_cut = wer(c["text"], whole), wer(c["text"], part)
    print(f"\n  {key}: whole {w_whole:.4f}  cut to {CUT_KEEP:.0%} {w_cut:.4f}\n    hyp {part}")
    assert w_cut >= CUT_MIN, f"{key}: cutting half the audio only cost {w_cut:.4f}: {part!r}"
    assert w_cut > w_whole + CUT_MIN, f"{key}: cut {w_cut:.4f} is not clearly worse than whole {w_whole:.4f}"
    assert _tail_present(c["text"], part) == 0, f"{key}: the cut clip still has its last words: {part!r}"
    assert _words(part)[:1] == _words(c["text"])[:1], f"{key}: the kept head is not the sentence's start"


# ------------------------------------------------------------------------------ long form (BUG-13)

def test_long_clip_transcribes_to_the_end(clips, asr):
    c, wav = clips["en_long"]
    seconds = wav.shape[0] / OUTPUT_SR
    hyp = asr(wav, "en")
    w = wer(c["text"], hyp)
    print(f"\n  en_long ({seconds:.1f} s, {len(_words(c['text']))} words): WER {w:.4f}, "
          f"last {TAIL_WORDS} words present {_tail_present(c['text'], hyp)}")
    assert seconds > 30
    assert w <= GOOD_MAX["en_long"], f"long known-good clip scored {w:.4f} > {GOOD_MAX['en_long']}"
    assert _tail_present(c["text"], hyp) == TAIL_WORDS, f"the transcript stops short: ...{hyp[-120:]!r}"


def test_long_clip_truncates_without_long_form(clips, asr):
    """The control: the same clip with long form OFF loses its tail. If this ever passes cleanly,
    Whisper stopped truncating and the branch above is no longer what makes the long test pass."""
    c, wav = clips["en_long"]
    hyp = asr(wav, "en", long_form=False)
    w = wer(c["text"], hyp)
    print(f"\n  en_long, long form OFF: WER {w:.4f}, tail present {_tail_present(c['text'], hyp)}")
    assert _tail_present(c["text"], hyp) == 0, "long form OFF still reached the end -- no truncation"
    assert w >= 0.10, f"truncation cost only {w:.4f}; expected the post-30 s words as deletions"


# ------------------------------------------------------------------------------ the report's scorer

@pytest.fixture(scope="module")
def report_scorer():
    spec = importlib.util.spec_from_file_location("_scorer", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report_score(sc, wav, text, tmp_path, name):
    """Run the quality report's transcribe_one on a wav written exactly as the generator writes it."""
    import wave

    path = str(tmp_path / f"{name}.wav")
    x = (wav.clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(OUTPUT_SR)
        f.writeframes(x.tobytes())
    hyp, _ = sc.transcribe_one({"text": text, "wav": path, "audio_s": wav.shape[0] / OUTPUT_SR})
    errs, n = sc.wer(text, hyp)
    return errs / max(n, 1), hyp


@pytest.mark.parametrize("key", ["en_medium", "en_long"])
def test_report_scorer_on_known_good(clips, report_scorer, tmp_path, key):
    """wer_longform is scored by this path, and its long cases are ~36 s, so it needs the same proof."""
    c, wav = clips[key]
    w, hyp = _report_score(report_scorer, wav, c["text"], tmp_path, key)
    print(f"\n  report scorer {key}: WER {w:.4f}, tail present {_tail_present(c['text'], hyp)}")
    assert w <= GOOD_MAX[key], f"report scorer: known-good {key} scored {w:.4f}: {hyp!r}"
    assert _tail_present(c["text"], hyp) == TAIL_WORDS, f"report scorer stops short: ...{hyp[-120:]!r}"


def test_report_scorer_on_known_bad(clips, report_scorer, tmp_path):
    c, wav = clips["en_medium"]
    w_sil, _ = _report_score(report_scorer, torch.zeros(8 * OUTPUT_SR), c["text"], tmp_path, "sil")
    w_cut, hyp = _report_score(report_scorer, wav[: int(wav.shape[0] * CUT_KEEP)], c["text"],
                               tmp_path, "cut")
    print(f"\n  report scorer: silence {w_sil:.4f}, cut tail {w_cut:.4f}")
    assert w_sil >= COLLAPSE, f"report scorer: silence scored {w_sil:.4f}"
    assert w_cut >= CUT_MIN, f"report scorer: cut tail scored {w_cut:.4f}: {hyp!r}"

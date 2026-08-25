# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""WER test: does the device actually say the words?

Every other gate compares numbers, so none of them can see whether the audio is intelligible — the
audio code cap cutting a sentence short, degeneration on repeated input, and the model inventing
speech past the last word are invisible to all of them.

Run-on that is SILENT is invisible to WER as well, since silence transcribes as nothing, so it is
measured directly rather than inferred from the transcript.

The model free-runs, so every draw realises the sentence differently and there is no canonical
transcript to diff against. WER is a metric on (audio, source text), so each run is scored against
the text it was given — invariant to how that particular draw happened to say it.

The voice comes from the checkpoint's own speakers_xtts.pth rather than a reference clip: a
synthetic waveform puts the speaker encoder outside its training distribution and the decoder
answers with non-speech. Latents also mean no DSP and no Block 1 or 2 work here, so a failure is
prefill, decode or the vocoder.

One case per supported language, each gated on its own, so a failure names the language instead of
burying it in a pooled average. English sweeps every speaker: that axis is language-independent —
the speaker embedding is the same tensor whatever the text says — so it is covered once, there, and
the other languages take a handful of speakers spread through d-vector space to average out draw
noise. Sweeping all of them per language would multiply the runtime by nine and measure the same
axis thirteen times.

The slowest test in the suite. Needs the Whisper weights (cached or downloadable), and the corpus
from language_corpus.WER_SENTENCES.

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_wer.py
"""
import functools
import math
import os
import unicodedata

import pytest
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from models.experimental.xtts_v2.frontend import ROMANIZERS, SUPPORTED_LANGUAGES, sinc_resample
from models.experimental.xtts_v2.tests.language_corpus import WER_SENTENCES
from models.experimental.xtts_v2.tt.ttnn_xtts_model import OUTPUT_SR, Voice, XttsV2

ASR_MODEL = "openai/whisper-large-v3"  # small hallucinates on short audio and is weak outside en
ASR_SR = 16000
SEED = 0  # one seed: every speaker/sentence pair is already a distinct draw
SPEAKERS_FILE = "speakers_xtts.pth"  # ships beside model.pth, the way vocab.json does
FULL_SWEEP_LANG = "en"
SPEAKERS_PER_LANG = 6

# (ceiling, collapse) per language, ordered by difficulty. Both are per language because the
# RECOGNISER's error rate is not: a bar that fits English fails the languages Whisper transcribes
# less well, through no fault of the device.
#
# ceiling  -- ~3x the language's baseline, with a floor so one unlucky draw cannot breach a
#             near-perfect language. FULL_SWEEP_LANG is tighter, running far more draws.
# collapse -- past this a run did not say the sentence. Tiered, not fitted per language: the worst
#             of a few dozen draws is too noisy to fit to.
LIMITS = {
    "en": (0.02, 0.30),
    "es": (0.03, 0.30),
    "de": (0.03, 0.30),
    "fr": (0.03, 0.30),
    "ja": (0.04, 0.30),
    "ko": (0.04, 0.30),
    "nl": (0.06, 0.30),
    "pl": (0.07, 0.50),
    "ar": (0.08, 0.30),
    "pt": (0.08, 0.50),
    "tr": (0.08, 0.50),
    "it": (0.09, 0.50),
    "ru": (0.10, 0.30),
    "zh": (0.11, 0.30),
    "hu": (0.15, 0.50),
    "hi": (0.20, 0.75),
}
# A mean over a language's runs barely moves when a few collapse, so the collapse count is asserted
# directly rather than left to the average to reveal.
MAX_DEGENERATE = 2

# The model sometimes keeps emitting codes after the sentence rather than STOP, and those codes
# vocode to silence. Nothing else sees it: WER hears nothing, and the duration check in
# test_all_languages_smoke compares audio length against code count, which stay consistent precisely
# because the model generated the extra codes.
OVER_RUN_SECONDS = 0.5  # a natural tail is a fraction of this; a run-on is several times it
# One bound for most languages: at this few runs per language the counts cannot separate one rate
# from another. FULL_SWEEP_LANG runs far more, so it carries a tighter bound.
MAX_OVER_RUN = 6
MAX_OVER_RUN_FULL_SWEEP = 3


# Optional orthography: marks a writer may or may not put in, which the ASR usually leaves out. Both
# spellings are correct and sound identical, so scoring them as substitutions charges the model for
# errors it did not make -- the same reason casefolding and punctuation-stripping happen at all.
# Every rule is scoped to one script, so Latin, Cyrillic and Greek text is provably untouched.
_DROP_MARKS = {
    "\u093c",  # Devanagari nukta: तेज़ and तेज are the same word
    *(chr(c) for c in range(0x64B, 0x653)),  # Arabic harakat, omitted in ordinary writing
    "\u0653",
    "\u0654",
    "\u0655",  # madda and hamza, which fold the alef variants together
}
_FOLD_CHARS = str.maketrans({"\u0901": "\u0902", "\u0629": "\u0647", "\u0649": "\u064a"})
# chandrabindu -> anusvara,  ta marbuta -> ha,  alef maqsura -> ya


def _words(s):
    """Casefold, drop punctuation and fold optional orthography, in ANY script.

    WER on raw text is dominated by commas and capitals. Three traps beyond that, all silent:

    An ASCII-only filter looks equivalent and is not -- it empties Arabic, Cyrillic and Devanagari
    completely, so both sides normalise to nothing and every comparison scores a free zero.

    Combining marks are not alnum, and Devanagari vowels are combining marks: dropping them cuts
    every word into fragments. Identical text still scores 0.000 because both sides fragment the
    same way, which is why this needs testing on text that DIFFERS.

    Optional marks (see above) are folded away. Without it the score measures spelling conventions
    as much as speech, and does so unevenly: scripts that mark more get charged more."""
    flat = s.casefold().replace("\u2019", "'").replace("\u02bc", "'")  # ASR emits curly apostrophes
    flat = unicodedata.normalize("NFD", flat)  # so precomposed forms expose their marks
    flat = "".join(c for c in flat if c not in _DROP_MARKS).translate(_FOLD_CHARS)
    flat = unicodedata.normalize("NFC", flat)  # composed and decomposed forms must compare equal
    keep = lambda c: c.isalnum() or c.isspace() or c == "'" or unicodedata.category(c) in ("Mn", "Mc")
    return "".join(c if keep(c) else " " for c in flat).split()


def _trailing_silence(wav):
    """Seconds of near-silence at the end of a waveform.

    RMS over short windows rather than raw samples: a lone nonzero sample is the noise floor, not
    speech. The loudness bar is relative to the clip's own peak as well as absolute, because
    speakers differ in level and a fixed bar alone would call a quiet speaker's tail silence.
    A clip with no loud window at all counts as silent throughout, which is a fault worth seeing."""
    n = int(OUTPUT_SR * 0.02)
    flat = wav.reshape(-1)
    if flat.numel() < n:
        return 0.0  # shorter than one window; the empty-audio contract is handled by the caller
    frames = flat[: flat.numel() // n * n].reshape(-1, n)
    rms = frames.pow(2).mean(1).sqrt()
    loud = (rms > max(0.01, 0.02 * rms.max().item())).nonzero()
    if not len(loud):
        return len(frames) * n / OUTPUT_SR
    return (len(frames) - 1 - loud[-1].item()) * n / OUTPUT_SR


def _edit_ratio(ref, hyp):
    """Levenshtein distance over two sequences, divided by the reference length."""
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[-1][-1] / max(len(ref), 1)


def _wer(reference, hypothesis):
    return _edit_ratio(_words(reference), _words(hypothesis))


def _cer(reference, hypothesis):
    """Error rate over characters, with spacing discarded.

    For scripts without dependable word boundaries a whitespace metric is unusable: where there are
    no spaces at all it sees one token per sentence, so any mistake scores 1.0. Korean does space,
    but variably enough that the reference and the ASR can both be right and still disagree."""
    return _edit_ratio("".join(_words(reference)), "".join(_words(hypothesis)))


# Scored by character, not by word. Their limits are therefore not comparable with the others':
# the same audio spreads its errors over more units.
CHAR_SCORED = ("ja", "ko", "zh")


@functools.lru_cache(maxsize=1)
def _simplifier():
    from opencc import OpenCC

    return OpenCC("t2s")


# Both sides pass through the same normaliser before scoring, so how the ASR chose to WRITE a
# correct utterance is not charged to the model.
#
# zh -- Whisper picks simplified or traditional per utterance and the difference lands on nearly
#       every character. Chinese only: Japanese kanji are a third variant this would corrupt.
# ja -- the same word may be written in kanji or kana, and homophones pick either. The fold is the
#       frontend's own romanizer, so it asks exactly what the model was given: the right sounds.
SCORE_FOLDS = {
    "zh": lambda t: _simplifier().convert(t),
    "ja": ROMANIZERS["ja"],
}


def _score(lang, reference, hypothesis):
    fold = SCORE_FOLDS.get(lang)
    if fold:
        reference, hypothesis = fold(reference), fold(hypothesis)
    return (_cer if lang in CHAR_SCORED else _wer)(reference, hypothesis)


class _Asr:
    """Whisper on CPU, greedy so the transcript is reproducible."""

    def __init__(self):
        self.proc = WhisperProcessor.from_pretrained(ASR_MODEL)
        # the large checkpoints ship fp16, which cannot run against fp32 features on CPU
        self.model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL, torch_dtype=torch.float32).eval()

    def __call__(self, wav, lang):
        # The model's own resampler rather than a new dependency: a fault in it raises the score,
        # which the ceiling catches.
        audio = sinc_resample(wav.reshape(1, -1), OUTPUT_SR, ASR_SR)
        feats = self.proc(audio[0].numpy(), sampling_rate=ASR_SR, return_tensors="pt").input_features
        with torch.no_grad():
            # told the language, not left to detect it: detection on a few seconds is unreliable,
            # and a wrong guess transcribes into the wrong script entirely
            ids = self.model.generate(feats, language=lang, task="transcribe", do_sample=False, num_beams=1)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _speakers(ckpt_path):
    """The checkpoint's built-in studio speakers -> {name: Voice}. Latents, so no DSP is involved."""
    raw = torch.load(os.path.join(os.path.dirname(ckpt_path), SPEAKERS_FILE), weights_only=False)
    return {
        name: Voice(gpt_cond_latent=d["gpt_cond_latent"], speaker_embedding=d["speaker_embedding"])
        for name, d in raw.items()
    }


def _representative(voices, k):
    """`k` speakers spread as widely as possible through d-vector space (farthest-point sampling).

    Taking the first k in checkpoint order would sample whatever the file happens to open with.
    Deterministic: seeded from the first speaker, then each pick is the one furthest from everything
    already picked."""
    names = list(voices)
    if k >= len(names):
        return voices
    emb = {n: voices[n].speaker_embedding.flatten().float() for n in names}
    picked = [names[0]]
    while len(picked) < k:
        rest = [n for n in names if n not in picked]
        picked.append(max(rest, key=lambda n: min(torch.dist(emb[n], emb[q]).item() for q in picked)))
    return {n: voices[n] for n in picked}


def run_language(lang, asr, tts, all_voices, verbose=True):
    """One language's matrix -> (passed, message). Speakers are swept in full for FULL_SWEEP_LANG."""
    voices = all_voices if lang == FULL_SWEEP_LANG else _representative(all_voices, SPEAKERS_PER_LANG)
    texts = WER_SENTENCES[lang]
    scores, over_run = {}, []
    for name, voice in voices.items():
        for text in texts:
            wav = tts.generate(text, voice, language=lang, seed=SEED)
            silence = _trailing_silence(wav)
            if silence > OVER_RUN_SECONDS:
                over_run.append(f"{name}/{text[:16]}:{silence:.1f}s")
            scores[(name, text)] = _score(lang, text, asr(wav, lang))
        if verbose:  # per speaker, so a long run shows progress and a failure names the speaker
            row = [scores[(name, t)] for t in texts]
            print(f"  {lang}  {name:24s} " + " ".join(f"{w:.3f}" for w in row) + f"  mean {sum(row) / len(row):.3f}")
    mean = sum(scores.values()) / len(scores)
    ceiling, collapse = LIMITS[lang]
    run_on_limit = MAX_OVER_RUN_FULL_SWEEP if lang == FULL_SWEEP_LANG else MAX_OVER_RUN
    metric = "CER" if lang in CHAR_SCORED else "WER"
    degenerate = [f"{n}/{t[:20]}" for (n, t), w in scores.items() if w >= collapse]
    msg = (
        f"{lang}: {len(voices)} speakers x {len(texts)} sentences, {metric} {mean:.4f} "
        f"(worst single {max(scores.values()):.3f}, perfect {sum(1 for w in scores.values() if w == 0)}"
        f"/{len(scores)}, degenerate {len(degenerate)}, over-run {len(over_run)}/{run_on_limit}) "
        f"ceiling {ceiling}, degenerate limit {MAX_DEGENERATE} at {metric} {collapse}"
        + (f"; over-ran: {over_run}" if over_run else "")
    )
    passed = mean <= ceiling and len(degenerate) <= MAX_DEGENERATE and len(over_run) <= run_on_limit
    return passed, msg


@pytest.fixture(scope="module")
def rig():
    """One model and one Whisper for the whole module: loading either per language would dominate."""
    asr = _Asr()
    tts = XttsV2()
    tts.warmup()
    yield asr, tts, _speakers(tts.ckpt_path)
    tts.close()


def test_wer_metric():
    """Standard WER, checked before it is used to judge anything. An over-long transcript can
    score above 1.0."""
    cases = (
        ("the cat sat down", "the cat sat down", 0.0),
        ("the cat sat down", "the dog sat down", 0.25),  # substitution
        ("the cat sat down", "the cat down", 0.25),  # deletion
        ("the cat sat down", "the cat sat right down", 0.25),  # insertion
        ("the cat sat down", "", 1.0),  # nothing transcribed
        ("The cat, sat down!", "the cat sat down", 0.0),  # punctuation and case ignored
        ("the cat", "the dog ran fast today", 2.0),
        # non-Latin scripts must survive normalisation rather than emptying to a free zero
        ("привет мир", "привет мир", 0.0),
        ("привет мир", "привет луна", 0.5),
        ("привет мир", "", 1.0),
        ("नमस्ते दुनिया", "नमस्ते दुनिया", 0.0),
        # Devanagari vowels are combining marks: these fail if they are dropped as punctuation,
        # because the words fragment and the denominator grows. Identical text cannot catch it.
        ("नमस्ते दुनिया", "नमस्ते चाँद", 0.5),
        ("मुझे यह किताब बहुत पसंद है", "मुझे यह किताब बहुत अच्छी है", 1 / 6),
        ("سوق الشتاء مبكرا اليوم", "سوق الصيف مبكرا اليوم", 0.25),  # Arabic, undiacritized
        # Optional orthography must fold: these are the SAME words spelled two legal ways, and the
        # ASR picks its own.
        ("तेज़ चाय", "तेज चाय", 0.0),  # nukta
        ("आँच पर", "आंच पर", 0.0),  # chandrabindu vs anusvara
        ("ज़ोर से", "जोर से", 0.0),
        ("سُوقُ الشِّتَاءِ", "سوق الشتاء", 0.0),  # harakat
        ("أهلا وسهلا", "اهلا وسهلا", 0.0),  # alef variants
        ("مدينة كبيرة", "مدينه كبيره", 0.0),  # ta marbuta
        # ...while real differences in those same scripts still score
        ("तेज़ चाय", "तेज़ दूध", 0.5),
        ("سوق الشتاء", "سوق الصيف", 0.5),
        # ...and the rules must not touch Latin: an accent is not optional orthography
        ("naive café", "naive cafe", 0.5),
        ("\u0915\u093f", "\u0915\u093f", 0.0),  # composed vs decomposed must not differ
        ("Grüße, Welt!", "grüße welt", 0.0),
        ("it doesn't matter", "it doesn\u2019t matter", 0.0),  # curly vs straight apostrophe
    )
    for reference, hypothesis, want in cases:
        assert _wer(reference, hypothesis) == want, f"WER({reference!r}, {hypothesis!r})"


def test_cer_metric():
    """The character metric, checked before it is used to judge anything. Spacing is discarded, so
    a difference only in spacing scores zero while a different character still counts."""
    cases = (
        ("的岛屿", "的岛屿", 0.0),
        ("旧地图上画着", "旧地图上画着", 0.0),
        ("旧地图上画着", "旧地图上画过", 1 / 6),  # one character of six
        ("旧地图上画着", "旧地图上画", 1 / 6),  # a dropped character
        ("こんにちは世界", "こんにちは世海", 1 / 7),
        # Korean spacing is optional in places, so the same sentence written two legal ways matches
        ("달려 있었기에", "달려있었기에", 0.0),
        ("네 개 층을", "네개 층을", 0.0),
        # ...but a genuinely different word still scores
        ("선원도 찾지", "선언도 찾지", 1 / 5),  # five characters once spacing is dropped
        # scripts that DO have word boundaries are not scored this way; _wer keeps them
        ("the cat sat", "the cat sat", 0.0),
    )
    for reference, hypothesis, want in cases:
        got = _cer(reference, hypothesis)
        assert abs(got - want) < 1e-9, f"CER({reference!r}, {hypothesis!r}) = {got}, want {want}"


def test_metric_choice_matches_the_script():
    """A space-less script scored by word would see one token per sentence: any error at all would
    score 1.0, and the collapse gate would call every imperfect run degenerate."""
    zh = ("旧地图上画着三座岛屿", "旧地图上画着三座岛")
    assert _wer(*zh) == 1.0, "word scoring is degenerate here, which is why CHAR_SCORED exists"
    assert _cer(*zh) < 0.2
    assert all(lang in SUPPORTED_LANGUAGES for lang in CHAR_SCORED)


def test_score_folds():
    """Each fold removes a way the ASR can write a CORRECT utterance differently, and none of them
    may hide a wrong one."""
    trad, simp = "在鐵路修到山谷以前很久", "在铁路修到山谷以前很久"
    assert _score("zh", simp, trad) == 0.0, "the same sentence in the other script is not an error"
    assert _cer(simp, trad) > 0.0, "...and it is the fold doing that; the raw characters differ"
    assert _score("zh", "时刻表取决于天气", "時刻標取決於天氣") > 0.0, "a mishearing must survive"

    for a, b in (("まったく", "全く"), ("いっそう", "一層"), ("やかん", "夜間")):
        assert _score("ja", a, b) == 0.0, f"{a}/{b} is one word written two ways"
    for a, b in (("資料", "修行"), ("満たし", "煮たし")):
        assert _score("ja", a, b) > 0.0, f"{a}/{b} is a mishearing and must survive"

    # the Chinese fold must not reach Japanese: its kanji are a third variant
    assert SCORE_FOLDS["zh"]("見た") != "見た" and SCORE_FOLDS["ja"] is not SCORE_FOLDS["zh"]


def test_trailing_silence_metric():
    """The instrument, checked before it is used to judge anything -- the same reason test_wer_metric
    exists. A quiet speaker must not read as silence, which is what the relative bar is for."""
    sr = OUTPUT_SR
    tone = lambda secs, amp=0.5: amp * torch.sin(2 * math.pi * 220 * torch.arange(int(secs * sr)) / sr)
    hush = lambda secs: torch.zeros(int(secs * sr))
    cases = (
        (tone(1.0), 0.0, 0.06, "speech to the very end"),
        (torch.cat([tone(1.0), hush(2.0)]), 1.94, 2.06, "two seconds of tail"),
        (torch.cat([tone(1.0), hush(0.3)]), 0.24, 0.36, "a short natural tail"),
        (torch.cat([tone(1.0, amp=0.02), hush(1.0)]), 0.94, 1.06, "quiet speaker, still speech"),
        (hush(1.0), 0.94, 1.06, "silent throughout"),
        (torch.cat([hush(2.0), tone(1.0)]), 0.0, 0.06, "leading silence is not trailing"),
    )
    for wav, lo, hi, why in cases:
        got = _trailing_silence(wav)
        assert lo <= got <= hi, f"{why}: expected {lo}-{hi}s, got {got:.3f}s"


def test_wer_corpus_covers_every_language():
    """A supported language with no sentences or no limits would pass by never being measured."""
    assert sorted(WER_SENTENCES) == sorted(SUPPORTED_LANGUAGES)
    assert sorted(LIMITS) == sorted(SUPPORTED_LANGUAGES)
    assert all(len(v) == 5 for v in WER_SENTENCES.values()), "five sentences per language"
    assert all(c < k for c, k in LIMITS.values()), "a ceiling above the collapse point gates nothing"


@pytest.mark.slow
@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_wer(lang, rig):
    passed, msg = run_language(lang, *rig)
    assert passed, f"past the ceiling, or too many degenerate or over-running runs -- {msg}"


if __name__ == "__main__":
    import sys

    asr, tts = _Asr(), XttsV2()
    tts.warmup()
    voices, rows = _speakers(tts.ckpt_path), []
    try:
        for lang in SUPPORTED_LANGUAGES:
            ok, msg = run_language(lang, asr, tts, voices)
            rows.append((ok, msg))
            print(("PASSED " if ok else "FAILED ") + msg, flush=True)  # as each finishes, not at the end
    finally:
        tts.close()
    print("\n=== summary ===")
    for ok, msg in rows:
        print(("PASSED " if ok else "FAILED ") + msg)
    sys.exit(0 if all(ok for ok, _ in rows) else 1)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end WER, one gate per (language, length band).

The full request path runs, the waveform is transcribed back with Whisper, and the words are scored
against each cell's ceiling. Per language because Whisper's own error rate differs by language; per
band because one wrong word is a different rate in a short sentence than in a long passage.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_wer_languages.py           # full gate
    pytest -svv models/experimental/voxtral_tts/tests/test_wer_languages.py -k hindi  # one language
    pytest -svv models/experimental/voxtral_tts/tests/test_wer_languages.py -m "not slow"  # metric only
"""

import math
import os
import unicodedata

import pytest

torch = pytest.importorskip("torch")

from models.experimental.voxtral_tts.reference.voxtral_common_ref import DEFAULT_CKPT  # noqa: E402
from models.experimental.voxtral_tts.tests.sentence_corpus import (  # noqa: E402
    BANDS,
    WER_SENTENCES,
    lang_of,
    wer_band,
)

ASR_MODEL = "openai/whisper-large-v3"  # smaller models hallucinate on short audio and are weak outside English
ASR_SR = 16000
OUTPUT_SR = 24000
# Extra seeds only for the cells whose WER moves with the seed.
SEEDS = {("ar", "long"): (0, 1, 2), ("hi", "long"): tuple(range(10))}
DEFAULT_SEEDS = (0,)
FULL_SWEEP_LANG = "en"

# A run at or past this did not say the sentence; it is counted against MAX_DEGENERATE, not averaged.
COLLAPSE = 0.30

# Ceiling per (language, band): three times the cell's mean WER, but no lower than the band's floor
# (about one wrong word in that band) and no higher than COLLAPSE.
CEILINGS = {
    ("ar", "short"): 0.25,
    ("ar", "medium"): 0.05,
    ("ar", "long"): 0.07,
    ("de", "short"): 0.25,
    ("de", "medium"): 0.03,
    ("de", "long"): 0.02,
    ("en", "short"): 0.25,
    ("en", "medium"): 0.03,
    ("en", "long"): 0.02,
    ("es", "short"): 0.25,
    ("es", "medium"): 0.03,
    ("es", "long"): 0.02,
    ("fr", "short"): 0.25,
    ("fr", "medium"): 0.03,
    ("fr", "long"): 0.02,
    ("hi", "short"): 0.25,
    ("hi", "medium"): 0.16,
    # the weakest cell; its WER moves with the seed, so SEEDS gives it ten
    ("hi", "long"): 0.30,
    ("it", "short"): 0.25,
    ("it", "medium"): 0.03,
    ("it", "long"): 0.02,
    ("nl", "short"): 0.25,
    ("nl", "medium"): 0.04,
    ("nl", "long"): 0.04,
    ("pt", "short"): 0.25,
    ("pt", "medium"): 0.03,
    ("pt", "long"): 0.02,
    ("en", "voice_sweep"): 0.03,
}

VOICE_SWEEP_LANG = "en"
VOICE_SWEEP_SENTENCES = 2  # breadth over voices, not depth over sentences

MAX_DEGENERATE = 2  # runs per cell allowed at or past COLLAPSE
# Per-cell overrides of MAX_DEGENERATE: hi/long degenerates on some seeds on any healthy build.
MAX_DEGENERATE_CELL = {("hi", "long"): 8}
# runs that hit the frame cap without [END_AUDIO]; WER cannot hear a missing tail
MAX_NON_TERMINATING = 2

LANG_NAMES = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "hi": "hindi",
    "ar": "arabic",
}

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT), reason=f"no checkpoint at {DEFAULT_CKPT}")

# --------------------------------------------------------------------------------------- metric

# Optional orthography, dropped or folded so two legal spellings score alike; each rule is scoped
# to one script.
_DROP_MARKS = {
    "़",  # Devanagari nukta: तेज़ and तेज are one word
    *(chr(c) for c in range(0x64B, 0x656)),  # Arabic harakat, madda, hamza -- omitted in prose
}
_FOLD_CHARS = str.maketrans({"ँ": "ं", "ة": "ه", "ى": "ي"})
#                            chandrabindu->anusvara, ta marbuta->ha, alef maqsura->ya


def _words(s):
    """Casefold, drop punctuation and fold optional orthography, in ANY script.

    Keeps combining marks, so non-Latin text survives.
    """
    flat = s.casefold().replace("’", "'").replace("ʼ", "'")  # ASR emits curly apostrophes
    flat = unicodedata.normalize("NFD", flat)  # expose precomposed marks
    flat = "".join(c for c in flat if c not in _DROP_MARKS).translate(_FOLD_CHARS)
    flat = unicodedata.normalize("NFC", flat)
    keep = lambda c: c.isalnum() or c.isspace() or c == "'" or unicodedata.category(c) in ("Mn", "Mc")
    return "".join(c if keep(c) else " " for c in flat).split()


def wer(reference, hypothesis):
    """Levenshtein distance over words, divided by the reference length. Can exceed 1.0."""
    ref, hyp = _words(reference), _words(hypothesis)
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[-1][-1] / max(len(ref), 1)


@pytest.mark.parametrize(
    "ref,hyp,exp",
    [
        ("the cat sat down", "the cat sat down", 0.0),
        ("the cat sat down", "the dog sat down", 0.25),  # substitution
        ("the cat sat down", "the cat down", 0.25),  # deletion
        ("the cat sat down", "the cat sat right down", 0.25),  # insertion
        ("the cat sat down", "", 1.0),  # nothing transcribed
        ("The cat, sat down!", "the cat sat down", 0.0),  # punctuation and case ignored
        # non-Latin scripts must survive normalisation rather than emptying to a free zero
        ("नमस्ते दुनिया", "नमस्ते दुनिया", 0.0),
        ("नमस्ते दुनिया", "नमस्ते चाँद", 0.5),
        ("मुझे यह किताब बहुत पसंद है", "मुझे यह किताब बहुत अच्छी है", 1 / 6),
        ("سوق الشتاء مبكرا اليوم", "سوق الصيف مبكرا اليوم", 0.25),
        # optional orthography folds: the same words spelled two legal ways score 0
        ("तेज़ हवा चली", "तेज हवा चली", 0.0),
        ("مرحبا كيف حالك", "مَرْحَبا كيف حالك", 0.0),
    ],
)
def test_wer_metric(ref, hyp, exp):
    """The metric, before it is used to judge anything."""
    assert wer(ref, hyp) == pytest.approx(exp, abs=1e-9)


def test_every_voice_language_has_wer_sentences():
    """A voice whose language has no WER text would silently go ungated."""
    from models.experimental.voxtral_tts.tests.reference_helpers import all_voices

    missing = sorted({lang_of(v) for v in all_voices()} - set(WER_SENTENCES))
    assert not missing, f"languages with voices but no WER sentences: {missing}"


def test_every_language_band_is_gated():
    """A cell with no ceiling would raise mid-run, after paying for the generation, instead of
    being gated. Checked host-side so the mistake costs nothing."""
    want = {(l, b) for l in WER_SENTENCES for b in BANDS} | {(VOICE_SWEEP_LANG, "voice_sweep")}
    missing, extra = sorted(want - set(CEILINGS)), sorted(set(CEILINGS) - want)
    assert not missing and not extra, f"missing ceilings {missing}, unexpected {extra}"


# ---------------------------------------------------------------------------------------- the run


def frame_budget(text):
    """Frame cap: ~18 chars/s at 12.5 frames/s, x2.2 margin, floor 320. Generation stops on
    [END_AUDIO], so a generous cap costs nothing; a tight one would fake a non-terminating run."""
    return max(320, int(math.ceil(len(text) / 18.0 * 12.5 * 2.2)))


class Asr:
    """Whisper on CPU, greedy so the transcript is reproducible."""

    def __init__(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.proc = WhisperProcessor.from_pretrained(ASR_MODEL)
        # the large checkpoints ship fp16, which cannot run against fp32 features on CPU
        self.model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL, torch_dtype=torch.float32).eval()

    def __call__(self, wav, lang, long_form=None):
        """`long_form` None decides by duration (the gate); False forces the truncating path for the
        calibration control."""
        audio = wav.reshape(1, -1)
        n = int(audio.shape[1] * ASR_SR / OUTPUT_SR)
        audio = torch.nn.functional.interpolate(audio.unsqueeze(0), size=n, mode="linear", align_corners=False).squeeze(
            0
        )
        # Whisper silently truncates past 30 s unless asked for long form
        long_form = n > 30 * ASR_SR if long_form is None else long_form
        kw = {"truncation": False, "padding": "longest", "return_attention_mask": True} if long_form else {}
        inp = self.proc(audio[0].numpy(), sampling_rate=ASR_SR, return_tensors="pt", **kw)
        gen = {"language": lang, "task": "transcribe", "do_sample": False, "num_beams": 1}
        if long_form:
            gen |= {"attention_mask": inp.attention_mask, "return_timestamps": True}
        with torch.no_grad():
            # told the language: detection on a few seconds is unreliable and can pick the wrong script
            ids = self.model.generate(inp.input_features, **gen)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def voices_for(lang, all_voices):
    """Every voice of this language. English is the full sweep; the others have one or two."""
    return tuple(v for v in all_voices if lang_of(v) == lang)


def run_language(
    lang,
    asr,
    pipe,
    band="medium",
    voices=None,
    max_sentences=None,
    collapse=COLLAPSE,
    seeds_override=None,
    verbose=True,
):
    """One (language, band) voice x sentence x seed matrix -> stats dict.

    No assertions, so the measurement probe and the gate share one code path.
    """
    from models.experimental.voxtral_tts.tests.reference_helpers import all_voices, corpus_embeds

    voices = voices if voices is not None else voices_for(lang, all_voices())
    texts = wer_band(lang, band)[:max_sentences]  # a wide voice sweep pays breadth, not depth
    seeds = seeds_override or SEEDS.get((lang, band), DEFAULT_SEEDS)
    scores, non_terminating = {}, []
    for voice in voices:
        for si, text in enumerate(texts):
            for sd in seeds:
                cap = frame_budget(text)
                embeds = corpus_embeds(text, voice, pipe.wb)
                pipe.backbone.reset()
                frames, _, _ = pipe.generate(embeds, max_frames=cap, seed=sd, verbose=False)
                if len(frames) >= cap:
                    non_terminating.append(f"{voice}/s{si}/seed{sd}")
                scores[(voice, si, sd)] = wer(text, asr(pipe.decode(frames), lang))
        if verbose:
            row = [scores[(voice, i, sd)] for i in range(len(texts)) for sd in seeds]
            print(
                f"  {lang}/{band:<6} {voice:18s} "
                + " ".join(f"{w:.3f}" for w in row)
                + f"   mean {sum(row) / len(row):.4f}",
                flush=True,
            )
    vals = list(scores.values())
    return {
        "lang": lang,
        "band": band,
        "n_voices": len(voices),
        "n_runs": len(vals),
        "mean": sum(vals) / len(vals),
        "worst": max(vals),
        "perfect": sum(1 for w in vals if w == 0),
        "degenerate": [f"{v}/s{i}/seed{sd}" for (v, i, sd), w in scores.items() if w >= collapse],
        "non_terminating": non_terminating,
        "collapse": collapse,
    }


@pytest.fixture(scope="module")
def rig():
    """One pipeline and one Whisper for the whole module: loading either per language would dominate."""
    ttnn = pytest.importorskip("ttnn")
    from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device

    dev = open_device()
    pipe = TtVoxtralPipeline(dev)
    pipe.warmup(verbose=False)
    yield Asr(), pipe
    pipe.close()
    ttnn.close_device(dev)


@pytest.mark.slow
# the long band outlasts the default 300 s timeout
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("band", BANDS)
@pytest.mark.parametrize("lang", sorted(WER_SENTENCES), ids=lambda l: LANG_NAMES[l])
def test_wer_per_language_band(rig, lang, band):
    """This cell's own ceiling, so a failure names both the language and the length."""
    asr, pipe = rig
    s = run_language(lang, asr, pipe, band=band)
    ceiling = CEILINGS[(lang, band)]
    n_seeds = len(SEEDS.get((lang, band), DEFAULT_SEEDS))
    per = s["n_runs"] // s["n_voices"] // n_seeds
    print(
        f"\n  {lang}/{band}: {s['n_voices']} voices x {per} sentences x {n_seeds} seed(s), "
        f"WER {s['mean']:.4f} (worst {s['worst']:.3f}, perfect {s['perfect']}/{s['n_runs']}, "
        f"degenerate {len(s['degenerate'])}, non-terminating {len(s['non_terminating'])}) "
        f"ceiling {ceiling}",
        flush=True,
    )
    assert (
        s["mean"] <= ceiling
    ), f"{lang}/{band}: mean WER {s['mean']:.4f} over {s['n_runs']} runs above ceiling {ceiling}"
    max_degenerate = MAX_DEGENERATE_CELL.get((lang, band), MAX_DEGENERATE)
    assert len(s["degenerate"]) <= max_degenerate, (
        f"{lang}/{band}: {len(s['degenerate'])} runs at or above WER {COLLAPSE} "
        f"(limit {max_degenerate}): {s['degenerate']}"
    )
    assert len(s["non_terminating"]) <= MAX_NON_TERMINATING, (
        f"{lang}/{band}: {len(s['non_terminating'])} runs hit the frame cap without [END_AUDIO] "
        f"(limit {MAX_NON_TERMINATING}): {s['non_terminating']}"
    )


# Every voice, on English only: other languages lose accuracy on a foreign-language preset, which is
# the model's cross-lingual limit, not a device defect.
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_wer_every_voice_english(rig):
    """All twenty presets on English, so a defect in one voice's prompt geometry is audible here."""
    from models.experimental.voxtral_tts.tests.reference_helpers import all_voices

    asr, pipe = rig
    voices = tuple(all_voices())
    s = run_language(VOICE_SWEEP_LANG, asr, pipe, band="medium", voices=voices, max_sentences=VOICE_SWEEP_SENTENCES)
    # one ceiling for the whole sweep
    ceiling = CEILINGS[("en", "voice_sweep")]
    print(
        f"\n  en/voice_sweep: {len(voices)} voices, WER {s['mean']:.4f} "
        f"(worst {s['worst']:.3f}, perfect {s['perfect']}/{s['n_runs']}, "
        f"degenerate {len(s['degenerate'])}) ceiling {ceiling}",
        flush=True,
    )
    assert s["mean"] <= ceiling, f"en/voice_sweep: mean WER {s['mean']:.4f} over {s['n_runs']} runs above {ceiling}"
    assert (
        len(s["degenerate"]) <= MAX_DEGENERATE
    ), f"en/voice_sweep: {len(s['degenerate'])} collapsed runs: {s['degenerate']}"
    assert (
        len(s["non_terminating"]) <= MAX_NON_TERMINATING
    ), f"en/voice_sweep: {len(s['non_terminating'])} hit the frame cap: {s['non_terminating']}"

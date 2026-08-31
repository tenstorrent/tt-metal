# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end WER, one gate per language.

Every other gate on this model is numerical -- PCC against the fp32 reference, block by block. This
one asks the question a listener actually has: does the audio say the sentence? The full request
path runs (prompt assembly, backbone, flow matching, codec), the waveform is transcribed back with
Whisper, and the words are scored.

Gated PER LANGUAGE rather than pooled, because the RECOGNISER's error rate is not language
independent: a bar that fits English fails the languages Whisper transcribes less well, through no
fault of the device. Pooling also hides direction -- a language collapsing entirely moves a pooled
mean by a fraction of a word.

English carries the most runs (every English voice x five sentences). The other languages run every
voice they have, which on this model is one or two, so their means rest on fewer draws and their
ceilings are correspondingly looser.

Sentences are the sibling xtts_v2 port's, unchanged, so the two ports' WER numbers compare directly.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_wer_languages.py           # ~30 min
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

ASR_MODEL = "openai/whisper-large-v3"  # small hallucinates on short audio and is weak outside en
ASR_SR = 16000
OUTPUT_SR = 24000
# One draw per (voice, sentence) is enough where the audio is stable, which is every cell whose runs
# come back perfect. It is NOT enough for a cell that free-runs long: hi/long generates ~45-60 s per
# utterance and its trajectory moves with the seed, so hi_male reads 0.3419 / 0.0513 / 0.0342 /
# 0.0342 / 0.2051 over seeds 0-4 -- a 10x spread in which one draw would decide half the cell's mean.
# Averaging three seeds there makes the statistic survive a reshuffle, which any numerics change
# causes. Listed per cell rather than per band because the other eight long cells measure
# 0.0000-0.0119 with every run perfect, so extra draws buy nothing there and cost ~48 min a run.
SEEDS = {("ar", "long"): (0, 1, 2), ("hi", "long"): (0, 1, 2)}
DEFAULT_SEEDS = (0,)
FULL_SWEEP_LANG = "en"

# Past this a run did not say the sentence, so it is counted rather than averaged -- a mean over ten
# runs barely moves when one collapses. Uniform across languages and bands: the worst single medium
# run across 100 was 0.135, so this keeps real headroom on the noisiest cell.
COLLAPSE = 0.30

# Ceiling per (language, band), from each cell's THREE-SEED mean -- 460 runs, whisper-large-v3. The
# table and the run that produced it are in VOXTRAL_TTS_STATUS.md.
#
# Derived from three seeds even though most cells GATE on one, because a single draw is a biased
# estimate of the cell it is judging: seed 0 flattered nl/medium (0.0091 against a true 0.0146, which
# left only 1.3x headroom at seed 1) and maligned ar/medium (0.0274 against 0.0173). Re-deriving from
# the 3-seed mean tightened two cells, loosened three and left twenty-two alone.
#
# Seed count per cell is set by MEASURED spread, not by band -- see SEEDS. 18 of 27 cells move by
# less than a fifth of their ceiling across seeds; those keep one seed and their tight ceilings.
# Seven move 20-50% and are listed below as watch. Two exceed 50% and carry three seeds.
#
# BANDED RATHER THAN POOLED because the bands are not comparable. A five-word sentence quantises to
# multiples of 0.20, so its rate is coarse however good the audio is, and pooling it with a 115-word
# passage lets the short band set the language's number.
#
# ~3x the cell's measured mean, floored per band because coarser bands need more room: one wrong
# word is 0.20 of a five-word sentence and 0.01 of a hundred-word one.
CEILINGS = {
    # ("lang", "band"): ceiling.   Rule: min(max(3 x measured mean, per-band floor), COLLAPSE).
    #
    # Capping at COLLAPSE matters. The 3x rule alone wanted 0.59 for ("hi", "long"), and a cell
    # whose MEAN may sit at 0.59 would pass with every one of its runs past the line at which we
    # declare the sentence unsaid. A ceiling above COLLAPSE is not a gate.
    #
    # Per-band floors, because one wrong word is a different rate in each band: 0.20 of a five-word
    # sentence, ~0.03 of a thirty-word one, ~0.01 of a hundred-word one. The short floor of 0.25
    # therefore buys exactly "one word may go"; two makes 0.40 and fails.
    #
    #   band          floor
    #   short         0.25
    #   medium        0.03
    #   long          0.02
    #   voice_sweep   0.03
    #
    # WATCH LIST -- these move 20-50% of their ceiling across seeds, so a failure here is worth
    # re-running across seeds before it is believed: ar/medium, de/medium, fr/long, hi/short,
    # it/medium, nl/medium, nl/long. nl/medium is the closest to its ceiling of the seven.
    ("ar", "short"): 0.25,          # measured 0.0000
    ("ar", "medium"): 0.05,        # 3-seed 0.0173 (seed0 read 0.0274)
    ("ar", "long"): 0.07,          # 3-seed 0.0238 (seed0 read 0.0119) -- 3 seeds, see SEEDS
    ("de", "short"): 0.25,          # measured 0.0000
    ("de", "medium"): 0.03,         # measured 0.0000
    ("de", "long"): 0.02,           # measured 0.0000
    ("en", "short"): 0.25,          # measured 0.0000
    ("en", "medium"): 0.03,         # measured 0.0010
    ("en", "long"): 0.02,           # measured 0.0000  (5/5 word-perfect, ~445 frames)
    ("es", "short"): 0.25,          # measured 0.0000
    ("es", "medium"): 0.03,         # measured 0.0000
    ("es", "long"): 0.02,           # measured 0.0000
    ("fr", "short"): 0.25,          # measured 0.0000
    ("fr", "medium"): 0.03,         # measured 0.0000
    ("fr", "long"): 0.02,           # measured 0.0049
    ("hi", "short"): 0.25,          # measured 0.0000
    ("hi", "medium"): 0.16,        # 3-seed 0.0533 (seed0 read 0.0615)
    # THE WEAKEST CELL, and the only one whose ceiling is the cap rather than 3x its mean. Measured
    # 0.1966: the mean of hi_female 0.0513 and hi_male 0.3419 on ONE seed each.
    #
    # THAT 0.3419 IS A TRAJECTORY, NOT A DEFECT, and the numbers to check before chasing it: hi_male
    # over seeds 0-4 reads 0.3419 / 0.0513 / 0.0342 / 0.0342 / 0.2051 -- a 10x spread, of which the
    # gate's seed 0 is the worst draw. The fp32 CPU reference reads 0.0256 on the same text and
    # voice, next to the device's BEST seeds, and hi_female matches the reference exactly (0.0513
    # both). So the device tracks the reference; one free-running path happened to degrade.
    #
    # Over-generation IS the model: the reference makes 736 frames (58.9 s) for a passage that
    # should run ~30 s, MORE than the device's 671. hi_male is unstable on both axes (frames
    # 573-748, +/-15%) where hi_female is not (510-523, +/-1.3%).
    #
    # The cell is reproducible -- fixed seed, deterministic device -- so it works as a regression
    # detector, but it is SENSITIVE: any numerics change reshuffles the trajectory and this WER can
    # move 0.03 -> 0.34 with nothing wrong. A failure here means re-run across seeds before
    # believing it.
    # Three seeds (see SEEDS): 0.1011 over 6 runs, against 0.1966 when one seed's bad draw carried
    # half the mean. The cap and the 3x rule now agree -- 3 x 0.1011 = 0.303 -- so this ceiling is
    # no longer the cap rescuing an unstable statistic. Degenerate 1 of 6, inside the limit of 2.
    ("hi", "long"): 0.30,           # measured 0.1011 over 3 seeds
    ("it", "short"): 0.25,          # measured 0.0000
    ("it", "medium"): 0.03,         # measured 0.0105
    ("it", "long"): 0.02,           # measured 0.0000
    ("nl", "short"): 0.25,          # measured 0.0000
    ("nl", "medium"): 0.04,        # 3-seed 0.0146 (seed0 read 0.0091) -- closest to its ceiling
    ("nl", "long"): 0.04,          # 3-seed 0.0127 (seed0 read 0.0095)
    ("pt", "short"): 0.25,          # measured 0.0000
    ("pt", "medium"): 0.03,         # measured 0.0077
    ("pt", "long"): 0.02,           # measured 0.0000
    ("en", "voice_sweep"): 0.03,    # measured 0.0032, 38/40 perfect, worst single 0.091
}

VOICE_SWEEP_LANG = "en"
VOICE_SWEEP_SENTENCES = 2  # breadth over voices, not depth over sentences

MAX_DEGENERATE = 2
# generate() stops on [END_AUDIO]; hitting the frame cap instead means the model never closed the
# utterance. WER hears nothing of a missing tail, so it is asserted separately.
MAX_NON_TERMINATING = 2

LANG_NAMES = {"en": "english", "de": "german", "fr": "french", "es": "spanish", "it": "italian",
              "pt": "portuguese", "nl": "dutch", "hi": "hindi", "ar": "arabic"}

pytestmark = pytest.mark.skipif(not os.path.exists(DEFAULT_CKPT),
                                reason=f"no checkpoint at {DEFAULT_CKPT}")

# --------------------------------------------------------------------------------------- metric

# Optional orthography: marks a writer may or may not put in and the ASR usually leaves out. Both
# spellings are correct and sound identical, so scoring them as substitutions charges the model for
# errors it did not make. Every rule is scoped to one script, so Latin text is provably untouched.
_DROP_MARKS = {
    "़",                                  # Devanagari nukta: तेज़ and तेज are one word
    *(chr(c) for c in range(0x64B, 0x656)),    # Arabic harakat, madda, hamza -- omitted in prose
}
_FOLD_CHARS = str.maketrans({"ँ": "ं", "ة": "ه", "ى": "ي"})
#                            chandrabindu->anusvara, ta marbuta->ha, alef maqsura->ya


def _words(s):
    """Casefold, drop punctuation and fold optional orthography, in ANY script.

    An ASCII-only filter looks equivalent and is not: it empties Arabic and Devanagari entirely, so
    both sides normalise to nothing and every comparison scores a free zero. Devanagari vowels are
    combining marks, so dropping marks as punctuation cuts words into fragments -- identical text
    still scores 0.000 because both sides fragment alike, which is why the metric is tested on text
    that DIFFERS.
    """
    flat = s.casefold().replace("’", "'").replace("ʼ", "'")  # ASR emits curly apostrophes
    flat = unicodedata.normalize("NFD", flat)                          # expose precomposed marks
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
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    return d[-1][-1] / max(len(ref), 1)


@pytest.mark.parametrize("ref,hyp,exp", [
    ("the cat sat down", "the cat sat down", 0.0),
    ("the cat sat down", "the dog sat down", 0.25),          # substitution
    ("the cat sat down", "the cat down", 0.25),              # deletion
    ("the cat sat down", "the cat sat right down", 0.25),    # insertion
    ("the cat sat down", "", 1.0),                           # nothing transcribed
    ("The cat, sat down!", "the cat sat down", 0.0),         # punctuation and case ignored
    # non-Latin scripts must survive normalisation rather than emptying to a free zero
    ("नमस्ते दुनिया", "नमस्ते दुनिया", 0.0),
    ("नमस्ते दुनिया", "नमस्ते चाँद", 0.5),
    ("मुझे यह किताब बहुत पसंद है", "मुझे यह किताब बहुत अच्छी है", 1 / 6),
    ("سوق الشتاء مبكرا اليوم", "سوق الصيف مبكرا اليوم", 0.25),
    # optional orthography folds: the same words spelled two legal ways score 0
    ("तेज़ हवा चली", "तेज हवा चली", 0.0),
    ("مرحبا كيف حالك", "مَرْحَبا كيف حالك", 0.0),
])
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
    """~18 chars/s of speech at 12.5 frames/s, x2.2 margin, floor 320 -- the generator's own rule.

    A CAP, not a cost: generation stops on [END_AUDIO] regardless, so over-provisioning is free
    while under-provisioning silently fakes a non-terminating model.
    """
    return max(320, int(math.ceil(len(text) / 18.0 * 12.5 * 2.2)))


class Asr:
    """Whisper on CPU, greedy so the transcript is reproducible."""

    def __init__(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.proc = WhisperProcessor.from_pretrained(ASR_MODEL)
        # the large checkpoints ship fp16, which cannot run against fp32 features on CPU
        self.model = WhisperForConditionalGeneration.from_pretrained(
            ASR_MODEL, torch_dtype=torch.float32).eval()

    def __call__(self, wav, lang):
        audio = wav.reshape(1, -1)
        n = int(audio.shape[1] * ASR_SR / OUTPUT_SR)
        audio = torch.nn.functional.interpolate(audio.unsqueeze(0), size=n, mode="linear",
                                                align_corners=False).squeeze(0)
        # PAST 30 s WHISPER MUST BE ASKED FOR LONG FORM. The processor's default pads or TRUNCATES
        # to 3000 mel frames, which is exactly 30 s, and says nothing when it cuts: a 36 s utterance
        # transcribes to its first 30 s and the rest scores as deletions. That reads as the model
        # losing the thread on long input -- it measured WER 0.245 on a passage that is actually
        # word-perfect. Under 30 s the default padding is required instead, because the encoder
        # wants the full 3000 frames.
        long_form = n > 30 * ASR_SR
        kw = ({"truncation": False, "padding": "longest", "return_attention_mask": True}
              if long_form else {})
        inp = self.proc(audio[0].numpy(), sampling_rate=ASR_SR, return_tensors="pt", **kw)
        gen = {"language": lang, "task": "transcribe", "do_sample": False, "num_beams": 1}
        if long_form:
            gen |= {"attention_mask": inp.attention_mask, "return_timestamps": True}
        with torch.no_grad():
            # told the language, not left to detect it: detection on a few seconds is unreliable
            # and a wrong guess transcribes into the wrong script entirely
            ids = self.model.generate(inp.input_features, **gen)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def voices_for(lang, all_voices):
    """Every voice of this language. English is the full sweep; the others have one or two."""
    return tuple(v for v in all_voices if lang_of(v) == lang)


def run_language(lang, asr, pipe, band="medium", voices=None, max_sentences=None,
                 collapse=COLLAPSE, seeds_override=None, verbose=True):
    """One (language, band) voice x sentence x seed matrix -> stats dict.

    No assertions, so the measurement probe and the gate share one code path.
    """
    from models.experimental.voxtral_tts.tests.reference_helpers import all_voices, corpus_embeds

    voices = voices if voices is not None else voices_for(lang, all_voices())
    texts = wer_band(lang, band)[:max_sentences]   # a wide voice sweep pays breadth, not depth
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
            print(f"  {lang}/{band:<6} {voice:18s} " + " ".join(f"{w:.3f}" for w in row)
                  + f"   mean {sum(row) / len(row):.4f}", flush=True)
    vals = list(scores.values())
    return {
        "lang": lang, "band": band, "n_voices": len(voices), "n_runs": len(vals),
        "mean": sum(vals) / len(vals), "worst": max(vals),
        "perfect": sum(1 for w in vals if w == 0),
        "degenerate": [f"{v}/s{i}/seed{sd}" for (v, i, sd), w in scores.items()
                       if w >= collapse],
        "non_terminating": non_terminating, "collapse": collapse,
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
# The default 300 s covers a two-voice medium cell and not the long band, where one utterance is
# ~36 s of audio and Whisper pays two 30 s windows for it.
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
    print(f"\n  {lang}/{band}: {s['n_voices']} voices x {per} sentences x {n_seeds} seed(s), "
          f"WER {s['mean']:.4f} (worst {s['worst']:.3f}, perfect {s['perfect']}/{s['n_runs']}, "
          f"degenerate {len(s['degenerate'])}, non-terminating {len(s['non_terminating'])}) "
          f"ceiling {ceiling}", flush=True)
    assert s["mean"] <= ceiling, (
        f"{lang}/{band}: mean WER {s['mean']:.4f} over {s['n_runs']} runs above ceiling {ceiling}")
    assert len(s["degenerate"]) <= MAX_DEGENERATE, (
        f"{lang}/{band}: {len(s['degenerate'])} runs at or above WER {COLLAPSE} "
        f"(limit {MAX_DEGENERATE}): {s['degenerate']}")
    assert len(s["non_terminating"]) <= MAX_NON_TERMINATING, (
        f"{lang}/{band}: {len(s['non_terminating'])} runs hit the frame cap without [END_AUDIO] "
        f"(limit {MAX_NON_TERMINATING}): {s['non_terminating']}")


# Every voice, not just the English ones. MEASURED: English survives a foreign preset -- 14 of the
# 15 non-English presets transcribed an English sentence at WER 0.000 and the fifteenth at 0.091,
# against 0.000 for the five English presets. The other languages do NOT survive it (French 0.0000
# native against 0.0476 borrowed, Hindi 0.0556 against 0.3148, with one borrowed run at 0.444 --
# past COLLAPSE on a healthy build), so this sweep is English-only by measurement, not by taste.
# Sweeping them too would gate cross-lingual preset transfer, which is a model property.
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_wer_every_voice_english(rig):
    """All twenty presets on English, so a defect in one voice's prompt geometry is audible here.

    Voice breadth and length breadth are crossed separately on purpose: the full matrix is twenty
    voices x nine languages x three bands, which measures little that these two do not and costs
    hours.
    """
    from models.experimental.voxtral_tts.tests.reference_helpers import all_voices

    asr, pipe = rig
    voices = tuple(all_voices())
    s = run_language(VOICE_SWEEP_LANG, asr, pipe, band="medium", voices=voices,
                     max_sentences=VOICE_SWEEP_SENTENCES)
    # one ceiling for the sweep, looser than the native-voice cell: fifteen of these presets are
    # carrying a language other than their own
    ceiling = CEILINGS[("en", "voice_sweep")]
    print(f"\n  en/voice_sweep: {len(voices)} voices, WER {s['mean']:.4f} "
          f"(worst {s['worst']:.3f}, perfect {s['perfect']}/{s['n_runs']}, "
          f"degenerate {len(s['degenerate'])}) ceiling {ceiling}", flush=True)
    assert s["mean"] <= ceiling, (
        f"en/voice_sweep: mean WER {s['mean']:.4f} over {s['n_runs']} runs above {ceiling}")
    assert len(s["degenerate"]) <= MAX_DEGENERATE, (
        f"en/voice_sweep: {len(s['degenerate'])} collapsed runs: {s['degenerate']}")
    assert len(s["non_terminating"]) <= MAX_NON_TERMINATING, (
        f"en/voice_sweep: {len(s['non_terminating'])} hit the frame cap: {s['non_terminating']}")

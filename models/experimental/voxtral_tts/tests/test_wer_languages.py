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
from models.experimental.voxtral_tts.tests.sentence_corpus import WER_SENTENCES, lang_of  # noqa: E402

ASR_MODEL = "openai/whisper-large-v3"  # small hallucinates on short audio and is weak outside en
ASR_SR = 16000
OUTPUT_SR = 24000
SEED = 0
FULL_SWEEP_LANG = "en"

# (ceiling, collapse) per language, MEASURED on this branch: 100 runs, whisper-large-v3, seed 0.
#
#         mean    worst  perfect   ceiling
#   en  0.0010    0.024    24/25      0.02
#   de  0.0000    0.000    10/10      0.03
#   es  0.0000    0.000    10/10      0.03
#   fr  0.0000    0.000    10/10      0.03
#   pt  0.0077    0.050     8/10      0.03
#   nl  0.0091    0.045     8/10      0.03
#   it  0.0105    0.067     8/10      0.04
#   ar  0.0274    0.087      3/5      0.09
#   hi  0.0615    0.135     1/10      0.20
#
# ceiling  -- ~3x the language's measured mean, floored at 0.03 so one unlucky draw cannot breach a
#             language that is currently perfect: with ten runs a single collapse to WER 0.10 moves
#             the mean to 0.010, so the floor absorbs three of them. English keeps 0.02 on 25 runs.
#             These are TIGHTER than the sibling xtts_v2 port's for pt, nl and it, because this
#             model measures better on them -- do not copy that table back over this one.
# collapse -- past this a run did not say the sentence, so it is counted rather than averaged; a
#             mean over ten runs barely moves when one collapses. Uniform 0.30: the worst single
#             run across all 100 was 0.135, so this has 2.2x headroom on the noisiest language and
#             does not need xtts_v2's per-language tiering.
LIMITS = {
    "en": (0.02, 0.30),
    "de": (0.03, 0.30),
    "es": (0.03, 0.30),
    "fr": (0.03, 0.30),
    "pt": (0.03, 0.30),
    "nl": (0.03, 0.30),
    "it": (0.04, 0.30),
    "ar": (0.09, 0.30),
    "hi": (0.20, 0.30),
}
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


def test_every_language_is_gated():
    """A language in the corpus with no LIMITS entry would raise mid-run instead of being gated."""
    assert sorted(WER_SENTENCES) == sorted(LIMITS), \
        f"corpus {sorted(WER_SENTENCES)} != gated {sorted(LIMITS)}"


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
        feats = self.proc(audio[0].numpy(), sampling_rate=ASR_SR,
                          return_tensors="pt").input_features
        with torch.no_grad():
            # told the language, not left to detect it: detection on a few seconds is unreliable
            # and a wrong guess transcribes into the wrong script entirely
            ids = self.model.generate(feats, language=lang, task="transcribe",
                                      do_sample=False, num_beams=1)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def voices_for(lang, all_voices):
    """Every voice of this language. English is the full sweep; the others have one or two."""
    return tuple(v for v in all_voices if lang_of(v) == lang)


def run_language(lang, asr, pipe, verbose=True):
    """One language's voice x sentence matrix -> stats dict. No assertions, so a probe can reuse it."""
    from models.experimental.voxtral_tts.tests.reference_helpers import all_voices, corpus_embeds

    voices = voices_for(lang, all_voices())
    scores, non_terminating = {}, []
    for voice in voices:
        for si, text in enumerate(WER_SENTENCES[lang]):
            cap = frame_budget(text)
            embeds = corpus_embeds(text, voice, pipe.wb)
            pipe.backbone.reset()
            frames, _, _ = pipe.generate(embeds, max_frames=cap, seed=SEED, verbose=False)
            if len(frames) >= cap:
                non_terminating.append(f"{voice}/s{si}")
            scores[(voice, si)] = wer(text, asr(pipe.decode(frames), lang))
        if verbose:
            row = [scores[(voice, i)] for i in range(len(WER_SENTENCES[lang]))]
            print(f"  {lang}  {voice:18s} " + " ".join(f"{w:.3f}" for w in row)
                  + f"   mean {sum(row) / len(row):.4f}", flush=True)
    vals = list(scores.values())
    ceiling, collapse = LIMITS[lang]
    return {
        "lang": lang, "n_voices": len(voices), "n_runs": len(vals),
        "mean": sum(vals) / len(vals), "worst": max(vals),
        "perfect": sum(1 for w in vals if w == 0),
        "degenerate": [f"{v}/s{i}" for (v, i), w in scores.items() if w >= collapse],
        "non_terminating": non_terminating, "ceiling": ceiling, "collapse": collapse,
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
# The default 300 s covers a two-voice language and not the English sweep's 25 runs, each of which
# generates an utterance and then transcribes it.
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("lang", sorted(WER_SENTENCES), ids=lambda l: LANG_NAMES[l])
def test_wer_per_language(rig, lang):
    """This language's own ceiling, collapse count and termination, so a failure names the language."""
    asr, pipe = rig
    s = run_language(lang, asr, pipe)
    limit = MAX_NON_TERMINATING
    print(f"\n  {lang}: {s['n_voices']} voices x {len(WER_SENTENCES[lang])} sentences, "
          f"WER {s['mean']:.4f} (worst {s['worst']:.3f}, perfect {s['perfect']}/{s['n_runs']}, "
          f"degenerate {len(s['degenerate'])}, non-terminating {len(s['non_terminating'])}) "
          f"ceiling {s['ceiling']}", flush=True)
    assert s["mean"] <= s["ceiling"], (
        f"{lang}: mean WER {s['mean']:.4f} over {s['n_runs']} runs above ceiling {s['ceiling']}")
    assert len(s["degenerate"]) <= MAX_DEGENERATE, (
        f"{lang}: {len(s['degenerate'])} runs at or above WER {s['collapse']} "
        f"(limit {MAX_DEGENERATE}): {s['degenerate']}")
    assert len(s["non_terminating"]) <= limit, (
        f"{lang}: {len(s['non_terminating'])} runs hit the frame cap without [END_AUDIO] "
        f"(limit {limit}): {s['non_terminating']}")

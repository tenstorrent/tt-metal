# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""WER test: does the device actually say the words?

Every other gate compares numbers, so none of them can see whether the audio is intelligible — the
audio code cap cutting a sentence short, degeneration on repeated input, and audio running past the
last word are invisible to all of them.

The model free-runs, so every draw realises the sentence differently and there is no canonical
transcript to diff against. WER is a metric on (audio, source text), so each run is scored against
the text it was given — invariant to how that particular draw happened to say it.

The voice comes from the checkpoint's own speakers_xtts.pth, one entry per built-in studio speaker,
rather than from a reference clip. That is not just convenience: conditioning on a synthetic
waveform puts the speaker encoder outside its training distribution, and the decoder can answer by
emitting non-speech for a whole utterance, or by running on past the end of the sentence. Built-in
latents avoid both. It also means this test does no DSP and no Block 1 or 2 work: the voice arrives
as tensors, so a failure here is prefill, decode or the vocoder.

Every speaker is swept, because the speaker embedding is the axis that produced that failure and
nothing else covered it. It also makes the mean robust — with this many samples one catastrophic
draw barely moves it, so the ceiling can stay tight enough to catch a rising failure RATE, which a
median would hide.

The slowest test in the suite. Needs the Whisper weights (cached or downloadable).

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_wer.py
"""
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from models.experimental.xtts_v2.frontend import sinc_resample
from models.experimental.xtts_v2.tt.ttnn_xtts_model import OUTPUT_SR, Voice, XttsV2

ASR_MODEL = "openai/whisper-large-v3"  # small hallucinates on short audio and is weak outside en
ASR_SR = 16000
SEED = 0  # one seed: every speaker/sentence pair is already a distinct draw
SPEAKERS_FILE = "speakers_xtts.pth"  # ships beside model.pth, the way vocab.json does
MAX_WER = 0.02  # several times the corpus baseline; the spread is in the bringup docs
# A third of the words wrong is far past anything mishearing explains -- ordinary ASR error costs a
# word or two. A run this bad did not say the sentence: non-speech, repetition, or babble running
# past the end.
DEGENERATE_WER = 0.3
# A mean over this many runs barely moves when a few collapse, so the collapse count is asserted
# directly rather than left to the average to reveal.
MAX_DEGENERATE = 2

# Long sentences, where the model is dependable, and nothing Whisper respells or renumbers
# ("harbour" -> "harbor", "nine" -> "9"), since that scores as an error without being one.
SENTENCES = (
    "The old map showed three islands that no sailor had ever found, and nobody wanted to be the "
    "first to erase them.",
    "Warm bread and strong coffee filled the small kitchen long before the sun came up, and the "
    "kettle went on again the moment the first pot stood empty.",
    "The archive filled four floors of a building designed for something else entirely, and every "
    "year the collection pressed a little harder against the walls until the shelves reached the "
    "ceiling.",
    "The bakery on the corner opens before dawn, and by six the whole street smells of warm bread, "
    "which is why the regulars arrive in the same order every morning without ever arranging it.",
    "Long before the railway reached the valley, the mail came over the mountain pass on horseback, "
    "and the timetable was a matter of weather rather than clocks, so the villagers learned to read "
    "the sky the way other people read a schedule.",
)


def _words(s):
    """Casefold and drop punctuation, keeping letters and digits in ANY script.

    WER on raw text is dominated by commas and capitals. An ASCII-only filter looks equivalent and
    is not: it empties Arabic, Cyrillic and Devanagari completely, so both sides normalise to
    nothing and every comparison scores a free zero."""
    flat = s.casefold().replace("\u2019", "'").replace("\u02bc", "'")  # ASR emits curly apostrophes
    return "".join(c if c.isalnum() or c.isspace() or c == "'" else " " for c in flat).split()


def _wer(reference, hypothesis):
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


class _Asr:
    """Whisper on CPU, greedy so the transcript is reproducible."""

    def __init__(self):
        self.proc = WhisperProcessor.from_pretrained(ASR_MODEL)
        # the large checkpoints ship fp16, which cannot run against fp32 features on CPU
        self.model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL, torch_dtype=torch.float32).eval()

    def __call__(self, wav):
        # The model's own resampler rather than a new dependency: a fault in it would raise the
        # score, which the ceiling catches.
        audio = sinc_resample(wav.reshape(1, -1), OUTPUT_SR, ASR_SR)
        feats = self.proc(audio[0].numpy(), sampling_rate=ASR_SR, return_tensors="pt").input_features
        with torch.no_grad():
            ids = self.model.generate(feats, language="en", task="transcribe", do_sample=False, num_beams=1)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _speakers(ckpt_path):
    """The checkpoint's built-in studio speakers -> {name: Voice}. Latents, so no DSP is involved."""
    import os

    raw = torch.load(os.path.join(os.path.dirname(ckpt_path), SPEAKERS_FILE), weights_only=False)
    return {
        name: Voice(gpt_cond_latent=d["gpt_cond_latent"], speaker_embedding=d["speaker_embedding"])
        for name, d in raw.items()
    }


def run_wer(verbose=True):
    asr = _Asr()
    tts = XttsV2()
    try:
        voices = _speakers(tts.ckpt_path)
        tts.warmup()
        device = {}
        for name, voice in voices.items():
            for text in SENTENCES:
                device[(name, text)] = asr(tts.generate(text, voice, seed=SEED))
            if verbose:  # per speaker, so a long run shows progress and a failure names the speaker
                row = [_wer(t, device[(name, t)]) for t in SENTENCES]
                print(f"  {name:24s} " + " ".join(f"{w:.3f}" for w in row) + f"  mean {sum(row) / len(row):.3f}")
    finally:
        tts.close()

    dev_scores = {k: _wer(k[1], t) for k, t in device.items()}
    dev_wer = sum(dev_scores.values()) / len(dev_scores)
    degenerate = [f"{n}/{t[:24]}" for (n, t), w in dev_scores.items() if w >= DEGENERATE_WER]
    msg = (
        f"{len(voices)} speakers x {len(SENTENCES)} sentences: device WER {dev_wer:.4f} "
        f"(worst single {max(dev_scores.values()):.3f}, degenerate {len(degenerate)}) "
        f"ceiling {MAX_WER}, degenerate limit {MAX_DEGENERATE} at WER {DEGENERATE_WER}"
    )
    return (dev_wer <= MAX_WER and len(degenerate) <= MAX_DEGENERATE), msg


def test_wer_metric():
    """The measuring instrument, checked before it is used to judge anything. Standard WER, so an
    over-long transcript can score above 1.0."""
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
        ("Grüße, Welt!", "grüße welt", 0.0),
        ("it doesn't matter", "it doesn\u2019t matter", 0.0),  # curly vs straight apostrophe
    )
    for reference, hypothesis, want in cases:
        assert _wer(reference, hypothesis) == want, f"WER({reference!r}, {hypothesis!r})"


def test_wer():
    passed, msg = run_wer()
    assert passed, f"the device is past the WER ceiling or produced degenerate audio: {msg}"


if __name__ == "__main__":
    import sys

    ok, msg = run_wer()
    print("\n" + ("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)

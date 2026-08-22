# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""WER test: does the device say the words, and say them as well as the CPU does?

Every other gate compares numbers, so none of them can see whether the audio is intelligible — the
605-code cap cutting a sentence short, degeneration on repeated input, and audio running past the
last word are invisible to all of them.

Both sides free-run, so they sample different code sequences and say the same sentence in different
ways. Comparing their transcripts to EACH OTHER would measure that divergence, not correctness. WER
is a metric on (audio, source text), so each side is scored against the text it was given and the
two SCORES are compared — invariant to how either one realised the sentence.

Sampling variance is larger than any regression worth catching, so scores are averaged over
sentences AND seeds, and the sentences run long. Short utterances are not merely noisier but
unusable as a gate: a six-word sentence measured 0.000, 0.667 and 0.000 across three seeds — a
healthy device wandering, because short text with a synthetic reference leaves the model too little
to hold on to. A ceiling loose enough to survive that would sit above every regression this exists
to catch.

So the absolute ceiling is the gate that bites. The CPU score sits beside it as a diagnostic,
separating "the model got worse" from "the device got worse"; it is not a stable measure of what a
sentence is worth, because the CPU free-runs too.

Needs Whisper weights (cached or downloadable). The CPU pipeline dominates the runtime.

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_wer.py
"""
import torch
from transformers import GPT2Model, WhisperForConditionalGeneration, WhisperProcessor  # noqa: F401

from models.experimental.xtts_v2.frontend import (
    assemble_prompt,
    conditioning_mels,
    sinc_resample,
    speaker_logmel,
)
from models.experimental.xtts_v2.reference.xtts_cond_ref import CondReference
from models.experimental.xtts_v2.reference.xtts_gpt_ref import build_reference
from models.experimental.xtts_v2.reference.xtts_hifigan_ref import HifiganReference
from models.experimental.xtts_v2.reference.xtts_speaker_ref import SpeakerReference
from models.experimental.xtts_v2.tests.reference_helpers import synthetic_speech
from models.experimental.xtts_v2.tt.ttnn_xtts_model import (
    GPT_MAX_AUDIO,
    HOP,
    OUTPUT_SR,
    START_AUDIO_TOKEN,
    STOP_AUDIO_TOKEN,
    XttsV2,
    _fade_out,
    _sample_token,
    _voc_bucket,
    _voc_input,
    _voc_pad,
)

ASR_MODEL = "openai/whisper-small"  # multilingual, so the same rig extends past English
ASR_SR = 16000
TOLERANCE = 0.05  # how much worse than the CPU the device may score, averaged over the corpus
MAX_WER = 0.10  # the gate that bites: several times the corpus baseline, well under a real fault
SEEDS = (0, 1, 2)  # each draws a different code path; the gate averages over them

# 22 to 42 words: the regime where the model is dependable (see the module docstring). Two further
# constraints, both learned by measuring: nothing Whisper respells or renumbers ("harbour" ->
# "harbor", "nine" -> "9") scores as an error without being one, and gendered pronouns come out
# ambiguous from the synthetic reference voice.
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
    nothing and every comparison scores a perfect 0.000."""
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
        self.model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL).eval()

    def __call__(self, wav):
        # The model's own resampler rather than a new dependency: a fault in it would move both
        # sides equally, and the absolute ceiling still catches it.
        audio = sinc_resample(wav.reshape(1, -1), OUTPUT_SR, ASR_SR)
        feats = self.proc(audio[0].numpy(), sampling_rate=ASR_SR, return_tensors="pt").input_features
        with torch.no_grad():
            ids = self.model.generate(feats, language="en", task="transcribe", do_sample=False, num_beams=1)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _cpu_generate(host, gpt, final_norm, voc, cond, spk, text, seed):
    """XttsV2.generate on CPU: prompt, sampled decode with a KV cache, then the reference vocoder.
    `host` is (tokenizer, tables, heads) — the pieces the shipped path also keeps off device."""
    tokenizer, tables, h = host
    prefix = assemble_prompt(tokenizer.encode(text, "en"), cond, tables)
    gen = torch.Generator().manual_seed(seed)
    step = (h["mel_emb"][START_AUDIO_TOKEN] + h["mel_pos"][0]).view(1, 1, -1)
    with torch.no_grad():
        out = gpt(inputs_embeds=torch.cat([prefix, step], dim=1), use_cache=True)
        past, last = out.past_key_values, final_norm(out.last_hidden_state[:, -1:])
        seen, codes, latents = {1, START_AUDIO_TOKEN}, [], []
        while len(codes) < GPT_MAX_AUDIO:
            nxt = _sample_token(last, seen, gen, h["mel_head_w"], h["mel_head_b"])
            if nxt == STOP_AUDIO_TOKEN:
                break
            codes.append(nxt)
            latents.append(last)
            seen.add(nxt)
            step = (h["mel_emb"][nxt] + h["mel_pos"][len(codes)]).view(1, 1, -1)
            out = gpt(inputs_embeds=step, past_key_values=past, use_cache=True)
            past, last = out.past_key_values, final_norm(out.last_hidden_state[:, -1:])
    if not codes:  # generate's empty-audio contract
        return torch.zeros(1, 1, 0)
    z = _voc_input(torch.cat(latents, dim=1))
    L = z.shape[-1]
    return _fade_out(voc(_voc_pad(z, _voc_bucket(L)), spk)[:, :, : L * HOP])


def run_wer(verbose=True):
    ref_wav, ref_sr = synthetic_speech(), 22050  # a clip's content only has to be voiced
    asr = _Asr()
    tts = XttsV2()
    try:
        tts.warmup()
        voice = tts.compute_voice(ref_wav, ref_sr)
        device = {(t, s): asr(tts.generate(t, voice, seed=s)) for t in SENTENCES for s in SEEDS}
        host = (tts.tokenizer, tts.tables, tts.heads)  # host-side pieces, kept past close()
    finally:
        tts.close()

    # The CPU baseline computes its own conditioning and speaker embedding, so it is a whole
    # pipeline rather than a device run with the tail replaced.
    gpt, final_norm = build_reference()
    mel = conditioning_mels(ref_wav, ref_sr, host[1].mel_stats)[0]
    _, cond = CondReference().get_style_emb(mel)
    spk = SpeakerReference().core(speaker_logmel(ref_wav, ref_sr), l2_norm=True).unsqueeze(-1)
    voc = HifiganReference()
    cpu = {(t, s): asr(_cpu_generate(host, gpt, final_norm, voc, cond, spk, t, s)) for t in SENTENCES for s in SEEDS}

    dev_scores = {k: _wer(k[0], v) for k, v in device.items()}
    cpu_scores = {k: _wer(k[0], v) for k, v in cpu.items()}
    if verbose:
        for i, text in enumerate(SENTENCES):
            d = [dev_scores[(text, s)] for s in SEEDS]
            c = [cpu_scores[(text, s)] for s in SEEDS]
            dev_col = " ".join(f"{w:.3f}" for w in d)
            cpu_col = " ".join(f"{w:.3f}" for w in c)
            print(f"  sentence {i} ({len(text.split()):2d} words)  device {dev_col}   cpu {cpu_col}")
            for s in SEEDS:  # only the misses are worth reading
                if dev_scores[(text, s)]:
                    print(f"      device seed{s}: {device[(text, s)]}")
    dev_wer = sum(dev_scores.values()) / len(dev_scores)
    cpu_wer = sum(cpu_scores.values()) / len(cpu_scores)
    worst = max(dev_scores.values())
    msg = (
        f"{len(SENTENCES)} sentences x {len(SEEDS)} seeds: device WER {dev_wer:.3f} "
        f"(worst single {worst:.3f}) vs cpu {cpu_wer:.3f}, ceiling {MAX_WER}"
    )
    return dev_wer <= cpu_wer + TOLERANCE and dev_wer <= MAX_WER, msg


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
        # non-Latin scripts must survive normalisation rather than emptying to a free 0.000
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
    assert passed, f"the device is less intelligible than the CPU, or past the ceiling: {msg}"


if __name__ == "__main__":
    import sys

    ok, msg = run_wer()
    print("\n" + ("PASSED " if ok else "FAILED ") + str(msg))
    sys.exit(0 if ok else 1)

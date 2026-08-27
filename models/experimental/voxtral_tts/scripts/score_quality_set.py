# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Score the generated audio set: ASR word error rate, plus a voice-identity check.

    python .../scripts/score_quality_set.py <results.json> [<more.json> ...]

WER is the only end-to-end number that means anything to a listener. The fp32 CPU reference scores
0.0% on natural text, so that is the bar -- not a PCC.

THREE MEASUREMENT TRAPS, ALL HIT DURING THIS WORK:

1. Whisper's encoder is a fixed 30 s window. The fixture paragraphs are 36-39 s, and a plain
   processor+generate call silently transcribes only the first 30 s, reporting the rest as
   deletions -- a ~20% WER floor that is pure harness. Long clips go through the `chunk_length_s`
   pipeline, which does overlapped long-form decoding.
2. Normalising with `[^a-z0-9' ]` erases Hindi and Arabic entirely and scores them 100% WER on
   perfect audio. Use the Unicode-category normaliser below.
3. A voice-name prefix is not a language. Fixture case 4 is English "Hello." spoken by `ar_male`;
   forcing Arabic decoding made Whisper hallucinate a filler word and report 100% WER on 1 word.
   Language comes from the TEXT, so it is detected here rather than taken from the voice name.

4. **A correct number reading scores as errors.** Fixture case 7 is "El niño comió 42 manzanas";
   the model says "cuarenta y dos", which is how you read that aloud, and Whisper writes the words.
   Against a reference containing "42" that is one deletion plus three insertions -- 50% WER on a
   6-word clip for behaviour that is right. Not corrected here (a multilingual number verbaliser is
   a dependency we do not have); just do not read such a case as a TTS defect without looking at
   the transcript.

Some fixture texts are deliberately adversarial (emoji, `!@#$%^&*()`, literal tab/newline escapes).
The model tries to VOCALISE those, which is reasonable behaviour but has no well-defined reference
transcript, so they are reported separately and excluded from the headline number.
"""

import json
import re
import sys
import unicodedata
import wave

import numpy as np
import torch

# Texts whose spoken form has no well-defined transcript -- scored, but not in the headline.
ADVERSARIAL = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]|[!@#$%^&*()]{3,}|\\[tn]|\t|\n")

# A THIRD bucket: prompts the MODEL is chaotic on, where a fixed reference transcript cannot
# measure an implementation. Currently just the one-word texts.
#
# Fixture case 4 is "Hello." -- one word of text carried on 74 prompt tokens, nearly all of it
# voice-preset conditioning. Free-running frame counts for it, 9 being correct:
#     fp32 CPU reference (no device, no precision loss)   81 /  8 / 57
#     tt_transformers + bf16 flow                          9 /  8 / 72 / 88 / 55 / 118
#     ours, several weight configurations                 45 /  8 / 39 / 69  ... 108 / 8 / 34 / 54
# Every implementation, including the pure-torch reference, lands anywhere from 8 to 183 frames
# depending only on the noise draw. The likely mechanism is that one word of text cannot compete
# with 74 tokens of voice conditioning at the last prefill position -- the ssinghal/voxtral_tts
# branch reports the same effect independently ("147 voice frames dilute text signal").
#
# Folding that into the headline is actively misleading: at 1 reference word a rambling take
# scores tens of thousands of percent and swamps 340 good words. It cost real debugging time
# twice, chasing a port defect that does not exist. Reported separately, like the adversarial set.
UNSTABLE_MAX_WORDS = 1

# A FOURTH split, inside the natural-text bucket, and the one that matters most for gating.
#
# MEASURED: with IDENTICAL CODE, changing only the generation seed, the natural-text headline
# lands on 1.76% / 2.06% / 0.88% (seeds 0/1/2). That is the ENTIRE range observed across every
# implementation variant tried in the optimization sweep, endpoints included -- so a single-seed
# headline cannot distinguish a code change from a reroll, and comparing two builds at one seed
# is worthless.
#
# The reason is the word counts, not the audio. Per case, same code, seeds 0/1/2:
#     case  7  spanish     6 words    50.0% /  0.0% /  0.0%
#     case  9  portuguese  6 words     0.0% / 83.3% /  0.0%
#     case 13  arabic      3 words    33.3% /  0.0% / 33.3%
#     case  6  german      7 words    28.6% / 28.6% / 28.6%   <- real, reproducible, not noise
#     cases 2,3 english  125 each      0.0% /  0.0% /  0.0%   <- 250 of the 340 words
# On a 6-word clip one Whisper disagreement is 17% and three is 50%, so 90 words of short prompts
# swing the 340-word aggregate by more than a full point while 250 words of English sit at zero.
#
# So the headline is split. LONG-FORM is the gate: it is 74% of the corpus, it has been 0.0% in
# every run ever measured, and a regression there is real. SHORT-PROMPT is reported next to it and
# is seed noise unless a case moves in the SAME direction across several seeds.
#
# For judging a numerical change (a fused matmul, a different norm, a new dtype), prefer the
# DETERMINISTIC gates instead -- teacher-forced PCC and worst-sample against the fp32 reference in
# tests/gates.py. Those feed both builds identical inputs, so no trajectory is involved. That
# is what actually caught the Block 1 sharded-norm regression (worst sample 1.06% -> 1.95%), while
# this metric said 2.06% for a change that was fine and 2.06% again for one that was not.
LONGFORM_MIN_WORDS = 20
# Scripts that pin the language without guessing.
SCRIPT_LANG = [("ऀॿ", "hindi"), ("؀ۿ", "arabic")]
# Latin-script languages still need a hint; take it from characteristic words in the fixture text.
# There is deliberately no `dutch` entry: the `nl_male` fixture text ("Tab\tand\nnewline handling.")
# is English, so it must be scored with the English model -- that is trap 3.
WORD_LANG = [("french", (" je ", " ça ", "bonjour", "appelle", "très")),
             ("german", ("grüße", "straße", "schön", " die ")),
             ("spanish", ("niño", "comió", "verdad", " el ")),
             ("italian", ("ciao", "però", "così", "vero")),
             ("portuguese", ("olá", "ação", "coração", "não"))]


def detect_lang(text):
    for rng, lang in SCRIPT_LANG:
        if any(rng[0] <= c <= rng[1] for c in text):
            return lang
    low = f" {text.lower()} "
    for lang, marks in WORD_LANG:
        if any(m in low for m in marks):
            return lang
    return None                                    # None -> English model


def read_wav(p):
    with wave.open(p, "rb") as f:
        a = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return a, f.getframerate()


def norm(s):
    """NFKC + casefold, drop punctuation/marks/symbols, collapse whitespace. Survives non-Latin
    scripts, unlike an [a-z0-9] class -- see trap 2."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(" " if unicodedata.category(c)[0] in "PZSC" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r), len(h)] / max(len(r), 1), len(r)


def transcribe(rows):
    """One ASR pipeline per (model, language); long clips use chunked long-form decoding."""
    from transformers import pipeline
    cache = {}

    for r in rows:
        lang = detect_lang(r["text"])
        model = "openai/whisper-small" if lang else "openai/whisper-base.en"
        long_form = r["audio_s"] > 28
        key = (model, long_form)
        if key not in cache:
            cache[key] = pipeline("automatic-speech-recognition", model=model,
                                  **({"chunk_length_s": 30, "stride_length_s": 5}
                                     if long_form else {}))
        a, sr = read_wav(r["wav"])
        if sr != 16000:
            import torchaudio
            a = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()
        kw = {"generate_kwargs": {"language": lang, "task": "transcribe"}} if lang else {}
        hyp = cache[key]({"raw": a.astype(np.float32), "sampling_rate": 16000}, **kw)["text"]
        yield r, lang or "english", hyp


def speaker_check(rows):
    """Are the voice presets distinguishable in the OUTPUT? Long-term average log-mel spectrum
    plus median F0 -- coarse proxies, but fixture cases 0 and 2 are the same voice reading
    DIFFERENT text, which makes a positive control: if that pair is not the most similar, voice
    conditioning is not landing and no PCC would have told us. Only the ranking is meaningful.
    """
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256,
                                               n_mels=80)
    by_case = {r["case"]: r for r in rows}
    sel = [c for c in (0, 1, 2, 3) if c in by_case]
    if len(sel) < 3 or 0 not in sel or 2 not in sel:
        print("\n  speaker check skipped (needs fixture cases 0-3)")
        return

    def ltas(x):
        m = mel(x)
        e = m.sum(0)
        v = torch.log(m[:, e > e.max() * 1e-3] + 1e-8).mean(1)
        return v - v.mean()                        # drop loudness, keep spectral shape

    def f0(x, sr=24000, lo=60, hi=400):
        win, hop, lag_lo, lag_hi = 1024, 256, sr // hi, sr // lo
        vals = []
        for i in range(0, len(x) - win, hop):
            f = x[i:i + win]
            if f.pow(2).mean().sqrt() < 1e-3:
                continue
            f = f - f.mean()
            ac = torch.nn.functional.conv1d(f.view(1, 1, -1), f.view(1, 1, -1),
                                            padding=win - 1).view(-1)[win - 1:]
            seg = ac[lag_lo:lag_hi]
            if len(seg) and ac[0] > 0 and seg.max() / ac[0] > 0.3:
                vals.append(sr / (int(seg.argmax()) + lag_lo))
        return float(np.median(vals)) if vals else float("nan")

    print("\n  === voice identity ===")
    vec, pitch = {}, {}
    for c in sel:
        x = torch.from_numpy(read_wav(by_case[c]["wav"])[0])
        vec[c], pitch[c] = ltas(x), f0(x)
        print(f"    case {c}  {by_case[c]['voice']:<16} median F0 {pitch[c]:6.1f} Hz")
    pairs = {(a, b): float(torch.nn.functional.cosine_similarity(vec[a], vec[b], dim=0))
             for a in sel for b in sel if a < b}
    print("    spectrum cosine similarity:")
    for (a, b), v in sorted(pairs.items(), key=lambda kv: -kv[1]):
        same = " <- SAME VOICE (positive control)" if (a, b) == (0, 2) else ""
        print(f"      case{a} vs case{b}  {v:.3f}  "
              f"({by_case[a]['voice']} / {by_case[b]['voice']}){same}")
    top = max(pairs, key=pairs.get)
    print(f"    => {'PASS' if top == (0, 2) else 'FAIL'}: the same-voice pair is "
          f"{'the most' if top == (0, 2) else 'NOT the most'} similar")
    print(f"    F0 spread across voices: {min(pitch.values()):.0f}-{max(pitch.values()):.0f} Hz")


def main():
    rows = [r for path in sys.argv[1:] for r in json.load(open(path))]
    if not rows:
        raise SystemExit("usage: score_quality_set.py <results.json> [...]")

    clean, adv, unstable = [0.0, 0], [0.0, 0], [0.0, 0]
    longform, shortform = [0.0, 0], [0.0, 0]   # the clean bucket, split by length
    table = []
    for r, lang, hyp in transcribe(rows):
        e, nw = wer(r["text"], hyp)
        if ADVERSARIAL.search(r["text"]):
            bucket, kind = adv, "adversarial"
        elif nw <= UNSTABLE_MAX_WORDS:
            bucket, kind = unstable, "model-unstable"
        else:
            bucket, kind = clean, ""
        bucket[0] += e * nw
        bucket[1] += nw
        if bucket is clean:
            (longform if nw >= LONGFORM_MIN_WORDS else shortform)[0] += e * nw
            (longform if nw >= LONGFORM_MIN_WORDS else shortform)[1] += nw
        table.append((r, lang, e, nw, kind))
        print(f"\n  case {r['case']:>2} {r['voice']:<16} lang={lang:<11} "
              f"{r['audio_s']:.1f}s / {r['frames']} frames"
              f"{f'   [{kind} text]' if kind else ''}")
        print(f"    REF: {norm(r['text'])[:110]}")
        print(f"    ASR: {norm(hyp)[:110]}")
        print(f"    WER: {e*100:.1f}%  ({nw} ref words)"
              f"{'  <- not terminated' if not r.get('terminated', True) else ''}")

    print("\n  === summary ===")
    for r, lang, e, nw, kind in table:
        print(f"    case {r['case']:>2}  {r['voice']:<16} {lang:<11} WER {e*100:>7.1f}%  "
              f"({nw:>3} words){f'  [{kind}]' if kind else ''}")
    print(f"\n  NATURAL-TEXT WER {clean[0]/max(clean[1],1)*100:.2f}% over {clean[1]} words "
          f"(reference scores 0.0%) -- split below, and read the split, not this")
    print(f"    long-form  {longform[0]/max(longform[1],1)*100:6.2f}% over {longform[1]:>3} words "
          f"= {round(longform[0]):g} wrong  <- THE GATE. Read the COUNT: at 298 words one word is"
          f" 0.34%, so quote errors, not a percentage that looks like precision it does not have.")
    print(f"    short      {shortform[0]/max(shortform[1],1)*100:6.2f}% over {shortform[1]:>3} words "
          f"<- SEED NOISE at this length. Same code, seeds 0/1/2, moved this bucket enough to "
          f"swing the headline 0.88-2.06%. Only believe a case that moves the same way across "
          f"several seeds (case 6 does).")
    if adv[1]:
        print(f"  adversarial-text WER {adv[0]/max(adv[1],1)*100:.2f}% over {adv[1]} words "
              f"(emoji/symbol/whitespace texts; the model vocalises them, so there is no "
              f"well-defined transcript)")
    if unstable[1]:
        print(f"  model-unstable WER {unstable[0]/max(unstable[1],1)*100:.2f}% over "
              f"{unstable[1]} words (one-word prompts; the MODEL is chaotic on these -- the fp32 "
              f"CPU reference is too -- so this measures the model, not the port)")
    not_done = [r["case"] for r in rows if not r.get("terminated", True)]
    print(f"  natural [END_AUDIO] termination: {len(rows)-len(not_done)}/{len(rows)} cases"
          + (f", MISSING on {not_done}" if not_done else ""))
    speaker_check(rows)


if __name__ == "__main__":
    main()

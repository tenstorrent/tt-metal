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

    clean, adv = [0.0, 0], [0.0, 0]
    table = []
    for r, lang, hyp in transcribe(rows):
        e, nw = wer(r["text"], hyp)
        bucket = adv if ADVERSARIAL.search(r["text"]) else clean
        bucket[0] += e * nw
        bucket[1] += nw
        table.append((r, lang, e, nw, bucket is adv))
        print(f"\n  case {r['case']:>2} {r['voice']:<16} lang={lang:<11} "
              f"{r['audio_s']:.1f}s / {r['frames']} frames"
              f"{'   [adversarial text]' if bucket is adv else ''}")
        print(f"    REF: {norm(r['text'])[:110]}")
        print(f"    ASR: {norm(hyp)[:110]}")
        print(f"    WER: {e*100:.1f}%  ({nw} ref words)"
              f"{'  <- not terminated' if not r.get('terminated', True) else ''}")

    print("\n  === summary ===")
    for r, lang, e, nw, is_adv in table:
        print(f"    case {r['case']:>2}  {r['voice']:<16} {lang:<11} WER {e*100:>7.1f}%  "
              f"({nw:>3} words){'  [adversarial]' if is_adv else ''}")
    print(f"\n  NATURAL-TEXT WER {clean[0]/max(clean[1],1)*100:.2f}% over {clean[1]} words "
          f"<- the headline; reference scores 0.0%")
    if adv[1]:
        print(f"  adversarial-text WER {adv[0]/max(adv[1],1)*100:.2f}% over {adv[1]} words "
              f"(emoji/symbol/whitespace texts; the model vocalises them, so there is no "
              f"well-defined transcript)")
    not_done = [r["case"] for r in rows if not r.get("terminated", True)]
    print(f"  natural [END_AUDIO] termination: {len(rows)-len(not_done)}/{len(rows)} cases"
          + (f", MISSING on {not_done}" if not_done else ""))
    speaker_check(rows)


if __name__ == "__main__":
    main()

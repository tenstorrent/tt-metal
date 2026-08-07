"""WER scoring for the quality set, resampling with scipy instead of torchaudio.

USE THIS ONE, not `score_quality_set.py`, on any box where torchaudio is unavailable — which is
every box this fork has run on. The repo scorer needs torchaudio purely to resample 24 kHz to
16 kHz for Whisper, and torchaudio's wheel is ABI-broken against this torch AND merely having it
importable breaks `transformers`, which takes the scorer down with it (STATUS.md §2). This is a
faithful port: identical ADVERSARIAL / UNSTABLE / LONGFORM bucketing, identical normaliser,
identical edit distance, identical language detection and Whisper model choice. The only change is
`scipy.signal.resample_poly` in place of the torchaudio call.

BECAUSE IT IS A PORT, IT IS ALSO A LIABILITY: a systematic bug here would pass every utterance
silently. It is committed rather than left in /tmp so it can be reviewed and diffed against the
original, which is the only defence available.

Prints the LONG-FORM error COUNT, which is the gate (§6.7): short prompts are seed noise, and the
adversarial / one-word buckets have no well-defined transcript.
"""
import json
import re
import sys
import unicodedata
import wave

import numpy as np
from scipy.signal import resample_poly

ADVERSARIAL = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]|[!@#$%^&*()]{3,}|\\[tn]|\t|\n")
UNSTABLE_MAX_WORDS = 1
LONGFORM_MIN_WORDS = 20
SCRIPT_LANG = [("ऀॿ", "hindi"), ("؀ۿ", "arabic")]
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
    return None


def read_wav(p):
    with wave.open(p, "rb") as f:
        a = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return a, f.getframerate()


def norm(s):
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
    return d[len(r), len(h)], len(r)


_CACHE = {}


def transcribe_one(r):
    from transformers import pipeline
    lang = detect_lang(r["text"])
    model = "openai/whisper-small" if lang else "openai/whisper-base.en"
    long_form = r["audio_s"] > 28
    key = (model, long_form)
    if key not in _CACHE:
        _CACHE[key] = pipeline("automatic-speech-recognition", model=model,
                               **({"chunk_length_s": 30, "stride_length_s": 5}
                                  if long_form else {}))
    a, sr = read_wav(r["wav"])
    if sr != 16000:
        a = resample_poly(a, 16000, sr).astype(np.float32)
    kw = {"generate_kwargs": {"language": lang, "task": "transcribe"}} if lang else {}
    return cache_get(key)({"raw": a, "sampling_rate": 16000}, **kw)["text"], lang or "english"


def cache_get(key):
    return _CACHE[key]


def score(path):
    rows = json.load(open(path))
    lf_err = lf_w = sf_err = sf_w = 0
    per_case = {}
    for r in rows:
        hyp, lang = transcribe_one(r)
        e, nw = wer(r["text"], hyp)
        if ADVERSARIAL.search(r["text"]):
            bucket = "adversarial"
        elif nw <= UNSTABLE_MAX_WORDS:
            bucket = "unstable"
        elif nw >= LONGFORM_MIN_WORDS:
            bucket = "longform"
            lf_err += e
            lf_w += nw
        else:
            bucket = "short"
            sf_err += e
            sf_w += nw
        per_case[r["case"]] = (bucket, e, nw, r["frames"], hyp)
    return dict(longform=(lf_err, lf_w), short=(sf_err, sf_w), per_case=per_case,
                frames={r["case"]: r["frames"] for r in rows},
                msf={r["case"]: r.get("gen_ms_per_frame", float("nan")) for r in rows},
                term=sum(1 for r in rows if r.get("terminated", True)), n=len(rows))


def main():
    out = {}
    for path in sys.argv[1:]:
        tag = path.split("results")[-1].replace(".json", "")
        out[tag] = score(path)
        s = out[tag]
        print(f"\n===== {tag} =====")
        print(f"  long-form {s['longform'][0]} wrong of {s['longform'][1]} words   |  "
              f"short {s['short'][0]} of {s['short'][1]}  |  "
              f"[END_AUDIO] {s['term']}/{s['n']}")
        for c, (b, e, nw, fr, hyp) in sorted(s["per_case"].items()):
            if b == "longform" and e:
                print(f"    case {c} ({fr} frames): {e} of {nw} -> {norm(hyp)[:90]}")
    print("\n\n===== LONG-FORM ERROR COUNTS (the gate) =====")
    print(f"{'tag':>10} {'wrong':>7} {'of':>6}   frames per long-form case")
    for tag, s in out.items():
        lf = [c for c, (b, *_) in s["per_case"].items() if b == "longform"]
        fr = " ".join(f"c{c}={s['frames'][c]}" for c in sorted(lf))
        print(f"{tag:>10} {s['longform'][0]:>7} {s['longform'][1]:>6}   {fr}")
    print(f"\n{'tag':>10}   mean gen ms/frame over long-form cases")
    for tag, s in out.items():
        lf = [c for c, (b, *_) in s["per_case"].items() if b == "longform"]
        v = [s["msf"][c] for c in lf]
        print(f"{tag:>10}   {sum(v) / len(v):.2f}")


if __name__ == "__main__":
    main()

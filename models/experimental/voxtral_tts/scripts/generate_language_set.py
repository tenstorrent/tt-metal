#!/usr/bin/env python3
"""Generate one audio clip per (language, voice, sentence) for the per-language MOS gate.

MOS is pooled today, so a language could turn robotic while staying perfectly intelligible and no
gate would fire. WER cannot see naturalness; this is the axis it misses.

Only sentences of ~20 words and up are used: the report's own scorer treats anything shorter as
unstable, and STATUS 6.7 records mos_mean/mos_min as report-only because short prompts dominate them.
That is the medium and long WER bands, so the clips are the same text the WER cells score.

Writes generated/lang_<tag>/<lang>_<voice>_s<i>.wav and a manifest carrying the language label, so
the scorer (which runs in the isolated MOS venv and cannot import ttnn) needs no model knowledge.

    scripts/generate_language_set.py --tag base
"""
import argparse, json, math, os, wave

import torch
import ttnn

from models.experimental.voxtral_tts.tests.reference_helpers import all_voices, corpus_embeds
from models.experimental.voxtral_tts.tests.sentence_corpus import WER_SENTENCES, lang_of, wer_band
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (FRAME_RATE, TtVoxtralPipeline,
                                                                      open_device)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(HERE, "generated")
MIN_WORDS = 19          # the shortest sentence in the medium band; below this MOS is noise
SR = 24000


def frame_budget(text):
    """The generator's own rule: a CAP, not a cost -- generation stops on [END_AUDIO] regardless."""
    return max(320, int(math.ceil(len(text) / 18.0 * FRAME_RATE * 2.2)))


def save_wav(wav, path):
    x = (wav.reshape(-1).clamp(-1, 1) * 32767).to(torch.int16).numpy()
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes(x.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--langs", default="all")
    args = ap.parse_args()

    langs = sorted(WER_SENTENCES) if args.langs == "all" else args.langs.split(",")
    out = os.path.join(GEN, f"lang_{args.tag}")
    os.makedirs(out, exist_ok=True)

    dev = open_device()
    try:
        pipe = TtVoxtralPipeline(dev)
        pipe.warmup(verbose=False)
        rows = []
        for lang in langs:
            texts = [t for b in ("medium", "long") for t in wer_band(lang, b)
                     if len(t.split()) >= MIN_WORDS]
            voices = [v for v in all_voices() if lang_of(v) == lang]
            for voice in voices:
                for i, text in enumerate(texts):
                    pipe.backbone.reset()
                    frames, _, _ = pipe.generate(corpus_embeds(text, voice, pipe.wb),
                                                 max_frames=frame_budget(text), seed=args.seed,
                                                 verbose=False)
                    wav = pipe.decode(frames)
                    name = f"{lang}_{voice}_s{i}.wav"
                    save_wav(wav, os.path.join(out, name))
                    rows.append({"file": name, "lang": lang, "voice": voice, "sentence": i,
                                 "words": len(text.split()), "frames": int(frames.shape[0]),
                                 "seconds": round(frames.shape[0] / FRAME_RATE, 2)})
                    print(f"  {lang}/{voice} s{i}: {frames.shape[0]} frames "
                          f"{frames.shape[0]/FRAME_RATE:.1f}s -> {name}", flush=True)
        man = os.path.join(out, "manifest.json")
        json.dump(rows, open(man, "w"), indent=1)
        print(f"\n{len(rows)} clips, manifest {man}")
        pipe.close()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

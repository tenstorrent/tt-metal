# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Build a self-contained smoke manifest for the S2TT benchmark.

Writes a few synthetic wavs and uses the HF model's own greedy output as the
reference translation for each. Running run_benchmark.py against this manifest
therefore measures TT-vs-HF agreement (BLEU/chrF near 100 means the TT pipeline
reproduces HF) and exercises the metric + RTF plumbing — without needing an
external speech dataset. CPU only.

    python3 models/demos/seamless_m4t_v2/evaluation/make_smoke_manifest.py --n 3
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _synth(seed, seconds, sr=16000):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    f1, f2 = 150 + 60 * rng.rand(), 300 + 120 * rng.rand()
    return (0.5 * np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * f2 * t)
            + 0.1 * rng.randn(t.size)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tgt_lang", default="jpn")
    ap.add_argument("--out_dir", default=os.path.dirname(__file__))
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    from models.demos.seamless_m4t_v2.tt.model_config import DEFAULT_MODEL_ID

    audio_dir = os.path.join(args.out_dir, "smoke_audio")
    os.makedirs(audio_dir, exist_ok=True)
    proc = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID)
    hf = SeamlessM4Tv2ForSpeechToText.from_pretrained(DEFAULT_MODEL_ID).eval().float()

    manifest = []
    for i in range(args.n):
        audio = _synth(seed=i, seconds=3.0 + i)
        path = os.path.join(audio_dir, f"smoke_{i}.wav")
        sf.write(path, audio, 16000)
        try:
            feats = proc(audios=audio, sampling_rate=16000, return_tensors="pt")["input_features"].float()
        except (TypeError, ValueError):
            feats = proc(audio=audio, sampling_rate=16000, return_tensors="pt")["input_features"].float()
        with torch.no_grad():
            out = hf.generate(input_features=feats, tgt_lang=args.tgt_lang, num_beams=1)
        ref = proc.decode(out[0].tolist(), skip_special_tokens=True)
        manifest.append({"audio": path, "ref": ref})
        print(f"smoke_{i}: ref={ref!r}")

    out_path = os.path.join(args.out_dir, "smoke_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path} ({len(manifest)} items)")


if __name__ == "__main__":
    main()

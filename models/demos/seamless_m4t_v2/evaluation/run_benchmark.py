# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Phase 8: S2TT quality + speed benchmark (English speech -> Japanese text).

Reads a JSON manifest of {"audio": <wav path>, "ref": <reference translation>}
entries, runs the TT pipeline, and reports:
  - BLEU and chrF (sacrebleu; chrF is tokenizer-free and well suited to Japanese;
    BLEU uses the ja-mecab tokenizer when available, else the default),
  - RTF (total audio seconds / total wall seconds) and per-clip latency.

Optionally also runs the HF reference for a side-by-side BLEU/chrF comparison.

Usage (inside the metalcon container, device on /dev/tenstorrent/2):
    python3 models/demos/seamless_m4t_v2/evaluation/run_benchmark.py \
        --manifest models/demos/seamless_m4t_v2/evaluation/smoke_manifest.json \
        --tgt_lang jpn --bucket_len 96 [--compare_hf] [--max_samples N]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import ttnn
from models.demos.seamless_m4t_v2.evaluation.metrics import score_corpus
from models.demos.seamless_m4t_v2.tt.generator import SeamlessS2TTGenerator


def _load_audio(path, target_sr=16000):
    import soundfile as sf

    audio, sr = sf.read(path)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, len(audio) / target_sr


def _scores(hyps, refs):
    s = score_corpus(hyps, refs)
    return s["bleu"], s["chrf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tgt_lang", default="jpn")
    ap.add_argument("--bucket_len", type=int, default=96)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--compare_hf", action="store_true")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    with open(args.manifest) as f:
        items = json.load(f)
    if args.max_samples:
        items = items[: args.max_samples]

    device = ttnn.open_device(device_id=0)
    try:
        gen = SeamlessS2TTGenerator.build(device)
        hf = gen_proc = None
        if args.compare_hf:
            from transformers import SeamlessM4Tv2ForSpeechToText

            hf = SeamlessM4Tv2ForSpeechToText.from_pretrained(gen.config.model_id).eval().float()

        hyps, refs, hf_hyps = [], [], []
        total_audio, total_wall = 0.0, 0.0
        for it in items:
            audio, dur = _load_audio(it["audio"])
            t0 = time.time()
            text, _ = gen.generate(audio, device, tgt_lang=args.tgt_lang, bucket_len=args.bucket_len)
            dt = time.time() - t0
            hyps.append(text)
            refs.append(it["ref"])
            total_audio += dur
            total_wall += dt
            print(f"[{dur:.1f}s audio / {dt:.2f}s] {text!r}")
            if hf is not None:
                import torch

                feats = gen.processor(audios=audio, sampling_rate=16000, return_tensors="pt")["input_features"].float()
                with torch.no_grad():
                    out = hf.generate(input_features=feats, tgt_lang=args.tgt_lang, num_beams=1)
                hf_hyps.append(gen.processor.decode(out[0].tolist(), skip_special_tokens=True))

        bleu, chrf = _scores(hyps, refs)
        rtf = total_audio / total_wall if total_wall else 0.0
        result = {
            "n": len(hyps), "bleu": bleu, "chrf": chrf,
            "rtf": rtf, "avg_latency_s": total_wall / max(len(hyps), 1),
            "bucket_len": args.bucket_len,
        }
        if hf is not None:
            result["hf_bleu"], result["hf_chrf"] = _scores(hf_hyps, refs)
        print("\n=== Results ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        with open(args.out, "w") as f:
            json.dump({"summary": result, "hyps": hyps, "refs": refs}, f, ensure_ascii=False, indent=2)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

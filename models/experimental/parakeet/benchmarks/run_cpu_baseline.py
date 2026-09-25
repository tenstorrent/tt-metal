# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""PyTorch FP32 CPU baseline for parakeet-tdt-0.6b-v3 on the TT host.

Timing protocol: mel_features_to_tokens_with_sync_excluding_load_preprocessing_progress_v1
(model load and mel preprocessing excluded; mel -> tokens timed end to end).
Input: harness inputs.npz with keys <case>__mel [B,T,128], <case>__mel_lengths [B],
<case>__attention_mask [B,T], <case>__audio_seconds [B].
Usage: python benchmarks/run_cpu_baseline.py --input /input --weights /weights --out baseline/cpu_fp32_<stage>
(moved from baseline/run_cpu_baseline.py; the recorded baseline/*.json were produced by the same code.)
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch


def load_cases(input_dir):
    data = np.load(os.path.join(input_dir, "inputs.npz"), allow_pickle=False)
    names = [k[: -len("__mel")] for k in data.keys() if k.endswith("__mel")]
    cases = []
    for n in names:
        mel = data[f"{n}__mel"].astype(np.float32)
        lens = data[f"{n}__mel_lengths"].astype(np.int64)
        mask = data[f"{n}__attention_mask"] if f"{n}__attention_mask" in data else (
            np.arange(mel.shape[1])[None, :] < lens[:, None]).astype(np.int64)
        audio = data[f"{n}__audio_seconds"] if f"{n}__audio_seconds" in data else lens * 0.01
        cases.append((n, mel, lens, mask.astype(np.int64), audio.astype(np.float64)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--weights", default="/weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cases = load_cases(args.input)

    from transformers import ParakeetForTDT

    t0 = time.perf_counter()
    model = ParakeetForTDT.from_pretrained(args.weights, dtype=torch.float32).eval()
    load_s = time.perf_counter() - t0
    print(f"[load] {load_s:.2f}s threads={torch.get_num_threads()}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    results = {"host": platform.node(), "python": sys.version.split()[0], "torch": torch.__version__,
               "threads": torch.get_num_threads(), "load_s": load_s, "repeats": args.repeats,
               "protocol": "mel_features_to_tokens_with_sync_excluding_load_preprocessing_progress_v1",
               "cases": {}}
    arrays = {}
    with torch.inference_mode():
        for name, mel, lens, mask_np, audio in cases:
            x = torch.from_numpy(mel)
            mask = torch.from_numpy(mask_np)
            enc_t, gen_t = [], []
            for _ in range(args.repeats):
                t = time.perf_counter()
                enc = model.encoder(input_features=x, attention_mask=mask).last_hidden_state
                enc_t.append(time.perf_counter() - t)
                t = time.perf_counter()
                out = model.generate(input_features=x, attention_mask=mask)
                gen_t.append(time.perf_counter() - t)
            seqs = (out.sequences if hasattr(out, "sequences") else out).cpu().numpy()
            arrays[f"{name}__encoder"] = enc.float().numpy()
            arrays[f"{name}__tokens"] = seqs
            if hasattr(out, "durations") and out.durations is not None:
                arrays[f"{name}__durations"] = out.durations.cpu().numpy()
            med = float(np.median(gen_t))
            results["cases"][name] = {"batch": int(mel.shape[0]), "frames": int(mel.shape[1]),
                                      "lengths": lens.tolist(), "audio_seconds": audio.tolist(),
                                      "encode_s": enc_t, "transcribe_s": gen_t, "transcribe_median_s": med,
                                      "rtf_audio_over_wall": float(audio.sum()) / med,
                                      "enc_shape": list(enc.shape), "tokens_shape": list(seqs.shape)}
            print(f"[case] {name} B={mel.shape[0]} T={mel.shape[1]} enc_med={np.median(enc_t):.3f}s "
                  f"gen_med={med:.3f}s tokens={seqs.shape} row0={seqs[0][:12].tolist()}", flush=True)
    np.savez_compressed(os.path.join(args.out, "oracle_cpu_fp32.npz"), **arrays)
    with open(os.path.join(args.out, "baseline.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("[saved]", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""Component check: TT greedy TDT tokens vs CPU FP32 transformers generate(), same job.

For every case in <input>/inputs.npz (or --cases): exact token match (padding 2 stripped per row),
first mismatch position, determinism across repeats, and synchronized wall time.
Usage: python tests/check_transcribe.py [--input /input] [--weights /weights] [--cases a,b] [--repeats 2]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def strip(row, pad):
    row = list(int(t) for t in row)
    while row and row[-1] == pad:
        row.pop()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--weights", default="/weights")
    ap.add_argument("--cases", default="")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    data = np.load(os.path.join(args.input, "inputs.npz"))
    names = [k[:-5] for k in data.keys() if k.endswith("__mel")]
    if args.cases:
        names = [n for n in args.cases.split(",") if n in names]

    from transformers import ParakeetForTDT

    model = ParakeetForTDT.from_pretrained(args.weights, dtype=torch.float32).eval()
    pad = model.generation_config.pad_token_id

    import backend as be

    import ttnn

    device = ttnn.open_device(device_id=0, **be.DEVICE_OPTIONS)
    all_ok = True
    try:
        with open(os.path.join(args.weights, "config.json")) as f:
            cfg = json.load(f)
        bk = be.create_backend(args.weights, cfg, device, precision="bf16")
        for n in names:
            mel = data[f"{n}__mel"].astype(np.float32)
            lens = data[f"{n}__mel_lengths"].astype(np.int64)
            mask = (np.arange(mel.shape[1])[None] < lens[:, None]).astype(np.int64)
            with torch.inference_mode():
                ref = model.generate(input_features=torch.from_numpy(mel), attention_mask=torch.from_numpy(mask))
            ref = (ref.sequences if hasattr(ref, "sequences") else ref).numpy()
            outs, times = [], []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                outs.append(np.asarray(bk.transcribe(mel, lens)["tokens"]))
                times.append(time.perf_counter() - t0)
            det = all(o.shape == outs[0].shape and (o == outs[0]).all() for o in outs)
            ok = True
            for b in range(mel.shape[0]):
                r, o = strip(ref[b], pad), strip(outs[0][b], pad)
                if r != o:
                    ok = False
                    k = next((i for i in range(min(len(r), len(o))) if r[i] != o[i]), min(len(r), len(o)))
                    print(
                        f"[mismatch] {n} row{b} at {k}: ref={r[max(0, k - 3):k + 4]} tt={o[max(0, k - 3):k + 4]} "
                        f"len ref={len(r)} tt={len(o)}",
                        flush=True,
                    )
            all_ok &= ok and det
            print(
                f"[case] {n} B={mel.shape[0]} T={mel.shape[1]} match={ok} deterministic={det} "
                f"shape tt={outs[0].shape} ref={ref.shape} times={[round(t, 3) for t in times]}",
                flush=True,
            )
        print(f"[summary] all_match={all_ok}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""Synchronized latency breakdown of the TT transcribe path (encoder vs TDT decode loop).

Warm, sync-bounded medians of transcribe and encode_device per case, then one instrumented
transcribe where encode_device / _decoder_step / _joint / _masked are each wrapped with
device synchronizes, plus a tiny upload / readback / idle-sync round-trip microbenchmark.
Usage: python benchmarks/profile_transcribe.py --input /input --cases long,short [--precision bf16]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

WRAPPED = ("encode_device", "_decoder_step", "_joint", "_masked")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.environ.get("PARAKEET_INPUT", "/input"))
    ap.add_argument("--weights", default="/weights")
    ap.add_argument("--cases", default="long,short")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    import ttnn
    import backend as be
    data = np.load(os.path.join(args.input, "inputs.npz"))
    with open(os.path.join(args.weights, "config.json")) as f:
        cfg = json.load(f)
    dev = ttnn.open_device(device_id=0, **be.DEVICE_OPTIONS)
    sync = lambda: ttnn.synchronize_device(dev)
    pc = time.perf_counter
    try:
        bk = be.create_backend(args.weights, cfg, dev, precision=args.precision)

        def timed(fn, n):
            ts = []
            for _ in range(n):
                sync()
                t = pc()
                r = fn()
                sync()
                ts.append(pc() - t)
            return float(np.median(ts)), r

        for case in args.cases.split(","):
            mel = data[f"{case}__mel"].astype(np.float32)
            lens = data[f"{case}__mel_lengths"].astype(np.int64)
            for _ in range(2):
                bk.transcribe(mel, lens)
            t_tr, out = timed(lambda: bk.transcribe(mel, lens), args.repeats)
            t_enc, _ = timed(lambda: bk.encode_device(mel, lens), args.repeats)
            steps = out["tokens"].shape[1] - 1
            T = {}

            def wrap(name, f):
                def g(*a, **k):
                    sync()
                    t = pc()
                    r = f(*a, **k)
                    sync()
                    T.setdefault(name, []).append(pc() - t)
                    return r
                return g

            for name in WRAPPED:
                setattr(bk, name, wrap(name, getattr(bk, name)))
            try:
                sync()
                t = pc()
                bk.transcribe(mel, lens)
                sync()
                t_inst = pc() - t
            finally:
                for name in WRAPPED:
                    delattr(bk, name)
            tot = {k: (len(v), sum(v)) for k, v in T.items()}
            acc = sum(s for _, s in tot.values())
            print(f"[{case}] T'={bk.cfg.sub_length(mel.shape[1])} steps={steps} warm_transcribe_p50={t_tr:.4f}s "
                  f"encode_p50={t_enc:.4f}s decode_loop(p50 diff)={t_tr - t_enc:.4f}s "
                  f"per_step={(t_tr - t_enc) / steps * 1e3:.2f}ms")
            print(f"[{case}] instrumented total={t_inst:.4f}s "
                  + " ".join(f"{k}: n={n} sum={s:.4f}s avg={s / n * 1e3:.2f}ms" for k, (n, s) in tot.items())
                  + f" host_other={t_inst - acc:.4f}s")
        # round trip: one tiny upload (token ids), one tiny readback (argmax result), idle sync
        y = ttnn.argmax(ttnn.to_layout(ttnn.from_torch(torch.randn(1, 1, 32, 8224), dtype=ttnn.float32,
                                                       layout=ttnn.TILE_LAYOUT, device=dev),
                                       ttnn.ROW_MAJOR_LAYOUT), dim=-1)
        n = 200
        sync()
        t = pc()
        for _ in range(n):
            bk._ids([5])
        sync()
        up = (pc() - t) / n
        t = pc()
        for _ in range(n):
            ttnn.to_torch(y)
        rd = (pc() - t) / n
        sync()
        t = pc()
        for _ in range(n):
            sync()
        sy = (pc() - t) / n
        print(f"[rt] upload_ids={up * 1e6:.1f}us readback_argmax={rd * 1e6:.1f}us sync_idle={sy * 1e6:.1f}us")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

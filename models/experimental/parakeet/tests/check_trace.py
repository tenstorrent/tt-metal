# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""Traced vs untraced encoder: outputs, tokens and synchronized encode time per case.

Runs the cases in an order that switches (B, Tp) so trace release / re-capture is exercised.
Usage: python tests/check_trace.py --input /input --cases short,long,short
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--weights", default="/weights")
    ap.add_argument("--cases", default="short,long,short")
    ap.add_argument("--precision", default="bf16")
    args = ap.parse_args()
    import ttnn
    import backend as be
    data = np.load(os.path.join(args.input, "inputs.npz"))
    with open(os.path.join(args.weights, "config.json")) as f:
        cfg = json.load(f)
    dev = ttnn.open_device(device_id=0, **be.DEVICE_OPTIONS)
    sync = lambda: ttnn.synchronize_device(dev)
    try:
        bk = be.create_backend(args.weights, cfg, dev, precision=args.precision, use_trace=True)

        def enc(mel, lens):
            sync()
            t = time.perf_counter()
            out = bk.encode(mel, lens)["encoder"]
            sync()
            return out, time.perf_counter() - t

        for case in args.cases.split(","):
            mel = data[f"{case}__mel"].astype(np.float32)
            lens = data[f"{case}__mel_lengths"].astype(np.int64)
            bk.use_trace = False
            ref, t_ref = enc(mel, lens)
            ref2, t_ref2 = enc(mel, lens)
            tok_ref = bk.transcribe(mel, lens)["tokens"]
            bk.use_trace = True
            first, t_first = enc(mel, lens)  # untraced run + capture
            outs = [enc(mel, lens) for _ in range(3)]  # replays
            tok_tr = bk.transcribe(mel, lens)["tokens"]
            rn = lambda a: float(np.sqrt(np.mean((a - ref) ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-12))
            diffs = [float(np.abs(o - ref).max()) for o, _ in outs]
            print(f"[{case}] shape={ref.shape} untraced={t_ref:.4f}/{t_ref2:.4f}s first+capture={t_first:.4f}s "
                  f"replay={','.join(f'{t:.4f}' for _, t in outs)}s")
            print(f"[{case}] first_maxdiff={float(np.abs(first - ref).max()):.3g} replay_maxdiff={diffs} "
                  f"replay_nrmse={rn(outs[-1][0]):.3g} untraced_repeat_maxdiff={float(np.abs(ref2 - ref).max()):.3g} "
                  f"tokens_equal={bool(np.array_equal(tok_ref, tok_tr))}")
        bk.release_trace()
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

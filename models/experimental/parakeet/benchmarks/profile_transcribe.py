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
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _instrument(backend, timers, sync):
    """Wrap the four hot-path methods with sync-bounded timers (direct refs, no dynamic attrs)."""
    pc = time.perf_counter

    def make_timed(name, fn):
        def timed(*args, **kwargs):
            sync()
            start = pc()
            result = fn(*args, **kwargs)
            sync()
            timers[name].append(pc() - start)
            return result

        return timed

    saved_encode = backend.encode_device
    saved_decoder_step = backend._decoder_step
    saved_joint = backend._joint
    saved_masked = backend._masked
    backend.encode_device = make_timed("encode_device", saved_encode)
    backend._decoder_step = make_timed("_decoder_step", saved_decoder_step)
    backend._joint = make_timed("_joint", saved_joint)
    backend._masked = make_timed("_masked", saved_masked)

    def restore():
        backend.encode_device = saved_encode
        backend._decoder_step = saved_decoder_step
        backend._joint = saved_joint
        backend._masked = saved_masked

    return restore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.environ.get("PARAKEET_INPUT", "/input"))
    ap.add_argument("--weights", default="/weights")
    ap.add_argument("--cases", default="long,short")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    import backend as be

    import ttnn

    data = np.load(os.path.join(args.input, "inputs.npz"))
    with open(os.path.join(args.weights, "config.json")) as f:
        cfg = json.load(f)
    dev = ttnn.open_device(device_id=0, **be.DEVICE_OPTIONS)

    def sync():
        ttnn.synchronize_device(dev)

    pc = time.perf_counter
    try:
        bk = be.create_backend(args.weights, cfg, dev, precision=args.precision)

        def timed(fn, n):
            ts = []
            result = None
            for _ in range(n):
                sync()
                start = pc()
                result = fn()
                sync()
                ts.append(pc() - start)
            return float(np.median(ts)), result

        for case in args.cases.split(","):
            mel = data[f"{case}__mel"].astype(np.float32)
            lens = data[f"{case}__mel_lengths"].astype(np.int64)
            for _ in range(2):
                bk.transcribe(mel, lens)
            t_tr, out = timed(lambda: bk.transcribe(mel, lens), args.repeats)
            t_enc, _ = timed(lambda: bk.encode_device(mel, lens), args.repeats)
            steps = out["tokens"].shape[1] - 1
            timers = defaultdict(list)

            restore = _instrument(bk, timers, sync)
            try:
                sync()
                start = pc()
                bk.transcribe(mel, lens)
                sync()
                t_inst = pc() - start
            finally:
                restore()
            tot = {k: (len(v), sum(v)) for k, v in timers.items()}
            acc = sum(s for _, s in tot.values())
            print(
                f"[{case}] T'={bk.cfg.sub_length(mel.shape[1])} steps={steps} warm_transcribe_p50={t_tr:.4f}s "
                f"encode_p50={t_enc:.4f}s decode_loop(p50 diff)={t_tr - t_enc:.4f}s "
                f"per_step={(t_tr - t_enc) / steps * 1e3:.2f}ms"
            )
            print(
                f"[{case}] instrumented total={t_inst:.4f}s "
                + " ".join(f"{k}: n={n} sum={s:.4f}s avg={s / n * 1e3:.2f}ms" for k, (n, s) in tot.items())
                + f" host_other={t_inst - acc:.4f}s"
            )
        # round trip: one tiny upload (token ids), one tiny readback (argmax result), idle sync
        y = ttnn.argmax(
            ttnn.to_layout(
                ttnn.from_torch(torch.randn(1, 1, 32, 8224), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            dim=-1,
        )
        n = 200
        sync()
        start = pc()
        for _ in range(n):
            bk._ids([5])
        sync()
        up = (pc() - start) / n
        start = pc()
        for _ in range(n):
            ttnn.to_torch(y)
        rd = (pc() - start) / n
        sync()
        start = pc()
        for _ in range(n):
            sync()
        sy = (pc() - start) / n
        print(f"[rt] upload_ids={up * 1e6:.1f}us readback_argmax={rd * 1e6:.1f}us sync_idle={sy * 1e6:.1f}us")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()

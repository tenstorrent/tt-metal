# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Burst and sustained traced latency of the text embedding forward (P150a throttle-aware).

The P150a trips its current limit after about 1 s of back-to-back replays and lowers AICLK, so one
number is not enough (see doc/optimized_text/README.md):

    burst      3 s idle, then `--burst` back-to-back replays timed with one sync; repeated
               `--reps` times; the median of the per-window means is reported
    sustained  `--soak` s of back-to-back replays, then `--reps` windows of `--sustained` replays
               each, timed with one sync per window; median of the window means

Device-only timing (inputs already resident, execute_trace + synchronize), like perf.py `device`.

Run:
    python models/demos/blackhole/embedding_gemma2/tests/perf/bench_sustained.py \
        --cases 1x128,1x512,1x2048,8x512 [--json out.jsonl] [--label name]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.embedding_gemma2.conftest import MODEL_ID, open_eg2_device
from models.demos.blackhole.embedding_gemma2.tests.common import make_batch
from models.demos.blackhole.embedding_gemma2.tt.text_encoder import TextEncoder
from models.demos.blackhole.embedding_gemma2.tt.weights import resolve_snapshot


def _window_ms(enc: TextEncoder, b: int, s: int, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        enc.execute_captured(b, s)
    ttnn.synchronize_device(enc.device)
    return (time.perf_counter() - t0) * 1e3 / n


def _soak(enc: TextEncoder, b: int, s: int, seconds: float) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        _window_ms(enc, b, s, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1x128,1x512,1x2048,8x512")
    ap.add_argument("--burst", type=int, default=20)
    ap.add_argument("--sustained", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--idle", type=float, default=3.0)
    ap.add_argument("--soak", type=float, default=10.0)
    ap.add_argument("--max-seq-len", type=int, default=0, help="RoPE table length (default: largest case)")
    ap.add_argument("--no-sustained", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cases = [tuple(int(v) for v in c.split("x")) for c in args.cases.split(",")]
    max_seq_len = args.max_seq_len or max(8192, max(s for _, s in cases))
    snapshot = resolve_snapshot(MODEL_ID)
    device = open_eg2_device(0)  # EMBEDDING_GEMMA2_GRID; default: the device's own grid
    try:
        enc = TextEncoder.from_pretrained(device, str(snapshot), max_seq_len=max_seq_len)
        enc.capture_all(cases)
        results = []
        for b, s in cases:
            ids, mask = make_batch(snapshot, [s] * b)
            host = enc.prepare_inputs(ids, mask)
            _, emb = enc.replay(host)
            out = TextEncoder.read_embedding(emb, b)
            assert torch.isfinite(out).all()
            _window_ms(enc, b, s, 3)  # warm

            burst = []
            for _ in range(args.reps):
                time.sleep(args.idle)
                burst.append(_window_ms(enc, b, s, args.burst))
            rec = {
                "label": args.label,
                "batch": b,
                "seq_len": s,
                "burst_ms_median": statistics.median(burst),
                "burst_ms_min": min(burst),
                "burst_ms_max": max(burst),
                "burst_windows": burst,
            }
            if not args.no_sustained:
                _soak(enc, b, s, args.soak)
                sus = [_window_ms(enc, b, s, args.sustained) for _ in range(args.reps)]
                rec.update(
                    sustained_ms_median=statistics.median(sus),
                    sustained_ms_min=min(sus),
                    sustained_ms_max=max(sus),
                    sustained_windows=sus,
                )
                time.sleep(args.idle)
            results.append(rec)
            msg = (
                f"BENCH {args.label} b{b} s{s}: burst median {rec['burst_ms_median']:.3f} ms "
                f"[{rec['burst_ms_min']:.3f}, {rec['burst_ms_max']:.3f}]"
            )
            if "sustained_ms_median" in rec:
                msg += (
                    f"; sustained median {rec['sustained_ms_median']:.3f} ms "
                    f"[{rec['sustained_ms_min']:.3f}, {rec['sustained_ms_max']:.3f}]"
                )
            print(msg, flush=True)
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        enc.release_traces()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()

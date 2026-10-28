# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Burst and sustained traced latency of the vision forward (patch embed -> 16 layers -> pool -> embed).

Same protocol as `bench_sustained.py` (text):

    burst      `--idle` s idle, then `--burst` back-to-back replays timed with one sync; repeated
               `--reps` times; the median of the per-window means is reported
    sustained  `--soak` s of back-to-back replays, then `--reps` windows of `--sustained` replays
               each, timed with one sync per window; median of the window means

Device-only timing: inputs already resident, `execute_trace` + one synchronize per window.
Inputs are the real processor outputs of the vision test cases (`vision_common.load_case`).

Run:
    python models/demos/embedding_gemma2/tests/perf/bench_vision.py \
        --cases 1x2560,4x2560,8x1280 [--json out.jsonl] [--label name]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import ttnn
from models.demos.embedding_gemma2.conftest import DEVICE_PARAMS, MODEL_ID
from models.demos.embedding_gemma2.tests.pcc.vision.vision_common import load_case
from models.demos.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder
from models.demos.embedding_gemma2.tt.weights import resolve_snapshot

# (batch, bucket) -> processor cases concatenated along the batch (as test_vision_perf.PERF_CASES)
CASE_INPUTS = {
    (1, 2560): ("landscape",),
    (2, 2560): ("square", "small"),
    (4, 2560): ("landscape", "portrait", "square", "small"),
    (8, 1280): ("video8",),
}


def _inputs(snapshot, b: int, s: int):
    names = CASE_INPUTS.get((b, s))
    if names is None:  # repeat the bucket's batch-1 case
        names = ("landscape",) * b if s == 2560 else None
    if names is None:
        raise ValueError(f"no input case for {b}x{s}")
    pvs, pids = zip(*(load_case(str(snapshot), c) for c in names))
    pv, pid = torch.cat(pvs), torch.cat(pids)
    if pv.shape[0] != b:
        raise ValueError(f"case {names} gives batch {pv.shape[0]}, wanted {b}")
    return pv, pid


def _window_ms(enc: VisionEncoder, b: int, s: int, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        enc.execute_captured(b, s)
    ttnn.synchronize_device(enc.device)
    return (time.perf_counter() - t0) * 1e3 / n


def _soak(enc: VisionEncoder, b: int, s: int, seconds: float) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        _window_ms(enc, b, s, 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1x2560,4x2560,8x1280")
    ap.add_argument("--burst", type=int, default=10)
    ap.add_argument("--sustained", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--idle", type=float, default=3.0)
    ap.add_argument("--soak", type=float, default=10.0)
    ap.add_argument("--no-sustained", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cases = [tuple(int(v) for v in c.split("x")) for c in args.cases.split(",")]
    snapshot = resolve_snapshot(MODEL_ID)
    device = ttnn.open_device(device_id=0, **DEVICE_PARAMS)
    try:
        enc = VisionEncoder.from_pretrained(device, str(snapshot))
        enc.capture_all(cases)
        for b, s in cases:
            host = enc.prepare_inputs(*_inputs(snapshot, b, s))
            assert (host.batch, host.seq_len) == (b, s)
            soft, _ = enc.replay(host)
            out = VisionEncoder.read_rows(soft, host.num_soft_tokens)
            assert all(torch.isfinite(o).all() for o in out)
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
            msg = (
                f"BENCH {args.label} b{b} s{s}: burst median {rec['burst_ms_median']:.3f} ms "
                f"[{rec['burst_ms_min']:.3f}, {rec['burst_ms_max']:.3f}] ({rec['burst_ms_median'] / b:.3f} ms/img)"
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

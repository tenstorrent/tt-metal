# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# Latency benchmark of the public create_backend(...).forecast path: CPU
# float32 arrays in, CPU quantile arrays out, device synchronized before and
# after each call; weight loading and the first (compiling) call are reported
# separately. Synthetic inputs cover B=2/T=512, B=1/T=512 and B=1/T=65.
#
#   python -m models.experimental.chronos2.benchmarks.benchmark_forecast \
#       --checkpoint /path/to/chronos-2 --reps 20 [--precision bf16] [--eager] [--json out.json]
#
# Results on Blackhole are in docs/validation.md.

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from models.experimental.chronos2.tt import DEFAULT_PRECISION, DEVICE_OPTIONS, SUPPORTED_PRECISIONS, create_backend

SHAPES = ((2, 512), (1, 512), (1, 65))


def _series(B, T, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    v = np.stack([10 + 3 * np.sin(2 * np.pi * t / 24 + b) + 0.3 * rng.standard_normal(T) for b in range(B)])
    return v.astype(np.float32), np.ones((B, T), np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--precision", choices=SUPPORTED_PRECISIONS, default=DEFAULT_PRECISION)
    ap.add_argument("--prediction-length", type=int, default=64)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--eager", action="store_true", help="trace_region_size=0 (no metal trace)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import ttnn

    options = {"trace_region_size": 0} if args.eager else dict(DEVICE_OPTIONS)
    dev = ttnn.open_device(device_id=args.device_id, **options)
    report = {"precision": args.precision, "options": options, "reps": args.reps, "cases": []}
    try:
        be = create_backend(args.checkpoint, None, dev, precision=args.precision, device_options=options)
        for B, T in SHAPES:
            v, m = _series(B, T, seed=T)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            q0 = be.forecast(v, m, args.prediction_length)["quantiles"]
            ttnn.synchronize_device(dev)
            first = time.perf_counter() - t0
            ts, equal = [], True
            for _ in range(args.reps):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                q = be.forecast(v, m, args.prediction_length)["quantiles"]
                ttnn.synchronize_device(dev)
                ts.append(time.perf_counter() - t0)
                equal &= bool(np.array_equal(q, q0))
            case = {
                "B": B,
                "T": T,
                "first_s": first,
                "p50_ms": float(np.median(ts) * 1e3),
                "min_ms": float(np.min(ts) * 1e3),
                "max_ms": float(np.max(ts) * 1e3),
                "repeat_equal": equal,
                "forecasts_per_s": B / float(np.median(ts)),
            }
            report["cases"].append(case)
            print(
                f"[{args.precision}{' eager' if args.eager else ''}] B={B} T={T} first={first:.2f}s "
                f"p50={case['p50_ms']:.2f}ms min={case['min_ms']:.2f} max={case['max_ms']:.2f} "
                f"eq={equal}",
                flush=True,
            )
        report["trace_stats"] = dict(be.executor.trace_stats)
        be.release()
    finally:
        ttnn.close_device(dev)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

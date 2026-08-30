# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reduce ``ops_perf_results_*.csv`` from the light profiler into a per-op breakdown.

Usage:  python3 summarize_attention_perf.py <iters> <label>=<csv> [<label>=<csv> ...]

**Aggregation.** The CSV carries one row per (op instance, device) and this model runs 32 devices in
lockstep, so summing device time would overcount by ~32x. An op instance does not finish until its
slowest participating device finishes, so each instance is reduced with MAX across devices — the
critical path — then instances are summed and divided by the iteration count.

Only rows between the ``start`` and ``stop`` signposts are counted, which drops weight loading, cache
allocation, RoPE table construction and the warmup iterations.
"""

import csv
import json
import sys
from collections import defaultdict

DUR = "DEVICE KERNEL DURATION [ns]"

# Coarse buckets, so the headline is "where does the time go" rather than 10 op names.
BUCKETS = {
    "RingJointSDPADeviceOperation": "SDPA (ring)",
    "ScaledDotProductAttentionDeviceOperation": "SDPA (dense)",
    "MatmulDeviceOperation": "Matmul (QKV + o_proj)",
    "ReduceScatterMinimalAsyncDeviceOperation": "CCL (TP reduce-scatter)",
    "AllGatherAsyncDeviceOperation": "CCL (all-gather)",
    "RotaryEmbeddingIndexedDeviceOperation": "RoPE (indexed)",
    "UpdatePaddedKvCacheDeviceOperation": "KV-cache write",
    "NlpCreateHeadsDeviceOperation": "Head split",
    "NLPConcatHeadsDeviceOperation": "Head concat",
    "TypecastDeviceOperation": "Typecast",
    "TilizeDeviceOperation": "Tilize",
    "GenericOpDeviceOperation": "Generic",
}


def summarize(csv_path, iters):
    rows = list(csv.DictReader(open(csv_path)))
    lo = hi = None
    for i, r in enumerate(rows):
        code = (r.get("OP CODE") or "").strip()
        if code == "start" and lo is None:
            lo = i
        elif code == "stop":
            hi = i
    assert lo is not None and hi is not None, f"signposts not found in {csv_path}"
    window = rows[lo + 1 : hi]

    # One row per (op instance, device). GLOBAL CALL COUNT is per-device, so it cannot group an
    # instance across devices. Instead do what models/tt_transformers merge_device_rows does: keep
    # each device's ops in dispatch order, then match positionally — the Nth op on every device is
    # the same instance — and reduce with MAX, since an instance ends when its slowest device ends.
    by_device = defaultdict(list)
    for r in window:
        code = (r.get("OP CODE") or "").strip()
        if not code or code in ("start", "stop"):
            continue
        if (r.get("OP TYPE") or "").strip() not in ("tt_dnn_device", ""):
            continue
        raw = (r.get(DUR) or "").strip()
        if not raw:
            continue
        try:
            ns = float(raw)
        except ValueError:
            continue
        by_device[r.get("DEVICE ID", "0")].append((code, ns))

    devs = sorted(by_device)
    assert devs, f"no device rows in {csv_path}"
    lens = {d: len(by_device[d]) for d in devs}
    n = min(lens.values())
    if len(set(lens.values())) != 1:
        print(
            f"  note: uneven op counts across devices {sorted(set(lens.values()))}; truncating to {n}", file=sys.stderr
        )

    by_op = defaultdict(lambda: {"ns": 0.0, "n": 0})
    skew_ns = 0.0
    for i in range(n):
        code = by_device[devs[0]][i][0]
        vals = []
        for d in devs:
            c2, ns = by_device[d][i]
            if c2 != code:  # sequences diverged; stop rather than mis-attribute
                n = i
                break
            vals.append(ns)
        else:
            by_op[code]["ns"] += max(vals)
            by_op[code]["n"] += 1
            skew_ns += max(vals) - min(vals)
            continue
        break

    out = []
    for code, v in by_op.items():
        out.append(
            {
                "op": code,
                "bucket": BUCKETS.get(code, code),
                "calls_per_iter": v["n"] / iters,
                "us_per_iter": v["ns"] / iters / 1000.0,
            }
        )
    out.sort(key=lambda d: -d["us_per_iter"])
    total = sum(d["us_per_iter"] for d in out)
    for d in out:
        d["pct"] = 100.0 * d["us_per_iter"] / total if total else 0.0
    return {
        "total_us_per_iter": total,
        "devices": len(devs),
        "device_skew_us_per_iter": skew_ns / iters / 1000.0,
        "ops": out,
    }


def main():
    iters = int(sys.argv[1])
    results = {}
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        results[label] = summarize(path, iters)

    for label, res in results.items():
        print(
            f"\n=== {label} — {res['total_us_per_iter']:.1f} us/iter "
            f"(device kernel, critical path over {res['devices']} devices; "
            f"slowest-minus-fastest device {res['device_skew_us_per_iter']:.1f} us/iter) ==="
        )
        print(f"{'bucket':<28} {'us/iter':>10} {'%':>7} {'calls':>7}")
        print("-" * 56)
        for d in res["ops"]:
            print(f"{d['bucket']:<28} {d['us_per_iter']:>10.1f} {d['pct']:>6.1f}% {d['calls_per_iter']:>7.1f}")
    print(json.dumps(results, indent=None), file=sys.stderr)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Bucket the device time of signposted decode windows from a Tracy ops report (ops_perf_results_*.csv).

    python profile_buckets.py <ops_perf_results.csv> [--steps N] [--device D] [--top 25] [--json out.json]

Each start/stop signpost pair is one window (enhancer_bench.py BENCH_PROFILE_STEPS; one per temperature, in
BENCH_TEMPS order). Per window and device: device kernel time by op class, the gaps between ops (op-to-op
latency), and their sum, all divided by ``--steps`` to give ms per decode step. Ops replayed from the decode
trace carry a METAL TRACE ID; ops without one ran eagerly (the device sampler), so they form their own split.
The table shows one device (``--device``, default the first); the spread across devices is printed below it.
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict

# First match wins; checked against the op code.
CLASSES = (
    ("matmul", r"matmul|linear"),
    ("ccl", r"allgather|all_gather|reducescatter|reduce_scatter|allreduce|all_reduce|broadcast|ccl|fabric"),
    ("norm", r"norm"),
    ("sdpa", r"scaleddotproduct|sdpa|flash"),
    ("kv cache", r"updatecache|update_cache|pagedupdate|paged_update|fillcache|fill_cache"),
    ("rope", r"rotary|rope"),
    ("sampling", r"sampling|topk|top_k|argmax|sort|softmax"),
    (
        "data movement",
        r"reshape|slice|concat|transpose|permute|memoryconfig|interleaved|sharded|copy|tilize|pad|typecast|"
        r"embedding|split|nlp|move|clone|repeat|gather|scatter|view",
    ),
    ("eltwise", r"binary|unary|eltwise|ternary|where|mul|add|sub|tanh|exp|gelu|silu|clamp|reduction|reduce"),
)


def op_class(code):
    c = code.lower()
    for name, pattern in CLASSES:
        if re.search(pattern, c):
            return name
    return "other"


def col(row, *names):
    for n in names:
        if n in row and row[n] not in ("", None):
            return row[n]
    return None


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def windows(rows):
    """Split rows into start..stop windows by signpost rows, in file order."""
    out, cur = [], None
    for r in rows:
        if (col(r, "OP TYPE") or "").lower() == "signpost":
            name = (col(r, "OP CODE") or "").strip()
            if name == "start":
                cur = []
            elif name == "stop" and cur is not None:
                out.append(cur)
                cur = None
            continue
        if cur is not None:
            cur.append(r)
    return out


def summarize(window, steps):
    per_dev = defaultdict(lambda: {"kernel": defaultdict(float), "gap": 0.0, "ops": 0, "per_op": defaultdict(float)})
    for r in window:
        dev = col(r, "DEVICE ID")
        if dev is None:
            continue
        code = col(r, "OP CODE") or "?"
        traced = col(r, "METAL TRACE ID") not in (None, "", "nan")
        cls = op_class(code) if traced else "eager (sampler)"
        k = num(col(r, "DEVICE KERNEL DURATION [ns]")) / 1e6
        d = per_dev[int(float(dev))]
        d["kernel"][cls] += k
        d["gap"] += num(col(r, "OP TO OP LATENCY [ns]")) / 1e6
        d["ops"] += 1
        d["per_op"][(cls, code)] += k
    res = {}
    for dev, d in per_dev.items():
        kernel = {c: v / steps for c, v in d["kernel"].items()}
        res[dev] = {
            "kernel_ms": kernel,
            "kernel_total_ms": sum(kernel.values()),
            "gap_ms": d["gap"] / steps,
            "total_ms": sum(kernel.values()) + d["gap"] / steps,
            "ops_per_step": d["ops"] / steps,
            "per_op_ms": {f"{c} | {code}": v / steps for (c, code), v in d["per_op"].items()},
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--steps", type=int, default=2, help="decode steps per window")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    with open(a.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    wins = windows(rows)
    print(f"{len(rows)} rows, {len(wins)} signposted windows, {a.steps} steps each")
    out = []
    for i, w in enumerate(wins):
        res = summarize(w, a.steps)
        if not res:
            print(f"window {i}: no device rows")
            continue
        dev = a.device if a.device in res else min(res)
        d = res[dev]
        print(f"\n=== window {i}: device {dev}, {d['ops_per_step']:.0f} ops/step ===")
        print(f"{'class':18s} {'ms/step':>9s} {'share':>7s}")
        for c, v in sorted(d["kernel_ms"].items(), key=lambda kv: -kv[1]):
            print(f"{c:18s} {v:9.3f} {100 * v / d['total_ms']:6.1f}%")
        print(f"{'op-to-op gaps':18s} {d['gap_ms']:9.3f} {100 * d['gap_ms'] / d['total_ms']:6.1f}%")
        print(f"{'total':18s} {d['total_ms']:9.3f}")
        totals = [r["total_ms"] for r in res.values()]
        kern = [r["kernel_total_ms"] for r in res.values()]
        print(
            f"across {len(res)} devices: total ms/step min {min(totals):.3f} median {statistics.median(totals):.3f} "
            f"max {max(totals):.3f}; kernel min {min(kern):.3f} max {max(kern):.3f}"
        )
        print(f"top {a.top} ops on device {dev} (ms/step):")
        for name, v in sorted(d["per_op_ms"].items(), key=lambda kv: -kv[1])[: a.top]:
            print(f"  {v:7.3f}  {name}")
        out.append({"window": i, "device": dev, "devices": {str(k): v for k, v in res.items()}})
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

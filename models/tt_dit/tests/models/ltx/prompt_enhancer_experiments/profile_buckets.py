# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Bucket the device time of the drained decode-step windows in a device profiler run.

    python profile_buckets.py <profiler dir or cpp_device_perf_report.csv> [--device D] [--top 25] [--json out.json]

Reads the per-op device report the profiler writes (.logs/cpp_device_perf_report.csv); Tracy's ops report and
the host signposts are not needed, which matters with every device zone on: the host capture stops at Tracy's
32K source-location limit long before the window.

Windows are found on the device timeline. test_enhancer_decode_step.py / enhancer_bench.py drain the profiler
(several seconds) right before and right after each window, so a window is a run of ops with a gap of at
least ``--drain-gap`` seconds on both sides that contains one complete replay of the decode trace. That run
is exactly one decode step: the trace replay plus the eager ops of the step (the device sampler).

Per window and device: device kernel time by op class (eager ops apart, summed over ops), the time some kernel
is running (union of kernel intervals; ops can overlap, so this can be less than the sum), the rest of the
step's span with no kernel running (launch, firmware, dispatch waits), and the span (first op start to last
op end). Firmware intervals of consecutive ops overlap, so firmware time is not summed.
"""

import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict

# First match wins; checked against the op name.
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
REPORT = "cpp_device_perf_report.csv"


def op_class(name):
    n = name.lower()
    for cls, pattern in CLASSES:
        if re.search(pattern, n):
            return cls
    return "other"


def find_report(path):
    if os.path.isfile(path):
        return path
    for root, _, files in os.walk(path):
        if REPORT in files:
            return os.path.join(root, REPORT)
    raise SystemExit(f"no {REPORT} under {path}")


def load(path):
    """Rows per device with cycle timestamps, sorted by start; plus cycles per ns per device."""
    per_dev = defaultdict(list)
    ratios = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r["DEVICE FW START CYCLE"]:
                continue
            start, end = int(r["DEVICE FW START CYCLE"]), int(r["DEVICE FW END CYCLE"])
            fw_ns = float(r["DEVICE FW DURATION [ns]"] or 0)
            if fw_ns > 1000:
                ratios[r["DEVICE ID"]].append((end - start) / fw_ns)
            per_dev[int(r["DEVICE ID"])].append(
                {
                    "name": r["OP NAME"],
                    "trace": r["METAL TRACE ID"],
                    "session": r["METAL TRACE REPLAY SESSION ID"],
                    "start": start,
                    "end": end,
                    "k_start": int(r["DEVICE KERNEL START CYCLE"] or start),
                    "k_end": int(r["DEVICE KERNEL END CYCLE"] or end),
                    "fw_ns": fw_ns,
                    "kernel_ns": float(r["DEVICE KERNEL DURATION [ns]"] or 0),
                }
            )
    cyc_per_ns = {int(d): statistics.median(v) for d, v in ratios.items()}
    for rows in per_dev.values():
        rows.sort(key=lambda r: r["start"])
    return per_dev, cyc_per_ns


def windows(rows, cyc_per_ns, drain_gap_s):
    """Runs of ops separated by drain-sized gaps that hold exactly one complete trace replay. The first and last
    runs have a drain on one side only, so they are never windows."""
    gap = drain_gap_s * 1e9 * cyc_per_ns
    runs, cur = [], [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r["start"] - prev["end"] > gap:
            runs.append(cur)
            cur = []
        cur.append(r)
    runs.append(cur)
    out = []
    for run in runs[1:-1]:
        sessions = {(r["trace"], r["session"]) for r in run if r["trace"]}
        if len(sessions) == 1:
            out.append(run)
    return out


def summarize(run, cyc_per_ns):
    kernel = defaultdict(float)
    per_op = defaultdict(float)
    per_op_calls = defaultdict(int)
    kern_total = 0.0
    for r in run:
        cls = op_class(r["name"]) if r["trace"] else f"eager: {op_class(r['name'])}"
        kernel[cls] += r["kernel_ns"] / 1e6
        per_op[f"{cls} | {r['name']}"] += r["kernel_ns"] / 1e6
        per_op_calls[f"{cls} | {r['name']}"] += 1
        kern_total += r["kernel_ns"] / 1e6
    busy = 0
    cur_s = cur_e = None
    for a, b in sorted((r["k_start"], r["k_end"]) for r in run):
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    busy += cur_e - cur_s
    busy_ms = busy / cyc_per_ns / 1e6
    span = (max(r["end"] for r in run) - run[0]["start"]) / cyc_per_ns / 1e6
    traced = [r for r in run if r["trace"]]
    trace_span = (max(r["end"] for r in traced) - min(r["start"] for r in traced)) / cyc_per_ns / 1e6
    return {
        "ops": len(run),
        "traced_ops": len(traced),
        "session": traced[0]["session"],
        "span_ms": span,
        "trace_span_ms": trace_span,
        "kernel_ms": dict(kernel),
        "kernel_total_ms": kern_total,
        "kernel_busy_ms": busy_ms,
        "no_kernel_ms": span - busy_ms,
        "per_op_ms": dict(per_op),
        "per_op_calls": dict(per_op_calls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--drain-gap", type=float, default=1.0, help="seconds; gaps this long are profiler drains")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    per_dev, cyc_per_ns = load(find_report(a.path))
    devices = sorted(per_dev)
    dev = a.device if a.device in per_dev else devices[0]
    per_window = defaultdict(dict)
    for d in devices:
        for i, run in enumerate(windows(per_dev[d], cyc_per_ns[d], a.drain_gap)):
            per_window[i][d] = summarize(run, cyc_per_ns[d])
    print(f"devices {devices}; {len(per_window)} windows (one decode step each)")

    for i, res in per_window.items():
        if dev not in res:
            continue
        s = res[dev]
        total = s["span_ms"]
        print(
            f"\n=== window {i}: device {dev}, trace session {s['session']}, {s['ops']} ops "
            f"({s['traced_ops']} traced) ==="
        )
        print(f"{'':24s} {'ms':>8s} {'share':>7s}")
        for cls, v in sorted(s["kernel_ms"].items(), key=lambda kv: -kv[1]):
            print(f"{cls:24s} {v:8.3f} {100 * v / total:6.1f}%")
        print(f"{'sum of kernels':24s} {s['kernel_total_ms']:8.3f}")
        print(f"{'some kernel running':24s} {s['kernel_busy_ms']:8.3f} {100 * s['kernel_busy_ms'] / total:6.1f}%")
        print(f"{'no kernel running':24s} {s['no_kernel_ms']:8.3f} {100 * s['no_kernel_ms'] / total:6.1f}%")
        print(f"{'step span':24s} {total:8.3f}   (decode trace alone {s['trace_span_ms']:.3f})")
        spans = [r["span_ms"] for r in res.values()]
        print(
            f"across {len(res)} devices: step span min {min(spans):.3f} median {statistics.median(spans):.3f} "
            f"max {max(spans):.3f} ms"
        )
        print(f"top {a.top} ops on device {dev} (ms summed over calls, calls, us per call):")
        for name, v in sorted(s["per_op_ms"].items(), key=lambda kv: -kv[1])[: a.top]:
            n = s["per_op_calls"][name]
            print(f"  {v:7.3f}  {n:5d}  {1000 * v / n:7.2f}  {name}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({str(i): {str(d): v for d, v in res.items()} for i, res in per_window.items()}, f, indent=1)


if __name__ == "__main__":
    main()

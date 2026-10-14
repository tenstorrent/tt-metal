# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Matmul, SDPA and CCL shares of layer time from test_prefill_layer_perf_chunk_n profiler runs.

    python models/demos/gemma4_d_p/scripts/layer_perf_shares.py <dir> [<dir> ...] --out shares.csv [--context 262144]

Each dir is a tracy -o directory (its reports/**/ops_perf_results_*.csv is read; no tt_perf_report needed). For every
signposted cell and device: layer span = last op's FW end - first op's FW start at 1.35 GHz; matmul times are the device
kernel durations of the layer's five matmuls in program order (QKV/QK, o_proj, gate, up, down). Medians over devices.

Also prints a whole-prefill estimate per chunk size C: context/C chunks x (50 sliding layers at chunk 1 + 10 global
layers at the mean of their first and last measured chunk). This is device time from the layer traces, not end-to-end.
"""

import argparse
import csv
import glob
import re
import statistics as st
from collections import defaultdict

FREQ_GHZ = 1.35
MATMULS = ["qkv", "o", "gate", "up", "down"]
SLIDING_LAYERS, GLOBAL_LAYERS = 50, 10
SIGNPOST = re.compile(r"^gemma4-layer-(global|local)-sz([0-9]+)-chunk([0-9]+)-(start|stop)$")


def cells_of(path):
    """{(layer_type, chunk_size, chunk_idx): {device: [op rows]}} of one ops CSV."""
    cells, cur = defaultdict(lambda: defaultdict(list)), None
    for r in csv.DictReader(open(path)):
        if r["OP TYPE"] == "signpost":
            m = SIGNPOST.match(r["OP CODE"])
            cur = (m.group(1), int(m.group(2)), int(m.group(3))) if m and m.group(4) == "start" else None
        elif cur and r["DEVICE KERNEL DURATION [ns]"]:
            cells[cur][r["DEVICE ID"]].append(r)
    return cells


def kernel_ns(rows, pred=lambda r: True):
    return sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows if pred(r))


def summarize(cell, devices):
    per = defaultdict(list)
    for rows in devices.values():
        start = min(int(r["DEVICE FW START CYCLE"]) for r in rows)
        end = max(int(r["DEVICE FW END CYCLE"]) for r in rows)
        matmuls = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows if r["OP CODE"].startswith("Matmul")]
        if len(matmuls) != len(MATMULS):
            raise ValueError(f"{cell}: expected {len(MATMULS)} matmuls per layer, found {len(matmuls)}")
        per["span"].append((end - start) / FREQ_GHZ)
        per["kernel_sum"].append(kernel_ns(rows))
        per["matmul"].append(sum(matmuls))
        per["sdpa"].append(kernel_ns(rows, lambda r: "SDPA" in r["OP CODE"]))
        per["ccl"].append(kernel_ns(rows, lambda r: "Gather" in r["OP CODE"] or "Scatter" in r["OP CODE"]))
        for name, t in zip(MATMULS, matmuls):
            per[name].append(t)
    layer, size, idx = cell
    row = dict(layer=layer, chunk=size, chunk_idx=idx, devices=len(devices))
    row.update({f"{k}_us": st.median(v) / 1e3 for k, v in per.items()})
    row["span_spread_pct"] = 100 * (max(per["span"]) - min(per["span"])) / st.median(per["span"])
    row["matmul_share_of_span_pct"] = 100 * row["matmul_us"] / row["span_us"]
    row["matmul_ns_per_token"] = row["matmul_us"] * 1e3 / (size // 8)
    return row


def prefill_estimate(rows, context):
    """{chunk: {metric: seconds}} over a whole prefill of `context` tokens."""
    out = {}
    for size in sorted({r["chunk"] for r in rows}):
        glob_ = sorted((r for r in rows if r["layer"] == "global" and r["chunk"] == size), key=lambda r: r["chunk_idx"])
        local = next((r for r in rows if r["layer"] == "local" and r["chunk"] == size and r["chunk_idx"] == 1), None)
        if not glob_ or local is None:
            continue
        n = context // size

        def total(key):
            return n * (SLIDING_LAYERS * local[key] + GLOBAL_LAYERS * (glob_[0][key] + glob_[-1][key]) / 2) / 1e6

        out[size] = {k: total(f"{k}_us") for k in ["span", "matmul", "sdpa", "ccl", *MATMULS]}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--context", type=int, default=262144)
    args = p.parse_args()

    rows = []
    for d in args.dirs:
        for path in sorted(glob.glob(f"{d}/reports/**/ops_perf_results_*.csv", recursive=True)):
            rows += [summarize(cell, devs) for cell, devs in cells_of(path).items()]
    if not rows:
        raise SystemExit(f"no signposted layer cells in {args.dirs}")
    rows.sort(key=lambda r: (r["layer"], r["chunk"], r["chunk_idx"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(
            f"{r['layer']:6s} C={r['chunk']:5d} chunk{r['chunk_idx']:<3d} span {r['span_us']:8.1f} us  "
            f"matmul {r['matmul_us']:7.1f} ({r['matmul_share_of_span_pct']:4.1f}%)  sdpa {r['sdpa_us']:7.1f}  "
            f"ccl {r['ccl_us']:6.1f}  | " + " ".join(f"{n} {r[n + '_us']:.1f}" for n in MATMULS)
        )
    print(f"\nWhole prefill of {args.context} tokens ({SLIDING_LAYERS} sliding + {GLOBAL_LAYERS} global layers):")
    for size, t in prefill_estimate(rows, args.context).items():
        print(
            f"C={size:5d}  total {t['span']:6.2f} s  matmul {t['matmul']:5.2f} s ({100 * t['matmul'] / t['span']:4.1f}%)  "
            f"sdpa {t['sdpa']:5.2f} s ({100 * t['sdpa'] / t['span']:4.1f}%)  ccl {t['ccl']:4.2f} s"
        )


if __name__ == "__main__":
    main()

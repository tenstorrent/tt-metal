#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Summarise a prefill-matrix results JSONL (matrix_producer.py output) as (cached x new) tables.

Idle-pipeline rows (mode "idle", one per timed iteration): median TTFT over iterations >= 1 (iteration 0 is the cold
one; cells run with ITERS=1 fall back to it) and new tok/s = new / median TTFT. Loaded rows (mode "loaded", one per
cell): steady-state aggregate tok/s (fill/drain excluded), aggregate incl. tails, and per-request TTFT under load
(median / p90). If a cell was measured more than once, the LAST loaded measurement wins. Stdlib only."""

import argparse
import csv
import json
import statistics as st

# Agent X dataset percentiles the default matrix values correspond to (see README).
NEW_PCT = {640: "p25", 1600: "p50", 3072: "p75", 6900: "p90", 51200: "p99"}
CACHED_PCT = {61440: "p25", 143360: "p50", 312320: "p75", 552960: "p90", 860160: "p99"}

LOADED_TABLES = [  # (title, JSONL key) -- the steady_* keys are absent when a stream had too few chunks
    ("steady-state aggregate NEW tok/s (fill/drain excluded)", "steady_new_tps"),
    ("steady-state PROCESSED tok/s (5120 per chunk)", "steady_processed_tps"),
    ("aggregate NEW tok/s incl. fill/drain", "aggregate_new_tps"),
    ("per-request TTFT ms median", "ttft_under_load_ms_median"),
    ("per-request TTFT ms p90", "ttft_under_load_ms_p90"),
]
CSV_LOADED_KEYS = [k for _, k in LOADED_TABLES]


def load(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def idle_ttfts(rows, iter0):
    """{(cached, new): [ttft_ms, ...]} from the idle rows: iterations >= 1, falling back to iteration 0 for cells that
    only have it; with ``iter0`` the cold iteration only."""
    per_iter = {}
    for r in rows:
        if r["mode"] == "idle":
            per_iter.setdefault((r["cached"], r["new"]), {}).setdefault(r["iter"], []).append(r["ttft_ms"])
    out = {}
    for k, v in per_iter.items():
        warm = [x for i, xs in sorted(v.items()) if i >= 1 for x in xs]
        vals = v.get(0, []) if iter0 else (warm or v.get(0, []))
        if vals:
            out[k] = vals
    return out


def _lbl(v, pct):
    return f"{v} ({pct[v]})" if v in pct else str(v)


def table(title, cells, cached, news):
    w = 12
    print(f"\n{title}   rows: cached tokens, cols: new tokens (Agent X percentiles in parentheses)")
    print(f"{'cached':>14} |" + "".join(f"{_lbl(n, NEW_PCT):>{w}}" for n in news))
    print("-" * (16 + w * len(news)))
    for c in cached:
        print(
            f"{_lbl(c, CACHED_PCT):>14} |"
            + "".join(f"{cells[(c, n)]:>{w}.0f}" if (c, n) in cells else f"{'-':>{w}}" for n in news)
        )


def write_csv(path, cached, news, med, last):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cached", "new", "idle_ttft_ms_median", "idle_new_tps"] + [f"loaded_{k}" for k in CSV_LOADED_KEYS])
        for c in cached:
            for n in news:
                m, lr = med.get((c, n)), last.get((c, n), {})
                row = [c, n, f"{m:.1f}" if m else "", f"{n / (m / 1000):.0f}" if m else ""]
                row += [f"{lr[k]:.0f}" if lr.get(k) is not None else "" for k in CSV_LOADED_KEYS]
                w.writerow(row)
    print(f"\nCSV written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results JSONL written by matrix_producer.py")
    ap.add_argument(
        "--iter0", action="store_true", help="idle tables from the cold iteration 0 instead of iterations >= 1"
    )
    ap.add_argument("--csv", metavar="OUT", help="also write one CSV row per (cached, new) cell")
    args = ap.parse_args()
    rows = load(args.results)
    med = {k: st.median(v) for k, v in idle_ttfts(rows, args.iter0).items()}
    loaded = [r for r in rows if r["mode"] == "loaded"]
    last = {(r["cached"], r["new"]): r for r in loaded}  # last measurement per cell wins
    cached = sorted({c for c, _ in med} | {c for c, _ in last})
    news = sorted({n for _, n in med} | {n for _, n in last})
    tag = " [iteration 0 only]" if args.iter0 else " [iterations >= 1]"
    if med:
        table("IDLE pipeline: TTFT ms (median)" + tag, med, cached, news)
        table(
            "IDLE pipeline: new tok/s (new / median TTFT)" + tag,
            {k: k[1] / (v / 1000) for k, v in med.items()},
            cached,
            news,
        )
    for title, key in LOADED_TABLES:
        cells = {k: r[key] for k, r in last.items() if r.get(key) is not None}
        if cells:
            table("LOADED pipeline: " + title, cells, cached, news)
    if loaded:
        print(
            f"\n(loaded pass: users={sorted({r['users'] for r in loaded})}, requests per cell {sorted({r['requests'] for r in loaded})})"
        )
    if args.csv:
        write_csv(args.csv, cached, news, med, last)


if __name__ == "__main__":
    main()

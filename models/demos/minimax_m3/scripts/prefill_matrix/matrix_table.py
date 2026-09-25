#!/usr/bin/env python3
"""Summarise a prefill-matrix results JSONL (matrix_producer.py output) as (cached x new) tables.

Idle-pipeline rows (mode "idle", one per timed iteration): median TTFT over iterations >= 1 (iteration 0 is the cold
one) and new tok/s = new / median TTFT. Loaded rows (mode "loaded", one per cell): steady-state aggregate tok/s
(fill/drain excluded), aggregate incl. tails, and per-request TTFT under load (median / p90).

Usage: matrix_table.py results.jsonl [--iter0] [--csv out.csv]"""

import csv
import json
import statistics as st
import sys


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for r in rows:
        r.setdefault("mode", "idle")
    return rows


# Agent X dataset percentiles the default matrix values correspond to (see README).
NEW_PCT = {640: "p25", 1600: "p50", 3072: "p75", 51200: "p99"}
CACHED_PCT = {61440: "p25", 143360: "p50", 312320: "p75", 552960: "p90"}


def _lbl(v, pct):
    return f"{v} ({pct[v]})" if v in pct else str(v)


def table(title, cells, cached, news, fmt):
    w = 12
    print(f"\n{title}   rows: cached tokens, cols: new tokens (Agent X percentiles in parentheses)")
    print(f"{'cached':>14} |" + "".join(f"{_lbl(n, NEW_PCT):>{w}}" for n in news))
    print("-" * (16 + w * len(news)))
    for c in cached:
        print(
            f"{_lbl(c, CACHED_PCT):>14} |"
            + "".join(f"{fmt(cells[(c, n)]):>{w}}" if (c, n) in cells else f"{'-':>{w}}" for n in news)
        )


def main():
    path = sys.argv[1]
    iter0 = "--iter0" in sys.argv
    csv_out = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    rows = load(path)
    idle = [r for r in rows if r["mode"] == "idle" and ((r["iter"] == 0) if iter0 else (r["iter"] >= 1))]
    loaded = [r for r in rows if r["mode"] == "loaded"]
    ttft = {}
    for r in idle:
        ttft.setdefault((r["cached"], r["new"]), []).append(r["ttft_ms"])
    cached = sorted({c for c, _ in ttft} | {r["cached"] for r in loaded})
    news = sorted({n for _, n in ttft} | {r["new"] for r in loaded})
    med = {k: st.median(v) for k, v in ttft.items()}
    tag = " [iteration 0 only]" if iter0 else " [iterations >= 1]"
    if med:
        table("IDLE pipeline: TTFT ms (median)" + tag, med, cached, news, lambda v: f"{v:.0f}")
        table(
            "IDLE pipeline: new tok/s (new / median TTFT)" + tag,
            {k: k[1] / (v / 1000) for k, v in med.items()},
            cached,
            news,
            lambda v: f"{v:.0f}",
        )
    if loaded:
        last = {}
        for r in loaded:  # keep the last measurement per cell if repeated
            last[(r["cached"], r["new"])] = r
        table(
            "LOADED pipeline: steady-state aggregate NEW tok/s (fill/drain excluded)",
            {k: r.get("steady_new_tps") for k, r in last.items() if r.get("steady_new_tps")},
            cached,
            news,
            lambda v: f"{v:.0f}",
        )
        table(
            "LOADED pipeline: steady-state PROCESSED tok/s (5120 per chunk)",
            {k: r.get("steady_processed_tps") for k, r in last.items() if r.get("steady_processed_tps")},
            cached,
            news,
            lambda v: f"{v:.0f}",
        )
        table(
            "LOADED pipeline: aggregate NEW tok/s incl. fill/drain",
            {k: r["aggregate_new_tps"] for k, r in last.items()},
            cached,
            news,
            lambda v: f"{v:.0f}",
        )
        table(
            "LOADED pipeline: per-request TTFT ms median",
            {k: r["ttft_under_load_ms_median"] for k, r in last.items()},
            cached,
            news,
            lambda v: f"{v:.0f}",
        )
        table(
            "LOADED pipeline: per-request TTFT ms p90",
            {k: r["ttft_under_load_ms_p90"] for k, r in last.items()},
            cached,
            news,
            lambda v: f"{v:.0f}",
        )
        u = {r["users"] for r in loaded}
        print(f"\n(loaded pass: users={sorted(u)}, requests per cell {sorted({r['requests'] for r in loaded})})")
    if csv_out:
        with open(csv_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "cached",
                    "new",
                    "idle_ttft_ms_median",
                    "idle_new_tps",
                    "loaded_steady_new_tps",
                    "loaded_steady_processed_tps",
                    "loaded_aggregate_new_tps",
                    "loaded_ttft_ms_median",
                    "loaded_ttft_ms_p90",
                ]
            )
            for c in cached:
                for n in news:
                    m = med.get((c, n))
                    lr = next((r for r in reversed(loaded) if (r["cached"], r["new"]) == (c, n)), None)
                    w.writerow(
                        [c, n, f"{m:.1f}" if m else "", f"{n / (m / 1000):.0f}" if m else ""]
                        + (
                            [
                                f"{lr.get('steady_new_tps', float('nan')):.0f}",
                                f"{lr.get('steady_processed_tps', float('nan')):.0f}",
                                f"{lr['aggregate_new_tps']:.0f}",
                                f"{lr['ttft_under_load_ms_median']:.0f}",
                                f"{lr['ttft_under_load_ms_p90']:.0f}",
                            ]
                            if lr
                            else ["", "", "", "", ""]
                        )
                    )
        print(f"\nCSV written to {csv_out}")


if __name__ == "__main__":
    main()

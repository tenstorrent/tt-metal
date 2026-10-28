# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Summarize a reduced-layer profile: section device times, the full-depth estimate, op categories, top ops.

    python models/experimental/kairos_forecast/tests/perf/summarize_profile.py <iter_dir> [ops_perf_results.csv]

Reads the Tracy ops CSV (default: the newest under <iter_dir>/tracy/reports), layer_counts.json and, when present,
bench.json (the measured target device time, and BENCH_LAYERS records for the reduced-depth check). Writes
summary.json and summary.md in <iter_dir>.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

KERNEL = "DEVICE KERNEL DURATION [ns]"
GAP = "OP TO OP LATENCY [ns]"
TOP = 15


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def sections(rows):
    """[(section name, [op rows])] between KAIROS_START and KAIROS_END, split at every signpost."""
    out, cur, on = [], None, False
    for r in rows:
        if r["OP TYPE"] == "signpost":
            name = r["OP CODE"]
            if name == "KAIROS_START":
                on = True
            elif name == "KAIROS_END":
                break
            elif on:
                cur = (name, [])
                out.append(cur)
        elif on and cur is not None:
            cur[1].append(r)
    return out


def main(iter_dir: Path, csv_path: Path | None = None):
    csv_path = csv_path or sorted(iter_dir.glob("tracy/reports/*/ops_perf_results_*.csv"))[-1]
    rows = list(csv.DictReader(csv_path.open()))
    meta = json.loads((iter_dir / "layer_counts.json").read_text())
    counts = meta["layer_counts"]
    per = defaultdict(list)
    ops = []
    for name, rs in sections(rows):
        per[name].append(sum(_f(r[KERNEL]) for r in rs) / 1e6)
        ops += [(name, r) for r in rs]
    sec = {}
    est = 0.0
    for name, times in per.items():
        c = counts.get(name, 1)
        profiled, full = (c if isinstance(c, list) else [1, 1])
        mean = sum(times) / len(times)
        sec[name] = {"profiled": len(times), "mean_ms": mean, "full_count": full, "full_ms": mean * full}
        est += mean * full
    reduced = sum(sum(t) for t in per.values())
    cats = defaultdict(float)
    for _name, r in ops:
        cats[r["OP CODE"]] += _f(r[KERNEL]) / 1e6
    top = sorted(ops, key=lambda x: -_f(x[1][KERNEL]))[:TOP]
    gaps = sum(_f(r[GAP]) for _n, r in ops) / 1e6
    summary = {"csv": str(csv_path), "shape": meta["shape"], "precision": meta["precision"], "rows": meta["rows"],
               "sections": sec, "reduced_kernel_ms": reduced, "full_estimate_ms": est, "op_to_op_gap_ms": gaps,
               "ops": len(ops), "categories_ms": dict(sorted(cats.items(), key=lambda kv: -kv[1]))}
    for bench in (iter_dir / "bench.json", iter_dir / "bench_reduced.json"):
        if not bench.is_file():
            continue
        for r in json.loads(bench.read_text()):
            if r.get("name") == meta["shape"] and r.get("precision") == meta["precision"] and r.get("test") == "generate":
                key = "measured_full_ms" if r.get("layers") == [6, 6] else "measured_reduced_ms"
                # the bench times every device forward of a generate(); paper (H <= 128) is one forward of both halves
                summary[key] = r["device_ms"] / len(r["rows_per_forward"])
    if "measured_full_ms" in summary:
        summary["estimate_vs_measured"] = est / summary["measured_full_ms"] - 1
    if "measured_reduced_ms" in summary:
        summary["reduced_vs_measured"] = reduced / summary["measured_reduced_ms"] - 1
    (iter_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [f"# Profile summary: {meta['shape']}/{meta['precision']}, {meta['rows']} rows", "",
             f"- reduced forward kernel time {reduced:.2f} ms over {len(ops)} ops; op-to-op gaps {gaps:.2f} ms "
             "(eager, not in the traced bench)",
             f"- full-depth estimate {est:.2f} ms"
             + (f"; measured {summary['measured_full_ms']:.2f} ms ({100 * summary['estimate_vs_measured']:+.1f} %)"
                if "measured_full_ms" in summary else ""), "",
             "| section | profiled | mean ms | x full | full ms |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['profiled']} | {v['mean_ms']:.3f} | {v['full_count']} | {v['full_ms']:.2f} |"
              for k, v in sec.items()]
    lines += ["", "| op | ms (reduced) | share |", "|---|---|---|"]
    lines += [f"| {k} | {v:.3f} | {100 * v / reduced:.1f} % |" for k, v in list(summary["categories_ms"].items())[:12]]
    lines += ["", f"Top {TOP} ops:", "", "| section | op | ms | cores | fidelity | in0 shape |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {r['OP CODE']} | {_f(r[KERNEL]) / 1e6:.3f} | {r['CORE COUNT']} | {r['MATH FIDELITY']} | "
              f"{r.get('INPUT_0_Y_PAD[LOGICAL]', '')} x {r.get('INPUT_0_X_PAD[LOGICAL]', '')} |" for n, r in top]
    (iter_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + len(sec) + 2]))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)

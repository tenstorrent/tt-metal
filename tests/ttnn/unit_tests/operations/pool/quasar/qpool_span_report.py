# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fit span(T) = prologue + marginal * T from a craq-sim per-dispatch perf trace produced by
test_qpool_span.py, per Debin's Quasar perf-measurement procedure:

  - span      : kernel envelope per iteration (here: the merged halo+pool dispatch clocks;
                CORES=1 makes the program envelope the per-cluster envelope)
  - T         : tiles per cluster (output sticks x in_ntiles_c), parsed from the phase labels
  - prologue  : fitted intercept — T-independent setup cost (grows with thread count)
  - marginal  : fitted slope — steady-state cycles per tile (lower is better)

Given ONE trace: prints the measured spans (median over iterations per T) and the fit.
Given TWO traces (baseline first, config second): additionally prints
  - marginal gain  = marginal_baseline / marginal_config   (asymptotic, prologue-free)
  - span gain      = span_baseline(T) / span_config(T) at each measured T (the real gain)
  - crossover T*   = (prologue_cfg - prologue_base) / (marginal_base - marginal_cfg),
                     the T above which the config wins end-to-end (when defined)

Usage:
  python qpool_span_report.py <trace.tsv>
  python qpool_span_report.py <baseline.tsv> <config.tsv> [baseline_name config_name]

SIM CLOCKS — relative comparison on the same sim build only; real numbers come from the
emulator with per-kernel profiler zones.
"""

import csv
import re
import statistics
import sys
from collections import defaultdict

LABEL_RE = re.compile(r"^t(\d+)_i\d+$")


def load_spans(path):
    """Returns {T: [span, ...]} from measured-iteration rows."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or "nodeid" not in rows[0] or "clocks" not in rows[0]:
        raise SystemExit(f"{path}: need a per-dispatch trace with nodeid + clocks columns")
    spans = defaultdict(list)
    for r in rows:
        m = LABEL_RE.match(r["nodeid"])
        if m:
            spans[int(m.group(1))].append(int(r["clocks"]))
    if len(spans) < 2:
        raise SystemExit(f"{path}: need >= 2 T points to fit a line (found {sorted(spans)})")
    return dict(spans)


def fit(spans):
    """Least-squares span(T) = prologue + marginal*T over per-T medians. Returns
    (prologue, marginal, r2, {T: median_span})."""
    med = {t: statistics.median(v) for t, v in spans.items()}
    ts, ys = zip(*sorted(med.items()))
    n = len(ts)
    mean_t, mean_y = sum(ts) / n, sum(ys) / n
    sxx = sum((t - mean_t) ** 2 for t in ts)
    sxy = sum((t - mean_t) * (y - mean_y) for t, y in zip(ts, ys))
    marginal = sxy / sxx
    prologue = mean_y - marginal * mean_t
    ss_res = sum((y - (prologue + marginal * t)) ** 2 for t, y in zip(ts, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return prologue, marginal, r2, med


def report_one(name, spans):
    prologue, marginal, r2, med = fit(spans)
    print(f"\n{name}:")
    print(f"  {'T (tiles)':>10} {'span med':>10} {'span/T':>10}  iters")
    for t in sorted(med):
        print(f"  {t:>10} {med[t]:>10.0f} {med[t] / t:>10.1f}  {sorted(spans[t])}")
    print(f"  fit: span(T) = {prologue:.0f} + {marginal:.1f} * T   (R^2 = {r2:.4f})")
    if r2 < 0.98:
        print("  WARNING: fit is not clean — something besides T scales; inspect before quoting numbers")
    return prologue, marginal, med


def main(argv):
    if len(argv) == 2:
        report_one(argv[1], load_spans(argv[1]))
        return 0
    base_name = argv[3] if len(argv) > 3 else "baseline"
    cfg_name = argv[4] if len(argv) > 4 else "config"
    p_b, m_b, med_b = report_one(base_name, load_spans(argv[1]))
    p_c, m_c, med_c = report_one(cfg_name, load_spans(argv[2]))

    print(f"\nGAINS ({base_name} vs {cfg_name}, >1 means {cfg_name} is faster; SIM-relative only):")
    print(f"  marginal gain (asymptotic): {m_b / m_c:.3f}x   ({m_b:.1f} -> {m_c:.1f} cycles/tile)")
    print(f"  prologue: {p_b:.0f} -> {p_c:.0f} cycles ({p_c - p_b:+.0f} threading tax)")
    for t in sorted(set(med_b) & set(med_c)):
        print(f"  span gain @ T={t:<5}: {med_b[t] / med_c[t]:.3f}x")
    if m_b > m_c:
        t_star = (p_c - p_b) / (m_b - m_c)
        print(f"  crossover T* = {t_star:.1f} tiles ({cfg_name} wins for T > T*)")
    else:
        print(f"  no crossover: {cfg_name} marginal is not better ({m_c:.1f} >= {m_b:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

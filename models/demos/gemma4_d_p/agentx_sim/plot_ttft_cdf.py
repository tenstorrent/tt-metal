#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTFT distributions (CDFs) of agentx_sim.py runs, one curve per run, to compare designs on one plot.

    agentx_sim.py --traces <dir>/traces.jsonl --config 8192x8192:fcfs,4096x4096:fcfs,2048x8192:pack \
        --concurrency 64 -j 3 --ttft-out ttft.jsonl
    plot_ttft_cdf.py ttft.jsonl --out ttft_cdf.png [--by-size] [--where concurrency=64 --where cache=inf]

Curves are labelled by the knobs that differ between the plotted runs; the ones they share go in the title. --by-size
draws one panel per new-token bucket (as in agentx_sim's summary), to show whether small requests wait behind big ones.
The TTFT axis is log, since TTFT spans several decades; --xmax S draws it linear from 0 to S s and cuts the tail off.
Needs matplotlib (not in the tt-metal environment: pip install matplotlib).
"""

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_sim import BUCKETS  # noqa: E402

KNOBS = ("config", "concurrency", "cache", "align", "mid_cost", "max_think_s", "decode_tps")
# Categorical palette in a fixed order, so a run keeps its color across plots of the same sweep.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8f5bd6", "#d6457a", "#4c9f9f", "#7a7a72"]


def matches(value, want):
    """A knob's value against a --where value: as numbers when both are (60 matches 60.0), else as strings."""
    try:
        return float(value) == float(want)
    except (TypeError, ValueError):
        return str(value) == want


def load_runs(paths, where):
    """Runs from --ttft-out files, kept when every --where KNOB=VALUE matches."""
    runs = []
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                r.setdefault("mid_cost", 0.0)  # exports from before --mid-cost
                if all(matches(r.get(k), v) for k, v in where):
                    runs.append(r)
    return runs


def design(cfg):
    """UNITxBUDGET:POLICY in words: 8192x8192:fcfs is "8k", 2048x8192:pack "2k x4 lanes, shortest-first"."""
    unit_budget, policy = cfg.split(":")
    unit, budget = (int(x) for x in unit_budget.split("x"))
    name = f"{unit // 1024}k" if unit % 1024 == 0 else str(unit)
    if budget > unit:
        name += f" x{budget // unit} lanes"
    return name + (", shortest-first" if policy == "pack" else "")


def knob(k, v):
    if k == "config":
        return design(v)
    if k == "cache":
        return "unlimited cache" if v == "inf" else f"{float(v) / 1e6:g}M-token cache"
    if k == "concurrency":
        return f"{v} sessions"
    if k == "align":
        return {"resume": "resume at the model's granularity"}.get(v, f"{v}-aligned resume")
    if k == "mid_cost":
        return f"mid-chunk cost +{100 * float(v):g}%"
    if k == "max_think_s":
        return f"think cap {float(v):g} s"
    return f"decode {float(v):g} tok/s"


def labels(runs):
    """(curve label per run, title suffix): the knobs that vary between runs label the curves, the rest are shared."""
    varying = [k for k in KNOBS if len({str(r.get(k)) for r in runs}) > 1] or ["config"]
    shared = [k for k in KNOBS if k not in varying and k in runs[0]]
    return [", ".join(knob(k, r[k]) for k in varying) for r in runs], ", ".join(knob(k, runs[0][k]) for k in shared)


def cdf(xs):
    xs = sorted(xs)
    return xs, [(i + 1) / len(xs) for i in range(len(xs))]


def plot(runs, out, by_size=False, title=None, xmax=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, shared = labels(runs)
    panels = [(name, lo, hi) for lo, hi, name in BUCKETS] if by_size else [("all requests", -1, 1 << 40)]
    fig, axes = plt.subplots(*((2, 2) if by_size else (1, 1)), figsize=(12, 8) if by_size else (8, 4.8), squeeze=False)
    for ax, (panel, lo, hi) in zip(axes.flat, panels):
        for i, (r, name) in enumerate(zip(runs, names)):
            ttft = [t for t, n in zip(r["ttft_s"], r["new_tokens"]) if lo < n <= hi]
            if ttft:
                ax.plot(*cdf(ttft), color=COLORS[i % len(COLORS)], label=name)
        if xmax:
            ax.set_xlim(0, xmax)
        else:
            ax.set_xscale("log")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 0.9, 0.99])
        ax.grid(True, alpha=0.4)
        ax.set_xlabel(f"TTFT (s, cut at {xmax:g} s)" if xmax else "TTFT (s, log scale)")
        ax.set_ylabel("fraction of requests")
        if by_size:
            ax.set_title(f"{panel} new tokens", loc="left", fontsize=11)
    axes.flat[0].legend(fontsize=9, loc="lower right", frameon=False)
    title = title or f"TTFT distribution, simulated: {shared}"
    fig.suptitle(textwrap.fill(title, 140 if by_size else 90), x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ttft", nargs="+", help="--ttft-out files of agentx_sim.py")
    ap.add_argument("--out", required=True, help="PNG to write")
    ap.add_argument("--by-size", action="store_true", help="one panel per new-token bucket")
    ap.add_argument("--where", action="append", default=[], help="KNOB=VALUE: plot only the runs that match")
    ap.add_argument("--title")
    ap.add_argument("--xmax", type=float, help="linear TTFT axis from 0 to XMAX s instead of log (the tail is cut off)")
    a = ap.parse_args()
    where = []
    for w in a.where:
        k, sep, v = w.partition("=")
        if not sep or k not in KNOBS:
            ap.error(f"bad --where {w!r}: want KNOB=VALUE, KNOB one of {', '.join(KNOBS)}")
        where.append((k, v))
    runs = load_runs(a.ttft, where)
    if not runs:
        raise SystemExit("no runs match")
    plot(runs, a.out, a.by_size, a.title, a.xmax)
    print(f"{len(runs)} runs -> {a.out}")


if __name__ == "__main__":
    main()

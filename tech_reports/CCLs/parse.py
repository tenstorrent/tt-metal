#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Turn one benchmark run into data/runs/<run>/cells.csv, one row per measured cell.

    python tech_reports/CCLs/parse.py data/runs/<run>

Reads the run's configs.jsonl, written by the test, and the profiler CSV that
`python -m tracy -o <run>/profiler` left beside it. Only measurements go into
the CSV; report.py merges the runs and derives every metric.
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

DURATION_COL = "DEVICE KERNEL DURATION [ns]"
DEVICE_COL = "DEVICE ID"
IDEAL_COL = "PM IDEAL [ns]"
ATTRS_COL = "ATTRIBUTES"


# ------------------------------------------------------- fabric discovery
#
# The link count comes from the op's own ATTRIBUTES. The key names are
# op-authored with no contract, so this tries the known names first, then falls
# back to a fuzzy match, and blanks rather than guesses when ambiguous. `ttnn`
# exposes no equivalent query: get_num_links is not bound, and
# len(get_forwarding_link_indices(...)) counts dispatch-reserved links too.
#
# Topology comes from the test (ttnn.get_usable_topology). The op's own topology
# attribute is only a cross-check: all_to_all records the requested one.


def _attr_map(attrs):
    """Key/value pairs from "{'k': 'v'; 'k': 'v'}". Values hold ';' themselves,
    so the string cannot be split on the separator."""
    return dict(re.findall(r"'(\w+)':\s*'([^']*)'", str(attrs)))


def _pick(amap, known, needle, exclude):
    for k in known:
        if k in amap:
            return amap[k]
    hits = {v for k, v in amap.items() if needle in k.lower() and exclude not in k.lower()}
    return hits.pop() if len(hits) == 1 else None


def _axis_value(raw, axis):
    """Per-axis attributes look like '{a; b}'. Scalars are returned as-is."""
    if raw is None:
        return None
    m = re.fullmatch(r"\{(.*)\}", raw.strip())
    if not m:
        return raw.strip()
    parts = [p.strip() for p in m.group(1).split(";")]
    return parts[axis] if axis < len(parts) else None


def discover(attrs, cluster_axis):
    """(links, topology name) from the op's attributes, either may be None."""
    amap = _attr_map(attrs)

    links = _axis_value(_pick(amap, ("axis_num_links", "num_links"), "link", "worker"), cluster_axis)
    try:
        links = int(links)
    except (TypeError, ValueError):
        links = None
    if links is not None and not 1 <= links <= 16:  # implausible, so do not trust it
        links = None

    topo = _axis_value(_pick(amap, ("axis_topology", "topology"), "topolog", "!none!"), cluster_axis)
    return links, topo.replace("Topology::", "") if topo else None


CSV_FIELDS = [
    "run",
    "arch",
    "mesh",
    "axis",
    "dtype",
    "page_size",
    "packet",
    "line_rate_gbps",
    "topology",
    "fabric",
    "resolved",
    "links",
    "memory",
    "op",
    "n",
    "target_bytes",
    "bytes",
    "count",
    "num_pages",
    "shape",
    "iters",
    "calls",
    "us",
    "ideal_us",
]


# ------------------------------------------------------------------- parsing


def measure(seg, iters, n, calls):
    """Mean per-call duration and roofline ideal, both in ns.

    One ttnn call is not always one kernel: all_reduce runs as reduce_scatter
    then all_gather. Rows of the stages interleave across devices, so a stage is
    identified by its position among one device's rows. Each stage's max over
    devices is its time, and the stages sum.

    A trace of several back-to-back calls drops its first call, which starts
    cold, so the rest measure calls that start with the devices already in step.

    PM IDEAL is the op's own roofline model. Ops without one report 1, so the
    ideal is dropped if any kernel in the call lacks a model.
    """
    # A missing device duration would sum to zero and publish infinite bandwidth.
    if seg.empty or seg[DURATION_COL].isna().any() or len(seg) % (iters * n * calls):
        return None, None
    rows = len(seg) // iters
    stages = rows // (n * calls)
    first = 1 if calls > 1 else 0
    totals, ideals = [], []
    for i in range(iters):
        it = seg.iloc[i * rows : (i + 1) * rows]
        k = it.groupby(DEVICE_COL).cumcount()
        by_stage = [k // stages, k % stages]
        totals.extend(it[DURATION_COL].astype(float).groupby(by_stage).max().groupby(level=0).sum().iloc[first:])
        if IDEAL_COL in it.columns:
            # 1 ns is the default for an op with no performance model
            per_stage = it[IDEAL_COL].map(_num).groupby(by_stage).max()
            ok = (per_stage > 1).groupby(level=0).all()
            per_call = per_stage.groupby(level=0).sum().where(ok)
            ideals.extend(per_call.iloc[first:])

    mean_dur = sum(totals) / len(totals)
    mean_ideal = sum(ideals) / len(ideals) if ideals and all(pd.notna(i) for i in ideals) else None
    return mean_dur, mean_ideal


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: parse.py <run dir>")
    run_dir = Path(sys.argv[1])
    cfg_path = run_dir / "configs.jsonl"
    if not cfg_path.exists():
        sys.exit(f"{cfg_path} not found. Run the perf test first.")
    cfgs = [json.loads(l) for l in cfg_path.read_text().splitlines() if l.strip()]
    csvs = sorted(run_dir.glob("profiler/**/ops_perf_results_*.csv"))
    if len(csvs) != 1:
        sys.exit(f"expected one profiler CSV under {run_dir}/profiler, found {len(csvs)}")
    print(f"configs  {cfg_path} ({len(cfgs)})")
    print(f"profiler {csvs[0]}")

    df = pd.read_csv(csvs[0], low_memory=False)
    starts = df.index[df["OP CODE"] == "start"].tolist()
    stops = df.index[df["OP CODE"] == "stop"].tolist()
    if not (len(starts) == len(stops) == len(cfgs)):
        sys.exit(f"{len(starts)} signpost pairs but {len(cfgs)} configs")

    dest = run_dir / "cells.csv"
    written = 0
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for cfg, a, b in zip(cfgs, starts, stops):
            seg = df.iloc[a + 1 : b]
            dur, ideal = measure(seg, cfg["iters"], cfg["n"], cfg.get("calls", 1))
            if dur is None:
                continue
            links = attr_topo = None
            if ATTRS_COL in seg.columns and len(seg):
                links, attr_topo = discover(seg[ATTRS_COL].iloc[0], cfg["cluster_axis"])
            if attr_topo and attr_topo != cfg["resolved"]:
                print(f"  {cfg['op']} n={cfg['n']}: op reports {attr_topo}, resolved {cfg['resolved']}")
            w.writerow(
                {
                    "run": run_dir.name,
                    "arch": cfg["arch"],
                    "mesh": "x".join(str(d) for d in cfg["mesh"]),
                    "axis": cfg["cluster_axis"],
                    "dtype": cfg["dtype"],
                    "page_size": cfg["page_size"],
                    "packet": cfg["packet"],
                    "line_rate_gbps": cfg["line_rate_gbps"],
                    "topology": cfg["topology"],
                    "fabric": cfg["fabric"],
                    "resolved": cfg["resolved"],
                    "links": links,
                    "memory": cfg["memory"],
                    "op": cfg["op"],
                    "n": cfg["n"],
                    "target_bytes": cfg["target_bytes"],
                    "bytes": cfg["bytes"],
                    "count": cfg["count"],
                    "num_pages": cfg["num_pages"],
                    "shape": cfg["shape"],
                    "iters": cfg["iters"],
                    "calls": cfg.get("calls", 1),
                    "us": f"{dur / 1000.0:.3f}",
                    "ideal_us": "" if ideal is None else f"{ideal / 1000.0:.3f}",
                }
            )
            written += 1
    print(f"wrote {dest} ({written} measurements)")


if __name__ == "__main__":
    main()

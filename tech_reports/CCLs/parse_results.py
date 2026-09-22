#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Turn the profiler output from a benchmark run into a markdown table and a CSV.

The markdown is for reading, the CSV is for plotting. Both hold the same rows.

    tech_reports/CCLs/run_bench.sh

Reads generated/ccl_bench_configs.jsonl, written by the test, and the newest
profiler CSV. Writes into data/, which is gitignored: the report carries these
numbers in synthesized form, and anyone can regenerate the rest.
"""

import csv
import glob
import json
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "data"
BYTE_TARGETS = [1 << k for k in range(10, 35)]
OP_ORDER = ["all_reduce", "all_to_all", "all_gather", "reduce_scatter"]

DURATION_COL = "DEVICE KERNEL DURATION [ns]"
IDEAL_COL = "PM IDEAL [ns]"
ATTRS_COL = "ATTRIBUTES"


# ------------------------------------------------------- fabric discovery
#
# Link count and resolved topology come from the op's own ATTRIBUTES, which is
# what the op actually used rather than what it would compute. The key names
# are op-authored with no contract, so this tries the known names first, then
# falls back to a fuzzy match, and blanks rather than guesses when ambiguous.
# `ttnn` exposes no equivalent query: get_num_links is not bound, and
# len(get_forwarding_link_indices(...)) counts dispatch-reserved links too.

WRAPPING = ("Ring", "Torus")
NON_WRAPPING = ("Linear", "Mesh")


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
    """(links, wraps, topology name), any of which may be None."""
    amap = _attr_map(attrs)

    links = _axis_value(_pick(amap, ("axis_num_links", "num_links"), "link", "worker"), cluster_axis)
    try:
        links = int(links)
    except (TypeError, ValueError):
        links = None
    if links is not None and not 1 <= links <= 16:   # implausible, so do not trust it
        links = None

    topo = _axis_value(_pick(amap, ("axis_topology", "topology"), "topolog", "!none!"), cluster_axis)
    wraps = None
    if topo:
        topo = topo.replace("Topology::", "")
        if topo in WRAPPING:
            wraps = True
        elif topo in NON_WRAPPING:
            wraps = False
        else:
            topo = None
    return links, wraps, topo


# ------------------------------------------------------------------ metrics

def busbw_factor(op, n):
    """nccl-tests bus bandwidth correction, doc/PERFORMANCE.md."""
    return 2.0 * (n - 1) / n if op == "all_reduce" else (n - 1) / n


def link_factor(op, n):
    """Bottleneck bytes as a multiple of the total array.

    all_to_all differs because each chunk has one destination, so relay hops on
    a ring are extra traffic rather than a substitute for a direct send.
    """
    if op == "all_reduce":
        return 2.0 * (n - 1) / n
    if op == "all_to_all":
        return (n // 2) * ((n + 1) // 2) / n
    return (n - 1) / n


def pct_of_line_rate(r, linkbw):
    """linkbw against the line rate the test recorded, as a percentage."""
    rate = r.get("line_rate_gbps")
    if linkbw is None or not rate:
        return ""
    return f"{linkbw / rate * 100.0:.1f}"


def row_metrics(r):
    """(algbw, busbw, linkbw) in GB/s. linkbw is None when links or the
    resolved topology could not be read from the op's attributes."""
    secs = r["us"] * 1e-6
    algbw = r["bytes"] / secs / 1e9
    busbw = algbw * busbw_factor(r["op"], r["n"])
    linkbw = None
    if r["links"] and r["wraps"] is not None:
        channels = (2 if r["wraps"] else 1) * r["links"]
        linkbw = r["bytes"] * link_factor(r["op"], r["n"]) / channels / secs / 1e9
    return algbw, busbw, linkbw


CSV_FIELDS = [
    "op", "n", "topology_requested", "topology_resolved", "wraps", "links",
    "directions", "memory", "dtype", "page_size", "packet", "target_bytes",
    "bytes", "count", "num_pages", "iters", "us",
    "algbw_gbps", "busbw_gbps", "linkbw_gbps", "line_rate_gbps",
    "pct_of_line_rate", "roofline_pct", "arch", "shape",
]


def write_csv(rows, dest):
    """One row per measured cell. The plots read this; the markdown is for
    reading."""
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows.values():
            algbw, busbw, linkbw = row_metrics(r)
            w.writerow({
                "op": r["op"], "n": r["n"],
                "topology_requested": r["topology"], "topology_resolved": r["topo"],
                "wraps": r["wraps"],
                "links": r["links"],
                "directions": None if r["wraps"] is None else (2 if r["wraps"] else 1),
                "memory": r["memory"], "dtype": r["dtype"], "page_size": r["page_size"],
                "packet": r["packet"], "target_bytes": r["target_bytes"],
                "bytes": r["bytes"], "count": r["count"], "num_pages": r["num_pages"],
                "iters": r["iters"], "us": f"{r['us']:.3f}",
                "algbw_gbps": f"{algbw:.4f}", "busbw_gbps": f"{busbw:.4f}",
                "linkbw_gbps": "" if linkbw is None else f"{linkbw:.4f}",
                "line_rate_gbps": r.get("line_rate_gbps"),
                "pct_of_line_rate": pct_of_line_rate(r, linkbw),
                "arch": r.get("arch"),
                "roofline_pct": "" if not r["roofline"] else f"{r['roofline']:.2f}",
                "shape": r["shape"],
            })


# ------------------------------------------------------------------- parsing

def repo_root():
    d = HERE
    while d != d.parent and not (d / "pytest.ini").exists():
        d = d.parent
    return d


def config_log():
    """Where the test wrote its per-cell metadata."""
    return OUT_DIR / "ccl_bench_configs.jsonl"


def newest_csv():
    dirs = sorted(d for d in glob.glob(str(repo_root() / "generated/profiler/reports/*"))
                  if os.path.isdir(d))
    if not dirs:
        return None
    hits = glob.glob(os.path.join(dirs[-1], "ops_perf_results_*.csv"))
    return hits[0] if hits else None


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def measure(seg, iters, n):
    """Mean per-iteration duration and roofline ideal, both in ns.

    One ttnn call is not always one kernel: all_reduce runs as reduce_scatter
    then all_gather. Rows arrive in blocks of n, one per device, per launch, so
    each block's max is that stage and the stages sum.

    PM IDEAL is the op's own roofline model. Ops without one report 1, so the
    ideal is dropped if any kernel in the call lacks a model.
    """
    if len(seg) % (iters * n):
        return None, None
    per_iter = len(seg) // (iters * n)
    dur = seg[DURATION_COL].astype(float).values
    ideal = seg[IDEAL_COL].values if IDEAL_COL in seg.columns else None

    totals, ideals = [], []
    for i in range(iters):
        base = i * per_iter * n
        totals.append(sum(max(dur[base + k * n:base + (k + 1) * n]) for k in range(per_iter)))
        if ideal is None:
            continue
        stages = []
        for k in range(per_iter):
            vals = [v for v in (_num(x) for x in ideal[base + k * n:base + (k + 1) * n]) if v]
            # 1 ns is the default for an op with no performance model
            stages.append(max(vals) if vals and max(vals) > 1 else None)
        ideals.append(None if any(s is None for s in stages) else sum(stages))

    mean_dur = sum(totals) / len(totals)
    mean_ideal = sum(ideals) / len(ideals) if ideals and all(i is not None for i in ideals) else None
    return mean_dur, mean_ideal


def main():
    cfg_path = config_log()
    if not cfg_path.exists():
        sys.exit(f"{cfg_path} not found. Run the perf test first.")
    cfgs = [json.loads(l) for l in cfg_path.read_text().splitlines() if l.strip()]
    csv_path = newest_csv()
    if not csv_path:
        sys.exit("no profiler CSV found. Was the test run under 'python -m tracy'?")
    print(f"configs  {cfg_path} ({len(cfgs)})")
    print(f"profiler {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    starts = df.index[df["OP CODE"] == "start"].tolist()
    stops = df.index[df["OP CODE"] == "stop"].tolist()
    if not (len(starts) == len(stops) == len(cfgs)):
        sys.exit(f"{len(starts)} signpost pairs but {len(cfgs)} configs. "
                 "Delete run_configs.jsonl and re-run the test.")

    rows, found = {}, OrderedDict()
    for cfg, a, b in zip(cfgs, starts, stops):
        seg = df.iloc[a + 1:b]
        dur, ideal = measure(seg, cfg["iters"], cfg["n"])
        if dur is None:
            continue
        links = wraps = topo = None
        if ATTRS_COL in seg.columns and len(seg):
            links, wraps, topo = discover(seg[ATTRS_COL].iloc[0], cfg["cluster_axis"])
        cfg.update(us=dur / 1000.0, links=links, wraps=wraps, topo=topo,
                   roofline=(ideal / dur * 100.0) if ideal else None)
        rows[(cfg["op"], cfg["n"], cfg["target_bytes"])] = cfg

    # One link count covers the run: it depends on the machine and the axis, not
    # on the collective. Ops disagree only by using fewer than they discovered.
    reported = sorted({r["links"] for r in rows.values() if r["links"]})
    shared = reported[0] if len(reported) == 1 else None
    if len(reported) > 1:
        print(f"\nops disagree on link count {reported}; linkbw left blank")
    missing = sorted({r["op"] for r in rows.values() if not r["links"]})
    for r in rows.values():
        r["links"] = r["links"] or shared
        found.setdefault((r["op"], r["n"]), (r["links"], r["topo"]))

    print(f"\nlink count {shared or '?'}, shared across the run")
    if missing:
        print(f"  not reported by: {', '.join(missing)}")
    for (op, n), (links, topo) in found.items():
        print(f"  {op:>15} n={n:<2} links={links if links else '?':<3} topology={topo or '?'}")

    h = cfgs[0]
    ns_seen = sorted({c["n"] for c in cfgs})
    out = [
        "# CCL benchmark", "", "```",
        f"mesh          {tuple(h['mesh'])}",
        f"topology      {h['topology']} (requested)",
        f"cluster axis  {h['cluster_axis']}",
        f"memory        {h['memory'].upper()} interleaved",
        f"dtype         {h['dtype']}  ({h['page_size']} B pages)",
        f"packet        {h['packet']} B",
        f"arch          {h.get('arch')}  line rate {h.get('line_rate_gbps')} GB/s per link per direction",
        "```", "",
        "Byte targets follow nccl-tests (`-b 1K -e 16G -f 2`), rounded to whole tiles.",
        "`size` is what was achieved. `algbw` is size/time. `busbw` is nccl's correction,",
        "`algbw * 2(n-1)/n` for all_reduce and `algbw * (n-1)/n` for the rest.",
        "",
        "`% line rate` is linkbw against the per-link line rate recorded above.",
        "",
        "`linkbw` is per link per direction. It divides the bottleneck traffic by the",
        "links carrying it, doubled on a wrapping axis. Topology comes from each op's own",
        "profiler attributes. The link count is shared across the run, and it is the count",
        "discovered rather than the count used: a program may clamp below it to fit its",
        "worker cores. Blank when neither could be read. all_to_all uses a different",
        "factor from busbw, because on a ring its chunks relay through intermediate",
        "chips instead of arriving in one hop.",
        "",
        "`roofline` is the op's own performance model over the measured time. The model",
        "takes the tightest of several ceilings and does not record which one bound, so",
        "rows compare within a size regime, not across the column. It is not link",
        "utilisation, and it is blank for ops with no model.",
        "",
        "Empty rows had no tiled shape at that device count: each device must hold at",
        "least one tile of the split, so the total array is at least 1024*n elements.",
        "", "```", "discovered per op:",
    ]
    for (op, n), (links, topo) in found.items():
        out.append(f"  {op:>15} n={n:<2} links={links if links else '?':<3} topology={topo or '?'}")
    out += ["```", ""]

    for op in OP_ORDER:
        if not any(c["op"] == op for c in cfgs):
            continue
        out += [f"## {op}", "",
                "| n | target (B) | size (B) | count | pages | time (us) | "
                "algbw (GB/s) | busbw (GB/s) | linkbw (GB/s) | % line rate | "
                "roofline (%) |",
                "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for n in ns_seen:
            for target in BYTE_TARGETS:
                r = rows.get((op, n, target))
                if r is None:
                    out.append(f"| {n} | {target} | | | | | | | | | |")
                    continue
                algbw, busbw, linkbw = row_metrics(r)
                link = "" if linkbw is None else f"{linkbw:.2f}"
                roof = f"{r['roofline']:.1f}" if r["roofline"] else ""
                out.append(
                    f"| {n} | {target} | {r['bytes']} | {r['count']} | {r['num_pages']} | "
                    f"{r['us']:.2f} | {algbw:.2f} | {busbw:.2f} | {link} | "
                    f"{pct_of_line_rate(r, linkbw)} | {roof} |")
        out.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = (f"results_{h.get('arch') or 'unknown'}_{h['topology']}_{h['memory']}"
            f"_{h['dtype']}_{h['packet']}")
    dest = OUT_DIR / f"{stem}.md"
    dest.write_text("\n".join(out) + "\n")
    csv_dest = OUT_DIR / f"{stem}.csv"
    write_csv(rows, csv_dest)
    print(f"\nwrote {dest} ({len(rows)} measurements)")
    print(f"wrote {csv_dest}")


if __name__ == "__main__":
    main()

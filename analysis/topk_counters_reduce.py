#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduce the archived TopK perf-counter capture (block T2.6 COUNTERS rerun) to busy fractions.

The multipass capture has no ops report (tt-metal issue 56712), so the cells are recovered from the
run host ID order and the per-op core count, which the campaign's cell list fixes: four
topk_large_indices cells at 110 cores, two single-core generic cells at 1, one multi-core generic at
33, one composite that emits three ops (prep 86, li 110, finish 8), then the gate and the grouped
top-k at 110. Five invocations per cell per pass, the first three warm-ups discarded as everywhere
in this campaign.

Every counter row carries its own `ref cnt`, the per-core profiled cycle count of that op, so a busy
fraction is value / ref cnt on the same core and needs no wall from elsewhere. Reported per cell and
counter as the median over the participating cores, then the median of the two timed invocations.

Usage: topk_counters_reduce.py <passes dir> [--out <prefix>]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

# cell_id, ops in call order as (core count, op label); the campaign's cells_COUNTERS.csv order
CELLS = [
    ("COUNTERS-282-li-R110-N16384-K512", [(110, "TopkLargeIndices")]),
    ("COUNTERS-283-li-R110-N16384-K2048", [(110, "TopkLargeIndices")]),
    ("COUNTERS-284-li-R110-N131072-K512", [(110, "TopkLargeIndices")]),
    ("COUNTERS-285-li-R110-N131072-K2048", [(110, "TopkLargeIndices")]),
    ("COUNTERS-286-topk-rows32-N4096-K32-gridsingle", [(1, "TopK single core")]),
    ("COUNTERS-287-topk-rows32-N4096-K512-gridsingle", [(1, "TopK single core")]),
    ("COUNTERS-288-topk-rows32-N16384-K32-gridfull", [(33, "TopK multi core")]),
    (
        "COUNTERS-289-topk-rows32-N65536-K128-gridNone",
        [(86, "TopkRoutePrep"), (110, "TopkLargeIndices"), (8, "TopkRouteFinish")],
    ),
    ("COUNTERS-290-gmg-B110-K8-softmaxFalse-experts256", [(110, "generalized_moe_gate")]),
    ("COUNTERS-291-mgt-T4096-N128-K4-groups1", [(110, "moe_grouped_topk")]),
]
INVOCATIONS = 5  # per cell per pass
DISCARD = 3  # warm-ups, as everywhere in this campaign


def read_pass(path: Path):
    """Yield dicts for one pass log. Line 1 is the ARCH banner, line 2 the header."""
    with open(path) as f:
        f.readline()
        hdr = [c.strip() for c in f.readline().split(",")]
        for row in csv.reader(f):
            if not row or len(row) < len(hdr) - 1:
                continue
            yield dict(zip(hdr, [x.strip() for x in row]))


def parse_meta(md: str):
    """'{"counter type":"FPU_COUNTER";"ref cnt":69449;"value":9454}' -> (name, ref, value)."""
    try:
        name = md.split('"counter type":"')[1].split('"')[0]
        ref = int(md.split('"ref cnt":')[1].split(";")[0].rstrip("}"))
        val = int(md.split('"value":')[1].split(";")[0].rstrip("}"))
    except (IndexError, ValueError):
        return None
    return name, ref, val


def collect(path: Path):
    """run host ID -> {'cores': n, 'ctr': {counter: [(ref, value) per core]}}, in first-seen order."""
    runs: "OrderedDict[str, dict]" = OrderedDict()
    for d in read_pass(path):
        rid = d["run host ID"]
        e = runs.setdefault(rid, {"cores": set(), "ctr": defaultdict(list)})
        e["cores"].add((d["core_x"], d["core_y"]))
        if d["timer_id"] != "9090":
            continue
        parsed = parse_meta(d.get("meta data", ""))
        if parsed is None:
            continue
        name, ref, val = parsed
        if ref > 0:
            e["ctr"][name].append(val / ref)
    for e in runs.values():
        e["cores"] = len(e["cores"])
    return runs


def assign(runs: "OrderedDict[str, dict]"):
    """Walk the run host IDs in order and hand them to the expected op sequence."""
    expected = []
    for cell, ops in CELLS:
        for _ in range(INVOCATIONS):
            for cores, label in ops:
                expected.append((cell, label, cores))
    rids = list(runs)
    if len(rids) != len(expected):
        raise SystemExit(f"{len(rids)} run host IDs against {len(expected)} expected ops; cell list is stale")
    out = []
    for rid, (cell, label, cores) in zip(rids, expected):
        got = runs[rid]["cores"]
        if got != cores:
            raise SystemExit(f"rid {rid}: expected {cores} cores for {cell} {label}, log has {got}")
        out.append((cell, label, rid))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("passes_dir")
    ap.add_argument("--out", default=None, help="write <out>.csv instead of printing a table")
    a = ap.parse_args()

    d = Path(a.passes_dir)
    files = sorted(d.glob("pass_*.csv"))
    if not files:
        raise SystemExit(f"no pass_*.csv under {d}")

    # (cell, op) -> counter -> list over passes of the per-invocation median-over-cores
    acc: dict = defaultdict(lambda: defaultdict(list))
    for f in files:
        runs = collect(f)
        seq = assign(runs)
        per_op: dict = defaultdict(list)  # (cell, op) -> [rid, ...] in call order
        for cell, label, rid in seq:
            per_op[(cell, label)].append(rid)
        for key, rids in per_op.items():
            for rid in rids[DISCARD:]:  # the timed invocations
                for name, fracs in runs[rid]["ctr"].items():
                    if fracs:
                        acc[key][name].append(statistics.median(fracs))
        print(
            f"  read {f.name}: {len(runs)} ops, " f"{len({n for r in runs.values() for n in r['ctr']})} counters",
            file=sys.stderr,
        )

    rows = []
    for (cell, label), counters in acc.items():
        for name, vals in sorted(counters.items()):
            rows.append(
                {
                    "cell": cell,
                    "op": label,
                    "counter": name,
                    "fraction_of_ref_cnt": round(statistics.median(vals), 4),
                    "n_samples": len(vals),
                }
            )
    if a.out:
        with open(a.out + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}.csv ({len(rows)} rows)")
    else:
        for r in rows:
            print(
                f"{r['cell'][:44]:44s} {r['op'][:18]:18s} {r['counter']:34s} "
                f"{r['fraction_of_ref_cnt']:.4f}  n={r['n_samples']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-op MoE statistics across ALL chunks, with the collective work/wait split and per-chip imbalance.

Why this exists, and why it is not `parse_zone_perf.py`
------------------------------------------------------
`parse_zone_perf.py` answers "where did the profiled chunk's time go": it charges only ops nested under
the `profiled_chunk` zone and quotes each zone on its own worst device. That is the right view for a
breakdown, and the wrong view for deciding whether a change worked, for two reasons.

1. **n = 1 per op.** Only the final chunk is inside `profiled_chunk`, so each op in each of the 3 sparse
   layers is a single draw. Those single draws swing violently — measured on dev 16, the same
   reduce-scatter came in at 148 / 1463 / 3580 us across three consecutive layers, a 24x spread. No
   per-layer estimate survives that, so a 30-130 us change is unmeasurable against it.

2. **MoE work is cache-depth independent.** dispatch / experts / combine never touch the KV cache, so
   every chunk in the capture is an equally valid sample of them — including the prefix-fill chunks,
   whose ops are in the same CSV (they are outside `profiled_chunk`, which is exactly why the other
   parser drops them, and they ARE drained to the CSV: profile_prefill.py reads the device profiler
   freely before the profiled chunk). A LAYERS=6 CACHE=25600 run therefore carries
   6 chunks x 3 sparse layers = **n=18** samples per op instead of n=1.

Both views are needed. Use the other one to find where time goes; use this one to decide whether a diff
moved it.

The work/wait split
-------------------
Every chip must participate in a collective, so across the 32 chips:

    work = MIN over devices     -- the chip that arrived LAST waited on nobody, so its measured
                                   duration is the collective's actual work
    wait = MEDIAN - MIN         -- what a typical chip spent blocked on its peers

This is what separates a genuinely slow collective from one that is merely absorbing upstream skew, and
it is the difference between "the reduce-scatter costs 1266 us" and "it costs 145 us and waits 879".

It is NOT valid for non-collectives: `UnifiedRoutedExpertFfn`'s min across chips has been measured at
4.7 us, which means some chip held a near-empty expert. For an independent per-chip op, min32 measures
load imbalance, not wait. Ops are classified below and the split is printed only where it means
something; use --imbalance for the other kind.

Usage
-----
    # the headline table: every MoE op, n=18 medians, tight enough to measure a 5-line change
    python3 moe_op_stats.py <ops_perf_results_*.csv>

    # + the collective work/wait split
    python3 moe_op_stats.py <csv> --work-wait

    # per-chip load for one op, per layer slot, and whether the hot chip is stable across chunks
    python3 moe_op_stats.py <csv> --imbalance UnifiedRoutedExpertFfn

    # compare two captures (before/after a change); prints per-op deltas with both n's
    python3 moe_op_stats.py <after.csv> --baseline <before.csv>

Op identity
-----------
An op is keyed on (layer-relative zone path, OP CODE, slot), where `slot` is its index within one visit
to that zone — so the 4 serialized `UnifiedRoutedExpertFfn` launches and the 2 `ag_kv` all-gathers stay
distinct rather than being averaged together. Samples are the (layer, chunk) visits of that zone.
"""

import argparse
import statistics
import sys
from collections import defaultdict

import pandas as pd

ZONE_START = "M3_ZONE_START"
ZONE_END = "M3_ZONE_END"
DURATION_COL = "DEVICE KERNEL DURATION [ns]"
BASE_COLS = ["OP CODE", "OP TYPE", "DEVICE ID", DURATION_COL]
# Stripped when it is the outermost frame: only the last chunk has it, and keeping it would make that
# chunk's ops a different identity from the same ops in every other chunk.
CHUNK_ROOT = "profiled_chunk"

# Ops where every chip participates, so min32 = the last arrival = real work (see module docstring).
# Substring match on OP CODE.
COLLECTIVE_MARKERS = (
    "ReduceScatter",
    "AllGather",
    "DispatchDeviceOperation",
    "CombineDeviceOperation",
    "AllReduce",
    "AllToAll",
)
# Independent per-chip work: min32 measures load imbalance, NOT wait. Never print a split for these.
NOT_COLLECTIVE = ("UnifiedRoutedExpertFfn", "RoutedExpertFfn", "Matmul", "Bincount")


def is_collective(op_code: str) -> bool:
    if any(m in op_code for m in NOT_COLLECTIVE):
        return False
    return any(m in op_code for m in COLLECTIVE_MARKERS)


def strip_layer_index(path: str) -> str:
    """`layer04_sparse/mlp/dispatch` -> `sparse/mlp/dispatch`, so the 57 sparse layers collapse into
    one row per op instead of 57. Also drops a leading `profiled_chunk` frame (last chunk only)."""
    parts = [p for p in path.split("/") if p]
    if parts and parts[0] == CHUNK_ROOT:
        parts = parts[1:]
    if parts and parts[0].startswith("layer"):
        tail = parts[0].split("_", 1)
        parts[0] = tail[1] if len(tail) > 1 else parts[0]
    return "/".join(parts)


class Collector:
    """Walks CSV rows in order, tracks the open zone stack, and buckets every op row into
    (identity, sample, device) -> ns. Rows are in host-enqueue order, so the ops between a zone's
    START and END signposts are exactly the ops that zone enqueued."""

    def __init__(self):
        self.stack = []  # [{"name", "path", "ops": {op_code: next_slot}}]
        self.visits = defaultdict(int)  # full zone path -> times entered so far
        # (rel_path, op_code, slot) -> {(full_path, visit): {device: ns}}
        self.samples = defaultdict(lambda: defaultdict(dict))
        self.unmatched_ends = 0
        self.rows_charged = 0
        self.rows_outside = 0

    def _push(self, name):
        parent = self.stack[-1]["path"] if self.stack else ""
        path = f"{parent}/{name}" if parent else name
        visit = self.visits[path]
        self.visits[path] += 1
        self.stack.append({"name": name, "path": path, "visit": visit, "ops": defaultdict(int)})

    def feed(self, row):
        op_type = row.get("OP TYPE")
        code = row.get("OP CODE")
        if isinstance(op_type, str) and op_type == "signpost":
            name = str(code)
            if name.startswith(ZONE_START):
                self._push(name[len(ZONE_START) :].strip())
            elif name.startswith(ZONE_END):
                ending = name[len(ZONE_END) :].strip()
                if self.stack and self.stack[-1]["name"] == ending:
                    self.stack.pop()
                elif any(f["name"] == ending for f in self.stack):
                    # Tolerate a dropped START/END (truncated CSV): unwind to the matching frame.
                    while self.stack and self.stack.pop()["name"] != ending:
                        pass
                    self.unmatched_ends += 1
                else:
                    self.unmatched_ends += 1
            return

        if not self.stack:
            self.rows_outside += 1
            return
        ns = row.get(DURATION_COL)
        if ns is None or pd.isna(ns) or float(ns) <= 0:
            return  # host-only op / no device kernel
        dev = row.get("DEVICE ID")
        if dev is None or pd.isna(dev):
            return

        frame = self.stack[-1]  # the innermost zone is where the op actually ran
        code = str(code)
        dev = int(dev)
        # Slot must be counted PER DEVICE. Every op appears once per device in the CSV, so a single
        # counter per op code would number them 0..(instances*32-1) and every (code, slot) pair would
        # then hold exactly one device -- making min == median == max and reporting every wait as zero.
        # Per-device counters give the k-th launch on dev A the same slot as the k-th launch on dev B,
        # which is what lets the 32 chips of one launch be compared against each other.
        slot = frame["ops"][(code, dev)]
        frame["ops"][(code, dev)] += 1
        ident = (strip_layer_index(frame["path"]), code, slot)
        self.samples[ident][(frame["path"], frame["visit"])][dev] = float(ns)
        self.rows_charged += 1


def per_sample_stats(by_sample, drop_first):
    """-> list of (sample_key, max_us, min_us, median_us) in visit order, first `drop_first` dropped.

    Ordering is by (full_path, visit) so dropping the first entries drops the earliest visit of EACH
    zone instance — i.e. the warmup pass, whose ops are cold (a program-cache miss shows up as
    dispatch 1874 us / combine 2503 us against a steady ~1010 / ~1383).
    """
    per_path = defaultdict(list)
    for (full_path, visit), devs in by_sample.items():
        per_path[full_path].append((visit, devs))
    out = []
    for full_path, entries in per_path.items():
        entries.sort(key=lambda e: e[0])
        for visit, devs in entries[drop_first:]:
            vals = sorted(devs.values())
            out.append(
                (
                    (full_path, visit),
                    max(vals) / 1000.0,
                    min(vals) / 1000.0,
                    statistics.median(vals) / 1000.0,
                    len(vals),
                )
            )
    return out


def summarize(collector, drop_first):
    rows = []
    for ident, by_sample in collector.samples.items():
        stats = per_sample_stats(by_sample, drop_first)
        if not stats:
            continue
        maxes = [s[1] for s in stats]
        mins = [s[2] for s in stats]
        meds = [s[3] for s in stats]
        ndev = max(s[4] for s in stats)
        rows.append(
            {
                "zone": ident[0],
                "op": ident[1],
                "slot": ident[2],
                "n": len(stats),
                "devices": ndev,
                # Median over samples of the per-sample MAX across chips: the mesh waits for the
                # slowest chip, so this is the wall-clock-relevant number.
                "med_max_us": statistics.median(maxes),
                "lo_max_us": min(maxes),
                "hi_max_us": max(maxes),
                "iqr_us": (
                    (statistics.quantiles(maxes, n=4)[2] - statistics.quantiles(maxes, n=4)[0])
                    if len(maxes) >= 4
                    else float("nan")
                ),
                "work_us": statistics.median(mins),
                "wait_us": statistics.median(meds) - statistics.median(mins),
                "collective": is_collective(ident[1]),
            }
        )
    rows.sort(key=lambda r: -r["med_max_us"])
    return rows


def print_table(rows, work_wait):
    hdr = f"{'zone':<34} {'op':<32} {'sl':>2} {'n':>3} {'median':>9} {'min':>9} {'max':>9} {'IQR':>8}"
    if work_wait:
        hdr += f" {'work':>8} {'wait':>8} {'wait%':>6}"
    print(hdr)
    print("-" * len(hdr))
    tot = 0.0
    for r in rows:
        line = (
            f"{r['zone'][:34]:<34} {r['op'][:32]:<32} {r['slot']:>2} {r['n']:>3} "
            f"{r['med_max_us']:>9.1f} {r['lo_max_us']:>9.1f} {r['hi_max_us']:>9.1f} "
            f"{r['iqr_us']:>8.1f}"
        )
        if work_wait:
            if r["collective"]:
                pct = 100.0 * r["wait_us"] / r["med_max_us"] if r["med_max_us"] else 0.0
                line += f" {r['work_us']:>8.1f} {r['wait_us']:>8.1f} {pct:>5.0f}%"
            else:
                # min32 here is load imbalance, not wait -- printing a split would invite a wrong read.
                line += f" {'-':>8} {'-':>8} {'-':>6}"
        print(line)
        tot += r["med_max_us"]
    print("-" * len(hdr))
    print(f"{'TOTAL (sum of medians)':<34} {'':<32} {'':>2} {'':>3} {tot:>9.1f} us")
    print(
        "  ^ an UPPER BOUND, not a per-layer total: each row is the median of the per-sample MAX across\n"
        "    chips, and for per-chip-independent ops (experts_mm) a different chip is the max in each\n"
        "    slot, so summing the slots over-counts. The valid per-layer figure for those is the max\n"
        "    over chips of the SUM over slots -- which is what --imbalance reports."
    )
    if work_wait:
        cw = sum(r["wait_us"] for r in rows if r["collective"])
        share = f"{100 * cw / tot:.0f}% of total" if tot else "no qualifying rows"
        print(f"{'collective barrier wait':<34} {'':<32} {'':>2} {'':>3} {cw:>9.1f} us  ({share})")
        print("\nwork = min across chips (the chip that arrived last, so it waited on nobody)")
        print("wait = median - min.  '-' means the op is per-chip independent, where min32 would be")
        print("       load imbalance rather than wait -- use --imbalance for those.")


def print_imbalance(collector, op_filter, drop_first):
    """Per-chip load for one op, per layer instance, and whether the hot chip is stable across chunks.

    Sums the op's slots (e.g. all 4 UnifiedRoutedExpertFfn launches) per (device, layer, chunk), which is
    the per-chip total the barrier downstream has to absorb.
    """
    # (full_path, visit) -> device -> summed ns
    per_visit = defaultdict(lambda: defaultdict(float))
    for (zone, op, _slot), by_sample in collector.samples.items():
        if op_filter not in op:
            continue
        for key, devs in by_sample.items():
            for d, ns in devs.items():
                per_visit[key][d] += ns
    if not per_visit:
        print(f"no op matching '{op_filter}' found")
        return

    by_path = defaultdict(list)
    for (full_path, visit), devs in per_visit.items():
        by_path[full_path].append((visit, devs))

    print(f"per-chip total for ops matching '{op_filter}', summed over slots\n")
    hdr = f"{'layer zone':<30} {'chunk':>5} {'fastest':>9} {'median':>9} {'slowest':>9} {'spread':>9} {'argmax':>7}"
    print(hdr)
    print("-" * len(hdr))
    argmax_by_path = defaultdict(list)
    for full_path in sorted(by_path):
        entries = sorted(by_path[full_path], key=lambda e: e[0])[drop_first:]
        for visit, devs in entries:
            vals = sorted(devs.values())
            hot = max(devs, key=lambda d: devs[d])
            argmax_by_path[full_path].append(hot)
            print(
                f"{full_path[:30]:<30} {visit:>5} {vals[0]/1000:>9.1f} "
                f"{statistics.median(vals)/1000:>9.1f} {vals[-1]/1000:>9.1f} "
                f"{(vals[-1]-vals[0])/1000:>9.1f} {hot:>7}"
            )
    print("-" * len(hdr))
    print("\nargmax stability (is the hot chip the same one every chunk?)")
    for full_path, hots in sorted(argmax_by_path.items()):
        if not hots:
            continue
        top = max(set(hots), key=hots.count)
        print(f"  {full_path[:44]:<44} dev {top:>3} in {hots.count(top)}/{len(hots)} chunks")
    print("\nA stable argmax means the imbalance is STRUCTURAL (static expert->chip placement + a fixed")
    print("gate), not noise -- so the chip holding that layer's hot expert is always its critical path.")


def load(csv_path, chunksize, drop_first):
    header = list(pd.read_csv(csv_path, nrows=0).columns)
    missing = [c for c in BASE_COLS if c not in header]
    if missing:
        sys.exit(f"ERROR: {csv_path} is missing expected column(s): {missing}")
    c = Collector()
    nrows = 0
    for chunk in pd.read_csv(csv_path, usecols=BASE_COLS, chunksize=chunksize, low_memory=False):
        for row in chunk.to_dict("records"):
            c.feed(row)
        nrows += len(chunk)
    print(
        f"[moe-stats] {csv_path}: {nrows} rows, {c.rows_charged} charged to a zone, "
        f"{c.rows_outside} outside any zone, {c.unmatched_ends} unmatched zone ends"
    )
    if c.stack:
        print(f"[moe-stats] WARNING: {len(c.stack)} zone(s) still open at EOF (truncated CSV?): {c.stack[-1]['path']}")
    return c, drop_first


def main():
    ap = argparse.ArgumentParser(
        description="Per-op MoE stats across ALL chunks (n=18 medians), with the collective work/wait split"
    )
    ap.add_argument("csv", help="ops_perf_results_*.csv from `python3 -m tracy -r ...`")
    ap.add_argument("--work-wait", action="store_true", help="add the min32/median collective work/wait split")
    ap.add_argument("--imbalance", metavar="OP", help="per-chip load for ops whose OP CODE contains OP")
    ap.add_argument("--baseline", metavar="CSV", help="second capture to diff against (before/after)")
    ap.add_argument(
        "--drop-first",
        type=int,
        default=1,
        help="drop the first N visits of each zone (default 1 = the warmup pass, whose program-cache "
        "misses make it a large outlier). Use 0 to keep everything.",
    )
    ap.add_argument("--min-us", type=float, default=5.0, help="hide ops below this median (default 5)")
    ap.add_argument("--chunksize", type=int, default=200_000, help="CSV streaming chunk size")
    args = ap.parse_args()

    coll, drop_first = load(args.csv, args.chunksize, args.drop_first)

    if args.imbalance:
        print()
        print_imbalance(coll, args.imbalance, drop_first)
        return 0

    rows = [r for r in summarize(coll, drop_first) if r["med_max_us"] >= args.min_us]
    if not args.baseline:
        print()
        print_table(rows, args.work_wait)
        ns = {r["n"] for r in rows}
        print(f"\nn per op: {sorted(ns)} samples (chunks x layers, after dropping the first {drop_first})")
        if max(ns, default=0) < 6:
            print("WARNING: fewer than 6 samples per op -- too few to resolve a <200 us change.")
        return 0

    base_coll, _ = load(args.baseline, args.chunksize, args.drop_first)
    base = {(r["zone"], r["op"], r["slot"]): r for r in summarize(base_coll, drop_first)}
    print()
    hdr = f"{'zone':<32} {'op':<30} {'sl':>2} {'base':>9} {'after':>9} {'delta':>9} {'%':>7} {'n b/a':>8}"
    print(hdr)
    print("-" * len(hdr))
    net = 0.0
    for r in rows:
        b = base.get((r["zone"], r["op"], r["slot"]))
        if b is None:
            print(
                f"{r['zone'][:32]:<32} {r['op'][:30]:<30} {r['slot']:>2} {'ABSENT':>9} "
                f"{r['med_max_us']:>9.1f} {r['med_max_us']:>+9.1f} {'':>7} {'-':>3}/{r['n']:<4}"
            )
            net += r["med_max_us"]
            continue
        d = r["med_max_us"] - b["med_max_us"]
        net += d
        print(
            f"{r['zone'][:32]:<32} {r['op'][:30]:<30} {r['slot']:>2} {b['med_max_us']:>9.1f} "
            f"{r['med_max_us']:>9.1f} {d:>+9.1f} {100*d/b['med_max_us']:>+6.1f}% "
            f"{b['n']:>3}/{r['n']:<4}"
        )
    for key, b in base.items():
        if b["med_max_us"] >= args.min_us and key not in {(r["zone"], r["op"], r["slot"]) for r in rows}:
            print(
                f"{b['zone'][:32]:<32} {b['op'][:30]:<30} {b['slot']:>2} {b['med_max_us']:>9.1f} "
                f"{'GONE':>9} {-b['med_max_us']:>+9.1f} {'':>7} {b['n']:>3}/{'-':<4}"
            )
            net -= b["med_max_us"]
    print("-" * len(hdr))
    print(f"{'NET':<32} {'':<30} {'':>2} {'':>9} {'':>9} {net:>+9.1f} us/layer")
    print("\nDo not scale this to wall-clock with a >1x multiplier: a measured 24 device-ms saving")
    print("delivered 0.3-2.1% end-to-end, so ~1.0x is the honest conversion until better sampled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

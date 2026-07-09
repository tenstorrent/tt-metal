#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parse a Tracy ops_perf_results CSV and report, per pipeline stage, how much of the
device timeline is real kernel compute (device-active) versus op-to-op gap, plus the
CCL / SDPA / matmul / tilize / typecast / norm class breakdown.

Purpose (LTX 3s opt loop): decide whether a stage's path to 3s is device-compute-bound
(need faster/sparser kernels) or gap-bound (need op-count reduction / fusion / host work).

The CSV holds every device op for the whole run, tagged by DEVICE ID and, for traced
regions, by (METAL TRACE ID, METAL TRACE REPLAY SESSION ID). Stage 1 denoise and stage 2
denoise each capture their own trace, so this script segments by trace id — no source
markers required — and auto-identifies the stage-2 denoise trace (the larger of the two
SDPA-bearing traces). Run it on the CSV; it does NOT touch the device.

Usage:
    python tmp/parse_tracy.py [CSV]                 # auto-find newest report, auto-pick s2
    python tmp/parse_tracy.py [CSV] --stage s1      # stage-1 denoise trace instead
    python tmp/parse_tracy.py [CSV] --trace <id>    # a specific METAL TRACE ID
    python tmp/parse_tracy.py [CSV] --session first  # first replay session instead of last
    python tmp/parse_tracy.py [CSV] --all           # dump every trace/session group

CSV columns used (see tools/tracy/process_ops_logs.py OPS_CSV_HEADER):
    OP CODE, DEVICE ID, GLOBAL CALL COUNT, METAL TRACE ID, METAL TRACE REPLAY SESSION ID,
    DEVICE FW DURATION [ns], DEVICE KERNEL DURATION [ns], OP TO OP LATENCY [ns],
    HOST START TS, HOST END TS, INPUTS.

CAVEAT (verify once against the first real CSV): the OP-CODE class matcher below uses
case-insensitive substrings. The exact device OP CODE strings (e.g. is it
"RingJointScaledDotProductAttention", "AllGatherMinimalMatmulAsync", ...) come out of the
C++ op registry; if the "OTHER" bucket is large, print the raw per-op-code table (always
shown) and extend CLASS_RULES.
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

# First match wins. Ordered so fused collectives land in CCL (the collective is the cost
# driver, matching context.md's "matmul/CCL" + "AG-fused matmul" framing) and fused
# norm/matmul land correctly. Matched against a lowercased, alnum-only op code.
CLASS_RULES = [
    ("CCL", ("allgather", "reducescatter", "allreduce", "alltoall", "ccl", "allbroadcast")),
    ("SDPA", ("scaleddotproduct", "ringjoint", "jointattention", "sdpa", "attention")),
    ("NORM", ("rmsnorm", "layernorm", "groupnorm", "batchnorm", "norm")),
    ("TYPECAST", ("typecast",)),
    ("TILIZE", ("tilize", "untilize")),  # ttnn.to_layout lowers to (un)tilize
    ("MATMUL", ("matmul", "linear", "moreh", "bmm")),
    # everything else -> OTHER (reshape/permute/transpose/binary/concat/slice/embedding/...)
]

CSV_GLOB = "generated/profiler/reports/*/ops_perf_results_*.csv"


def _classify(op_code: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", (op_code or "").lower())
    for name, subs in CLASS_RULES:
        if any(s in key for s in subs):
            return name
    return "OTHER"


def _num(row: dict, col: str) -> float:
    v = row.get(col, "")
    if v is None:
        return 0.0
    v = str(v).strip()
    if v == "" or v.lower() in ("nan", "n/a", "-"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 0.0


def _int_or_none(row: dict, col: str):
    v = str(row.get(col, "") or "").strip()
    if v == "" or v.lower() in ("nan", "n/a", "-"):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _find_csv(path_arg):
    if path_arg:
        if os.path.isdir(path_arg):
            cands = sorted(glob.glob(os.path.join(path_arg, "**", "ops_perf_results_*.csv"), recursive=True))
            if not cands:
                sys.exit(f"no ops_perf_results_*.csv under {path_arg}")
            return cands[-1]
        return path_arg
    root = os.environ.get("TT_METAL_HOME", os.getcwd())
    cands = sorted(glob.glob(os.path.join(root, CSV_GLOB)))
    if not cands:
        sys.exit(f"no CSV found under {os.path.join(root, CSV_GLOB)} — pass the path explicitly")
    return cands[-1]


def _fmt_us(ns: float) -> str:
    return f"{ns / 1000.0:11.1f}"


def _seq_hint(rows):
    """Largest integer seen in any INPUTS cell of an SDPA op — a proxy for the video
    sequence length, so the bigger (full-res) trace can be flagged as stage 2."""
    best = 0
    for r in rows:
        if _classify(r.get("OP CODE", "")) != "SDPA":
            continue
        for m in re.findall(r"\d+", r.get("INPUTS", "") or ""):
            best = max(best, int(m))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "csv", nargs="?", help="ops_perf_results_*.csv (or a report dir). Default: newest under TT_METAL_HOME."
    )
    ap.add_argument("--stage", choices=["s1", "s2"], default="s2", help="which denoise trace to focus (default s2)")
    ap.add_argument("--trace", type=int, default=None, help="focus a specific METAL TRACE ID (overrides --stage)")
    ap.add_argument("--session", default="last", help="'last' (steady-state gen#1), 'first', or a session id")
    ap.add_argument("--device", type=int, default=None, help="focus one DEVICE ID (default: the slowest device)")
    ap.add_argument("--top", type=int, default=15, help="rows in the per-op-code table (default 15)")
    ap.add_argument("--all", action="store_true", help="print every trace/session group summary and exit")
    args = ap.parse_args()

    csv_path = _find_csv(args.csv)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{csv_path} has no data rows")
    print(f"CSV: {csv_path}\nrows: {len(rows)}")

    devices = sorted({_int_or_none(r, "DEVICE ID") for r in rows} - {None})
    print(f"devices: {len(devices)} {devices[:8]}{'...' if len(devices) > 8 else ''}")

    # ---- group by (trace id, session id); None trace => eager -----------------------------
    groups = defaultdict(list)  # (trace_id, session_id) -> rows
    for r in rows:
        tid = _int_or_none(r, "METAL TRACE ID")
        sid = _int_or_none(r, "METAL TRACE REPLAY SESSION ID")
        groups[(tid, sid)].append(r)

    # ---- dropped-op detection -------------------------------------------------------------
    # The device profiler DRAM buffer holds --op-support-count programs before it wraps and
    # silently drops ops. GLOBAL CALL COUNT is a GLOBAL id shared across a mesh op's per-device
    # rows, and sub-mesh/eager ops legitimately skip devices, so whole-CSV per-device contiguity
    # is NOT a reliable signal. Two robust, semantics-based checks instead:
    #   (1) A trace replays exactly what it captured, so every session of a given trace must have
    #       the same op count per device. A short session => ops were dropped in it.
    #   (2) Within one (trace, session, device), the captured op sequence is contiguous, so a
    #       GLOBAL CALL COUNT gap there is a real drop.
    # Either firing => raise --op-support-count (or shorten the schedule); numbers are understated.
    drop_msgs = []
    by_trace = defaultdict(lambda: defaultdict(dict))  # tid -> dev -> {sid: n_ops}
    for (tid, sid), grp in groups.items():
        if tid is None:
            continue
        for r in grp:
            d = _int_or_none(r, "DEVICE ID")
            by_trace[tid][d][sid] = by_trace[tid][d].get(sid, 0) + 1
    for tid, per_dev in by_trace.items():
        for d, sess in per_dev.items():
            counts = set(sess.values())
            if len(counts) > 1:
                drop_msgs.append(
                    f"trace {tid} dev {d}: session op-counts differ {sorted(sess.items())} "
                    f"(a short session = dropped ops)"
                )
    for (tid, sid), grp in groups.items():
        if tid is None:
            continue
        per_dev_calls = defaultdict(list)
        for r in grp:
            d = _int_or_none(r, "DEVICE ID")
            c = _int_or_none(r, "GLOBAL CALL COUNT")
            if d is not None and c is not None:
                per_dev_calls[d].append(c)
        for d, cs in per_dev_calls.items():
            cs = sorted(set(cs))
            missing = (cs[-1] - cs[0] + 1) - len(cs)
            if missing > 0:
                drop_msgs.append(
                    f"trace {tid} session {sid} dev {d}: {missing} GLOBAL CALL COUNT gaps " f"in [{cs[0]},{cs[-1]}]"
                )
    if drop_msgs:
        print("\n!!! POSSIBLE DROPPED OPS — numbers may be UNDERSTATED (raise --op-support-count):")
        for m in drop_msgs[:8]:
            print(f"    {m}")
    else:
        print("no dropped-op signal (trace sessions are complete and contiguous)")

    def group_stats(grp_rows):
        """Per-device aggregation, then summarize across devices. device-active = FW ns;
        gap = op-to-op latency; timeline ~= active + gap; the stage is bound by the slowest dev."""
        per_dev = defaultdict(lambda: {"n": 0, "fw": 0.0, "kern": 0.0, "gap": 0.0})
        for r in grp_rows:
            d = _int_or_none(r, "DEVICE ID")
            a = per_dev[d]
            a["n"] += 1
            a["fw"] += _num(r, "DEVICE FW DURATION [ns]")
            a["kern"] += _num(r, "DEVICE KERNEL DURATION [ns]")
            a["gap"] += _num(r, "OP TO OP LATENCY [ns]")
        return per_dev

    def summarize(grp_rows, dev=None):
        per_dev = group_stats(grp_rows)
        if dev is None:  # slowest device = max(active+gap)
            dev = max(per_dev, key=lambda d: per_dev[d]["fw"] + per_dev[d]["gap"])
        a = per_dev[dev]
        timeline = a["fw"] + a["gap"]
        return dev, a, timeline, per_dev

    # ---- overview table of every group ----------------------------------------------------
    def group_label(key):
        tid, sid = key
        if tid is None:
            return "eager (untraced)"
        return f"trace {tid} / session {sid}"

    print("\n=== all trace/session groups (slowest device) ===")
    print(f"{'group':<34}{'ops':>7}{'active_us':>13}{'gap_us':>12}{'active%':>9}{'sdpa_seq':>10}")
    order = sorted(groups, key=lambda k: (k[0] is None, k[0] or 0, k[1] or 0))
    sdpa_traces = {}  # trace_id -> max device-active across its sessions (denoise stages)
    for key in order:
        grp = groups[key]
        dev, a, timeline, _ = summarize(grp)
        frac = 100.0 * a["fw"] / timeline if timeline else 0.0
        seq = _seq_hint(grp)  # displayed only; NOT used to pick s1/s2 (auto-pick ranks by device-active)
        has_sdpa = any(_classify(r.get("OP CODE", "")) == "SDPA" for r in grp)
        print(f"{group_label(key):<34}{a['n']:>7}{_fmt_us(a['fw'])}{_fmt_us(a['gap'])}{frac:>8.1f}%{seq:>10}")
        if has_sdpa and key[0] is not None:
            sdpa_traces[key[0]] = max(sdpa_traces.get(key[0], 0.0), a["fw"])

    if args.all:
        return

    # ---- pick the focus group -------------------------------------------------------------
    if args.trace is not None:
        focus_tid = args.trace
    else:
        if not sdpa_traces:
            sys.exit("no SDPA-bearing (denoise) trace found; pass --trace <id> or use --all to inspect")
        ranked = sorted(sdpa_traces, key=lambda t: sdpa_traces[t])  # ascending device-active
        # s2 = larger (full-res) denoise trace; s1 = smaller.
        focus_tid = ranked[-1] if args.stage == "s2" else ranked[0]

    sessions = sorted({k[1] for k in groups if k[0] == focus_tid}, key=lambda s: (s is None, s))
    if not sessions:
        sys.exit(f"trace {focus_tid} not found")
    if args.session == "last":
        focus_sid = sessions[-1]
    elif args.session == "first":
        focus_sid = sessions[0]
    else:
        focus_sid = int(args.session)
    grp = groups[(focus_tid, focus_sid)]
    dev, a, timeline, per_dev = summarize(grp, dev=args.device)

    tag = args.stage.upper() if args.trace is None else f"trace {focus_tid}"
    print(f"\n=== FOCUS: {tag}  (trace {focus_tid}, session {focus_sid}, device {dev}) ===")
    print(f"  ops on device {dev}       : {a['n']}")
    print(f"  device-active (FW)        : {a['fw']/1000:.1f} us   ({a['fw']/1e6:.2f} ms)")
    print(f"  device-active (kernel)    : {a['kern']/1000:.1f} us")
    print(f"  op-to-op gap (device idle): {a['gap']/1000:.1f} us   ({a['gap']/1e6:.2f} ms)")
    print(f"  device timeline (act+gap) : {timeline/1000:.1f} us   ({timeline/1e6:.2f} ms)")
    print(
        f"  ACTIVE FRACTION           : {100.0*a['fw']/timeline:.1f}%  "
        f"(low => gap/op-count bound; high => kernel-compute bound)"
    )
    if len(per_dev) > 1:
        acts = [per_dev[d]["fw"] for d in per_dev]
        print(
            f"  across {len(per_dev)} devices: active_us min={min(acts)/1000:.1f} "
            f"max={max(acts)/1000:.1f} (stage is bound by the slowest)"
        )

    # HOST-side span (present for eager ops; degenerate under trace replay where the host
    # dispatches the whole trace as one execute_trace). Reported when available so the
    # host-idle-vs-device split is visible for the eager capture pass.
    hs = [_num(r, "HOST START TS") for r in grp if _int_or_none(r, "DEVICE ID") == dev and _num(r, "HOST START TS")]
    he = [_num(r, "HOST END TS") for r in grp if _int_or_none(r, "DEVICE ID") == dev and _num(r, "HOST END TS")]
    if hs and he:
        host_span = max(he) - min(hs)
        print(
            f"  host dispatch span        : {host_span/1000:.1f} us "
            f"(host-idle vs device = {max(0.0,(host_span-timeline))/1000:.1f} us)"
        )

    # ---- class breakdown ------------------------------------------------------------------
    cls = defaultdict(lambda: {"n": 0, "fw": 0.0})
    for r in grp:
        if _int_or_none(r, "DEVICE ID") != dev:
            continue
        c = cls[_classify(r.get("OP CODE", ""))]
        c["n"] += 1
        c["fw"] += _num(r, "DEVICE FW DURATION [ns]")
    tot = sum(c["fw"] for c in cls.values()) or 1.0
    print(f"\n  {'class':<10}{'ops':>7}{'active_us':>13}{'% of active':>13}")
    for name in ["CCL", "SDPA", "MATMUL", "TILIZE", "TYPECAST", "NORM", "OTHER"]:
        if name in cls:
            c = cls[name]
            print(f"  {name:<10}{c['n']:>7}{_fmt_us(c['fw'])}{100.0*c['fw']/tot:>12.1f}%")

    # ---- raw per-op-code table (ground truth; verify the classifier against this) ----------
    codes = defaultdict(lambda: {"n": 0, "fw": 0.0})
    for r in grp:
        if _int_or_none(r, "DEVICE ID") != dev:
            continue
        c = codes[r.get("OP CODE", "?")]
        c["n"] += 1
        c["fw"] += _num(r, "DEVICE FW DURATION [ns]")
    print(f"\n  top {args.top} op codes by device-active (raw — classifier check):")
    print(f"  {'op code':<44}{'class':<10}{'ops':>6}{'active_us':>13}{'%':>7}")
    for code, c in sorted(codes.items(), key=lambda kv: -kv[1]["fw"])[: args.top]:
        print(f"  {code[:43]:<44}{_classify(code):<10}{c['n']:>6}{_fmt_us(c['fw'])}{100.0*c['fw']/tot:>6.1f}%")


if __name__ == "__main__":
    main()

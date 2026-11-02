#!/usr/bin/env python3
"""Per-step budget of a TRACED decode profile (ops CSV): device kernel time and in-trace op-to-op gap.

Like budget_table.py, one decode step is the block between consecutive Embeddings ops (setup rows before
the first one are excluded). Differences: generic_op rows are split by their compute-kernel source (the A3
matmuls, prep and dequant kernels are all GenericOp), and the OP TO OP gap is reported next to kernel time,
since inside a trace the gap is device-side dispatch, not host time.

Usage: trace_budget.py <ops_perf_results.csv> [step_index] [--by-shape]
"""
import collections
import csv
import os
import sys

path = sys.argv[1]
which = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
by_shape = "--by-shape" in sys.argv


def ns(r, k):
    v = (r.get(k) or "").strip()
    return float(v) if v not in ("", "-") else 0.0


def key(r):
    c = r["OP CODE"].replace("DeviceOperation", "")
    if c.startswith("GenericOp") and "COMPUTE KERNEL SOURCE" not in r:  # device-side CSV: no kernel sources
        c = f"GenericOp:{r.get('CORE COUNT')}cores"
    elif c.startswith("GenericOp"):
        src = os.path.basename((r.get("COMPUTE KERNEL SOURCE") or "").strip("[]' ").split("'")[0]) or "?"
        c = f"GenericOp:{src}"
    if by_shape:
        shp = "x".join((r.get(f"INPUT_0_{d}_PAD[LOGICAL]") or "?").split("[")[-1].rstrip("]") for d in "WZYX")
        c = f"{c} {shp} {r.get('INPUT_0_DATATYPE', '')}->{r.get('OUTPUT_0_DATATYPE', '')}"
    return c


rows = list(csv.DictReader(open(path)))
if "OP TYPE" in rows[0]:  # host+device ops CSV: a step is the block between Embeddings ops
    rows = [r for r in rows if r["OP TYPE"] != "signpost"]
    starts = [i for i, r in enumerate(rows) if "Embedding" in r["OP CODE"]]
    if len(starts) < 2:
        raise SystemExit(f"need >=2 Embeddings ops to bound a decode step, found {len(starts)}")
    which = (len(starts) - 1) // 2 if which is None else which  # a middle step by default
    step = rows[starts[which] : starts[which + 1]]
    print(f"{path}\n  {len(starts) - 1} steps, using step {which}: {len(step)} ops")
else:  # device-side CSV of a traced run: a step is one trace replay session
    for r in rows:
        r["OP CODE"] = r["OP NAME"]
    sess = collections.defaultdict(list)
    for r in rows:
        if r["METAL TRACE REPLAY SESSION ID"]:
            sess[int(r["METAL TRACE REPLAY SESSION ID"])].append(r)
    which = sorted(sess)[0] if which is None else which
    step = sorted(sess[which], key=lambda r: ns(r, "DEVICE FW START CYCLE"))
    print(f"{path}\n  replay sessions {sorted((k, len(v)) for k, v in sess.items())}; using session {which}: {len(step)} ops")

K = "DEVICE KERNEL DURATION [ns]"
G = "OP TO OP LATENCY [ns]"
kern, gap, cnt = collections.Counter(), collections.Counter(), collections.Counter()
for i, r in enumerate(step):
    k = key(r)
    kern[k] += ns(r, K)
    gap[k] += ns(r, G) if i else 0.0  # the first op's gap is the time since the previous step, not dispatch
    cnt[k] += 1
tk, tg = sum(kern.values()), sum(gap.values())
print(f"  kernel {tk / 1e6:.2f} ms/step, op-to-op gap {tg / 1e6:.2f} ms/step, total {(tk + tg) / 1e6:.2f}\n")
print(f"  {'op':<70}{'n':>5}{'kern ms':>9}{'gap ms':>8}{'sum ms':>8}{'us/op':>7}")
for k, _ in sorted(kern.items(), key=lambda kv: -(kv[1] + gap[kv[0]]))[:40]:
    print(f"  {k[:70]:<70}{cnt[k]:5d}{kern[k] / 1e6:9.3f}{gap[k] / 1e6:8.3f}{(kern[k] + gap[k]) / 1e6:8.3f}"
          f"{kern[k] / cnt[k] / 1e3:7.1f}")

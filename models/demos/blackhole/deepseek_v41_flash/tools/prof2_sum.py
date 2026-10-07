"""python prof2_sum.py <profiler dir> [--ops] [--layers 0,1] -- per-layer / per-category device kernel time of the chunk profiled by tests/test_prefill_prof.py.
Per recorded call: kernel duration mean over the 32 devices (and max over the 32 devices). The category comes from the op, the first non-ttnn caller frame and the
nearest prefill_*.py frame (recorder caller string 'inner<outer'); see ``cat``."""

import csv
import json
import sys
from collections import defaultdict

d = sys.argv[1]
show_ops = "--ops" in sys.argv
want = None
if "--layers" in sys.argv:
    want = sys.argv[sys.argv.index("--layers") + 1].split(",")
rows = json.load(open(d + "/rows.json"))
K = "DEVICE KERNEL DURATION [ns]"
dur = defaultdict(dict)
for r in csv.DictReader(open(d + "/.logs/cpp_device_perf_report.csv")):
    if r["METAL TRACE ID"]:
        continue
    dur[int(float(r["GLOBAL CALL COUNT"])) >> 10][r["DEVICE ID"]] = float(r[K] or 0)


def cat(c):
    op, cl = c["op"], c["caller"]
    first, _, outer = cl.partition("<")
    outer = outer or first
    f_file, f_line, f_fn = (first.split(":") + ["", ""])[:3]
    o_file, o_line, o_fn = (outer.split(":") + ["", ""])[:3]
    o_line = int(o_line) if o_line.isdigit() else 0
    if f_file.startswith("engram") or (o_file == "prefill_model.py" and o_line < 262):
        return "engram (device, incl. its CCLs)"
    if "deepseek_prefill" in op:
        short = op.split(".")[-1]
        return {
            "masked_bincount": "moe.count",
            "offset_cumsum": "moe.count",
            "dispatch": "moe.dispatch",
            "unified_routed_expert_moe": "moe.experts",
            "moe_fused_swiglu": "moe.experts",
            "combine": "moe.combine",
            "post_combine_reduce": "moe.post_combine_reduce",
        }.get(short, "moe.other")
    if o_file == "prefill_unified_moe.py":
        if o_fn in ("route_cols",) or f_file.startswith("router"):
            return "moe.router+routing gather"
        if o_fn == "reduce_scatter_tokens":
            return "moe.reduce_scatter+all_to_all"
        if o_fn == "moe_cols":
            return "moe.hidden all_gather + layout"
        return "moe.layout/other"
    if f_file.startswith("router") and o_file == "prefill_layer.py":
        return "moe.router+routing gather"
    if o_fn == "shared_big":
        return "shared_expert"
    if o_fn in ("_moe_unified",):
        return "moe.layout/other"
    if f_file.startswith("mhc"):
        return "mhc kernels (mixes/collapse/expand)"
    if o_file == "prefill_layer.py":
        return "mhc/attn-in glue (allgather, slice, concat)" if "allgather" in first else "layer glue (slice/concat)"
    if o_file == "prefill_sparse.py":
        if o_fn == "select_dyn" or o_fn == "add_keys_dyn":
            return "attn.indexer (score, topk, keys)"
        if o_fn == "attend_dyn":
            return "attn.sparse_sdpa" if "sparse_sdpa" in op else "attn.sparse glue (compact, kv table, q pad)"
        return "attn.sparse other"
    if o_file == "prefill_attention.py":
        if o_fn == "_compress":
            return "attn.compressor"
        if o_fn == "_reduce_scatter_tokens":
            return "attn.reduce_scatter (CCL)"
        if f_file == "pf_tune.py":
            return "attn.projections (pf_tune linear / rope / fp4 sim)"
        return "attn.other (norms, heads, rope, concat)"
    if f_file == "pf_tune.py":
        return "attn.projections (pf_tune linear / rope / fp4 sim)"
    if o_file == "prefill_dyn.py":
        return "dyn masks"
    return "other:" + f_file


tot = defaultdict(lambda: [0.0, 0.0, 0])
nl = 0
for L, calls in rows.items():
    if want and L not in want:
        continue
    nl += 1
    bycat = defaultdict(lambda: [0.0, 0.0, 0])
    byop = defaultdict(lambda: [0.0, 0.0, 0])
    sm = sx = 0.0
    for c in calls:
        mean = mx = 0.0
        for i in range(c["id0"], c["id1"]):
            v = list(dur.get(i, {0: 0.0}).values())
            mean += sum(v) / len(v)
            mx += max(v)
        mean /= 1e6
        mx /= 1e6
        k = cat(c)
        for t in (bycat[k], tot[(L, k)]):
            t[0] += mean
            t[1] += mx
            t[2] += 1
        key = c["op"].replace("ttnn.", "") + " @ " + c["caller"]
        for t in (byop[key],):
            t[0] += mean
            t[1] += mx
            t[2] += 1
        sm += mean
        sx += mx
    print(f"==== layer {L}: {len(calls)} calls, kernel sum mean {sm:.2f} ms, max-over-devices {sx:.2f} ms")
    for k, v in sorted(bycat.items(), key=lambda x: -x[1][0]):
        print(f"   {k:48s} mean {v[0]:8.3f} max {v[1]:8.3f} ms  ({v[2]} calls)")
    if show_ops:
        print("   -- top ops")
        for k, v in sorted(byop.items(), key=lambda x: -x[1][0])[: int(__import__("os").environ.get("TOPN", "25"))]:
            print(f"      {v[0]:8.3f} {v[1]:8.3f} ms x{v[2]:3d} {k}")

if "--json" in sys.argv:
    out = defaultdict(dict)
    for (L, k), v in tot.items():
        out[L][k] = v
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"))

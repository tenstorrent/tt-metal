"""Device programs per decode layer from a layer_profile.py tracy run, labelled by plan scope (PLAN.md proofs).

  python ops_scope.py <ops_perf_results_*.csv> <out.txt>       (prints the per-scope counts, writes the op list)

Device 0, ops between the LAYER_<i>_START / _END signposts. The anchors below match the batch-1 TP expert path
(GeneralizedRouter + indexed sparse_matmul experts); another router / expert path needs its own anchors, and the
script raises StopIteration rather than mislabel when an anchor is missing.

Scope rules (sliding and full layers have the same structure):
  bench   - the layer_bench input reshard (DRAM interleaved -> sharded); in the model the residual arrives sharded
  router  - from the router's scaled projection input (first FP32 BinaryNg after the FP32 unshard) up to the first
            SparseMatmul, except the expert-input multiply (FP32 x FP32 -> BFP8), which is WS2 code
  experts - from the first SparseMatmul through the expert mix matmul (in0 width 32, logical 8)
  ws2     - everything else
"""
import csv
import sys
from collections import defaultdict

path, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(path)))
sections, cur = defaultdict(list), None
for r in rows:
    if r.get("OP TYPE") == "signpost":
        code = r["OP CODE"]
        cur = code[: -len("_START")] if code.endswith("_START") else None
        continue
    if cur and r["DEVICE ID"] == "0":
        sections[cur].append(r)


def f(x):
    try:
        return float(x)
    except ValueError:
        return 0.0


def desc(o):
    shape = "x".join(o.get(f"INPUT_0_{d}_PAD[LOGICAL]", "") for d in "WZYX")
    dt = " ".join(o.get(f"INPUT_{j}_DATATYPE", "") for j in range(2) if o.get(f"INPUT_{j}_DATATYPE"))
    return f"{o['OP CODE'][:38]:38} {f(o['DEVICE KERNEL DURATION [ns]'])/1000:7.2f} us  cores={o.get('CORE COUNT',''):>3}  in0={shape}  {dt} -> {o.get('OUTPUT_0_DATATYPE','')}"


lines = []
for name, ops in sections.items():
    labels = ["ws2"] * len(ops)
    if ops and "InterleavedToSharded" in ops[0]["OP CODE"] and "DRAM" in ops[0].get("INPUT_0_MEMORY", ""):
        labels[0] = "bench"
    first_sparse = next(i for i, o in enumerate(ops) if o["OP CODE"].startswith("SparseMatmul"))
    unshard = next(
        i
        for i, o in enumerate(ops)
        if o["OP CODE"].startswith("ShardedToInterleaved") and o.get("OUTPUT_0_DATATYPE") == "FLOAT32"
    )
    for i in range(unshard + 1, first_sparse):
        o = ops[i]
        expert_input = o["OP CODE"].startswith("BinaryNg") and o.get("OUTPUT_0_DATATYPE") == "BFLOAT8_B"
        labels[i] = "ws2" if expert_input else "router"
    mix = next(
        i
        for i in range(first_sparse, len(ops))
        if ops[i]["OP CODE"].startswith("Matmul") and ops[i].get("INPUT_0_X_PAD[LOGICAL]", "").startswith("32[")
    )
    for i in range(first_sparse, mix + 1):
        labels[i] = "experts"
    count = defaultdict(int)
    time = defaultdict(float)
    for o, lab in zip(ops, labels):
        count[lab] += 1
        time[lab] += f(o["DEVICE KERNEL DURATION [ns]"]) / 1000
    lines.append(f"== {name}: {len(ops)} programs in the profile")
    for lab in ("ws2", "router", "experts", "bench"):
        lines.append(f"   {lab:8s} {count[lab]:3d} programs  {time[lab]:7.1f} us device")
    lines.append(f"   WS2 scope in the model chain (no bench reshard): {count['ws2']} programs")
    for i, (o, lab) in enumerate(zip(ops, labels)):
        lines.append(f"{i:3d} {lab:8s} {desc(o)}")
    lines.append("")
open(out, "w").write("\n".join(lines))
print("\n".join(l for l in lines if l.startswith("==") or l.startswith("   ")))

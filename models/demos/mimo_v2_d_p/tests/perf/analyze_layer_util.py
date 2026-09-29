# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Math utilization of each part of a decoder layer (test_layer_perf.py --profile CSVs), MiMo-V2 on the 2x2.

Every op is priced at its own math fidelity (peak per core 4096 FLOP / cycle at LoFi, / 2 HiFi2, / 3 HiFi3, / 4 HiFi4,
1.35 GHz, the whole 110-core grid): matmuls from their shapes and the profiler's fidelity column, SDPA at HiFi2
(QK^T 192 + PV 128 per score, causal keys = prefix + half the chunk; SWA keys = the 128 window), the routed experts at
LoFi (bf4 weights) with the balanced row count 4 x tokens per chip (8 experts x 64 / 256 of the 2 x S column tokens).
Times are the slowest chip's (the layer's critical path): per op the kernel time on the chip with the largest busy
time, averaged over the measured iterations. Parts: attention matmuls (QKV, o_proj), SDPA (ring SDPA + the K-split
merge), dense MLP, router matmul, routed experts, and the non-math rest (norms, rope, heads, KV writes, CCLs, MoE data
movement, residual adds).

    python models/demos/mimo_v2_d_p/tests/perf/analyze_layer_util.py <ops_perf_results.csv> [...]
"""

import collections
import csv
import re
import sys

MHZ = 1350.0
CORES = 110
FLOP_PER_CYC = {"LoFi": 4096, "HiFi2": 2048, "HiFi3": 4096 / 3, "HiFi4": 1024}
H, I_MOE, DK, DV, NQ_LOCAL, WINDOW, SP = 4096, 2048, 192, 128, 32, 128, 2


def peak(fid):
    return CORES * FLOP_PER_CYC[fid] * MHZ * 1e6


def dim(r, key):
    v = r.get(key, "") or "0"
    return int(v.split("[")[0])


def load(path):
    tags, cur, it = collections.OrderedDict(), None, collections.Counter()
    for r in csv.DictReader(open(path)):
        if r["OP TYPE"] == "signpost":
            c = r["OP CODE"]
            if c.endswith("_start"):
                cur = c[: -len("_start")]
                it[cur] += 1
                tags.setdefault(cur, collections.defaultdict(lambda: collections.defaultdict(list)))
            else:
                cur = None
            continue
        if cur:
            tags[cur][it[cur]][int(r["DEVICE ID"])].append(r)
    return tags


def classify(ops, kind, S, kv_actual):
    """[(part, time_us, flops, fidelity)] for one chip's op list."""
    out = []
    after_sdpa = False
    for r in ops:
        op = r["OP CODE"].replace("DeviceOperation", "")
        t = float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3
        if op.startswith("Matmul"):
            M = dim(r, "INPUT_0_Y_PAD[LOGICAL]") * dim(r, "INPUT_0_Z_PAD[LOGICAL]")
            K, N = dim(r, "INPUT_0_X_PAD[LOGICAL]"), dim(r, "INPUT_1_X_PAD[LOGICAL]")
            fid = r["MATH FIDELITY"] or "HiFi2"
            part = "router matmul" if N == 256 else "dense MLP" if N == 8192 or K == 8192 else "attention matmuls"
            out.append((part, t, 2 * M * K * N, fid))
        elif "KSplitMerge" in op:  # the K-split merge (sdpa_k_split_merge): SDPA time, no math
            out.append(("SDPA", t, 0, "HiFi2"))
            continue
        elif "SDPA" in op:
            chunk = S * SP
            keys = WINDOW if kind == "SWA" else kv_actual + chunk / 2
            out.append(("SDPA", t, 2 * NQ_LOCAL * S * keys * (DK + DV), "HiFi2"))
            after_sdpa = True
            continue
        elif op.startswith("GenericOp") and after_sdpa:
            out.append(("SDPA", t, 0, "HiFi2"))  # the K-split merge (older runs: a generic_op)
        elif op.startswith("FlatRoutedExpert") or op.startswith("UnifiedRoutedExpert"):
            out.append(("routed experts", t, 4 * S * 6 * H * I_MOE, "LoFi"))
        elif op.startswith("BinaryNg") and out and out[-1][0] == "dense MLP":
            out.append(("dense MLP", t, 0, "HiFi2"))  # SiLU * up
        else:
            out.append(("rest (no math)", t, 0, "LoFi"))
        after_sdpa = False
    return out


def main(paths):
    rows = []
    for path in paths:
        for tag, iters in load(path).items():
            m = re.match(r"L(\d+)_(GA|SWA)_C(\d+)_ctx(\d+)", tag)
            if not m:
                continue
            L, kind, S, ctx = int(m[1]), m[2], int(m[3]), int(m[4])
            kv_actual = ctx - S * SP
            acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0])  # part -> time, flops, ideal time
            for devs in iters.values():
                slow = max(devs, key=lambda d: sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in devs[d]))
                for part, t, fl, fid in classify(devs[slow], kind, S, kv_actual):
                    a = acc[part]
                    a[0] += t / len(iters)
                    a[1] += fl / len(iters)
                    a[2] += (fl / peak(fid) * 1e6) / len(iters)
            rows.append((L, kind, S, ctx, acc))
    order = ["SDPA", "attention matmuls", "dense MLP", "router matmul", "routed experts", "rest (no math)"]
    for L, kind, S, ctx, acc in sorted(rows, key=lambda r: (r[0], r[2], r[3])):
        tot = sum(a[0] for a in acc.values())
        ideal = sum(a[2] for a in acc.values())
        print(
            f"\nL{L} {kind} {'dense' if L == 0 else 'MoE'}  {S} tok/chip  ctx {ctx // 1024}K: layer {tot / 1e3:.2f} ms, "
            f"math util (fidelity-priced) {100 * ideal / tot:.1f}%"
        )
        for part in order:
            if part not in acc:
                continue
            t, fl, it_ = acc[part]
            util = f"util {100 * it_ / t:5.1f}%  ({fl / t / 1e6:6.1f} TFLOP/s)" if fl else ""
            print(f"   {part:<18s} {t / 1e3:7.2f} ms  {100 * t / tot:5.1f}% of layer  {util}")


if __name__ == "__main__":
    main(sys.argv[1:])

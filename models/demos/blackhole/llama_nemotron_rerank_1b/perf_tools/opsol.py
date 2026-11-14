# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Per-op cycles vs speed of light between the last start/stop signposts (ops_perf_results CSV).
# Ideal cycles: matmul 2*M*K*N*passes / (4096 * cores), passes LoFi 1 / HiFi2 2 / HiFi4 4;
# SDPA 4*B*Hq*S*S*D*passes / (4096 * cores); data-movement ops: DRAM bytes / bytes-per-cycle.
import csv
import sys
from collections import defaultdict

CORES, FLOP_PER_CYCLE = int(sys.argv[3]) if len(sys.argv) > 3 else 130, 4096
B, S = map(int, sys.argv[2].split("x"))
rows = list(csv.DictReader(open(sys.argv[1])))
st = max(i for i, r in enumerate(rows) if r["OP CODE"] == "start")
en = min(i for i, r in enumerate(rows) if r["OP CODE"] == "stop" and i > st)
ops = [r for r in rows[st + 1 : en] if r["OP TYPE"] != "signpost"]
PASSES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}


def dim(r, k):
    v = r.get(k, "") or "0"
    return int(v.split("[")[0]) if v.split("[")[0].isdigit() else 0


agg = defaultdict(lambda: [0, 0.0, 0.0])
tot_c = tot_i = 0.0
mhz = []
for r in ops:
    fw_c = float(r["DEVICE FW END CYCLE"]) - float(r["DEVICE FW START CYCLE"])
    ns = float(r["DEVICE FW DURATION [ns]"] or 0)
    if ns:
        mhz.append(fw_c / ns * 1e3)
    # Kernel duration, not the FW window: an op on fewer cores than the next lets the idle cores
    # start the next op's firmware early, so FW windows overlap and double count.
    c = float(r["DEVICE KERNEL DURATION [ns]"] or 0) * (fw_c / ns if ns else 1.35)
    name, ideal = r["OP CODE"], 0.0
    f = PASSES.get(r["MATH FIDELITY"], 2)
    # The model's fused matmuls run as generic ops with the weight as input 1: x's width equals the
    # weight's K (fused_add_rmsnorm's two inputs are both [rows, 2048]).
    k1, n1 = dim(r, "INPUT_1_Y_PAD[LOGICAL]"), dim(r, "INPUT_1_X_PAD[LOGICAL]")
    if name.startswith("GenericOp") and k1 == dim(r, "INPUT_0_X_PAD[LOGICAL]") and k1 >= 2048:
        name = "Matmul(fused) " + ("QKV+RoPE" if n1 == 3072 else "+add+RMSNorm")
    # tt/custom_ops/sdpa_mask_cache runs SDPA as a generic op: Q [B, 32 heads, S, 64] is input 0.
    if (
        name.startswith("GenericOp")
        and dim(r, "INPUT_0_Z_PAD[LOGICAL]") == 32
        and dim(r, "INPUT_0_X_PAD[LOGICAL]") == 64
    ):
        name = "SDPAOperation (sdpa_mask_cache)"
    if "Matmul" in name:
        M = dim(r, "INPUT_0_Y_PAD[LOGICAL]") * dim(r, "INPUT_0_Z_PAD[LOGICAL]") * dim(r, "INPUT_0_W_PAD[LOGICAL]")
        K, N = dim(r, "INPUT_1_Y_PAD[LOGICAL]"), dim(r, "INPUT_1_X_PAD[LOGICAL]")
        ideal = 2 * M * K * N * f / (FLOP_PER_CYCLE * CORES)
        name += f" {K}x{N} {r['MATH FIDELITY']}"
    elif "SDPA" in name:
        ideal = 4 * B * 32 * S * S * 64 * 4 / (FLOP_PER_CYCLE * CORES)
    a = agg[name]
    a[0] += 1
    a[1] += c
    a[2] += ideal
    tot_c += c
    tot_i += ideal
print(
    f"{B}x{S} [kernel-duration cycles]: profiler clock ~{sorted(mhz)[len(mhz)//2]:.0f} MHz | total {tot_c/1e6:.2f} Mcycles, compute SoL {tot_i/1e6:.2f} Mcycles -> {100*tot_i/tot_c:.1f}% of SoL"
)
for name, (n, c, i) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print(
        f"  {c/1e6:7.2f} Mcyc {100*c/tot_c:5.1f}%  n={n:<3} {('SoL %5.1f%%' % (100*i/c)) if i else '  (no compute SoL)'}  {name}"
    )

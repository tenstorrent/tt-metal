#!/usr/bin/env python3
# Parse the single tracy ops_perf CSV + manifest into a FLUX (S,Pk) composition report.
import csv
import glob
import json
import os
import statistics
import sys

RUNDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flux_runs"
MANIFEST = sys.argv[2] if len(sys.argv) > 2 else "/tmp/flux_manifest.json"
MD = sys.argv[3] if len(sys.argv) > 3 else "/localdev/cglagovich/tt-metal/minimal_matmul_nslice_kpar_flux.md"

man = json.load(open(MANIFEST))
# Parse the RAW device perf report (robust to the ops_perf post-process crash). Every row here is a
# 64-core minimal_matmul device op (readbacks don't appear); GLOBAL CALL COUNT gives execution order.
csvs = [RUNDIR + "/.logs/cpp_device_perf_report.csv"]
if not os.path.exists(csvs[-1]):
    csvs = sorted(glob.glob(RUNDIR + "/**/cpp_device_perf_report.csv", recursive=True))
if not csvs:
    print("NO CSV in", RUNDIR)
    sys.exit(1)
rows = list(csv.DictReader(open(csvs[-1])))
DUR = "DEVICE KERNEL DURATION [ns]"
GCC = "GLOBAL CALL COUNT"
mm = [r for r in rows if r.get(DUR, "").strip()]
mm.sort(key=lambda r: int(r[GCC]))

# Order-based attribution: the program-cache HASH is per-SHAPE (it does not encode the env-derived
# S/Pk), so it can't separate combos -- but the rows are in execution order and each SUCCESSFUL combo
# emits exactly DROP(=1 pcc + WARMUP) + REPS contiguous minimal-matmul rows. Walk successful combos in
# order, take their chunk, drop the warmup head, median the rest. Failed combos emit 0 rows (skipped).
REPS = int(os.environ.get("FC_REPS", "20"))
WARMUP = int(os.environ.get("FC_WARMUP", "3"))
DROP = 1 + WARMUP
CHUNK = DROP + REPS
succ = [m for m in man if m["ok"]]
expect = CHUNK * len(succ)
if len(mm) != expect:
    print(f"WARN: {len(mm)} minimal rows vs expected {expect} ({CHUNK}/combo x {len(succ)}) — attribution may drift")
i = 0
for m in succ:
    chunk = mm[i : i + CHUNK]
    i += CHUNK
    v = [float(r[DUR]) for r in chunk[DROP:]] or [float(r[DUR]) for r in chunk]
    m["dur_ns"] = statistics.median(v) if v else None

# group by shape, write md
by = {}
for m in man:
    by.setdefault((m["M"], m["K"], m["N"]), []).append(m)

L = [
    "# FLUX (big-shape sweep): composing N-slicing (S) with K-par (Pk)\n",
    "Device kernel duration (median over cache-hit reps, single device session under tracy). Auto block "
    "sizer ON. `S=auto` = today's production N-slicer (Pk=1); Pk>1 = fused plan-B column reduction. Only "
    "shapes whose auto-slicer engages (S>1) are included. Combos restricted to S*Pk in {4,8} with no "
    "M-padding and Pk | K_tiles. WH B0 8x8.\n",
]
best_rows = []
detail = []
for (M, K, N), rs in sorted(by.items()):
    auto = next((r for r in rs if r["S"] == "auto"), None)
    adur = auto["dur_ns"] if auto and auto.get("dur_ns") else None
    out_t = rs[0]["out_tiles"]
    detail.append(f"\n## {M}x{K}x{N}  (out_tiles={out_t}, Kt={rs[0]['Kt']}, Mt={rs[0]['Mt']})\n")
    detail.append("| S | Pk | device us | PCC | vs auto |")
    detail.append("|---|---|---|---|---|")
    ok = [r for r in rs if r.get("dur_ns") and (r["pcc"] or 0) > 0.99]
    bestc = min(ok, key=lambda x: x["dur_ns"]) if ok else None
    for r in sorted(rs, key=lambda x: (x["S"] != "auto", str(x["S"]), x["Pk"])):
        us = f"{r['dur_ns']/1000:.2f}" if r.get("dur_ns") else ("FAIL" if not r["ok"] else "—")
        spd = f"{adur/r['dur_ns']:.2f}x" if (adur and r.get("dur_ns")) else "—"
        star = " ⭐" if r is bestc else ""
        detail.append(f"| {r['S']} | {r['Pk']} | {us} | {r['pcc']} | {spd}{star} |")
    if bestc and adur:
        best_rows.append((M, K, N, out_t, bestc["S"], bestc["Pk"], bestc["dur_ns"], adur / bestc["dur_ns"]))

L.append("\n## Best combo per shape (sorted by speedup)\n")
L.append("| shape | out tiles | best S | best Pk | best us | auto us | speedup |")
L.append("|---|---|---|---|---|---|---|")
for M, K, N, ot, S, Pk, d, sp in sorted(best_rows, key=lambda x: -x[7]):
    adur = d * sp
    L.append(f"| {M}x{K}x{N} | {ot} | {S} | {Pk} | {d/1000:.2f} | {adur/1000:.2f} | **{sp:.2f}x** |")
L += detail
open(MD, "w").write("\n".join(L) + "\n")
print(
    "wrote",
    MD,
    "| best speedups:",
    ", ".join(f"{M}x{K}x{N}:{sp:.2f}x" for (M, K, N, ot, S, Pk, d, sp) in sorted(best_rows, key=lambda x: -x[7])[:6]),
)

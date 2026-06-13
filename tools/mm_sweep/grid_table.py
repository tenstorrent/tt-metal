#!/usr/bin/env python3
# Build the full WH grid-sweep markdown (ALL shapes, ascending util) from test_grid output.
# Usage: grid_table.py <run_dir> <manifest.json> <out.md> [grid_y] [grid_x]
# Per-shape device rows in cpp_device_perf_report.csv: 1 build + [1 ttnn.matmul ref if K-par] + WARMUP +
# REPS; the timed minimal reps are the last REPS of each chunk.
import csv, json, math, statistics, sys, os

RUN = sys.argv[1] if len(sys.argv) > 1 else "/tmp/grid2_runs"
MAN = sys.argv[2] if len(sys.argv) > 2 else "/tmp/grid2_manifest.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/localdev/cglagovich/tt-metal/minimal_matmul_grid_sweep.md"
GY = int(sys.argv[4]) if len(sys.argv) > 4 else 8
GX = int(sys.argv[5]) if len(sys.argv) > 5 else 8
REPS = int(os.environ.get("FC_REPS", "10"))
WARMUP = 2

man = json.load(open(MAN))
rows = [
    r
    for r in csv.DictReader(open(RUN + "/.logs/cpp_device_perf_report.csv"))
    if r.get("DEVICE KERNEL DURATION [ns]", "").strip()
]
rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
ok = [m for m in man if m["ok"]]  # manifest order == execution order
exp = sum((1 + WARMUP + REPS) + (1 if m["pred_Pk"] > 1 else 0) for m in ok)
if len(rows) != exp:
    print(f"WARN rows={len(rows)} expected={exp} — attribution may drift")
i = 0
for m in ok:
    clen = 1 + WARMUP + REPS + (1 if m["pred_Pk"] > 1 else 0)
    chunk = rows[i : i + clen]
    i += clen
    reps = chunk[-REPS:]  # last REPS rows are always the minimal timing reps
    m["us"] = statistics.median(float(r["DEVICE KERNEL DURATION [ns]"]) for r in reps) / 1000.0
    m["thru"] = (m["Mt"] * m["Nt"] * m["Kt"]) / m["us"]

peak = max(m["thru"] for m in ok)
for m in ok:
    m["util"] = m["thru"] / peak
    m["orient"] = "T" if m["Mt"] > m["Nt"] else "N"  # transpose (M>N) vs normal


def geo(xs):
    return math.exp(sum(math.log(max(x, 1e-9)) for x in xs) / len(xs)) if xs else 0


# ---- summary ----
kp = [m for m in ok if m["pred_Pk"] > 1]
chk = [m for m in ok if m["pcc"] is not None]
bad = [m for m in chk if m["pcc"] < 0.99]
L = [
    "# minimal_matmul WH grid sweep — all shapes, ascending util\n",
    f"Dense geometric M/K/N grid (32..8192), default auto path (heuristic on). util = tile-MAC "
    f"throughput / observed peak ({peak:.0f} tile-MAC/us). **orient**: N=normal(M<=N), T=transpose(M>N). "
    f"**S**=N/M-slice (num_slices), **Pk**=K-split (num_k_slices); S1/Pk1 = no slicing/no K-par. PCC vs "
    f"ttnn.matmul for every K-par shape, vs torch for small shapes (blank = skipped, plain large path).\n",
    f"WH 8x8. shapes={len(ok)} (0 failures). K-par engaged={len(kp)} ({100*len(kp)/len(ok):.0f}%). "
    f"PCC-checked={len(chk)} (K-par unverified={sum(1 for m in kp if m['pcc'] is None)}), bad_pcc={len(bad)}.\n",
]
# util by out_tiles bucket
L.append("## util geomean by output size (grid-fill)\n")
L.append("| out_tiles | n | util geomean | max |")
L.append("|---|---|---|---|")
bk = [
    ("1-8", 1, 8),
    ("9-32", 9, 32),
    ("33-64", 33, 64),
    ("65-256", 65, 256),
    ("257-1024", 257, 1024),
    ("1025-4096", 1025, 4096),
    ("4097+", 4097, 10**9),
]
for name, lo, hi in bk:
    g = [m for m in ok if lo <= m["out_tiles"] <= hi]
    if g:
        L.append(f"| {name} | {len(g)} | {geo([m['util'] for m in g]):.3f} | {max(m['util'] for m in g):.3f} |")
L.append(
    f"\nK-par util geomean={geo([m['util'] for m in kp]):.3f} (n={len(kp)}); "
    f"non-K-par={geo([m['util'] for m in ok if m['pred_Pk']==1]):.3f}.\n"
)
if bad:
    L.append("\n⚠ **PCC < 0.99:** " + ", ".join(f"{m['M']}x{m['K']}x{m['N']}={m['pcc']}" for m in bad) + "\n")

# ---- full table ----
L.append("## All shapes (ascending util)\n")
L.append("| M | K | N | out | Kt | orient | S | Pk | util | us | pcc |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for m in sorted(ok, key=lambda x: x["util"]):
    pcc = "" if m["pcc"] is None else f"{m['pcc']:.5f}"
    L.append(
        f"| {m['M']} | {m['K']} | {m['N']} | {m['out_tiles']} | {m['Kt']} | {m['orient']} | "
        f"{m['pred_S']} | {m['pred_Pk']} | {m['util']:.3f} | {m['us']:.1f} | {pcc} |"
    )
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}: {len(ok)} shapes, peak={peak:.0f}, K-par={len(kp)}, bad_pcc={len(bad)}")
print(
    "worst 10:",
    [
        (m["M"], m["K"], m["N"], round(m["util"], 3), f"S{m['pred_S']}Pk{m['pred_Pk']}")
        for m in sorted(ok, key=lambda x: x["util"])[:10]
    ],
)

#!/usr/bin/env python3
# Join test_grid device times with the manifest and compute compute-efficiency to surface
# underperforming shapes. Usage: grid_parse.py <run_dir> <manifest.json> <out.csv>
import csv, json, math, statistics, sys, os

RUNDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/grid_runs"
MAN = sys.argv[2] if len(sys.argv) > 2 else "/tmp/grid_manifest.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/grid_results.csv"
REPS = int(os.environ.get("FC_REPS", "10"))
WARMUP = 2
DROP, CHUNK = 1 + WARMUP, 1 + WARMUP + REPS

man = json.load(open(MAN))
rows = [
    r
    for r in csv.DictReader(open(RUNDIR + "/.logs/cpp_device_perf_report.csv"))
    if r.get("DEVICE KERNEL DURATION [ns]", "").strip()
]
rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
succ = [m for m in man if m["ok"]]
if len(rows) != CHUNK * len(succ):
    print(f"WARN rows={len(rows)} expected={CHUNK*len(succ)} ({CHUNK}/shape x {len(succ)}) — attribution may drift")
i = 0
for m in succ:
    ch = rows[i : i + CHUNK]
    i += CHUNK
    m["us"] = statistics.median(float(r["DEVICE KERNEL DURATION [ns]"]) for r in ch[DROP:]) / 1000.0
    # tile-MAC throughput (work-per-time); peak-normalized below to get a util proxy
    m["tmac"] = m["Mt"] * m["Nt"] * m["Kt"]
    m["thru"] = m["tmac"] / m["us"]

ok = [m for m in man if m.get("us")]
peak = max(m["thru"] for m in ok)  # observed peak tile-MAC/us (compute-bound large squares)
for m in ok:
    m["util"] = m["thru"] / peak

cols = ["M", "K", "N", "Mt", "Kt", "Nt", "out_tiles", "pred_S", "pred_Pk", "macs", "us", "thru", "util", "pcc"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for m in sorted(man, key=lambda x: (x["M"], x["K"], x["N"])):
        if m.get("us"):
            w.writerow({**m, "us": round(m["us"], 2), "thru": round(m["thru"], 1), "util": round(m["util"], 3)})

bad_pcc = [m for m in ok if m["pcc"] is not None and m["pcc"] < 0.99]
fails = [m for m in man if not m["ok"]]
kpar = [m for m in ok if m["pred_Pk"] > 1]
print(
    f"shapes ok={len(ok)} failed={len(fails)}  peak={peak:.0f} tile-MAC/us  PCC-checked={sum(1 for m in ok if m['pcc'] is not None)} bad_pcc={len(bad_pcc)}"
)
print(
    f"K-par engaged on {len(kpar)}/{len(ok)} shapes; util geomean(all)={math.exp(sum(math.log(m['util']) for m in ok)/len(ok)):.3f}"
)
print("\nLowest-util shapes (worst 20, util = thru / observed-peak):")
for m in sorted(ok, key=lambda x: x["util"])[:20]:
    print(
        f"  {m['M']:>5}x{m['K']:>5}x{m['N']:<5} out={m['out_tiles']:>5} Kt={m['Kt']:>4} S{m['pred_S']}Pk{m['pred_Pk']}  {m['us']:>8.1f}us  util={m['util']:.3f}"
    )
if bad_pcc:
    print("\n⚠ PCC < 0.99:", [(m["M"], m["K"], m["N"], m["pcc"]) for m in bad_pcc])
if fails:
    print(f"\nfailed shapes ({len(fails)}):", [(m["M"], m["K"], m["N"], m.get("err", "")[:40]) for m in fails[:10]])
print(f"\nwrote {OUT}")

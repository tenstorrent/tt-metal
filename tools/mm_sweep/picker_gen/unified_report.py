#!/usr/bin/env python3
"""ONE table: every corpus + HeyGen shape re-measured on HEAD, vs the pre-fix reports.

Byte accounting matches prod_sweep_report.py exactly (in0 duplicated Ns times, in1 once, out once);
FPU% is against the full 110-core grid at bf16 HiFi2 (2048 FLOP/cycle/core, 1.35 GHz).
"""
import json, os, re, statistics, sys

S = os.path.dirname(os.path.abspath(__file__))
old = json.load(open(os.path.join(S, "old.json")))
corpus, heygen = old["corpus"], old["heygen"]
PEAK = 512.0
CORE_PEAK = (4096 / 2) * 1.35e9
GRID_PEAK = 110 * CORE_PEAK
CFG_RE = re.compile(r"pick=\((\d+),(\d+),(\d+),(\d+),(\d+)\) cores=(\d+) reduction=(\S+) placement=(\S+)")
cd = lambda v: -(-v // 32)

rows, failed = [], []
for line in open(os.path.join(S, "head_sweep.jsonl")):
    js, _, cfgpart = line.partition("||CFG||")
    r = json.loads(js[js.find("{"):])
    M, K, N = r["M"], r["K"], r["N"]
    name = "%dx%dx%d" % (M, K, N)
    src = "corpus" if name in corpus else ("HeyGen" if name in heygen else "new")
    o = corpus.get(name) or heygen.get(name)
    if r.get("outcome") != "ok":
        failed.append((name, src, str(r.get("err"))[:95]))
        continue
    m = CFG_RE.search(cfgpart)
    g = m.groups() if m else ("?",) * 8
    Pk, Ns = (int(g[0]), int(g[1])) if m else (1, 1)
    wall = r["median_us"]
    eff = (Ns * M * K * 2) + (K * N * 2) + (M * N * 2)
    eff_gbps = eff / (wall * 1e-6) / 1e9
    flops = 2.0 * M * N * K
    bm = r.get("block_medians", [])
    rows.append(dict(
        name=name, src=src, Mt=cd(M),
        cfg="%s,%s,%s,%s,%s" % g[:5] if m else "?", cores=g[5], red=g[6], place=g[7],
        wall=wall, eff=eff_gbps, pct=100 * eff_gbps / PEAK,
        tflops=flops / (wall * 1e-6) / 1e12,
        fpu=100.0 * flops / GRID_PEAK / (wall * 1e-6),
        pcc=r.get("pcc", float("nan")),
        spread=100.0 * (max(bm) - min(bm)) / statistics.median(bm) if len(bm) > 1 else 0.0,
        old_us=o["us"] if o else None, old_cfg=o["cfg"] if o else None,
        old_red=o["red"] if o else None,
        delta=(100.0 * (wall - o["us"]) / o["us"]) if o else None,
        moved=(o is not None and m is not None and o["cfg"] != "%s,%s,%s,%s,%s" % g[:5]),
    ))

rows.sort(key=lambda z: z["eff"])
H = ["shape","src","Mt","Pk,Ns,Sm,kb,nsb","core","reduction","placement","dev us","was us","Δ%",
     "eff GB/s","%pk","TFLOP/s","FPU%","PCC","blk%"]
print("| " + " | ".join(H) + " |")
print("|" + "|".join(["---"] * len(H)) + "|")
for r in rows:
    was = "%.2f" % r["old_us"] if r["old_us"] else "—"
    dl = ("%+.1f%%" % r["delta"]) if r["delta"] is not None else "new"
    star = " *" if r["moved"] else ""
    print("| %s | %s | %d | %s%s | %s | %s | %s | %.2f | %s | %s | %.1f | %.0f%% | %.1f | %.1f%% | %.5f | %.1f |" % (
        r["name"], r["src"], r["Mt"], r["cfg"], star, r["cores"], r["red"], r["place"],
        r["wall"], was, dl, r["eff"], r["pct"], r["tflops"], r["fpu"], r["pcc"], r["spread"]))

n = len(rows)
d = [r["delta"] for r in rows if r["delta"] is not None]
print("\n**%d shapes measured on HEAD** (%d corpus, %d HeyGen, %d newly runnable). All at defaults (config=None)." % (
    n, sum(1 for r in rows if r["src"] == "corpus"), sum(1 for r in rows if r["src"] == "HeyGen"),
    sum(1 for r in rows if r["src"] == "new")))
print("\n- effective DRAM BW: min %.1f, median %.1f, max %.1f GB/s (peak %.0f); median %.0f%% of peak" % (
    rows[0]["eff"], statistics.median([r["eff"] for r in rows]), rows[-1]["eff"], PEAK,
    statistics.median([r["pct"] for r in rows])))
print("- FPU vs full 110-core grid: median %.1f%%, max %.1f%% (%s)" % (
    statistics.median([r["fpu"] for r in rows]), max(r["fpu"] for r in rows),
    max(rows, key=lambda z: z["fpu"])["name"]))
print("- correctness: %d/%d PCC >= 0.999 (min %.5f)" % (
    sum(1 for r in rows if r["pcc"] >= 0.999), n, min(r["pcc"] for r in rows)))
print("- stability: median block spread %.1f%%, worst %.1f%% (%s)" % (
    statistics.median([r["spread"] for r in rows]), max(r["spread"] for r in rows),
    max(rows, key=lambda z: z["spread"])["name"]))
if d:
    d_sorted = sorted(d)
    print("- **vs the pre-fix reports (%d comparable): median %+.1f%%, best %+.1f%%, worst %+.1f%%**" % (
        len(d), d_sorted[len(d_sorted) // 2], d_sorted[0], d_sorted[-1]))
    reg = [r for r in rows if r["delta"] is not None and r["delta"] > 3.0]
    imp = [r for r in rows if r["delta"] is not None and r["delta"] < -3.0]
    print("  - beyond +-3%%: %d slower, %d faster" % (len(reg), len(imp)))
    for r in sorted(reg, key=lambda z: -z["delta"])[:8]:
        print("    - SLOWER %s %+.1f%% (%.2f -> %.2f us)" % (r["name"], r["delta"], r["old_us"], r["wall"]))
    for r in sorted(imp, key=lambda z: z["delta"])[:8]:
        print("    - FASTER %s %+.1f%% (%.2f -> %.2f us)" % (r["name"], r["delta"], r["old_us"], r["wall"]))
mv = [r for r in rows if r["moved"]]
print("- config changes (*): %d" % len(mv))
for r in mv:
    print("    - %s: %s/%s -> %s/%s" % (r["name"], r["old_cfg"], r["old_red"], r["cfg"], r["red"]))
red = [r for r in rows if r["old_red"] and r["old_red"] != r["red"]]
print("- reduction-strategy changes: %d%s" % (len(red), "" if not red else ""))
for r in red:
    print("    - %s: %s -> %s" % (r["name"], r["old_red"], r["red"]))
if failed:
    print("\n**Not measured (%d):**\n" % len(failed))
    for nm, src, e in failed:
        print("- `%s` (%s): %s" % (nm, src, e))

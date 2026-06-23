#!/usr/bin/env python3
# Re-runnable analysis of the joint (S,Pk,blocking) sweep results. Safe to run while the sweep is still
# going (just reads whatever results_*.jsonl exist so far). Answers:
#   Q1 - which shapes underperform on ABSOLUTE flop utilization, and what's the pattern?
#   Q2 - which blockings / partitioning (S, Pk) are optimal across shapes, and what does it indicate?
#
#   python tools/mm_sweep/analyze_sweep.py [--md OUT.md]
#
# Peak (per user): 2048 FLOP/cycle/core * 64 cores (8x8) * 1 GHz = 131.072 TFLOP/s. We always divide by
# the FULL-grid peak even if a shape only fills part of the grid -- under-filling the grid is itself a
# utilization loss we want to see. Matmul work = 2*M*K*N FLOPs. util% = work / (peak * seconds).
import glob, json, math, os, sys, statistics
from collections import defaultdict

OUT_DIR = "/localdev/cglagovich/mm_jointsweep"
FILES = sorted(glob.glob(os.path.join(OUT_DIR, "results_*.jsonl")))
GX = GY = 8
FLOP_PER_CYCLE_PER_CORE = 2048
FREQ_HZ = 1e9
PEAK_FLOPS = FLOP_PER_CYCLE_PER_CORE * GX * GY * FREQ_HZ  # 1.31072e14
PCC_MIN = 0.98


def load():
    """All ok, timed, PCC-passing config records, deduped by (shape, S, Pk, blk) keeping fastest."""
    best = {}
    for f in FILES:
        for ln in open(f):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if not r.get("ok") or r.get("us") is None:
                continue
            if r.get("pcc") is not None and r["pcc"] < PCC_MIN:
                continue
            k = (r["M"], r["K"], r["N"], str(r["S"]), str(r["Pk"]), r["blk"])
            if k not in best or r["us"] < best[k]["us"]:
                best[k] = r
    return list(best.values())


def util_pct(M, K, N, us):
    return 100.0 * (2.0 * M * K * N) / (PEAK_FLOPS * us * 1e-6)


def per_shape(recs):
    """For each shape: best config (min us), heuristic/auto config, and all configs."""
    byshape = defaultdict(list)
    for r in recs:
        byshape[(r["M"], r["K"], r["N"])].append(r)
    out = {}
    for sh, rs in byshape.items():
        best = min(rs, key=lambda r: r["us"])
        heur = next((r for r in rs if r["blk"] == "heuristic"), None)
        out[sh] = {
            "M": sh[0],
            "K": sh[1],
            "N": sh[2],
            "Mt": sh[0] // 32,
            "Kt": sh[1] // 32,
            "Nt": sh[2] // 32,
            "n_cfgs": len(rs),
            "best": best,
            "best_us": best["us"],
            "best_util": util_pct(*sh, best["us"]),
            "heur_us": heur["us"] if heur else None,
            "heur_util": util_pct(*sh, heur["us"]) if heur else None,
        }
    return out


def histo(label, vals, edges):
    """Simple text histogram of values into [edges] buckets (counts)."""
    buckets = [0] * (len(edges) + 1)
    for v in vals:
        placed = False
        for i, e in enumerate(edges):
            if v < e:
                buckets[i] += 1
                placed = True
                break
        if not placed:
            buckets[-1] += 1
    labels = [f"<{edges[0]}"] + [f"{edges[i-1]}-{edges[i]}" for i in range(1, len(edges))] + [f">={edges[-1]}"]
    return label, list(zip(labels, buckets))


def main():
    md = []

    def emit(s=""):
        print(s)
        md.append(s)

    recs = load()
    shapes = per_shape(recs)
    n = len(shapes)
    emit(f"# minimal_matmul sweep analysis")
    emit(
        f"\nPeak = {FLOP_PER_CYCLE_PER_CORE} FLOP/cyc/core x {GX*GY} cores x {FREQ_HZ/1e9:.0f}GHz = "
        f"{PEAK_FLOPS/1e12:.1f} TFLOP/s. util% = 2*M*K*N / (peak * time)."
    )
    emit(f"Loaded {len(recs)} configs over {n} shapes (best-per-shape used for util).\n")

    vals = sorted(shapes.values(), key=lambda s: s["best_util"])
    utils = [s["best_util"] for s in vals]
    emit(f"## Utilization distribution (best config per shape)")
    emit(
        f"- median best-util: {statistics.median(utils):.1f}%   mean: {statistics.mean(utils):.1f}%   "
        f"max: {max(utils):.1f}%   min: {min(utils):.1f}%"
    )
    _, hb = histo("util", utils, [5, 10, 20, 30, 40, 50, 60, 70])
    emit("- histogram: " + "  ".join(f"{lbl}:{cnt}" for lbl, cnt in hb))

    # ---- Q1: underperformers + pattern correlation ----
    emit(f"\n## Q1 - absolute FLOP-utilization underperformers\n")
    emit("### Utilization CEILING (top 12 shapes) - how close does the op ever get to peak?")
    emit("| shape | Mt x Nt x Kt | best util% | best µs | best cfg |")
    emit("|---|---|---|---|---|")
    for s in sorted(vals, key=lambda s: -s["best_util"])[:12]:
        b = s["best"]
        emit(
            f"| {s['M']}x{s['K']}x{s['N']} | {s['Mt']}x{s['Nt']}x{s['Kt']} | {s['best_util']:.1f} | "
            f"{s['best_us']:.1f} | S{b['S']},Pk{b['Pk']},{b['blk']} |"
        )

    # Residual: util vs the MEDIAN util of shapes in the same total-work decile. residual<1 => the shape
    # underperforms peers of similar size (genuine inefficiency, not just being small).
    work_edges = [0.5, 4, 32, 256, 2048]

    def work_bin(s):
        w = 2 * s["M"] * s["K"] * s["N"] / 1e9
        for i, e in enumerate(work_edges):
            if w < e:
                return i
        return len(work_edges)

    bin_med = {}
    binmap = defaultdict(list)
    for s in vals:
        binmap[work_bin(s)].append(s["best_util"])
    for bk, us in binmap.items():
        bin_med[bk] = statistics.median(us)
    for s in vals:
        s["_resid"] = s["best_util"] / max(1e-9, bin_med[work_bin(s)])
    # only consider shapes with non-trivial work (>=4 GFLOP) so "underperform" means real inefficiency
    big = [s for s in vals if 2 * s["M"] * s["K"] * s["N"] / 1e9 >= 4.0]
    emit(f"\n### Underperformers RELATIVE to their size class (work >=4 GFLOP, residual = util / size-class-median)")
    emit("Low residual = inefficient vs similar-sized shapes (not merely small). Worst 20:")
    emit("| shape | Mt x Nt x Kt | out_tiles | util% | size-class median% | residual | best cfg |")
    emit("|---|---|---|---|---|---|---|")
    for s in sorted(big, key=lambda s: s["_resid"])[:20]:
        b = s["best"]
        emit(
            f"| {s['M']}x{s['K']}x{s['N']} | {s['Mt']}x{s['Nt']}x{s['Kt']} | {s['Mt']*s['Nt']} | "
            f"{s['best_util']:.1f} | {bin_med[work_bin(s)]:.1f} | {s['_resid']:.2f} | "
            f"S{b['S']},Pk{b['Pk']},{b['blk']} |"
        )

    emit("\n### Worst 25 shapes by best-achievable util (absolute - dominated by tiny shapes)")
    emit("| shape | Mt x Nt x Kt | out_tiles | best util% | best µs | best cfg (S,Pk,blk) |")
    emit("|---|---|---|---|---|---|")
    for s in vals[:25]:
        b = s["best"]
        emit(
            f"| {s['M']}x{s['K']}x{s['N']} | {s['Mt']}x{s['Nt']}x{s['Kt']} | {s['Mt']*s['Nt']} | "
            f"{s['best_util']:.1f} | {s['best_us']:.1f} | S{b['S']},Pk{b['Pk']},{b['blk']} |"
        )

    # pattern correlations: median util binned by a feature
    def binned(feature_fn, edges, name):
        groups = defaultdict(list)
        labels = [f"<{edges[0]}"] + [f"{edges[i-1]}-{edges[i]}" for i in range(1, len(edges))] + [f">={edges[-1]}"]
        for s in vals:
            fv = feature_fn(s)
            idx = len(edges)
            for i, e in enumerate(edges):
                if fv < e:
                    idx = i
                    break
            groups[labels[idx]].append(s["best_util"])
        emit(f"\n### util vs {name}")
        emit(f"| {name} bucket | #shapes | median util% | mean util% |")
        emit("|---|---|---|---|")
        for lbl in labels:
            g = groups.get(lbl, [])
            if g:
                emit(f"| {lbl} | {len(g)} | {statistics.median(g):.1f} | {statistics.mean(g):.1f} |")

    binned(lambda s: s["Kt"], [2, 4, 8, 16, 32, 64, 128], "Kt (K-depth, tiles)")
    binned(lambda s: s["Mt"] * s["Nt"], [4, 16, 64, 256, 1024, 4096], "out_tiles (Mt*Nt)")
    binned(
        lambda s: max(s["Mt"], s["Nt"]) / max(1, min(s["Mt"], s["Nt"])),
        [1.5, 3, 8, 32, 128],
        "aspect (max/min of Mt,Nt)",
    )
    binned(lambda s: 2 * s["M"] * s["K"] * s["N"] / 1e9, [0.5, 4, 32, 256, 2048], "total work (GFLOP)")

    # ---- Q2: optimal blocking / partitioning patterns ----
    emit(f"\n## Q2 - optimal blocking & partitioning patterns\n")

    def dist(key_fn, name):
        c = defaultdict(int)
        for s in vals:
            c[key_fn(s["best"])] += 1
        emit(f"### optimal {name} (count of shapes where it's the best)")
        emit("  " + "  ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])))

    dist(lambda b: f"S{b['S']}", "num_slices S")
    dist(lambda b: f"Pk{b['Pk']}", "k-parallel Pk")
    dist(lambda b: f"S{b['S']}/Pk{b['Pk']}", "(S,Pk) combo")
    dist(lambda b: ("heuristic" if b["blk"] == "heuristic" else "pinned"), "auto-heuristic vs pinned block won")
    dist(lambda b: f"Kblk{b.get('Kblk', '?')}", "K_block")

    # heuristic gap: how far is the auto/heuristic config from the best, per shape
    gaps = [(s, s["heur_us"] / s["best_us"]) for s in vals if s["heur_us"]]
    if gaps:
        ratios = [g for _, g in gaps]
        emit(f"\n### Heuristic gap (auto config µs / best config µs; 1.0 = heuristic already optimal)")
        emit(
            f"- shapes with heuristic data: {len(gaps)}   median ratio: {statistics.median(ratios):.2f}x   "
            f"mean: {statistics.mean(ratios):.2f}x   worst: {max(ratios):.2f}x"
        )
        _, hg = histo("gap", ratios, [1.05, 1.2, 1.5, 2.0, 3.0])
        emit("- histogram: " + "  ".join(f"{lbl}:{cnt}" for lbl, cnt in hg))
        emit("\n#### Worst 20 heuristic misses (biggest best-vs-heuristic speedup left on the table)")
        emit("| shape | Mt x Nt x Kt | heur µs | best µs | speedup | best cfg | heur util% -> best util% |")
        emit("|---|---|---|---|---|---|---|")
        for s, g in sorted(gaps, key=lambda x: -x[1])[:20]:
            b = s["best"]
            emit(
                f"| {s['M']}x{s['K']}x{s['N']} | {s['Mt']}x{s['Nt']}x{s['Kt']} | {s['heur_us']:.1f} | "
                f"{s['best_us']:.1f} | {g:.2f}x | S{b['S']},Pk{b['Pk']},{b['blk']} | "
                f"{s['heur_util']:.1f} -> {s['best_util']:.1f} |"
            )

    if "--md" in sys.argv:
        path = sys.argv[sys.argv.index("--md") + 1]
        open(path, "w").write("\n".join(md) + "\n")
        print(f"\n[wrote {path}]")

    # Full per-shape table, ascending util, for browsing.
    tpath = (
        sys.argv[sys.argv.index("--table") + 1] if "--table" in sys.argv else os.path.join(OUT_DIR, "results_table.md")
    )
    t = [
        "# minimal_matmul sweep - all shapes by ascending FLOP utilization",
        f"\nPeak {PEAK_FLOPS/1e12:.1f} TFLOP/s (8x8, 2048 FLOP/cyc/core, 1GHz). util% = 2*M*K*N/(peak*time).",
        f"{n} shapes, best config per shape. heur_gap = heuristic_µs / best_µs (1.0 = heuristic optimal).\n",
        "| # | shape (MxKxN) | Mt x Nt x Kt | out_tiles | work GFLOP | best util% | best µs | "
        "best S | best Pk | best blk | sbh x sbw | heur util% | heur_gap | #cfgs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(sorted(vals, key=lambda s: s["best_util"]), 1):
        b = s["best"]
        work = 2 * s["M"] * s["K"] * s["N"] / 1e9
        sb = f"{b.get('sbh','?')}x{b.get('sbw','?')}" if b["blk"] not in ("heuristic", "auto") else "-"
        hg = f"{s['heur_us']/s['best_us']:.2f}" if s["heur_us"] else "-"
        hu = f"{s['heur_util']:.1f}" if s["heur_util"] is not None else "-"
        t.append(
            f"| {i} | {s['M']}x{s['K']}x{s['N']} | {s['Mt']}x{s['Nt']}x{s['Kt']} | {s['Mt']*s['Nt']} | "
            f"{work:.2f} | {s['best_util']:.2f} | {s['best_us']:.1f} | {b['S']} | {b['Pk']} | "
            f"{b['blk']} | {sb} | {hu} | {hg} | {s['n_cfgs']} |"
        )
    open(tpath, "w").write("\n".join(t) + "\n")
    print(f"[wrote {tpath}  ({n} shapes)]")


if __name__ == "__main__":
    main()

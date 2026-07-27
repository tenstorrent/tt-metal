#!/usr/bin/env python3
"""Offline analysis of the generator-driven campaign (results_v2).

Two deliverables (instruction section 4):
  1. OPTIMUM GAPS: how far the current production picker is from the measured optimum over the
     generator-selected candidate set, per train/val/holdout split and the FLUX/LTX subset.
  2. PRUNING AUDIT: for each shape the generator adds 8 deterministic candidates drawn from the PRUNED
     (not-selected-by-the-models) feasible space. If a pruned/audit candidate is faster than the best
     model-selected ("structured") candidate, the pruning excluded a real winner -> flagged.

Per-config wall = median over ALL ok timed samples (initial + rerun merged), so near-winners use their
stability-rerun samples. The production config is the candidate tagged 'explicit' (we always include the
real production pick); if absent we fall back to the offline production_pick mirror.
"""
import argparse, json, math, os, statistics, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = f"{HERE}/results_v2"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import regime_a_model as model  # noqa: E402

TILE = 32


def load_manifest():
    return {(_s["M"], _s["K"], _s["N"]): _s for _s in json.load(open(f"{HERE}/corpus_v2_manifest.json"))["shapes"]}


def load_shape(M, K, N):
    """cfg(tuple Ns,Pk,Sm,kb,nsb) -> {'wall':median_us,'pcc':min,'reasons':[...],'n_samples':k,
    'spread_pct':max relaunch spread,'gen':{...}}. Merges initial + rerun samples."""
    p = f"{RESULTS}/v2_{M}x{K}x{N}.jsonl"
    if not os.path.exists(p):
        return {}
    samples = defaultdict(list)
    relaunch_meds = defaultdict(list)
    pccs = defaultdict(list)
    gen = {}
    reasons = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        c = tuple(r["cfg"])
        if r["outcome"] == "ok" and r.get("samples"):
            samples[c].extend(r["samples"])
            relaunch_meds[c].append(statistics.median(r["samples"]))
        if r.get("pcc") is not None:
            pccs[c].append(r["pcc"])
        if r.get("gen"):
            gen[c] = r["gen"]
            reasons[c] = r["gen"].get("reasons", [])
    out = {}
    for c, s in samples.items():
        if not s:
            continue
        meds = relaunch_meds[c]
        spread = (max(meds) - min(meds)) / min(meds) * 100 if len(meds) > 1 and min(meds) > 0 else 0.0
        out[c] = {"wall": statistics.median(s), "pcc": (min(pccs[c]) if pccs[c] else None),
                  "reasons": reasons.get(c, []), "n_samples": len(s), "n_relaunch": len(meds),
                  "spread_pct": round(spread, 2), "gen": gen.get(c, {})}
    return out


def prod_cfg(shape_row, data):
    """The production pick as a candidate tuple: the one tagged 'explicit', else the offline mirror."""
    for c, d in data.items():
        if "explicit" in d["reasons"]:
            return c
    if shape_row.get("prod_pick"):
        return tuple(shape_row["prod_pick"])
    Mt, Kt, Nt = shape_row["Mt"], shape_row["Kt"], shape_row["Nt"]
    pk = model.production_pick(Mt, Kt, Nt)
    return (pk[1], pk[0], pk[2], pk[3], pk[4])


def geo(x):
    m = math.exp(statistics.mean(math.log(1 + v) for v in x)) - 1 if x else 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    manifest = load_manifest()

    rows = []
    for (M, K, N), srow in manifest.items():
        data = load_shape(M, K, N)
        oks = {c: d for c, d in data.items() if d["wall"]}
        if not oks:
            continue
        opt_cfg = min(oks, key=lambda c: oks[c]["wall"])
        opt = oks[opt_cfg]["wall"]
        pc = prod_cfg(srow, oks)
        prod_w = oks[pc]["wall"] if pc in oks else None
        prod_gap = (prod_w / opt - 1.0) if prod_w else None
        # pruning audit
        audit = {c: d for c, d in oks.items() if d["reasons"] == ["audit:pruned"]}
        struct = {c: d for c, d in oks.items() if d["reasons"] != ["audit:pruned"]}
        best_struct = min(struct.values(), key=lambda d: d["wall"])["wall"] if struct else None
        best_audit = min(audit.values(), key=lambda d: d["wall"])["wall"] if audit else None
        audit_beats = (best_audit is not None and best_struct is not None and best_audit < best_struct)
        audit_gain = ((best_struct - best_audit) / best_struct * 100) if audit_beats else 0.0
        rows.append({
            "shape": f"{M}x{K}x{N}", "split": srow["split"], "Mt": srow["Mt"],
            "tags": srow["tags"], "fluxltx": "fluxltx" in srow["tags"],
            "n_ok": len(oks), "opt_cfg": opt_cfg, "opt_us": round(opt, 3),
            "prod_cfg": pc, "prod_us": (round(prod_w, 3) if prod_w else None),
            "prod_gap": prod_gap, "n_audit": len(audit), "audit_beats_struct": audit_beats,
            "audit_gain_pct": round(audit_gain, 2), "opt_is_audit": (opt_cfg in audit),
            "winner_spread_pct": oks[opt_cfg]["spread_pct"], "winner_pcc": oks[opt_cfg]["pcc"],
        })

    def agg(sel):
        g = [r["prod_gap"] for r in sel if r["prod_gap"] is not None]
        if not g:
            return None
        return dict(n=len(g), geomean=geo(g) * 100, median=statistics.median(g) * 100,
                    worst=max(g) * 100, gt3=sum(1 for x in g if x > 0.03), gt5=sum(1 for x in g if x > 0.05))

    print(f"analyzed {len(rows)} shapes\n")
    print("===== PRODUCTION picker gap vs measured optimum (generator candidate set) =====")
    subs = [("ALL", lambda r: True), ("train", lambda r: r["split"] == "train"),
            ("val", lambda r: r["split"] == "val"), ("holdout", lambda r: r["split"] == "holdout"),
            ("fluxltx", lambda r: r["fluxltx"])]
    md_gap = []
    for name, pred in subs:
        a = agg([r for r in rows if pred(r)])
        if a:
            print(f"  {name:8s} n={a['n']:3d}  geomean {a['geomean']:5.1f}%  median {a['median']:5.1f}%  "
                  f"worst {a['worst']:5.1f}%  >3%={a['gt3']:2d} >5%={a['gt5']:2d}")
            md_gap.append(f"| {name} | {a['n']} | {a['geomean']:.1f}% | {a['median']:.1f}% | {a['worst']:.1f}% | {a['gt3']} | {a['gt5']} |")
    worst = sorted((r for r in rows if r["prod_gap"] is not None), key=lambda r: -r["prod_gap"])[:12]
    print("\n  largest production gaps (opportunity for a picker change):")
    for r in worst:
        print(f"    {r['shape']:16s} [{r['split']:7s}] Mt{r['Mt']} gap {r['prod_gap']*100:5.1f}%  "
              f"prod{list(r['prod_cfg'])} -> opt{list(r['opt_cfg'])}")

    print("\n===== PRUNING AUDIT (did the 8 pruned/audit candidates beat the model-selected set?) =====")
    beats = [r for r in rows if r["audit_beats_struct"]]
    opt_audit = [r for r in rows if r["opt_is_audit"]]
    print(f"  shapes where an audit(pruned) cfg beat best structured: {len(beats)}/{len(rows)}")
    print(f"  shapes where the overall optimum IS an audit cfg:        {len(opt_audit)}/{len(rows)}")
    if beats:
        print("  audit-beats-structured detail:")
        for r in sorted(beats, key=lambda z: -z["audit_gain_pct"]):
            print(f"    {r['shape']:16s} [{r['split']:7s}] audit faster by {r['audit_gain_pct']:.2f}%")
    # stability sanity
    spreads = [r["winner_spread_pct"] for r in rows if r["winner_spread_pct"]]
    if spreads:
        print(f"\n  winner rerun spread: median {statistics.median(spreads):.2f}%  "
              f"max {max(spreads):.2f}%  (n with >=2 relaunches: {len(spreads)})")
    pccs = [r["winner_pcc"] for r in rows if r["winner_pcc"] is not None]
    if pccs:
        print(f"  winner PCC: min {min(pccs):.5f}  ({sum(1 for p in pccs if p>=0.99)}/{len(pccs)} >= 0.99)")

    if args.md:
        with open(args.md, "w") as f:
            f.write("# Generator-driven campaign: optimum gaps + pruning audit\n\n")
            f.write(f"{len(rows)} shapes measured (results_v2). Per-config wall = median over ok samples "
                    "(initial + rerun).\n\n## Production picker gap vs measured optimum\n\n")
            f.write("| subset | n | geomean | median | worst | >3% | >5% |\n|---|---|---|---|---|---|---|\n")
            f.write("\n".join(md_gap) + "\n\n")
            f.write("### Largest production gaps\n\n| shape | split | Mt | gap | prod cfg | opt cfg |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in worst:
                f.write(f"| {r['shape']} | {r['split']} | {r['Mt']} | {r['prod_gap']*100:.1f}% | "
                        f"{list(r['prod_cfg'])} | {list(r['opt_cfg'])} |\n")
            f.write(f"\n## Pruning audit\n\naudit-beats-structured: **{len(beats)}/{len(rows)}**; "
                    f"optimum-is-audit: **{len(opt_audit)}/{len(rows)}**.\n")
            if beats:
                f.write("\n| shape | split | audit faster by |\n|---|---|---|\n")
                for r in sorted(beats, key=lambda z: -z["audit_gain_pct"]):
                    f.write(f"| {r['shape']} | {r['split']} | {r['audit_gain_pct']:.2f}% |\n")
            if spreads:
                f.write(f"\nWinner rerun spread: median {statistics.median(spreads):.2f}%, max {max(spreads):.2f}%. "
                        f"Winner PCC min {min(pccs):.5f}.\n")
        print(f"\nwrote {args.md}")
    # dump machine-readable summary
    json.dump({"rows": rows}, open(f"{HERE}/analysis_v2_summary.json", "w"), indent=2, default=list)


if __name__ == "__main__":
    main()

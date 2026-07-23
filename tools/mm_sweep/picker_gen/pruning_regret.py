#!/usr/bin/env python3
"""Pruning-regret evaluation of the theory-guided generator against the aborted campaign's EXHAUSTIVE
shapes (the completed-shape calibration data).

For each shape that was swept exhaustively (every feasible (Pk,Ns,Sm,kb) x nsb-lattice), we:
  - take the measured optimum over ALL measured ok configs,
  - run select_candidates(M,K,N, budget, audit, include=[production_pick]),
  - restrict to the OVERLAP (generator selection intersect measured configs),
  - report regret = best_overlap / best_all - 1, whether the optimum was retained, and what fraction of
    within-2%-of-optimum configs the generator kept.

The generator's cfg order is (Ns,Pk,Sm,kb,nsb); we build measured tuples from the JSONL's NAMED fields
so there is no tuple-order ambiguity.
"""
import argparse, json, os, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = f"{HERE}/results"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # tools/mm_sweep for the generator
import regime_a_model as model  # noqa: E402
import regime_a_candidate_generator as cg  # noqa: E402

TILE = 32


def load_measured(M, K, N):
    """cfg (Ns,Pk,Sm,kb,nsb) -> median wall_us over ok records. Baseline 'None' excluded."""
    p = f"{RESULTS}/sweep_{M}x{K}x{N}.jsonl"
    if not os.path.exists(p):
        return {}
    walls = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("cfgkey") == "None" or r.get("outcome") != "ok" or not r.get("wall_us"):
            continue
        cfg = (r["Ns"], r["Pk"], r["Sm"], r["kb"], r["nsb"])  # generator order, from NAMED fields
        walls.setdefault(cfg, []).append(r["wall_us"])
    return {c: statistics.median(v) for c, v in walls.items()}


def exhaustive_shapes():
    """Shapes whose JSONL is complete vs the lattice target (from the aborted campaign)."""
    import corpus as corpus_mod
    out = []
    for sp, lst in corpus_mod.SWEPT.items():
        for (M, K, N) in lst:
            Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
            tgt = len(model.enumerate_feasible(Mt, Kt, Nt, nsb_mode="lattice")) + 1
            p = f"{RESULTS}/sweep_{M}x{K}x{N}.jsonl"
            n = sum(1 for _ in open(p)) if os.path.exists(p) else 0
            if n >= tgt:
                out.append((M, K, N, sp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=96)
    ap.add_argument("--audit", type=int, default=8)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    rows = []
    for (M, K, N, sp) in exhaustive_shapes():
        Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
        measured = load_measured(M, K, N)
        if not measured:
            continue
        pk = model.production_pick(Mt, Kt, Nt)  # (Pk,Ns,Sm,kb,nsb,source)
        prod_cfg = (pk[1], pk[0], pk[2], pk[3], pk[4])  # -> (Ns,Pk,Sm,kb,nsb)
        sel, _reasons, stats = cg.select_candidates(M, K, N, budget=args.budget, audit=args.audit,
                                                    include=[prod_cfg])
        sel_cfgs = {g.cfg for g in sel}
        overlap = sel_cfgs & measured.keys()
        best_all_cfg = min(measured, key=measured.get)
        best_all = measured[best_all_cfg]
        # within-2% set (measured)
        within2 = {c for c, w in measured.items() if w <= best_all * 1.02}
        within2_kept = within2 & sel_cfgs
        opt_retained = best_all_cfg in sel_cfgs
        if overlap:
            best_ov_cfg = min(overlap, key=measured.get)
            regret = measured[best_ov_cfg] / best_all - 1.0
        else:
            best_ov_cfg, regret = None, None
        rows.append({
            "shape": f"{M}x{K}x{N}", "split": sp, "Mt": Mt, "measured": len(measured),
            "selected": len(sel_cfgs), "overlap": len(overlap),
            "opt_retained": opt_retained, "opt_cfg": best_all_cfg,
            "regret_pct": (regret * 100 if regret is not None else None),
            "within2_total": len(within2), "within2_kept": len(within2_kept),
        })

    print(f"pruning-regret over {len(rows)} exhaustive shapes (budget={args.budget} audit={args.audit})\n")
    hdr = f"{'shape':16s} {'split':7s} Mt {'meas':>5s} {'sel':>4s} {'ovlp':>4s} {'opt?':>4s} {'regret':>7s} {'w2kept':>8s}"
    print(hdr)
    for r in sorted(rows, key=lambda z: (z["regret_pct"] is None, -(z["regret_pct"] or 0))):
        reg = f"{r['regret_pct']:+.2f}%" if r["regret_pct"] is not None else "n/a"
        print(f"{r['shape']:16s} {r['split']:7s} {r['Mt']:2d} {r['measured']:5d} {r['selected']:4d} "
              f"{r['overlap']:4d} {'Y' if r['opt_retained'] else 'N':>4s} {reg:>7s} "
              f"{str(r['within2_kept'])+'/'+str(r['within2_total']):>8s}")
    regs = [r["regret_pct"] for r in rows if r["regret_pct"] is not None]
    n_opt = sum(1 for r in rows if r["opt_retained"])
    n_w2_full = sum(1 for r in rows if r["within2_kept"] == r["within2_total"])
    print(f"\noptimum retained: {n_opt}/{len(rows)}   all-within-2% retained: {n_w2_full}/{len(rows)}")
    if regs:
        print(f"regret: median {statistics.median(regs):+.2f}%  worst {max(regs):+.2f}%  "
              f"within1%={sum(1 for r in regs if r<=1.0)}/{len(regs)}  within2%={sum(1 for r in regs if r<=2.0)}/{len(regs)}")

    if args.md:
        with open(args.md, "w") as f:
            f.write("# Pruning-regret vs aborted exhaustive shapes\n\n")
            f.write(f"budget={args.budget} audit={args.audit}; {len(rows)} exhaustive shapes.\n\n")
            f.write("| shape | split | Mt | measured | selected | overlap | opt retained | regret | within-2% kept |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in sorted(rows, key=lambda z: (z["regret_pct"] is None, -(z["regret_pct"] or 0))):
                reg = f"{r['regret_pct']:+.2f}%" if r["regret_pct"] is not None else "n/a"
                f.write(f"| {r['shape']} | {r['split']} | {r['Mt']} | {r['measured']} | {r['selected']} | "
                        f"{r['overlap']} | {'Y' if r['opt_retained'] else 'N'} | {reg} | "
                        f"{r['within2_kept']}/{r['within2_total']} |\n")
            f.write(f"\n**optimum retained {n_opt}/{len(rows)}; all-within-2% retained {n_w2_full}/{len(rows)}**")
            if regs:
                f.write(f"; median regret {statistics.median(regs):+.2f}%, worst {max(regs):+.2f}%.\n")
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()

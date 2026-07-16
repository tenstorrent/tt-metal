#!/usr/bin/env python3
# Picker-sensitive corpus config sweep under the CURRENT production default (pareto ring order + pipelined
# drain). Checks whether the best (Ns,Pk,Sm,kb,nsb) per shape has shifted from the picker's current choice.
#
# The full planner-feasible space is ~13.7k configs across these shapes — intractable to device-run
# exhaustively. This is a PRINCIPLED BOUNDED search (not a top-10 re-rank): per shape the candidate set is
#   {prior-sweep configs with old us <= 1.5x that shape's best old us, capped at 25 by old us}   (broad seed)
#   ∪ {the current picker config}
#   ∪ {±1-step neighbours of the picker config over Ns/Pk/Sm/kb/nsb, planner-feasible}            (local search)
# so it can discover configs beyond the prior top-10 (ranks 11-25) AND configs the prior sweep never measured
# (the neighbourhood). Each candidate is run once (median of RA_ITERS timed iters) under the pareto default;
# any candidate that beats the picker config is re-run with 3 relaunches to confirm it clears noise.
import json, os, statistics, sys

sys.path.insert(0, os.path.dirname(__file__))
import regime_a_diag_suite as ds
import regime_a_bench as rb

HERE = os.path.dirname(__file__)
PRIOR = json.load(open(f"{HERE}/fluxltx_regimeA_sweep.json"))

# (label, M, K, N). 2 Mt=8 primaries + Mt>=4 FLUX/LTX + Pk>1/Sm>1 six-shape parity cases.
SHAPES = [
    ("256x2048x1024", 256, 2048, 1024),
    ("256x6144x768", 256, 6144, 768),
    ("128x6144x768", 128, 6144, 768),
    ("128x15360x768", 128, 15360, 768),
    ("128x6144x2304", 128, 6144, 2304),
    ("128x6144x4608", 128, 6144, 4608),
    ("128x2304x6144", 128, 2304, 6144),
    ("512x6144x1536", 512, 6144, 1536),
    ("32x6144x4608", 32, 6144, 4608),
    ("64x6144x4608", 64, 6144, 4608),
    ("256x6144x4608", 256, 6144, 4608),
    ("32x6080x4640", 32, 6080, 4640),
]
CAP = 25


def prior_cfgs(M, K, N):
    rows = [r for r in PRIOR if r["M"] == M and r["K"] == K and r["N"] == N and r.get("us")]
    if not rows:
        return []
    rows.sort(key=lambda r: r["us"])
    best = rows[0]["us"]
    keep = [r for r in rows if r["us"] <= 1.5 * best][:CAP]
    return [(r["Ns"], r["Pk"], r["Sm"], r["kb"], r["nsb"]) for r in keep]


def neighbours(cfg):
    Ns, Pk, Sm, kb, nsb = cfg
    out = set()
    for dv, base in ((0, Ns), (1, Pk), (2, Sm), (3, kb), (4, nsb)):
        for step in (-1, 1):
            c = list(cfg)
            if dv == 3:  # kb doubles/halves
                c[3] = base * 2 if step > 0 else max(1, base // 2)
            else:
                c[dv] = base + step
            if all(x >= 1 for x in c):
                out.add(tuple(c))
    return out


def main():
    out = []
    for label, M, K, N in SHAPES:
        picker = tuple(rb.auto_config(M, K, N))
        cands = set(prior_cfgs(M, K, N))
        cands.add(picker)
        cands |= neighbours(picker)
        # keep only planner-feasible
        feas = [c for c in cands if rb.planner_feasible(M, K, N, c)[0]]
        print(
            f"\n=== {label} picker={picker} candidates={len(feas)} (of {len(cands)} incl. infeasible) ===", flush=True
        )
        rows = []
        for c in sorted(feas):
            r = ds.run_one(M, K, N, c, 0, iters=8)  # mask 0 = pareto default
            med = r["wall_us"] if r.get("ok") else None
            rows.append({"cfg": list(c), "us": med})
            tag = " <== picker" if c == picker else ""
            print(f"  {c} us={med if med is None else round(med,2)}{tag}", flush=True)
        ok = [r for r in rows if r["us"] is not None]
        ok.sort(key=lambda r: r["us"])
        picker_us = next((r["us"] for r in rows if tuple(r["cfg"]) == picker), None)
        best = ok[0] if ok else None
        moved = bool(best and tuple(best["cfg"]) != picker)
        confirm = None
        if moved and picker_us and best["us"] and best["us"] < 0.98 * picker_us:
            # confirm the flip with 3 relaunches of best vs picker
            bw = statistics.median([ds.run_one(M, K, N, tuple(best["cfg"]), 0)["wall_us"] for _ in range(3)])
            pw = statistics.median([ds.run_one(M, K, N, picker, 0)["wall_us"] for _ in range(3)])
            confirm = {"best_cfg": best["cfg"], "best_us": bw, "picker_us": pw, "gain_pct": (1 - bw / pw) * 100}
            print(
                f"  CONFIRM best={tuple(best['cfg'])} {bw:.2f}us vs picker {pw:.2f}us ({confirm['gain_pct']:+.1f}%)",
                flush=True,
            )
        out.append(
            {
                "label": label,
                "M": M,
                "K": K,
                "N": N,
                "picker": list(picker),
                "picker_us": picker_us,
                "best_cfg": best["cfg"] if best else None,
                "best_us": best["us"] if best else None,
                "moved": moved,
                "confirm": confirm,
                "n_candidates": len(feas),
                "rows": rows,
            }
        )
        print(
            f"  -> best={tuple(best['cfg']) if best else None} {best and round(best['us'],2)}us vs "
            f"picker {picker_us and round(picker_us,2)}us "
            f"({'UNCHANGED' if not moved else 'single-pass move; ' + ('CONFIRMED' if confirm and confirm['gain_pct']>2 else 'within noise / unconfirmed')})",
            flush=True,
        )
        json.dump(out, open(f"{HERE}/regime_a_ring_corpus_sweep.json", "w"), indent=2)
    print("CORPUS SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()

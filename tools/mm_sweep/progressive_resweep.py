#!/usr/bin/env python3
# Follow-up re-sweep for the progressive-cumulative-in0-wait change: does the best (Pk,Ns,Sm,kb,nsb) config
# change for the two Mt=8 primary targets now that the resident-in0 startup barrier is gone? Progressive is
# the DEFAULT (mask 0) so this measures the shipping op. We take the top-K configs from the pre-change sweep
# (regime_a_sweep_MxKxN.json, ranked under the OLD full-wait) + the current picker winner, re-run each under
# progressive (mask 0, 3x relaunch, median device-profiler kernel us), and report whether the winner moved.
import json, os, statistics, sys

sys.path.insert(0, os.path.dirname(__file__))
import regime_a_diag_suite as ds

HERE = os.path.dirname(__file__)
PRIMARIES = [
    ("256x2048x1024", 256, 2048, 1024, (1, 4, 2, 2, 2)),  # current picker winner
    ("256x6144x768", 256, 6144, 768, (1, 12, 1, 2, 1)),
]
TOPK = 10


def main():
    out = []
    for label, M, K, N, winner in PRIMARIES:
        d = ds._load_sweep(M, K, N)
        cands = []
        if d is not None:
            res = [r for r in d["results"] if r.get("cls") == "ok"]
            res.sort(key=lambda r: r["us_med"])
            for r in res[:TOPK]:
                cands.append(tuple(r["cfg"]))
        if tuple(winner) not in cands:
            cands.insert(0, tuple(winner))
        print(f"\n=== {label} re-sweep ({len(cands)} configs) winner={winner}", flush=True)
        rows = []
        for cfg in cands:
            runs = [ds.run_one(M, K, N, cfg, 0) for _ in range(3)]  # mask 0 = progressive (default)
            oks = [x for x in runs if x.get("ok") and x["wall_us"]]
            walls = sorted(x["wall_us"] for x in oks)
            med = statistics.median(walls) if walls else None
            rows.append({"cfg": list(cfg), "med_us": med, "walls": walls, "n_ok": len(oks)})
            print(
                f"  cfg={cfg} prog_med={med if med is None else round(med,2)}us "
                f"all={[round(w,2) for w in walls]} ({len(oks)}/3)",
                flush=True,
            )
        ok_rows = [r for r in rows if r["med_us"] is not None]
        ok_rows.sort(key=lambda r: r["med_us"])
        best = ok_rows[0] if ok_rows else None
        w_med = next((r["med_us"] for r in rows if tuple(r["cfg"]) == tuple(winner)), None)
        out.append(
            {
                "label": label,
                "M": M,
                "K": K,
                "N": N,
                "winner_cfg": list(winner),
                "winner_prog_med_us": w_med,
                "best_cfg": best["cfg"] if best else None,
                "best_prog_med_us": best["med_us"] if best else None,
                "winner_moved": bool(best and tuple(best["cfg"]) != tuple(winner)),
                "rows": rows,
            }
        )
        if best:
            gain = (1 - best["med_us"] / w_med) * 100 if w_med else None
            print(
                f"  -> best={tuple(best['cfg'])} {round(best['med_us'],2)}us vs winner {w_med and round(w_med,2)}us "
                f"({'winner unchanged' if not out[-1]['winner_moved'] else f'MOVED, {gain:.1f}% faster'})",
                flush=True,
            )
        json.dump(out, open(f"{HERE}/regime_a_progressive_resweep.json", "w"), indent=2)
    print("RESWEEP DONE", flush=True)


if __name__ == "__main__":
    main()

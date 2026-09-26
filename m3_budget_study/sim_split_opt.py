#!/usr/bin/env python3
"""Hill-climb the layer split for one layout: from the best of candidate_splits(), repeatedly move one layer
between neighbouring stages while simulated useful tok/s improves (policy fcfs at one W, same traffic).

  sim_split_opt.py --coeffs C --stages S --pipelines P --hop-ms H [--width 4096] [--n 4000]
"""
import argparse, importlib.util, json

spec = importlib.util.spec_from_file_location("sim", "m3_budget_sim.py")
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--stages", type=int, required=True)
    ap.add_argument("--pipelines", type=int, default=1)
    ap.add_argument("--hop-ms", type=float, required=True)
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    C = dict(json.load(open(a.coeffs)), hop_ms=a.hop_ms)
    reqs = sim.sample_requests(a.n, 0, 1_048_576, align_recompute=True)
    reqs = [r for r in reqs if not (r[2] <= 1000 and r[1] <= 65_536)]
    useful = sum(r[2] for r in reqs)
    cache = {}

    def score(sp):
        key = tuple(sp)
        if key not in cache:
            M = sim.CostModel(C, list(sp), embed_stage0_only=True)
            mk, util, fms, _ = sim.run_policy("fcfs", reqs, a.width, M, P=a.pipelines)
            cache[key] = useful / mk * 1000
        return cache[key]

    best = max(sim.candidate_splits(a.stages), key=score)
    print(f"start {best} {score(best):.0f} tok/s")
    improved = True
    while improved:
        improved = False
        for i in range(a.stages - 1):
            for d in (1, -1):
                sp = list(best)
                sp[i] -= d
                sp[i + 1] += d
                if min(sp) < 1:
                    continue
                if score(sp) > score(best) * 1.001:
                    best, improved = sp, True
                    print(f"  -> {best} {score(best):.0f} tok/s")
    print(f"BEST {','.join(map(str, best))} {score(best):.0f}")


if __name__ == "__main__":
    main()

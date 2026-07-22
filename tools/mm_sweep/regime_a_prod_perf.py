#!/usr/bin/env python3
# Production perf re-measurement of the CLEANED-UP op vs the collected pre-cleanup baseline.
# Measures the PUBLIC ttnn op (ttnn.experimental.regime_a_matmul, config=None -> auto-picker -> linear-chain
# reduction) kernel wall across the 60-shape Mt<=8 corpus, one shape per subprocess (regime_a_prod_perf_worker),
# and compares to regime_a_current_perf.json (the collected baseline, which is the CHAIN configuration).
# Expectation: the diag/reduction-experiment removal + hardcoding + factory restructure is perf-NEUTRAL on the
# chain, so every shape should match the baseline within run-to-run noise. (The 5 shapes that briefly ran
# reduce-scatter as the production default before this cleanup are now back on the chain = back to this
# baseline; that intentional -5..-9% reduce-scatter removal is documented in LTX_FLUX_OPT_LOG.md.)
import json, os, subprocess, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("TT_METAL_HOME", os.path.abspath(f"{HERE}/../.."))
WORKER = f"{HERE}/regime_a_prod_perf_worker.py"
ITERS = 8
# The 5 shapes that had reduce-scatter as the pre-cleanup production default (now reverted to chain).
RS_SHAPES = {(64, 2048, 1024), (128, 2048, 1024), (128, 2048, 2048), (256, 2048, 1024), (256, 2048, 2048)}


def measure(M, K, N):
    env = dict(os.environ)
    env.update(TT_METAL_DEVICE_PROFILER="1", TT_METAL_HOME=ROOT, ARCH_NAME="blackhole")
    try:
        r = subprocess.run(
            [sys.executable, WORKER, str(M), str(K), str(N), str(ITERS)],
            env=env,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "regime_a_prod_perf_worker"], capture_output=True)
        return None
    for line in r.stdout.splitlines():
        if line.startswith("WALL_US="):
            v = line.split("=", 1)[1].strip()
            return float(v) if v not in ("None", "") else None
    return None


def main():
    base = json.load(open(f"{HERE}/regime_a_current_perf.json"))["mt8"]
    out = {"shapes": []}
    print(f"{'shape':16s} {'base_us':>9s} {'cur_us':>9s} {'delta%':>8s}  note", flush=True)
    for r in sorted(base, key=lambda z: (z["K"], z["M"], z["N"])):
        M, K, N, bmed = r["M"], r["K"], r["N"], r["us_med"]
        cur = measure(M, K, N)
        delta = ((cur - bmed) / bmed * 100.0) if (cur and bmed) else None
        rs = (M, K, N) in RS_SHAPES
        note = "was-reduce-scatter(now chain)" if rs else ""
        rec = {"M": M, "K": K, "N": N, "base_us": bmed, "cur_us": cur, "delta_pct": delta, "rs_shape": rs}
        out["shapes"].append(rec)
        ds = f"{delta:+.2f}" if delta is not None else "n/a"
        print(
            f"{f'{M}x{K}x{N}':16s} {bmed:>9.2f} {str(round(cur,2) if cur else None):>9s} {ds:>8s}  {note}", flush=True
        )
        json.dump(out, open(f"{HERE}/regime_a_prod_perf_results.json", "w"), indent=2)
    ok = [s for s in out["shapes"] if s["delta_pct"] is not None]
    deltas = [s["delta_pct"] for s in ok]
    print("\n===== SUMMARY: cleaned-up production (chain) vs pre-cleanup baseline (chain) =====", flush=True)
    print(f"measured {len(ok)}/{len(out['shapes'])}", flush=True)
    if deltas:
        worst = max(deltas, key=abs)
        within2 = sum(1 for d in deltas if abs(d) <= 2.0)
        within3 = sum(1 for d in deltas if abs(d) <= 3.0)
        print(
            f"mean delta {statistics.mean(deltas):+.2f}%  median {statistics.median(deltas):+.2f}%  "
            f"worst |delta| {worst:+.2f}%  within2%={within2}/{len(deltas)}  within3%={within3}/{len(deltas)}",
            flush=True,
        )
        big = sorted((s for s in ok if abs(s["delta_pct"]) > 3.0), key=lambda z: z["delta_pct"])
        if big:
            print("shapes >3% off baseline:", flush=True)
            for s in big:
                print(
                    f"  {s['M']}x{s['K']}x{s['N']}  base={s['base_us']:.2f} cur={s['cur_us']:.2f} "
                    f"delta={s['delta_pct']:+.2f}%  {'(was reduce-scatter)' if s['rs_shape'] else ''}",
                    flush=True,
                )
    print(f"wrote {HERE}/regime_a_prod_perf_results.json", flush=True)


if __name__ == "__main__":
    main()

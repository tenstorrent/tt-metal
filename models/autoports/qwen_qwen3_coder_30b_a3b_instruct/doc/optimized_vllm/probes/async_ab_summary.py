# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The async-scheduling A/B, re-measured with per-token ITLs retained.

Why this exists
---------------

Stage 09 first published the async-scheduling win as **1.754 ms/token (8.9 %)**,
taken as the difference of the two legs' ``mean_tpot_ms``. The stage-09 review
showed that number is outlier-driven.

The evidence was already in the artifact, in a form that had to be *solved for*
rather than read. ``bench/single_user_no_async_vllm_result.json`` reports
``mean_itl 21.5553``, ``median_itl 20.2449``, ``p99_itl 21.0446`` and
``std_itl 14.1009``. **``p99 < mean`` is impossible without a large excess above
the 99th percentile.** Solving the moments for 127 ITLs gives 126 tokens at
~20.30 ms plus a single token at **~180 ms** -- which reproduces the published
``std`` to two decimals. The async leg's ``std`` over the same workload is
0.4025, so the leg that decided the conclusion was carrying ~35x the dispersion
of the leg it was compared against, and that was never classified.

That single ~180 ms stall inflates the mean of a 127-token run by
``(180 - 20.3) / 127`` = **1.26 ms/token**, which is 72 % of the published
1.754 ms "win".

The original run's summary JSON kept only moments, which is why this had to be
inferred. So the re-measurement passes ``--save-detailed`` through to
``vllm bench serve`` and **retains every per-token ITL**, repeats each leg, and
adds a longer-output run so no single stall can carry a mean.

What it reports
---------------

Per repeat: n, mean, median, trimmed mean, std, p99, min, max, and an explicit
**stall list** (ITLs above ``--stall-factor`` x the run's own median). Per leg:
the pooled steady-state median across repeats. Then the three framings of the
same A/B -- mean TPOT (what stage 09 published), median ITL (robust to stalls),
and end-to-end request latency (immune to where a stall lands, because a stall
inside the decode phase is already inside e2e).

Writes ``bench/async_ab_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
BENCH = STAGE / "bench"


def describe(path: Path, stall_factor: float) -> dict:
    raw = json.loads(path.read_text())
    itls = [v * 1000.0 for v in raw["itls"][0]]
    med = st.median(itls)
    ordered = sorted(itls)
    stalls = [
        {"index": i, "ms": round(v, 3), "x_median": round(v / med, 2)}
        for i, v in enumerate(itls)
        if v > stall_factor * med
    ]
    # Mean with the stalls removed -- what the mean would have said without them.
    kept = [v for v in itls if v <= stall_factor * med]
    return {
        "artifact": str(path.relative_to(STAGE)),
        "n_itls": len(itls),
        "mean_ms": st.mean(itls),
        "median_ms": med,
        "mean_excluding_stalls_ms": st.mean(kept),
        "std_ms": st.pstdev(itls),
        "std_excluding_stalls_ms": st.pstdev(kept) if len(kept) > 1 else 0.0,
        "min_ms": min(itls),
        "max_ms": max(itls),
        "p99_ms": ordered[max(0, int(0.99 * len(ordered)) - 1)],
        "stall_count": len(stalls),
        "stalls": stalls,
        "ttft_ms": raw["mean_ttft_ms"],
        "e2e_ms": raw["mean_e2el_ms"],
        "tpot_mean_ms": raw["mean_tpot_ms"],
        "output_tokens": raw["total_output_tokens"],
    }


def leg_summary(runs: list[dict]) -> dict:
    return {
        "repeats": len(runs),
        "pooled_median_itl_ms": st.median([r["median_ms"] for r in runs]),
        "mean_of_mean_itl_ms": st.mean([r["mean_ms"] for r in runs]),
        "spread_of_median_itl_ms": (max(r["median_ms"] for r in runs) - min(r["median_ms"] for r in runs)),
        "max_std_ms": max(r["std_ms"] for r in runs),
        "total_stalls": sum(r["stall_count"] for r in runs),
        "median_ttft_ms": st.median([r["ttft_ms"] for r in runs]),
        "median_e2e_ms": st.median([r["e2e_ms"] for r in runs]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stall-factor", type=float, default=2.0, help="ITL above this multiple of the run median is a stall"
    )
    args = ap.parse_args()

    root = BENCH / "async_ab"
    legs = {}
    for leg in ("async_on", "async_off"):
        runs_128 = [describe(p, args.stall_factor) for p in sorted(root.glob(f"{leg}_128_r*_result.json"))]
        runs_512 = [describe(p, args.stall_factor) for p in sorted(root.glob(f"{leg}_512_r*_result.json"))]
        if not runs_128:
            raise SystemExit(f"no 128-token repeats found for {leg} under {root}")
        legs[leg] = {
            "runs_128": runs_128,
            "runs_512": runs_512,
            "summary_128": leg_summary(runs_128),
            "summary_512": leg_summary(runs_512) if runs_512 else None,
        }

    on, off = legs["async_on"], legs["async_off"]
    on128, off128 = on["summary_128"], off["summary_128"]

    def delta(a, b):
        return {"async_off": b, "async_on": a, "gain": b - a, "gain_pct": 100.0 * (b - a) / b}

    comparison = {
        "median_itl_128": delta(on128["pooled_median_itl_ms"], off128["pooled_median_itl_ms"]),
        "mean_itl_128": delta(on128["mean_of_mean_itl_ms"], off128["mean_of_mean_itl_ms"]),
        "e2e_128": delta(on128["median_e2e_ms"], off128["median_e2e_ms"]),
        "ttft_128": delta(on128["median_ttft_ms"], off128["median_ttft_ms"]),
    }
    if on["summary_512"] and off["summary_512"]:
        comparison["median_itl_512"] = delta(
            on["summary_512"]["pooled_median_itl_ms"], off["summary_512"]["pooled_median_itl_ms"]
        )
        comparison["e2e_512"] = delta(on["summary_512"]["median_e2e_ms"], off["summary_512"]["median_e2e_ms"])

    # The original stage-09 legs, kept as evidence and re-described from their
    # moments (they have no retained per-token data -- that is the defect).
    original = {}
    for name, fn in (
        ("async_on", "single_user_after_vllm_result.json"),
        ("async_off", "single_user_no_async_vllm_result.json"),
    ):
        raw = json.loads((BENCH / fn).read_text())
        original[name] = {
            "artifact": f"bench/{fn}",
            "mean_itl_ms": raw["mean_itl_ms"],
            "median_itl_ms": raw["median_itl_ms"],
            "p99_itl_ms": raw["p99_itl_ms"],
            "std_itl_ms": raw["std_itl_ms"],
            "ttft_ms": raw["mean_ttft_ms"],
            "e2e_ms": raw["mean_e2el_ms"],
            "per_token_itls_retained": "itls" in raw,
        }
    off_o = original["async_off"]
    n = 127
    # Solve the two moments for "n-1 tokens at x, one at y".
    lo, hi = off_o["mean_itl_ms"], 5000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        x = (n * off_o["mean_itl_ms"] - mid) / (n - 1)
        vals = [x] * (n - 1) + [mid]
        (lo, hi) = (mid, hi) if st.pstdev(vals) < off_o["std_itl_ms"] else (lo, mid)
    y = (lo + hi) / 2
    x = (n * off_o["mean_itl_ms"] - y) / (n - 1)
    original["async_off_outlier_inference"] = {
        "why": "p99 < mean is only possible with a large excess above the 99th percentile",
        "p99_below_mean_by_ms": off_o["mean_itl_ms"] - off_o["p99_itl_ms"],
        "model": f"{n - 1} ITLs at ~{x:.2f} ms plus one at ~{y:.1f} ms",
        "implied_steady_state_ms": x,
        "implied_stall_ms": y,
        "reproduces_published_std": off_o["std_itl_ms"],
        "measured_median_ms": off_o["median_itl_ms"],
        "mean_inflation_ms_per_token": (y - x) / n,
        "share_of_published_1754_gain": (y - x) / n / 1.7543572756140051,
        "dispersion_ratio_vs_async_on": off_o["std_itl_ms"] / original["async_on"]["std_itl_ms"],
    }

    out = {
        "what": "async-scheduling A/B re-measured with per-token ITLs retained (--save-detailed), each leg repeated",
        "workload": "128-token input, max_num_seqs=1, ignore_eos, greedy, --max-concurrency 1; 128- and 512-token outputs",
        "stall_factor": args.stall_factor,
        "legs": legs,
        "comparison": comparison,
        "original_stage09_legs": original,
    }
    dest = BENCH / "async_ab_summary.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"{'leg':<10} {'run':<10} {'n':>5} {'mean':>9} {'median':>9} {'std':>8} {'p99':>8} {'stalls':>7}")
    for leg in ("async_on", "async_off"):
        for key in ("runs_128", "runs_512"):
            for r in legs[leg][key]:
                print(
                    f"{leg:<10} {Path(r['artifact']).stem.replace(leg + '_', '')[:10]:<10} {r['n_itls']:>5} "
                    f"{r['mean_ms']:>9.4f} {r['median_ms']:>9.4f} {r['std_ms']:>8.4f} {r['p99_ms']:>8.4f} {r['stall_count']:>7}"
                )
    print()
    for name, d in comparison.items():
        print(
            f"{name:<18} off {d['async_off']:>10.4f}  on {d['async_on']:>10.4f}  gain {d['gain']:>8.4f} ({d['gain_pct']:.2f} %)"
        )
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()

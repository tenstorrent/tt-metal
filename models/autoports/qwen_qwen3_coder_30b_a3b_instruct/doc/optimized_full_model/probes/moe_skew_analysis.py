# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Is the MoE reduce-scatter's 27.88 us mean fixable, or is it arithmetic?

Stage 06 part 1 measured the two per-layer collectives and found them unalike:
the attention-side ``ReduceScatterMinimalAsync`` averages 12.54 us with no tail,
the MoE-side one averages 27.88 us over a 6.51-74.19 us range at the *same*
128 KB. Part 1 attributed the difference to expert-routing skew. This script
tests that attribution and then asks whether any layout change can remove it.

It is **pure analysis of the archived 48-layer profile**
(``ops_perf_full_model_48layer_decode.csv.gz``, one complete real decode
iteration, 4 devices, boundary-verified on ten exact op tallies). No device is
opened and no model is run, so it costs nothing to re-run and it cannot drift
from the numbers it explains.

Three things it establishes.

1. **The per-die active-expert count is recoverable from the profile.** The
   gate/up ``SparseMatmul`` duration quantises exactly: ``t = 29.4 us + 6.85 us
   * k``, where ``k`` is the number of this die's 32 local experts that the
   global top-8 selected. The 29.4 us floor is the ``nnz=None`` dynamic-sparsity
   scan the EP design is forced into (32 slots x 0.79 us; see
   ``multichip_decoder.py``'s "``nnz`` contract" section) and it is paid whether
   or not any expert fires. The recovery is **self-validating**: rounding
   ``(t - 29.4)/6.85`` and summing over the four dies gives **exactly 8 in all
   48 layers**, which is the top-8 the router selected. A wrong step size would
   not do that.

2. **The MoE reduce-scatter is a pure wait.** Regressing each die's MoE-side
   reduce-scatter duration on how far that die finished behind the slowest die
   in the same layer gives a correlation of **0.989** and a slope of
   **1.05 us per us**. The attention-side reduce-scatter regressed against the
   same lag gives 0.092. The collective is not slow; it is where the dies that
   finished early stand and wait.

3. **The skew is combinatorial, and the permutation lever is zero by
   construction if routing is exchangeable over experts.** Under EP=4 the top-8
   lands in 4 windows of 32 experts. If the router selected uniformly, the
   per-die counts would be ``multinomial(8, 1/4)``, whose expected maximum is
   3.538 against a mean of 2 -- an unavoidable 1.538 experts of imbalance per
   layer, i.e. ~10.5 us, because a collective waits for the maximum and not the
   mean. **The argument that matters is exchangeability, not the measured
   mean.** A permutation of experts across dies is a relabelling; if no expert
   is systematically hotter than another, every relabelling induces the *same*
   distribution of per-die counts, so the expected saving is exactly 0 by
   construction. The measured mean maximum, 3.438 against 3.538, is *not* the
   argument -- it is less than one standard error below the expectation
   (z = -0.9, p ~ 0.4) and the script says so rather than reading it as the
   shipped layout beating an arbitrary partition.

   What exchangeability rests on is that per-expert hotness does not *persist
   across tokens*, and that is invisible in this script: the archived profile is
   **one decode token**, so the 192 die-layer counts are 48 layers of a single
   token, and within each layer the four counts sum to 8 by construction. Every
   p-value here is a failure to reject uniformity, never evidence for it. The
   chi-square is published with its degrees of freedom and its p-value, counts
   of ``k >= 5`` are **pooled** rather than dropped (the dropped bin used to be
   the most anti-uniform cell in the table), and the per-die marginal over all
   48 layers -- which this script never used to compute -- is published too.
   ``moe_routing_across_tokens_probe.py`` is the probe that samples many tokens
   and settles the persistence question empirically.

What would actually remove the skew is making each die's cost independent of how
many of its experts fired -- which under the SPMD ``nnz=None`` contract means
computing all 32 local experts every layer, 4x the expert FLOPs to save 0.5 ms.

Usage::

    python moe_skew_analysis.py [--csv ../ops_perf_full_model_48layer_decode.csv.gz]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "ops_perf_full_model_48layer_decode.csv.gz"

NUM_DEVICES = 4
NUM_LAYERS = 48
TOP_K = 8
#: us per additional locally-active expert in the gate/up ``SparseMatmul``.
#: Recovered from the profile and validated by the per-layer sum being exactly 8.
EXPERT_STEP_US = 6.85


def load(csv_path: Path):
    with gzip.open(csv_path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    sm = {d: [] for d in range(NUM_DEVICES)}
    rs = {d: [] for d in range(NUM_DEVICES)}
    ag = {d: [] for d in range(NUM_DEVICES)}
    for r in rows:
        d = int(r["DEVICE ID"])
        us = int(r["DEVICE KERNEL DURATION [ns]"]) / 1000.0
        code = r["OP CODE"]
        if code == "SparseMatmulDeviceOperation":
            sm[d].append(us)
        elif code == "ReduceScatterMinimalAsyncDeviceOperation":
            rs[d].append(us)
        elif code == "AllGatherAsyncDeviceOperation":
            ag[d].append(us)
    S = np.array([sm[d] for d in range(NUM_DEVICES)]).reshape(NUM_DEVICES, NUM_LAYERS, 2)
    R = np.array([rs[d] for d in range(NUM_DEVICES)]).reshape(NUM_DEVICES, NUM_LAYERS, 2)
    A = np.array([ag[d] for d in range(NUM_DEVICES)]).reshape(NUM_DEVICES, NUM_LAYERS, 2)
    return S, R, A


#: Counts of ``k >= POOL_FROM`` are pooled into one chi-square bin rather than
#: dropped. The previous version skipped any bin whose expectation was below 1,
#: which threw away k=6: expected 0.74, **observed 3** -- the single most
#: anti-uniform cell in the table was the one being filtered out, and dropping it
#: made the test look better than it was. Pooling is the textbook fix and keeps
#: every observation in the statistic.
POOL_FROM = 5


def chi2_sf(x: float, df: int) -> float:
    """Upper tail of the chi-square distribution, in pure Python.

    ``Q(df/2, x/2)`` by the Numerical Recipes series/continued-fraction pair, so
    this script keeps its "no dependency beyond numpy" property and the p-values
    it publishes can be re-derived by reading it.
    """
    a, y = df / 2.0, x / 2.0
    if y <= 0.0:
        return 1.0
    log_gamma_a = math.lgamma(a)
    if y < a + 1.0:  # series for the lower tail, then complement
        term = 1.0 / a
        total = term
        n = a
        for _ in range(1000):
            n += 1.0
            term *= y / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-y + a * math.log(y) - log_gamma_a)
    # continued fraction for the upper tail (modified Lentz)
    tiny = 1e-300
    b = y + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-y + a * math.log(y) - log_gamma_a) * h


def multinomial_reference(n=TOP_K, bins=NUM_DEVICES):
    """Exact distribution of ``max`` and of a marginal count for uniform routing."""
    emax = 0.0
    max_pmf = Counter()
    marginal = Counter()
    for comp in itertools.product(range(n + 1), repeat=bins):
        if sum(comp) != n:
            continue
        p = math.factorial(n) / math.prod(math.factorial(c) for c in comp) * ((1.0 / bins) ** n)
        emax += p * max(comp)
        max_pmf[max(comp)] += p
        for c in comp:
            marginal[c] += p / bins
    return emax, dict(sorted(max_pmf.items())), dict(sorted(marginal.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=HERE / "moe_skew_analysis.json")
    args = ap.parse_args()

    S, R, A = load(args.csv)
    gate_up, down = S[:, :, 0], S[:, :, 1]
    attn_rs, moe_rs = R[:, :, 0], R[:, :, 1]
    out = {}

    # -- 1. recover the per-die active-expert count -------------------------
    base = gate_up.min()
    k = np.round((gate_up - base) / EXPERT_STEP_US).astype(int)
    per_layer_sum = k.sum(axis=0)
    ok = bool((per_layer_sum == TOP_K).all())
    out["expert_count_recovery"] = {
        "base_us": float(base),
        "step_us": EXPERT_STEP_US,
        "per_layer_sum_over_dies": sorted(Counter(per_layer_sum.tolist()).items()),
        "sums_to_top_k_in_every_layer": ok,
    }
    print(f"active-expert recovery: base {base:.2f} us + {EXPERT_STEP_US} us/expert")
    print(f"  per-layer sum over 4 dies == {TOP_K} in every layer: {ok}")
    assert ok, "the step size does not reproduce the router's top-8; do not trust anything below"

    # -- 2. the MoE reduce-scatter is a wait, not a collective --------------
    per_die_moe_us = gate_up + down
    lag = per_die_moe_us.max(axis=0)[None, :] - per_die_moe_us
    corr_moe = float(np.corrcoef(lag.flatten(), moe_rs.flatten())[0, 1])
    corr_attn = float(np.corrcoef(lag.flatten(), attn_rs.flatten())[0, 1])
    slope = float(np.polyfit(lag.flatten(), moe_rs.flatten(), 1)[0])
    out["reduce_scatter_is_wait"] = {
        "attn_rs_mean_us": float(attn_rs.mean()),
        "moe_rs_mean_us": float(moe_rs.mean()),
        "moe_rs_min_us": float(moe_rs.min()),
        "moe_rs_max_us": float(moe_rs.max()),
        "corr_lag_vs_moe_rs": corr_moe,
        "corr_lag_vs_attn_rs": corr_attn,
        "slope_us_moe_rs_per_us_lag": slope,
        "all_gather_mean_us": [float(A[:, :, 0].mean()), float(A[:, :, 1].mean())],
    }
    print(
        f"attn RS {attn_rs.mean():.2f} us | MoE RS {moe_rs.mean():.2f} us "
        f"(min {moe_rs.min():.2f}, max {moe_rs.max():.2f})"
    )
    print(f"  corr(lag, MoE RS) = {corr_moe:.3f}, slope {slope:.3f} us/us")
    print(f"  corr(lag, attn RS) = {corr_attn:.3f}   <- the attention collective does not see the skew")

    # the layer where the dies happened to balance: its MoE RS should read like
    # the attention one, which is the floor the collective actually runs at.
    spread = per_die_moe_us.max(axis=0) - per_die_moe_us.min(axis=0)
    best = int(np.argmin(spread))
    out["most_balanced_layer"] = {
        "layer": best,
        "work_spread_us": float(spread[best]),
        "moe_rs_us_all_dies": [float(v) for v in moe_rs[:, best]],
        "attn_rs_mean_us": float(attn_rs.mean()),
    }
    print(
        f"  most balanced layer {best}: work spread {spread[best]:.2f} us, "
        f"MoE RS {[round(float(v),2) for v in moe_rs[:, best]]} us -- the collective's floor"
    )

    # -- 3. is the skew fixable by permuting experts across dies? -----------
    emax, max_pmf, marginal = multinomial_reference()
    measured_max = k.max(axis=0)
    n = k.size
    obs = Counter(k.flatten().tolist())

    # Chi-square of the per-die count marginal against Binomial(8, 1/4), with
    # every k >= POOL_FROM pooled into one bin. df = bins - 1: the reference
    # distribution is fully specified, no parameter is estimated, and the one
    # degree of freedom lost is the total.
    pooled_expected: dict[int, float] = {}
    pooled_observed: dict[int, int] = {}
    for c, p in marginal.items():
        key = min(c, POOL_FROM)
        pooled_expected[key] = pooled_expected.get(key, 0.0) + p * n
    for c, o in obs.items():
        key = min(c, POOL_FROM)
        pooled_observed[key] = pooled_observed.get(key, 0) + o
    chi2 = sum((pooled_observed.get(key, 0) - exp) ** 2 / exp for key, exp in pooled_expected.items())
    dof = len(pooled_expected) - 1
    p_value = chi2_sf(chi2, dof)

    # The per-die marginal over the whole iteration, which the previous version
    # never computed at all. This is the axis a permutation targets: not "does
    # the count histogram look Binomial" but "does one die fire more often than
    # the others across all 48 layers".
    per_die_total = k.sum(axis=1)
    per_die_expected = NUM_LAYERS * TOP_K / NUM_DEVICES
    per_die_chi2 = float(((per_die_total - per_die_expected) ** 2 / per_die_expected).sum())
    per_die_dof = NUM_DEVICES - 1
    per_die_p = chi2_sf(per_die_chi2, per_die_dof)

    out["skew_is_combinatorial"] = {
        "measured_k_histogram": sorted(obs.items()),
        "uniform_k_marginal": {str(a): round(b, 4) for a, b in marginal.items()},
        "measured_mean_max_k": float(measured_max.mean()),
        "uniform_expected_max_k": emax,
        "uniform_max_pmf": {str(a): round(b, 4) for a, b in max_pmf.items()},
        "chi2_vs_uniform": chi2,
        "chi2_pooled_from_k": POOL_FROM,
        "chi2_observed_pooled": {str(a): b for a, b in sorted(pooled_observed.items())},
        "chi2_expected_pooled": {str(a): round(b, 4) for a, b in sorted(pooled_expected.items())},
        "chi2_bins": len(pooled_expected),
        "chi2_df": dof,
        "chi2_p_value": p_value,
        "per_die_total_active_experts": [int(v) for v in per_die_total],
        "per_die_expected": per_die_expected,
        "per_die_chi2": per_die_chi2,
        "per_die_df": per_die_dof,
        "per_die_p_value": per_die_p,
        "perfect_balance_max_k": TOP_K / NUM_DEVICES,
        "independence_caveat": (
            f"The {n} die-layer counts are NOT {n} independent Binomial(8, 1/4) draws: within each "
            f"layer the {NUM_DEVICES} counts sum to {TOP_K} by construction, and all {NUM_LAYERS} "
            "layers come from a single decode token. The effective sample is 48 layers of one "
            "token. Both p-values below are therefore optimistic about their own precision, and "
            "neither is evidence *for* uniformity -- only a failure to reject it. "
            "moe_routing_across_tokens_probe.py is the probe that samples many tokens."
        ),
        "why_the_lever_looked_like_zero_and_is_not": (
            "The sound argument does not need the routing to be uniform. If routing is "
            "exchangeable over experts -- no expert is systematically hotter than another -- then "
            "every relabelling of experts to dies induces the identical distribution of per-die "
            "counts, so the expected saving from a permutation is exactly 0 by construction and "
            "not by measurement. What that rests on is exchangeability, which this one-token "
            "sample can only fail to reject; persistent per-expert hotness across tokens is the "
            "one thing that would break it and it is invisible here. "
            "IT IS BROKEN. moe_routing_across_tokens_probe.py sampled 128 tokens on each of three "
            "prompts and found hotness that persists strongly (a layer's 8 hottest experts take "
            "47.5-57.4% of its selections against 6.2% under uniform routing), so routing is NOT "
            "exchangeable, the argument above does not apply, and the lever is NOT zero -- it is "
            "0.024-0.112 ms/iteration held out on routing the fit never saw. This key used to be "
            "named why_the_lever_is_zero, which contradicted its own last sentence and the stage's "
            "conclusion; the round-2 review caught the name."
        ),
    }
    print(f"per-die active-expert counts: {sorted(obs.items())}")
    print(f"  uniform marginal:           {[(a, round(b,3)) for a,b in marginal.items() if b > 0.001]}")
    print(
        f"  measured mean max_k {measured_max.mean():.3f} vs uniform {emax:.3f} "
        f"(perfect balance {TOP_K/NUM_DEVICES:.1f})"
    )
    print(f"  chi2 vs uniform = {chi2:.2f}, df {dof}, p = {p_value:.4f} " f"(k >= {POOL_FROM} pooled, not dropped)")
    print(f"  per-die totals over {NUM_LAYERS} layers: {per_die_total.tolist()} vs {per_die_expected:.0f} expected")
    print(f"    chi2 = {per_die_chi2:.2f}, df {per_die_dof}, p = {per_die_p:.4f}")
    print("  neither p-value is evidence FOR uniformity; the sample is 48 layers of ONE token.")

    # -- what it costs, and what a fix would be worth -----------------------
    measured_idle_us = float((measured_max.mean() - TOP_K / NUM_DEVICES) * EXPERT_STEP_US)
    uniform_idle_us = float((emax - TOP_K / NUM_DEVICES) * EXPERT_STEP_US)
    out["budget"] = {
        "measured_idle_us_per_layer": measured_idle_us,
        "measured_idle_ms_per_iteration": measured_idle_us * NUM_LAYERS / 1000.0,
        "uniform_routing_floor_us_per_layer": uniform_idle_us,
        "uniform_routing_floor_ms_per_iteration": uniform_idle_us * NUM_LAYERS / 1000.0,
        "moe_rs_total_ms_per_iteration": float(moe_rs.mean(axis=0).sum() / 1000.0),
        "attn_rs_total_ms_per_iteration": float(attn_rs.mean(axis=0).sum() / 1000.0),
        "moe_rs_excess_over_attn_ms": float((moe_rs.mean(axis=0) - attn_rs.mean()).sum() / 1000.0),
        "cost_of_removing_it_by_computing_all_experts": (
            f"{NUM_DEVICES * 32 / TOP_K:.0f}x the expert FLOPs "
            f"({NUM_DEVICES*32} expert-slots per layer instead of {TOP_K})"
        ),
    }
    print()
    print(
        f"idle from skew: measured {measured_idle_us:.2f} us/layer = "
        f"{measured_idle_us*NUM_LAYERS/1000:.3f} ms/iteration"
    )
    print(
        f"floor under *perfectly uniform* routing: {uniform_idle_us:.2f} us/layer = "
        f"{uniform_idle_us*NUM_LAYERS/1000:.3f} ms/iteration"
    )
    # The claim that used to stand here -- "the shipped layout is already
    # 0.69 us/layer better than the expectation for an arbitrary partition, so a
    # permutation is negative in expectation" -- was noise read as signal. The
    # measured mean max over 48 layers is 3.438 against 3.538 expected, and the
    # standard error of that mean is sd(max)/sqrt(48) ~ 0.12, so the difference
    # is about z = -0.9, p ~ 0.4. It is deleted rather than softened.
    se_max = float(measured_max.std(ddof=1) / math.sqrt(NUM_LAYERS))
    z = (measured_max.mean() - emax) / se_max
    out["budget"]["shipped_vs_uniform_expectation"] = {
        "measured_mean_max_k": float(measured_max.mean()),
        "uniform_expected_max_k": emax,
        "sd_of_max_over_layers": float(measured_max.std(ddof=1)),
        "standard_error_of_the_mean": se_max,
        "z": float(z),
        "two_sided_p": float(2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))),
        "reading": (
            "The measured mean maximum is below the uniform expectation by less than one "
            "standard error. This is NOT evidence that the shipped contiguous windows beat an "
            "arbitrary partition, and the earlier claim that it was has been withdrawn."
        ),
    }
    print(
        f"  -> measured mean max {measured_max.mean():.3f} vs uniform expectation {emax:.3f}: "
        f"z = {z:.2f}, p = {2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0)):.2f} -- indistinguishable."
    )
    print("  -> a permutation is worth 0 IF routing is exchangeable over experts. Nothing in this")
    print("     one-token sample establishes that; moe_routing_across_tokens_probe.py measures it,")
    print("     and it finds hotness that persists, so the lever is NOT zero. See that probe.")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

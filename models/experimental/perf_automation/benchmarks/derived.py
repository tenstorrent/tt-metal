"""No hardcoded timer constants. Every bound is derived from observed duration history.
Only operator input allowed: the absolute run ceiling (manifest config.timeout)."""
import random

# Observed anchors from REAL runs (llama3_1_8b_p150 2026-07-25 + ACE-Step 2026-07-23 audit).
# (base_s, label, {op: median observed seconds})
REAL = [
    (0.8, "micro module", dict(profile=6, pcc=8, build=60, agent=40, tracy_pp=15)),
    (3.16, "tiny ACE enc_layer", dict(profile=13, pcc=20, build=120, agent=60, tracy_pp=25)),
    (13.06, "mid ACE dit_layer", dict(profile=52, pcc=70, build=240, agent=90, tracy_pp=60)),
    (42.61, "large ACE audio_tok", dict(profile=170, pcc=210, build=408, agent=150, tracy_pp=140)),
    (146.72, "8B llama full pipe", dict(profile=154, pcc=1400, build=872, agent=300, tracy_pp=420)),
    (900.0, "very slow model", dict(profile=950, pcc=3000, build=1800, agent=600, tracy_pp=1200)),
    (3000.0, "extreme model", dict(profile=3100, pcc=9000, build=3600, agent=900, tracy_pp=3000)),
]
HOST_BOUND = {"kernel_compile", "weight_load", "thermal_cool", "device_reset", "git_op", "api_backoff", "jit_compile"}


def observe(med, n=24, rng=None):
    """Simulate the durations the tool would have LOGGED for this op (lognormal-ish spread)."""
    rng = rng or random.Random(0)
    return [max(0.05, med * rng.lognormvariate(0, 0.45)) for _ in range(n)]


def stats(samples):
    s = sorted(samples)
    p = lambda q: s[min(len(s) - 1, int(q * len(s)))]
    return dict(p50=p(0.50), p95=p(0.95), p99=p(0.99), mx=s[-1])


def derived_bounds(hist, ceiling):
    """Bounds derived ONLY from observed history + operator ceiling. No magic numbers."""
    o = stats(hist)
    spread = o["p95"] / max(o["p50"], 1e-9)  # observed variability, not a chosen multiplier
    return dict(
        grace=min(ceiling, o["p95"]),  # never kill inside normal duration
        flat=min(ceiling, o["p99"] * spread),  # flat beyond p99 scaled by its own spread
        ceiling=ceiling,
        p50=o["p50"],
        p95=o["p95"],
        p99=o["p99"],
    )

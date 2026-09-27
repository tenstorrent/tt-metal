"""Does the CFM's trace cache thrash under streaming's bucketed growing prefix, and what capacity would help?

Recreated 2026-09-25 (the 2026-09-23 original was lost with an instance, never committed). Pure Python, no device.
The access trace is the REAL streaming schedule -- `real_hop_schedule_lengths` / `session_lengths` imported from
../perf_2026_09_23/bucket_sizing_simulation.py (single source of truth, not re-derived) -- mapped to the CFM's own
trace key, the mel-rate bucket (= token-rate linear-step-64 bucket x token_mel_ratio 2). Each call either hits a
resident trace (steady-state replay cost) or captures (capture cost), under an LRU of the given capacity -- the
same policy `TtCausalConditionalCFM._traces` implements.

Costs are MEASURED, real checkpoint, streaming traced 10-step solve (cfm_streaming_real_checkpoint_check.py,
2026-09-25), at mel buckets 384 / 768 / 1536, linearly interpolated in T between them and linearly extrapolated
outside (flagged in the output: the real schedule's buckets span 256..1792). "capture" is the recapture-after-
release cost with kernels and conv weights already warm -- the cost every LRU miss after the first utterance pays.

What this does NOT model: allocation safety. Capacity > 1 means capturing while other traces are resident, which
the trace allocation tracker refuses (see TtCausalConditionalCFM's docstring) -- this answers "would it help", not
"is it safe as built".

Run: /home/user/tt-metal/python_env/bin/python cfm_trace_cache_thrashing_simulation.py   (no device)
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf_2026_09_23"))
sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from bucket_sizing_simulation import linear_bucket, session_lengths  # noqa: E402

TOKEN_MEL_RATIO = 2
N_UTTERANCES, SECONDS = 5, 30

# Measured 2026-09-25 (cfm_streaming_real_checkpoint_check.py), ms per 10-step traced streaming solve.
MEASURED = {
    # mel bucket: (recapture with warm kernels/weights, steady-state replay)
    384: (744.4, 501.6),
    768: (1071.9, 820.4),
    1536: (2063.9, 1628.4),
}


def cost(t_len: int, which: int) -> tuple[float, bool]:
    """Linear interpolation of MEASURED[.][which] in T; also returns whether it extrapolated."""
    pts = sorted((t, v[which]) for t, v in MEASURED.items())
    if t_len <= pts[0][0]:
        (t0, v0), (t1, v1) = pts[0], pts[1]
    elif t_len >= pts[-1][0]:
        (t0, v0), (t1, v1) = pts[-2], pts[-1]
    else:
        (t0, v0), (t1, v1) = next((a, b) for a, b in zip(pts, pts[1:]) if a[0] <= t_len <= b[0])
    extrapolated = not (pts[0][0] <= t_len <= pts[-1][0])
    return v0 + (v1 - v0) * (t_len - t0) / (t1 - t0), extrapolated


def simulate(keys_by_utt: list[list[int]], capacity: int):
    lru: OrderedDict[int, None] = OrderedDict()
    captures = hits = 0
    total_ms = capture_overhead_ms = 0.0
    for keys in keys_by_utt:
        for k in keys:
            steady, _ = cost(k, 1)
            if k in lru:
                lru.move_to_end(k)
                hits += 1
                total_ms += steady
            else:
                while len(lru) >= capacity:
                    lru.popitem(last=False)
                lru[k] = None
                captures += 1
                cap, _ = cost(k, 0)
                total_ms += cap
                capture_overhead_ms += cap - steady
    return captures, hits, total_ms, capture_overhead_ms


def main():
    assert all(v[0] is not None for v in MEASURED.values()), "fill MEASURED from the real-checkpoint run first"
    for persist, label in ((False, "hop reset per utterance (intended)"), (True, "hop persists (upstream literal)")):
        session = session_lengths(N_UTTERANCES, SECONDS, persist_hop=persist)
        keys_by_utt = [[linear_bucket(l, 64) * TOKEN_MEL_RATIO for l in lengths] for lengths in session]
        distinct = sorted({k for keys in keys_by_utt for k in keys})
        extrap = [k for k in distinct if cost(k, 0)[1]]
        within = sum(sum(1 for a, b in zip(keys, keys[1:]) if a == b) for keys in keys_by_utt)
        print(f"\n=== {label}: {N_UTTERANCES}x {SECONDS}s utterances ===")
        for i, keys in enumerate(keys_by_utt):
            print(f"utt {i}: {len(keys)} CFM calls, mel buckets {keys}")
        print(f"distinct mel buckets in session: {len(distinct)} {distinct}")
        print(f"consecutive same-bucket calls within an utterance (single-slot's only possible hits): {within}")
        if extrap:
            print(f"NOTE: costs extrapolated (outside measured 384..1536) for buckets {extrap}")
        print("\n| LRU capacity | captures | hits | total CFM ms | capture overhead ms | overhead / total |")
        print("|---|---|---|---|---|---|")
        for cap in sorted({1, 2, 4, 8, len(set(keys_by_utt[-1])), len(distinct)}):
            c, h, tot, ov = simulate(keys_by_utt, cap)
            print(f"| {cap} | {c} | {h} | {tot:.0f} | {ov:.0f} | {100 * ov / tot:.1f}% |")


if __name__ == "__main__":
    main()

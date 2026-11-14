# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Spec-decode round model with MEASURED costs (task mtp-device; replaces the 130 us 'other' assumption of speedup.py). Run: python -m ...spec.speedup_measured [a1 a2 a3]
Measured inputs (every number has a source):
 plain trace-only step B=4 (integ m1, flags HY4_POOL=1 SCALED_FP8 OPT=all IHC_FUSED SKINNY XMerge): 62.2 / 67.6 / 69.5 ms at 2k / 64k / 1M          [program/integ/STEPS.md "FINAL measured plain decode step"]
 MoE routed+shared per layer, global rows T = 4 / 8 / 16: 363 / 430 / 584 us                                                                    [layer-moe, speedup.py MOE_US]
 attention chain per FULL-indexer layer (score+select+filter+attend) S = 1 / 2 / 4 rows, ctx 2k / 64k / 1M, us (verify-rows RESULT.md, merged code):
     2k 256/260/284   64k 347/369/397   1M 431/452/513 ; shared (IndexCache) layers: attend only 149/147/166 (2k), 146/148/165 (64k), 153/160/187 (1M)
 MTP layer pass WITHOUT routed experts (this task, bench2/bench3 logs): S=1 and S=4 rows, see MTP_US below; lm_head + greedy sampler 0.62 ms (bench: layer+head+sample minus layer)
"""
import sys

from models.demos.blackhole.hy4_preview.spec.policy import expected_tokens

CTX = ("2k", "64k", "1M")
PLAIN = dict(zip(CTX, (62.2, 67.6, 69.5)))
MOE = {4: 363.0, 8: 430.0, 16: 584.0}
FULL = {  # ctx -> {S: us per full layer}  (verify-rows RESULT.md latency table, compact attend + A1 merge, merged b633624e336)
    "2k": {1: 256.0, 2: 260.0, 4: 284.0},
    "64k": {1: 347.0, 2: 369.0, 4: 397.0},
    "1M": {1: 431.0, 2: 452.0, 4: 513.0},
}
SHARED_ATT = {
    "2k": {1: 149.0, 2: 147.0, 4: 166.0},
    "64k": {1: 146.0, 2: 148.0, 4: 165.0},
    "1M": {1: 153.0, 2: 160.0, 4: 187.0},
}
# MTP layer (no routed experts) us per call, S rows: filled from bench2 (S=1) / bench3 (S=4); S=2 interpolated
MTP_US = {"2k": {1: 1482.0, 4: 1650.0}, "64k": {1: 1596.0, 4: 1778.0}, "1M": {1: 2330.0, 4: 2508.0}}
HEAD_SAMPLE_MS = 0.62
ROUND_OVH_MS = 0.91  # accept + derive x k + feedback glue in the trace: measured tail (7722 us, k=3, no routed experts) minus 3 MTP passes (3 x 1650) minus 3 heads (3 x 620)


def moe(T):
    ts = sorted(MOE)
    for a, b in zip(ts, ts[1:]):
        if T <= b:
            return MOE[a] + (MOE[b] - MOE[a]) * (T - a) / (b - a)
    return MOE[16]


def verify_ms(ctx, k):
    S = k + 1
    dm = 78 * (moe(4 * S) - moe(4)) / 1e3
    interp = lambda tab: tab[S] if S in tab else tab[1] + (tab[4] - tab[1]) * (S - 1) / 3
    da = (21 * (interp(FULL[ctx]) - FULL[ctx][1]) + 57 * (interp(SHARED_ATT[ctx]) - SHARED_ATT[ctx][1])) / 1e3
    return PLAIN[ctx] + dm + da


def mtp_ms(ctx, S):
    t = MTP_US[ctx]
    us = t[1] + (t[4] - t[1]) * (S - 1) / 3
    return (us + moe(4 * S)) / 1e3 + HEAD_SAMPLE_MS


def round_ms(ctx, k):
    if k == 0:
        return PLAIN[ctx] + mtp_ms(ctx, 1) + 0.3
    # all k MTP passes run at S = k+1 rows (one XMerge S per process: the recursive steps carry filler rows), each followed by a head + greedy
    return verify_ms(ctx, k) + k * mtp_ms(ctx, k + 1) + ROUND_OVH_MS


def report(p):
    out = [f"conditional acceptance p = {p}; E(k) = {[round(expected_tokens(p, k), 2) for k in (1, 2, 3)]}"]
    out.append("ctx | plain ms | k=1 round ms -> ms/token | k=2 | k=3 | best speedup vs plain")
    for ctx in CTX:
        cells, best = [], 0
        for k in (1, 2, 3):
            e = expected_tokens(p, k)
            r = round_ms(ctx, k)
            cells.append(f"{r:.1f} -> {r/e:.1f}")
            best = max(best, PLAIN[ctx] / (r / e))
        out.append(f"{ctx} | {PLAIN[ctx]} | " + " | ".join(cells) + f" | {best:.2f}x")
    return "\n".join(out)


if __name__ == "__main__":
    p = [float(x) for x in sys.argv[1:4]] or [0.775, 0.63, 0.63]
    print(report(p))

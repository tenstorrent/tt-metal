#!/usr/bin/env python3
"""Packing / pipeline simulator for the M3 prefill budget study.

Replays an AgentX-like request mix through an in-order N-stage prefill pipeline and
compares packing policies (fcfs, bucket, cost) and forward widths.
Per-layer costs come from coeffs.json (fitted from the single-galaxy measurements).
The DEFAULT_COEFFS below are PLACEHOLDERS so the script runs; replace them.

  python3 m3_budget_sim.py --coeffs coeffs.json --stages 8 --n 4000
"""
import argparse, json, math, random
from collections import deque

SEG = 2048  # max tokens per segment (= KV period today)
PAD = 2048  # each segment is padded to a multiple of PAD (2048 today, 128 with paged KV)
TRACE_BUCKET = 2048  # forward width is rounded up to a multiple of this (trace shapes)
DENSE_LAYERS = {0, 1, 2}  # full-attention layers in M3
N_LAYERS = 60

# Per-layer cost model, ms, for one forward on one (4,4) SP=4 stage:
#   layer_ms = a + b*W_padded + sum_i (c*p_i*h_i + d*h_i + e*n_i)
# W_padded = forward width after trace-bucket rounding, n_i = real tokens of segment i,
# p_i = padded(n_i) = rows attention actually computes, h_i = its cached_len (history).
# a,b: per-layer fixed + per-token cost; c: segment rows x history (attention / indexer compute,
# pad rows included; charged for at least p0 rows, since few rows per chip leave cores idle); d: history-only (e.g. index_k gather); e: per real token (MoE skips pad rows).
# stage_ms = overhead + overhead_per_token*W_padded + sum over the stage's layers of layer_ms
DEFAULT_COEFFS = {  # PLACEHOLDERS - replace with fitted values
    "sparse": {"a": 2.2, "b": 1.8e-3, "c": 1.0e-9, "d": 1.0e-6, "e": 0.0},
    "dense": {"a": 2.2, "b": 1.8e-3, "c": 3.0e-8, "d": 1.0e-6, "e": 0.0},
    "stage_overhead_ms": 0.0,
    "hop_ms": 0.0,
}

# ---------------------------------------------------------------- traffic
NEW_Q = [(0, 32), (0.25, 640), (0.5, 1600), (0.9, 5888), (0.99, 50000), (0.999, 100000), (1, 1_000_000)]
HIST_Q = [(0, 0), (0.05, 0), (0.25, 60_000), (0.5, 142_000), (0.75, 310_000), (0.9, 549_000), (1, 1_000_000)]


def qsample(rng, qs):
    """Piecewise log-linear interpolation between published quantiles (linear next to 0)."""
    u = rng.random()
    for (p0, v0), (p1, v1) in zip(qs, qs[1:]):
        if u <= p1:
            t = (u - p0) / (p1 - p0) if p1 > p0 else 0.0
            if v0 > 0 and v1 > 0:
                return int(math.exp(math.log(v0) + t * (math.log(v1) - math.log(v0))))
            return int(v0 + t * (v1 - v0))
    return int(qs[-1][1])


def sample_requests(n, seed, max_ctx, align_recompute=False):
    """-> [(new, hist)]; with align_recompute -> [(new + hist % SEG, aligned hist, useful new)]: a request whose
    history is not SEG-aligned restarts at the aligned boundary and recomputes the remainder (work, not useful)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        new = max(1, qsample(rng, NEW_Q))
        hist_raw = qsample(rng, HIST_Q)
        hist = (hist_raw // SEG) * SEG  # today: continue only from a SEG boundary
        new = min(new, max_ctx - hist)
        if new > 0:
            if align_recompute:
                out.append((new + (hist_raw - hist), hist, new))
            else:
                out.append((new, hist))
    return out


def route_to_prefill(reqs, dec_max_new, dec_max_hist):
    return [r for r in reqs if not (r[0] <= dec_max_new and r[1] <= dec_max_hist)]


def to_segments(rid, new, hist):
    segs, off = [], 0
    while off < new:
        n = min(SEG, new - off)
        segs.append({"rid": rid, "h": hist + off, "n": n})
        off += SEG
    return segs


# ---------------------------------------------------------------- cost
class CostModel:
    """stage_ms(fwd) = overhead + sum_layers(a + b*W) + sum_segs sum_layers(c*padded(n)*h + d*h + e*n)."""

    def __init__(self, C, split, embed_stage0_only=False):
        self.C, self.stages = C, []
        start = 0
        for nl in split:
            layers = range(start, start + nl)
            start += nl
            kinds = [C["dense"] if L in DENSE_LAYERS else C["sparse"] for L in layers]
            self.stages.append(kinds)
        assert start == N_LAYERS, split
        self.S = len(split)
        # seg_a: fixed attention cost per packed segment; charged per segment, so remove one from `a`.
        self.A = [C.get("stage_overhead_ms", 0.0) + sum(k["a"] - k.get("seg_a", 0.0) for k in ks) for ks in self.stages]
        o = C.get("stage_overhead_per_token_ms", 0.0)
        self.Bw = [
            (o if (s == 0 or not embed_stage0_only) else 0.0) + sum(k["b"] for k in ks)
            for s, ks in enumerate(self.stages)
        ]

    def seg_vec(self, seg):
        n, h = seg["n"], seg["h"]
        p = padded(n)
        return [
            sum(k.get("seg_a", 0.0) + k["c"] * max(p, k.get("p0", 0)) * h + k["d"] * h + k["e"] * n for k in ks)
            for ks in self.stages
        ]

    def fwd_vec(self, W, segsum):
        return [self.A[s] + self.Bw[s] * W + segsum[s] for s in range(self.S)]


# ---------------------------------------------------------------- packing
def bucket_of(h, edges=(16_384, 142_000, 310_000)):
    return sum(h >= e for e in edges)


def padded(n):
    return -(-n // PAD) * PAD


def pack(policy, reqs, W_max, M, lookahead=64, budget_ms=None):
    """reqs: (new, hist) in arrival order -> list of (segments, seg_cost_sum, W_forward).
    Only the next unscheduled segment of each request is eligible; a forward may take
    several consecutive segments of one request (depth-first). The scheduler looks at the
    first `lookahead` requests in the queue.
      fcfs   : fill the width in arrival order
      bucket : fill only from requests whose next segment is in the same history bucket
      cost   : like fcfs, but stop adding when the forward's slowest stage would exceed the
               cost of a full cold forward (deep segments displace tokens), or budget_ms if given"""
    streams = [deque(to_segments(i, r[0], r[1])) for i, r in enumerate(reqs)]
    for st in streams:
        for seg in st:
            seg["v"] = M.seg_vec(seg)
    active = deque(range(len(streams)))
    zero = [0.0] * M.S
    B_full = W_max // SEG
    full_cold = [sum(x) for x in zip(*([M.seg_vec({"n": SEG, "h": 0})] * B_full))]
    target = budget_ms if budget_ms is not None else max(M.fwd_vec(W_max, full_cold))
    fwds = []
    while active:
        head = [active.popleft() for _ in range(min(len(active), lookahead))]
        window = head
        if policy == "bucket":
            b = bucket_of(streams[head[0]][0]["h"])
            window = [i for i in head if bucket_of(streams[i][0]["h"]) == b]
        fwd, ssum, used = [], zero[:], 0
        for i in window:
            st = streams[i]
            while st and used + padded(st[0]["n"]) <= W_max:
                seg = st[0]
                trial = [x + y for x, y in zip(ssum, seg["v"])]
                if policy == "cost" and fwd:
                    Wt = -(-(used + padded(seg["n"])) // TRACE_BUCKET) * TRACE_BUCKET
                    if max(M.fwd_vec(Wt, trial)) > target:
                        break
                fwd.append(st.popleft())
                ssum = trial
                used += padded(seg["n"])
            if used + PAD > W_max:
                break
        keep = [i for i in head if streams[i]]
        active.extendleft(reversed(keep))
        W_fwd = -(-used // TRACE_BUCKET) * TRACE_BUCKET
        fwds.append((fwd, ssum, W_fwd))
    return fwds


# ---------------------------------------------------------------- pipeline
def flowshop(fwds, M):
    """In-order pipeline: forward f enters stage s when stage s is free and f left stage s-1."""
    hop = M.C.get("hop_ms", 0.0)
    done, busy, fwd_ms = [0.0] * M.S, [0.0] * M.S, []
    for fwd, ssum, W in fwds:
        cs = M.fwd_vec(W, ssum)
        prev = 0.0
        for s in range(M.S):
            start = max(done[s], prev + (hop if s else 0.0))
            done[s] = start + cs[s]
            busy[s] += cs[s]
            prev = done[s]
        fwd_ms.append(max(cs))
    mk = done[-1]
    return mk, [b / mk for b in busy], fwd_ms


def route(reqs, P, M, W):
    """Split requests over P pipelines: each goes to the pipeline with the least predicted queued cost."""
    if P == 1:
        return [reqs]
    load, out = [0.0] * P, [[] for _ in range(P)]
    for r in reqs:
        cost = 0.0
        for seg in to_segments(0, r[0], r[1]):
            v = M.seg_vec(seg)
            p = padded(seg["n"])
            cost += max(v[s] + (M.Bw[s] + M.A[s] / W) * p for s in range(M.S))
        i = min(range(P), key=load.__getitem__)
        out[i].append(r)
        load[i] += cost
    return out


def run_policy(pol, reqs, W, M, P=1, budget_ms=None):
    """-> (makespan ms, per-stage utilisation, per-forward slowest-stage ms, padded forward tokens)."""
    mks, busy, fms, padded_tok = [], [0.0] * M.S, [], 0
    for sub in route(reqs, P, M, W):
        if not sub:
            continue
        fw = pack(pol, sub, W, M, budget_ms=budget_ms)
        mk, util, f = flowshop(fw, M)
        mks.append(mk)
        busy = [b + u * mk for b, u in zip(busy, util)]
        fms += f
        padded_tok += sum(x[2] for x in fw)
    mk = max(mks)
    return mk, [b / sum(mks) for b in busy], fms, padded_tok


def candidate_splits(S):
    """Even split, plus stage 0 = k layers (k = the dense layers .. the even share) with the rest even."""

    def even(n, k):
        base, extra = divmod(n, k)
        return [base + (1 if i < extra else 0) for i in range(k)]

    cands = [even(N_LAYERS, S)]
    if S > 1:
        for first in range(len(DENSE_LAYERS), -(-N_LAYERS // S) + 1):
            rest = even(N_LAYERS - first, S - 1)
            cands.append([first] + sorted(rest, reverse=True))
    nd = len(DENSE_LAYERS)
    if S >= 3:  # dense layers spread over the leading stages (a deep request's dense cost split across stages)
        cands.append([2, 1] + sorted(even(N_LAYERS - 3, S - 2), reverse=True))
        cands.append([1, 2] + sorted(even(N_LAYERS - 3, S - 2), reverse=True))
    if S >= 4:
        cands.append([1] * nd + sorted(even(N_LAYERS - nd, S - nd), reverse=True))
        cands.append(
            [1, 1, 1 + (N_LAYERS - nd) // S] + sorted(even(N_LAYERS - nd - (N_LAYERS - nd) // S, S - 3), reverse=True)
        )
    if S >= 8:  # long pipelines: also try two light leading stages
        cands.append([len(DENSE_LAYERS)] * 2 + sorted(even(N_LAYERS - 2 * len(DENSE_LAYERS), S - 2), reverse=True))
    out = []
    for c in cands:
        if c not in out and all(x > 0 for x in c):
            out.append(c)
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", default=None)
    ap.add_argument("--stages", type=int, default=8)
    ap.add_argument("--split", default=None, help="comma list of layers per stage, e.g. 8,8,8,8,7,7,7,7")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--widths", default="2048,4096,6144,8192")
    ap.add_argument("--dec-max-new", type=int, default=1000)
    ap.add_argument("--dec-max-hist", type=int, default=65_536)
    ap.add_argument("--max-ctx", type=int, default=1_048_576)
    ap.add_argument("--paged", action="store_true", help="pad segments to 128 instead of 2048 (paged-KV what-if)")
    ap.add_argument("--pipelines", type=int, default=1, help="P independent pipelines, least-queued-cost routing")
    ap.add_argument("--align-recompute", action="store_true", help="recompute hist %% 2048 tokens (work, not useful)")
    ap.add_argument("--embed-stage0-only", action="store_true", help="charge the per-token overhead o to stage 0 only")
    ap.add_argument("--budget-ms", type=float, default=None, help="cost policy: per-forward stage budget in ms")
    ap.add_argument("--latency", action="store_true", help="add hot-request latency p50/p99 columns")
    ap.add_argument("--split-search", action="store_true", help="try candidate_splits(stages) and report each")
    ap.add_argument("--policies", default="fcfs,bucket,cost")
    ap.add_argument("--hop-ms", type=float, default=None, help="override the coeffs file's hop_ms")
    a = ap.parse_args()
    global PAD
    if a.paged:
        PAD = 128
    C = json.load(open(a.coeffs)) if a.coeffs else DEFAULT_COEFFS
    if a.split:
        split = [int(x) for x in a.split.split(",")]
    else:
        base, extra = divmod(N_LAYERS, a.stages)
        split = [base + (1 if i < extra else 0) for i in range(a.stages)]
    extended = (
        a.pipelines > 1
        or a.align_recompute
        or a.embed_stage0_only
        or a.budget_ms is not None
        or a.latency
        or a.split_search
        or a.policies != "fcfs,bucket,cost"
        or a.hop_ms is not None
    )
    if extended:
        return main_extended(a, C, split)
    M = CostModel(C, split)
    reqs = route_to_prefill(sample_requests(a.n, a.seed, a.max_ctx), a.dec_max_new, a.dec_max_hist)
    real = sum(n for n, _ in reqs)
    print(f"split={split} prefill requests={len(reqs)} real tokens={real:,}")
    print(f"{'W':>6} {'policy':>7} {'tok/s':>9} {'fwd p50 ms':>10} {'fwd p99 ms':>10} {'fill %':>7}  stage utilisation")
    for W in [int(x) for x in a.widths.split(",")]:
        for pol in ("fcfs", "bucket", "cost"):
            fw = pack(pol, reqs, W, M)
            mk, util, fms = flowshop(fw, M)
            fill = real / sum(f[2] for f in fw) * 100
            print(
                f"{W:6d} {pol:>7} {real / mk * 1000:9.0f} {pct(fms, 50):10.1f} {pct(fms, 99):10.1f} {fill:7.1f}  "
                + " ".join(f"{u*100:3.0f}" for u in util)
            )


def main_extended(a, C, split):
    """The --pipelines / --align-recompute / --embed-stage0-only / --budget-ms / --latency / --split-search path."""
    if a.hop_ms is not None:
        C = dict(C, hop_ms=a.hop_ms)
    hop = C.get("hop_ms", 0.0)
    reqs = sample_requests(a.n, a.seed, a.max_ctx, align_recompute=a.align_recompute)
    reqs = [r if len(r) == 3 else (r[0], r[1], r[0]) for r in reqs]
    reqs = [r for r in reqs if not (r[2] <= a.dec_max_new and r[1] <= a.dec_max_hist)]  # route on useful new tokens
    useful = sum(r[2] for r in reqs)
    work = sum(r[0] for r in reqs)
    splits = candidate_splits(len(split)) if a.split_search else [split]
    print(
        f"stages={len(split)} pipelines={a.pipelines} hop={hop} ms prefill requests={len(reqs)} useful tokens={useful:,} "
        f"work tokens={work:,} (recompute {100 * (work - useful) / work:.1f}%) budget_ms={a.budget_ms}"
    )
    head = f"{'W':>6} {'policy':>7} {'tok/s':>9} {'fwd p50':>8} {'fwd p99':>8} {'fill %':>7}"
    if a.latency:
        head += f" {'hot p50 ms':>10} {'hot p99 ms':>10}"
    for sp in splits:
        M = CostModel(C, sp, embed_stage0_only=a.embed_stage0_only)
        print(f"split={sp}")
        print(head + "  stage utilisation")
        for W in [int(x) for x in a.widths.split(",")]:
            for pol in a.policies.split(","):
                mk, util, fms, ptok = run_policy(
                    pol, reqs, W, M, P=a.pipelines, budget_ms=a.budget_ms if pol == "cost" else None
                )
                line = (
                    f"{W:6d} {pol:>7} {useful / mk * 1000:9.0f} {pct(fms, 50):8.1f} {pct(fms, 99):8.1f} "
                    f"{work / ptok * 100:7.1f}"
                )
                if a.latency:
                    S = M.S
                    line += f" {(S + 1) * pct(fms, 50) + (S - 1) * hop:10.0f} {(S + 1) * pct(fms, 99) + (S - 1) * hop:10.0f}"
                print(line + "  " + " ".join(f"{u * 100:3.0f}" for u in util))


if __name__ == "__main__":
    main()

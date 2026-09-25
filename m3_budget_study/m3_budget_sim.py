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
#   layer_ms = a + b*W_padded + sum_i (c*n_i*h_i + d*h_i + e*n_i)
# W_padded = forward width after trace-bucket rounding, n_i = real tokens of segment i,
# h_i = its cached_len (history). a,b: per-layer fixed + per-token cost; c: new x history
# (attention / indexer compute); d: history-only (e.g. index_k gather); e: per real token.
# stage_ms = overhead + sum over the stage's layers of layer_ms
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


def sample_requests(n, seed, max_ctx):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        new = max(1, qsample(rng, NEW_Q))
        hist = qsample(rng, HIST_Q)
        hist = (hist // SEG) * SEG  # today: continue only from a SEG boundary
        new = min(new, max_ctx - hist)
        if new > 0:
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
    """stage_ms(fwd) = overhead + sum_layers(a + b*W) + sum_segs sum_layers(c*n*h + d*h + e*n)."""

    def __init__(self, C, split):
        self.C, self.stages = C, []
        start = 0
        for nl in split:
            layers = range(start, start + nl)
            start += nl
            kinds = [C["dense"] if L in DENSE_LAYERS else C["sparse"] for L in layers]
            self.stages.append(kinds)
        assert start == N_LAYERS, split
        self.S = len(split)
        self.A = [C.get("stage_overhead_ms", 0.0) + sum(k["a"] for k in ks) for ks in self.stages]
        self.Bw = [sum(k["b"] for k in ks) for ks in self.stages]

    def seg_vec(self, seg):
        n, h = seg["n"], seg["h"]
        return [sum(k["c"] * n * h + k["d"] * h + k["e"] * n for k in ks) for ks in self.stages]

    def fwd_vec(self, W, segsum):
        return [self.A[s] + self.Bw[s] * W + segsum[s] for s in range(self.S)]


# ---------------------------------------------------------------- packing
def bucket_of(h, edges=(16_384, 142_000, 310_000)):
    return sum(h >= e for e in edges)


def padded(n):
    return -(-n // PAD) * PAD


def pack(policy, reqs, W_max, M, lookahead=64):
    """reqs: (new, hist) in arrival order -> list of (segments, seg_cost_sum, W_forward).
    Only the next unscheduled segment of each request is eligible; a forward may take
    several consecutive segments of one request (depth-first). The scheduler looks at the
    first `lookahead` requests in the queue.
      fcfs   : fill the width in arrival order
      bucket : fill only from requests whose next segment is in the same history bucket
      cost   : like fcfs, but stop adding when the forward's slowest stage would exceed the
               cost of a full cold forward (deep segments displace tokens)"""
    streams = [deque(to_segments(i, n, h)) for i, (n, h) in enumerate(reqs)]
    for st in streams:
        for seg in st:
            seg["v"] = M.seg_vec(seg)
    active = deque(range(len(streams)))
    zero = [0.0] * M.S
    B_full = W_max // SEG
    full_cold = [sum(x) for x in zip(*([M.seg_vec({"n": SEG, "h": 0})] * B_full))]
    target = max(M.fwd_vec(W_max, full_cold))
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


if __name__ == "__main__":
    main()

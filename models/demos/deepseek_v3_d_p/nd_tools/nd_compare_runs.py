#!/usr/bin/env python3
"""Compare ND fingerprints between two prefill log files (across-process) or
across iterations within one log (within-process).

Across-process: two fresh processes, identical input + fixed seed, same iter
index -> the FIRST stage whose sha differs is where run-to-run non-determinism
is born. Sidesteps the Gumbel-sampling iteration confound entirely.

Usage:
  nd_compare_runs.py <logA> <logB> [--iter N]      # across-process at iter N (default 0)
  nd_compare_runs.py <log> --within                # within-process, consecutive iters
"""
import re
import sys
from collections import defaultdict

LAYER_RE = re.compile(r"\[NDPROBE\]\s+iter=(\d+)\s+stage=\s*(\S+)\s+sha=(\S+)\s+norm=([\d.eE+-]+)\s+nonfinite=(-?\d+)")
# baseline line (prev is None) prints the same shape; diff line prints sha a->b. Capture baseline form.
MOE_RE = re.compile(
    r"\[NDPROBE-MOE\]\s+layer=(\d+)\s+iter=(\d+)\s+stage=\s*(\S+)\s+(?:scope=\S+\s+)?sha=(\S+)\s+norm=([\d.eE+-]+)\s+sum=(\S+)\s+nonfinite=(-?\d+)"
)


def parse(path):
    # layers[iter][stage] = (sha, norm, nonfinite); moe[(layer,iter)][stage] = (sha, norm, sum, nonfinite)
    layers = defaultdict(dict)
    moe = defaultdict(dict)
    order_layers = defaultdict(list)
    order_moe = defaultdict(list)
    with open(path, errors="replace") as f:
        for line in f:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = LAYER_RE.search(line)
            if m:
                it, stage, sha, norm, nf = m.groups()
                it = int(it)
                if stage not in layers[it]:
                    order_layers[it].append(stage)
                layers[it][stage] = (sha, float(norm), int(nf))
                continue
            m = MOE_RE.search(line)
            if m:
                layer, it, stage, sha, norm, ssum, nf = m.groups()
                key = (int(layer), int(it))
                if stage not in moe[key]:
                    order_moe[key].append(stage)
                # keep index 2 == nonfinite to match the layer tuple for the shared printer
                moe[key][stage] = (sha, float(norm), int(nf), ssum)
    return layers, moe, order_layers, order_moe


def cmp_stage_maps(order, a, b, label):
    print(f"\n=== {label} ===")
    first = None
    for stage in order:
        av = a.get(stage)
        bv = b.get(stage)
        if av is None or bv is None:
            print(f"  {stage:>18}  (missing in one run: A={av is not None} B={bv is not None})")
            continue
        same = av[0] == bv[0]
        tag = "SAME" if same else "DIFF"
        if not same and first is None:
            first = stage
        extra = ""
        if not same:
            extra = f"  normA={av[1]:.6f} normB={bv[1]:.6f} dnorm={bv[1]-av[1]:+.3e} nfA={av[2]} nfB={bv[2]}"
        print(f"  {stage:>18}  {tag}  shaA={av[0]} shaB={bv[0]}{extra}")
    if first:
        print(f"  --> FIRST DIVERGENCE: {first}")
    else:
        print(f"  --> identical across the two (no divergence at this granularity)")
    return first


def main():
    args = sys.argv[1:]
    if "--within" in args:
        args.remove("--within")
        path = args[0]
        layers, moe, ol, om = parse(path)
        iters = sorted(layers)
        print(f"within-process: iters found = {iters}")
        for i in range(1, len(iters)):
            a, b = iters[i - 1], iters[i]
            cmp_stage_maps(ol[b] or ol[a], layers[a], layers[b], f"layers iter{a} vs iter{b}")
        # MoE within-process
        mlayers = sorted({k[0] for k in moe})
        for L in mlayers:
            its = sorted(it for (ll, it) in moe if ll == L)
            for i in range(1, len(its)):
                a, b = its[i - 1], its[i]
                cmp_stage_maps(om[(L, b)] or om[(L, a)], moe[(L, a)], moe[(L, b)], f"MoE layer{L} iter{a} vs iter{b}")
        return

    it = 0
    if "--iter" in args:
        k = args.index("--iter")
        it = int(args[k + 1])
        del args[k : k + 2]
    logA, logB = args[0], args[1]
    la, ma, ola, oma = parse(logA)
    lb, mb, olb, omb = parse(logB)
    print(f"Comparing across-process at iter={it}\n  A={logA}\n  B={logB}")
    cmp_stage_maps(ola.get(it) or olb.get(it) or [], la.get(it, {}), lb.get(it, {}), f"per-layer @ iter{it}")
    mlayers = sorted({k[0] for k in ma} | {k[0] for k in mb})
    for L in mlayers:
        cmp_stage_maps(
            oma.get((L, it)) or omb.get((L, it)) or [],
            ma.get((L, it), {}),
            mb.get((L, it), {}),
            f"MoE layer{L} @ iter{it}",
        )


if __name__ == "__main__":
    main()

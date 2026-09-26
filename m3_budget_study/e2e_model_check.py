#!/usr/bin/env python3
"""Part B sanity: predicted pipeline tok/s = W / slowest stage (cost model, one plain chunk per forward, embedding
on stage 0 only) vs the measured open-loop and K = stages tok/s.   e2e_model_check.py results_sp2/e2e/summary.txt"""
import json, re, sys

H = {"cold": 0, "h141": 139264, "h549": 548864}


def stage_ms(C, first, cnt, W, h, embed):
    t = C.get("stage_overhead_per_token_ms", 0.0) * W if embed else 0.0
    for L in range(first, first + cnt):
        k = C["dense"] if L < 3 else C["sparse"]
        t += k["a"] + k["b"] * W + k["c"] * max(W, k.get("p0", 0)) * h + k["d"] * h + k["e"] * W
    return t


def main(summary):
    coeffs = {4: json.load(open("results_sp2/coeffs_sp2.json")), 2: json.load(open("results/coeffs.json"))}
    sess = None
    print(
        f"{'session':34} {'stream':6} {'pred tok/s':>10} {'open':>7} {'err':>7} {'K=stages':>8} {'err':>7}  predicted stage ms"
    )
    meas = {}
    for line in open(summary):
        if line.startswith("## "):
            sess = line[3:].strip()
        elif sess and (m := re.match(r"\s+(\w+?)_(open|k\d)\s+chunks.*tok/s\s+(\d+)", line)):
            meas[(sess, m[1], m[2])] = int(m[3])
    for s in sorted({k[0] for k in meas}):
        if "sync1" in s:
            continue
        r, W = int(s[1]), int(re.search(r"_w(\d+)", s)[1])
        counts = [int(x) for x in s.split("split")[1].split("-")] if "split" in s else [60 // r] * r
        starts = [sum(counts[:i]) for i in range(r)]
        for st in ("cold", "h141", "h549"):
            if (s, st, "open") not in meas:
                continue
            ms = [stage_ms(coeffs[r], a, c, W, H[st], i == 0) for i, (a, c) in enumerate(zip(starts, counts))]
            pred = W / max(ms) * 1000
            o, k = meas[(s, st, "open")], meas.get((s, st, f"k{r}"))
            print(
                f"{s:34} {st:6} {pred:10.0f} {o:7d} {(o - pred) / pred * 100:+6.1f}% {k or 0:8d} "
                f"{((k - pred) / pred * 100) if k else 0:+6.1f}%  " + " ".join(f"{x:.0f}" for x in ms)
            )


if __name__ == "__main__":
    main(sys.argv[1])

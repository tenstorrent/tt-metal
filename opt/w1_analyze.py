"""W1 traced A/B: steady-state STEP_MS for LTX_DEDUP_GATE_GATHER 0 vs 1.

Trace-capture steps are the ones that pay JIT/capture (>2000 ms) and the 2-step warmup generate();
both are discarded. What survives is replay-only steady state, where sigma ~ 0.1-0.4 ms.
"""
import re
import statistics as st
import sys


def parse(path):
    stage, rows = None, []
    for ln in open(path, errors="ignore"):
        m = re.search(r"shapes: vN=(\d+)", ln)
        if m:
            stage = "S1" if m.group(1) == "9728" else "S2"
        m = re.search(r"Step (\d+)/(\d+):.*STEP_MS=([\d.]+)", ln)
        if m:
            rows.append((stage, int(m.group(2)), int(m.group(1)), float(m.group(3))))
    return rows


def steady(rows, stage):
    # n in (8, 3) = the real generate(); n==2 is the trace warmup. <2000 ms = replay, not capture.
    return [v for s, n, _, v in rows if s == stage and n in (8, 3) and v < 2000]


a, b = parse(sys.argv[1]), parse(sys.argv[2])
print(f"{'stage':6} {'DEDUP=0 (off)':>22} {'DEDUP=1 (on)':>22} {'delta':>12} {'%':>8}")
for stage in ("S1", "S2"):
    va, vb = steady(a, stage), steady(b, stage)
    if not va or not vb:
        print(f"{stage:6} MISSING (off n={len(va)}, on n={len(vb)})")
        continue
    ma, mb = st.mean(va), st.mean(vb)
    sa = st.pstdev(va) if len(va) > 1 else 0.0
    sb = st.pstdev(vb) if len(vb) > 1 else 0.0
    d = mb - ma
    # sigma of the difference of two means, from the pooled per-step spread
    se = ((sa**2 / max(len(va), 1)) + (sb**2 / max(len(vb), 1))) ** 0.5
    nsig = abs(d) / se if se > 0 else float("inf")
    print(
        f"{stage:6} {ma:10.2f} +-{sa:5.2f} (n={len(va)}) {mb:10.2f} +-{sb:5.2f} (n={len(vb)}) "
        f"{d:+9.2f} ms {100 * d / ma:+7.2f}%   [{nsig:.0f} sigma]"
    )
    print(f"       off: {[round(x, 1) for x in va]}")
    print(f"       on : {[round(x, 1) for x in vb]}")

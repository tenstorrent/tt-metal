"""Summarize a pwr_ab.sh output dir: per-layer-step savings (TtMoe - block) by layer group and type, and telemetry
(AICLK, TDP, TDC, VCORE, temperature, clock-limiting arbiter) over the chunk loop, by context bucket.

usage: pwr_ab_summary.py OUT_DIR
Layer timestamps: the producer's CHUNK_RETIRE lines (time.monotonic seconds); telemetry: time.monotonic_ns.
A step gap(l) = ack(l) - ack(l-1) = layer l-1's FFN + layer l's attention. GLM-5.3 full-indexer layers are 0, 1, 2
and every 4th from 6; dense FFN layers 0-2.
"""

import bisect
import collections
import re
import statistics as st
import sys

D = sys.argv[1]
FULL = {0, 1, 2} | set(range(6, 78, 4))
R = re.compile(r"CHUNK_RETIRE slot=\d+ start=(\d+) end=\d+ first=([\d.]+) retire=([\d.]+) warmup=0 acks=\d+ layers=(\S+)")
BUCKETS = (("0-50k", 0, 10), ("200-500k", 40, 100), ("800k-1M", 160, 205))
GROUPS = (("layers 1-18", 1, 19), ("19-37", 19, 38), ("38-57", 38, 58), ("58-76", 58, 77))


def runs(v):
    rows = {}
    for line in open(f"{D}/runner_{v}.log", errors="replace"):
        m = R.search(line)
        if m:
            acks = {int(it.split("@")[0].split("-")[0]): float(it.split("@")[1]) for it in m[4].split(",")}
            rows[int(m[1])] = (float(m[2]), float(m[3]), acks)
    return [rows[k] for k in sorted(rows)]


A, B = runs("ttmoe"), runs("block")
print(f"\n==== per-step saving (TtMoe - block), ms; chunks ttmoe {len(A)} block {len(B)} ====")
for lbl, lo, hi in BUCKETS:
    def g(rows, l):
        return st.mean(r[2][l] - r[2][l - 1] for r in rows[lo:hi])
    parts = []
    for gl, a, b in GROUPS:
        cat = collections.defaultdict(list)
        for l in range(a, b):
            k = "dense" if l <= 3 else ("full" if l in FULL else "shared")
            cat[k].append((g(A, l) - g(B, l), g(A, l), g(B, l)))
        parts.append(gl + ": " + ", ".join(f"{k} n={len(x)} {st.mean(y[0] for y in x):+.2f} ({st.mean(y[1] for y in x):.2f}->{st.mean(y[2] for y in x):.2f})" for k, x in cat.items()))
    print(f"{lbl}:\n  " + "\n  ".join(parts))

try:
    lines = open(f"{D}/telemetry/telem.csv")
except OSError:
    print("no telemetry"); sys.exit(0)
ser = collections.defaultdict(list)
limits = []
for line in lines:
    if line.startswith("# limits"):
        limits.append(line.strip()); continue
    if line[0] in "#t":
        continue
    t, c, k, v = line.rstrip().split(",")
    ser[(k, int(c))].append((int(t) / 1e9, int(v)))
print("\n==== chip limits (first 4 chips; all distinct values) ====")
for l in limits[:4]:
    print(" ", l)
dist = collections.Counter(" ".join(x for x in l.split()[4:] if not x.startswith(("ASIC_LOCATION", "FLASH", "CM_", "ETH_"))) for l in limits)
for k, n in dist.items():
    print(f"  {n} chips: {k}")
chips = sorted({c for _, c in ser})


def avg(k, c, a, b):
    s = ser[(k, c)]
    if not s or b <= a:
        return float("nan")
    ts = [x[0] for x in s]
    i = max(bisect.bisect_right(ts, a) - 1, 0)
    tot, t = 0.0, a
    while t < b and i < len(s):
        nt = min(s[i + 1][0] if i + 1 < len(s) else b, b)
        nt = max(nt, t)
        tot += s[i][1] * (nt - t)
        t = nt
        i += 1
    return tot / (b - a)


print("\n==== telemetry over the chunk loop (chip mean) ====")
for v, rows in (("ttmoe", A), ("block", B)):
    for lbl, lo, hi in BUCKETS:
        sel = rows[lo:hi]
        if not sel:
            continue
        a, b = sel[0][0], sel[-1][1]
        out = {k: st.mean(avg(k, c, a, b) for c in chips) for k in ("aiclk", "tdp", "tdc", "vcore", "temp")}
        arb = collections.Counter()
        for c in chips:
            s = ser[("arb_max", c)]
            for i, (t, x) in enumerate(s):
                t2 = s[i + 1][0] if i + 1 < len(s) else b
                l2, h2 = max(t, a), min(t2, b)
                if h2 > l2:
                    arb[x >> 16] += h2 - l2
        tot = sum(arb.values()) or 1
        arbs = " ".join(f"#{k}:{100 * w / tot:.0f}%" for k, w in arb.most_common(3))
        print(f"  {v:6} {lbl:9} AICLK {out['aiclk']:.0f} MHz  TDP {out['tdp']:.1f} W  TDC {out['tdc']:.1f} A  VCORE {out['vcore']:.0f} mV  "
              f"temp {out['temp'] / 65536:.1f} C  limiter {arbs}  (arbiter 1 = TDP; 7 = GDDR-thermal, seen at the 1350 MHz fmax, i.e. no throttling)")

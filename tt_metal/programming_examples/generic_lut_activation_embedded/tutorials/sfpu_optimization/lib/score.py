"""Score one rung: correctness (max abs err vs the exact reference) + static analysis.

Usage: score.py <rung> <dump_csv> [<trisc_obj>]
Prints: "<fma> <sfpu_insns|NA> <max_abs_err>"

The "reference" IS the benchmark function (a piecewise polynomial for p* rungs, a
piecewise rational P(x)/Q(x) for r* rungs). We evaluate it exactly at each measured
input x rather than looking up a precomputed CSV, so the check is robust to any
sampling-grid offset and to segment-boundary discontinuities.
"""
import sys
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TUT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from static_analysis import fma_count, count_sfpu_insns  # noqa: E402

rung = sys.argv[1]
dump = sys.argv[2]
obj = sys.argv[3] if len(sys.argv) > 3 else ""
is_rational = rung.startswith("r")

hdr_name = "bench_rational_lut.h" if is_rational else "bench_lut.h"
hdr = open(os.path.join(TUT, "kernels/common", hdr_name)).read()


def _arr(name):
    return [float(v.strip().rstrip("f")) for v in re.search(name + r"\s*=\s*\{\{([^}]*)\}\}", hdr).group(1).split(",")]


def _int(name):
    return int(re.search(name + r"\s*=\s*(\d+)", hdr).group(1))


def _seg(x, bounds, num):
    s = 0
    for i in range(num):
        if x >= bounds[i]:
            s = i
    return s


def _poly_gt():
    num = _int("BENCH_NUM_SEGMENTS")
    maxd = _int("BENCH_MAX_DEGREE")
    lut = _arr("BENCH_LUT")
    bounds, coeffs = lut[: num + 1], lut[num + 1 :]

    def gt(x):
        base = _seg(x, bounds, num) * (maxd + 1)
        acc = 0.0
        for d in range(maxd, -1, -1):
            acc = acc * x + coeffs[base + d]
        return acc

    return gt


def _rational_gt():
    nseg = _int("BENCH_R_NUM_SEGMENTS")
    nd = _int("BENCH_R_NUM_DEGREE")
    dd = _int("BENCH_R_DEN_DEGREE")
    lut = _arr("BENCH_R_LUT")
    bounds = lut[: nseg + 1]
    body = lut[nseg + 1 :]
    stride = (nd + 1) + (dd + 1)

    def gt(x):
        s = _seg(x, bounds, nseg)
        base = s * stride
        p = 0.0
        for d in range(nd, -1, -1):
            p = p * x + body[base + d]
        q = 0.0
        for d in range(dd, -1, -1):
            q = q * x + body[base + (nd + 1) + d]
        return p / q

    return gt


# Correctness
err = 9.99
try:
    gt = _rational_gt() if is_rational else _poly_gt()
    e = 0.0
    n = 0
    for a, b in list(csv.reader(open(dump)))[1:]:
        e = max(e, abs(float(b) - gt(float(a))))
        n += 1
    err = e if n else 9.99
except Exception:
    err = 9.99

# Static analysis: per-element FMA count of the predicated cascade.
parity = ("parity" in rung) or ("adaptive" in rung) or ("deferred" in rung)
if is_rational:
    nseg = _int("BENCH_R_NUM_SEGMENTS")
    nd = _int("BENCH_R_NUM_DEGREE")
    dd = _int("BENCH_R_DEN_DEGREE")
    import math

    if parity:
        per = math.ceil(nd / 2) + math.ceil(dd / 2)
    else:
        per = nd + dd
    fma = per * nseg  # reciprocal counted separately (see results notes)
else:
    try:
        sd = [int(x) for x in re.search(r"BENCH_SEGMENT_DEGREES\[\d+\]\s*=\s*\{([^}]*)\}", hdr).group(1).split(",")]
    except Exception:
        sd = [8]
    degs = sd if "adaptive" in rung else [max(sd)] * len(sd)
    fma = fma_count(degs, parity=parity)

insns = count_sfpu_insns(obj)
print(f"{fma} {insns if insns is not None else 'NA'} {err:.3e}")

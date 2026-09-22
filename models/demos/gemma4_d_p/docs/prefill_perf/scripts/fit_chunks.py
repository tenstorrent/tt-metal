#!/usr/bin/env python3
"""Fit T = N*a + slope*N(N-1)/2 from a text_demo_prefill.py run log.

Standalone: stdlib only, no skill, no third-party imports.
    python3 fit_chunks.py <run.log> [<run.log> ...] [--isl 262144]

Parses the harness's own per-chunk device times:
    [traced_perf] chunk 3/32 [16384, 24576) device=267.0ms (30686 tok/s) ...
and least-squares fits  t_i = a + slope*i  per chunk size.

  a      = first-chunk time / per-chunk floor (ms)
  slope  = extra cost per additional chunk of history (ms per chunk index)
  R^2    = straight-line goodness of fit; 1.0 is perfect
"""
import re
import sys

RE = re.compile(r"\[traced_perf\] chunk (\d+)/(\d+) \[(\d+), (\d+)\) device=([0-9.]+)ms")


def fit(ts):
    n = len(ts)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ts) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ts))
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx == 0:
        return ts[0], None, None
    slope = sxy / sxx
    a = my - slope * mx
    ss_res = sum((y - (a + slope * x)) ** 2 for x, y in zip(xs, ts))
    ss_tot = sum((y - my) ** 2 for y in ts)
    return a, slope, (1 - ss_res / ss_tot) if ss_tot else 1.0


def main(argv):
    isl = 262144
    if "--isl" in argv:
        i = argv.index("--isl")
        isl = int(argv[i + 1])
        del argv[i : i + 2]
    series = {}
    for path in argv:
        for line in open(path, errors="replace"):
            m = RE.search(line)
            if m:
                start, end, ms = int(m.group(3)), int(m.group(4)), float(m.group(5))
                series.setdefault(end - start, []).append(ms)
    if not series:
        print("no [traced_perf] chunk lines found")
        return 1
    print(
        f"{'chunk':>7} {'pts':>4} {'a (ms)':>9} {'slope':>9} {'R^2':>8} "
        f"{'floor':>8} {'prefix':>8} {'total':>8}   (ISL {isl:,})"
    )
    print("-" * 78)
    for C in sorted(series):
        a, s, r2 = fit(series[C])
        if s is None:
            print(f"{C:>7} {len(series[C]):>4} {a:9.1f} {'UNVERIFIABLE (1 chunk)':>27}")
            continue
        N = isl // C
        fl = N * a / 1000.0
        pf = s * N * (N - 1) / 2 / 1000.0
        print(f"{C:>7} {len(series[C]):>4} {a:9.1f} {s:9.3f} {r2:8.5f} " f"{fl:7.2f}s {pf:7.2f}s {fl+pf:7.2f}s")
    print("\n  a = first-chunk time / per-chunk floor;  slope = ms added per chunk of history")
    print("  floor = N*a ;  prefix = slope*N(N-1)/2 ;  total = floor + prefix")
    return 0


sys.exit(main(sys.argv[1:]))

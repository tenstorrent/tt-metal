# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer math utilization with and without the host: the fidelity-priced ideal math time of each layer (from a
profiled test_layer_perf.py run, see analyze_layer_util.py) against

* the profiled device kernel time (sum of the slowest chip's op kernels: gaps between ops excluded),
* the traced replay time (device only, no host in the loop: includes dispatch gaps between ops),
* the eager wall time (host dispatch + device, the device idle at the start of the layer),

plus the host enqueue time of the layer call (the MIMO_PERF_OUT lines of test_layer_perf.py with MIMO_PERF_TRACE).

    python models/demos/mimo_v2_d_p/tests/perf/report_layer_util_e2e.py <perf_out.txt> <ops_perf_results.csv> [...]
"""

import collections
import re
import sys

from models.demos.mimo_v2_d_p.tests.perf.analyze_layer_util import SP, classify, load, peak


def perf_lines(path):
    """tag -> {trace, wall, host} ms from the MIMO_PERF_OUT file."""
    out = collections.defaultdict(dict)
    for line in open(path):
        m = re.match(r"(TRACE|WALL|HOST) (\S+): .*?median ([\d.]+) ms", line)
        if m:
            out[m[2]][m[1].lower()] = float(m[3])
    return out


def ideal_and_kernel(csv_paths):
    """tag -> (ideal ms, kernel ms, {part: (ms, ideal ms)})."""
    res = {}
    for path in csv_paths:
        for tag, iters in load(path).items():
            m = re.match(r"L(\d+)_(GA|SWA)_C(\d+)_ctx(\d+)", tag)
            if not m:
                continue
            kind, S, ctx = m[2], int(m[3]), int(m[4])
            parts = collections.defaultdict(lambda: [0.0, 0.0])
            for devs in iters.values():
                slow = max(devs, key=lambda d: sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in devs[d]))
                for part, t, fl, fid in classify(devs[slow], kind, S, ctx - S * SP):
                    parts[part][0] += t / len(iters) / 1e3
                    parts[part][1] += fl / peak(fid) * 1e3 / len(iters)
            res[tag] = (sum(p[1] for p in parts.values()), sum(p[0] for p in parts.values()), dict(parts))
    return res


def main(perf, csvs):
    p, u = perf_lines(perf), ideal_and_kernel(csvs)
    print(
        "| layer | tok/chip | context | ideal math | kernel sum | traced (device) | eager wall | host enqueue | "
        "util kernels | util device | util e2e | host visible |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    key = lambda t: (
        int(re.match(r"L(\d+)", t)[1]),
        int(re.search(r"_C(\d+)", t)[1]),
        int(re.search(r"ctx(\d+)", t)[1]),
    )
    for tag in sorted(set(p) & set(u), key=key):
        ideal, ker, _ = u[tag]
        tr, wa, ho = p[tag].get("trace"), p[tag].get("wall"), p[tag].get("host")
        L, kind = re.match(r"L(\d+)_(GA|SWA)", tag).groups()
        S, ctx = key(tag)[1], key(tag)[2]
        print(
            f"| L{L} {kind} | {S} | {ctx // 1024}K | {ideal:.2f} ms | {ker:.2f} ms | {tr:.2f} ms | {wa:.2f} ms | "
            f"{ho:.2f} ms | {100 * ideal / ker:.0f}% | {100 * ideal / tr:.0f}% | {100 * ideal / wa:.0f}% | "
            f"{wa - tr:+.2f} ms ({100 * (wa - tr) / wa:.0f}%) |"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

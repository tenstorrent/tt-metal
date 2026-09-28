# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-time views of each signposted tag (test_layer_perf.py --profile), per measured iteration, averaged:

  per-op max   sum over ops of the slowest device's kernel duration (analyze_tags.py's total). Over-counts on a mesh:
               a collective's kernel time on one device includes waiting for its peers, and the slowest device
               differs op to op.
  busy         per device, the sum of its kernel durations (ops are serialized on a device): mean / max over devices.
  span         per device, first op's FW start to last op's FW end (busy + op-to-op gaps: dispatch, host): max.

    python models/demos/mimo_v2_d_p/tests/perf/analyze_device_time.py <ops_perf_results.csv>
"""

import collections
import csv
import sys

MHZ = 1350.0


def main(path):
    tags = collections.OrderedDict()
    cur, it = None, collections.Counter()
    for r in csv.DictReader(open(path)):
        if r["OP TYPE"] == "signpost":
            c = r["OP CODE"]
            if c.endswith("_start"):
                cur = c[: -len("_start")]
                it[cur] += 1
            elif c.endswith("_end"):
                cur = None
            continue
        if cur is None:
            continue
        d = tags.setdefault(cur, collections.defaultdict(lambda: collections.defaultdict(list)))
        d[it[cur]][int(r["DEVICE ID"])].append(
            (
                float(r["DEVICE KERNEL DURATION [ns]"] or 0) / 1e3,
                float(r["DEVICE FW START CYCLE"] or "nan"),
                float(r["DEVICE FW END CYCLE"] or "nan"),
            )
        )
    for tag, iters in tags.items():
        opmax, busy_mean, busy_max, span = [], [], [], []
        for devs in iters.values():
            n = min(len(v) for v in devs.values())
            opmax.append(sum(max(v[j][0] for v in devs.values()) for j in range(n)))
            busy = [sum(o[0] for o in v) for v in devs.values()]
            busy_mean.append(sum(busy) / len(busy))
            busy_max.append(max(busy))
            span.append(max((max(o[2] for o in v) - min(o[1] for o in v)) / MHZ for v in devs.values()))
        m = lambda xs: sum(xs) / len(xs) / 1e3
        print(
            f"{tag:<44s} per-op max {m(opmax):7.2f} ms | busy mean {m(busy_mean):7.2f} max {m(busy_max):7.2f} ms"
            f" | span {m(span):7.2f} ms"
        )


if __name__ == "__main__":
    main(sys.argv[1])

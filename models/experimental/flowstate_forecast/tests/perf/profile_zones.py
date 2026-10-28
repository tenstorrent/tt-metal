# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Per-zone device time of a reduced-layer profile (test_profile_forward.py) and the full-model estimate.

    python profile_zones.py <ops_perf_results.csv> <layer_counts.json> [--out zones.json]

Sums DEVICE KERNEL DURATION per signpost zone (``<zone>_start`` .. ``<zone>_end``), multiplies by layer_counts,
and lists the top ops per zone by device time."""

import argparse
import csv
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("counts")
    ap.add_argument("--out")
    a = ap.parse_args()
    counts = json.load(open(a.counts))["counts"]
    zone, ns, ops = None, defaultdict(float), defaultdict(lambda: defaultdict(float))
    for r in csv.DictReader(open(a.csv)):
        code = r.get("OP CODE", "")
        if r.get("OP TYPE") == "signpost":
            if code.endswith("_start") and code[:-6] in counts:
                zone = code[:-6]
            elif code.endswith("_end"):
                zone = None
            continue
        d = r.get("DEVICE KERNEL DURATION [ns]") or ""
        if zone and d:
            ns[zone] += float(d)
            ops[zone][code] += float(d)
    est = {z: ns[z] * counts[z] / 1e6 for z in counts}
    out = {
        "zone_ms_once": {z: ns[z] / 1e6 for z in counts},
        "counts": counts,
        "zone_ms_total": est,
        "estimate_ms": sum(est.values()),
        "top_ops": {z: sorted(((k, v / 1e6) for k, v in ops[z].items()), key=lambda t: -t[1])[:8] for z in counts},
    }
    out["profile_stamp"] = {k: v for k, v in json.load(open(a.counts)).items() if k.startswith(("git_", "started_at"))}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

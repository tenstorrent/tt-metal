# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: median DEVICE KERNEL DURATION per (setting, placement) of a profiled
test_g6_reshard_noc_policy_perf.py run. Rows map to test ids in collection order.

    python .../probes/ab_table.py <ops_perf_results.csv> <ids.txt>
"""

import collections
import csv
import re
import statistics
import sys


def main(csv_path, ids_path):
    ids = [line.strip() for line in open(ids_path) if line.strip()]
    rows = [r for r in csv.DictReader(open(csv_path)) if r["OP CODE"] == "GenericOpDeviceOperation"]
    assert len(rows) == len(ids), f"{len(rows)} op rows vs {len(ids)} test ids"
    groups = collections.OrderedDict()
    for test_id, row in zip(ids, rows):
        m = re.search(r"setting=([^-\]]+)-name=([^-\]]+)", test_id)
        groups.setdefault((m.group(1), m.group(2)), []).append(
            (int(row["DEVICE KERNEL DURATION [ns]"]), int(row["DEVICE KERNEL FIRST TO LAST START [ns]"] or 0))
        )
    for (setting, name), samples in groups.items():
        durs = [d for d, _ in samples]
        skews = [s for _, s in samples]
        print(
            f"{setting:16s} {name:3s} median {int(statistics.median(durs)):5d} ns"
            f"  min {min(durs):5d}  max {max(durs):5d}  n={len(durs)}"
            f"  | median launch skew {int(statistics.median(skews))} ns  samples {durs}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

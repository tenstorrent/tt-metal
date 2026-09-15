# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize active-core math counters, rejecting 32-bit reference wraps."""

import csv
import json
import statistics
import sys
from collections import defaultdict


def analyze(path):
    records = defaultdict(dict)
    with open(path) as source:
        header = source.readline().strip()
        for raw in csv.DictReader(source):
            row = {key.strip(): value.strip() for key, value in raw.items()}
            key = (int(row["run host ID"]), int(row["core_x"]), int(row["core_y"]))
            if row["RISC processor type"] == "TRISC_1" and row["zone name"] == "TRISC-KERNEL":
                records[key][row["type"]] = int(row["time[cycles since reset]"])
            if row["timer_id"] == "9090":
                data = json.loads(row["meta data"].replace(";", ","))
                records[key][data["counter type"]] = data
    runs = defaultdict(list)
    for (run, x, y), record in records.items():
        if "FPU_COUNTER" not in record:
            continue
        duration = record["ZONE_END"] - record["ZONE_START"]
        reference = record["FPU_COUNTER"]["ref cnt"]
        assert 0 < duration < 2**32, (run, x, y, duration)
        assert abs(duration - reference) < 10000, (run, x, y, duration, reference)
        for counter in ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"):
            assert record[counter]["ref cnt"] == reference
            assert 0 <= record[counter]["value"] <= reference
        runs[run].append(record)
    for run, cores in sorted(runs.items()):
        result = dict(path=path, header=header, run=run, active_cores=len(cores), reference_wraps=0)
        for counter in ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"):
            fractions = [100 * c[counter]["value"] / c[counter]["ref cnt"] for c in cores]
            result[counter] = dict(
                mean_pct=statistics.mean(fractions),
                median_pct=statistics.median(fractions),
                min_pct=min(fractions),
                max_pct=max(fractions),
            )
        result["overlap_mean_pct"] = (
            result["FPU_COUNTER"]["mean_pct"] + result["SFPU_COUNTER"]["mean_pct"] - result["MATH_COUNTER"]["mean_pct"]
        )
        result["neither_math_unit_mean_pct"] = 100 - result["MATH_COUNTER"]["mean_pct"]
        print(json.dumps(result))


for path in sys.argv[1:]:
    analyze(path)

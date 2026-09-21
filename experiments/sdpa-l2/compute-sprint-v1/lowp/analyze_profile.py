# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read hardware activity separately from uninstrumented useful throughput."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def analyze(path, q_repeats=8, k_chunks=512):
    groups = defaultdict(dict)
    with path.open() as source:
        header = source.readline()
        mhz = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header)[1])
        for raw in csv.DictReader(source):
            row = {key.strip(): value.strip() for key, value in raw.items()}
            key = tuple(row[name] for name in (
                "PCIe slot", "core_x", "core_y", "run host ID", "trace id", "trace id counter"))
            if row["RISC processor type"] == "TRISC_1" and row["zone name"] in ("SDPA_SPRINT_LOWP", "TRISC-KERNEL"):
                groups[key][(row["zone name"], row["type"])] = int(row["time[cycles since reset]"])
            if row["timer_id"] == "9090":
                counter = json.loads(row["meta data"].replace(";", ","))
                groups[key][counter["counter type"]] = counter
    results = []
    for key, group in groups.items():
        if ("SDPA_SPRINT_LOWP", "ZONE_START") not in group:
            continue
        cycles = group[("SDPA_SPRINT_LOWP", "ZONE_END")] - group[("SDPA_SPRINT_LOWP", "ZONE_START")]
        whole = group[("TRISC-KERNEL", "ZONE_END")] - group[("TRISC-KERNEL", "ZONE_START")]
        assert 0 < cycles <= whole < 2**32, (key, cycles, whole)
        result = dict(path=str(path), run_key=key, profiler_reported_clock_mhz=mhz,
            zone_cycles=cycles, whole_kernel_cycles=whole,
            useful_flops_per_zone_cycle=(4*256*512*128*q_repeats*k_chunks)/cycles)
        for name in ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"):
            counter = group[name]
            assert abs(counter["ref cnt"]-whole) < 10000, (key, name, counter, whole)
            assert 0 <= counter["value"] <= counter["ref cnt"]
            result[name] = counter
            result[name+"_pct"] = 100*counter["value"]/counter["ref cnt"]
        result["both_pct"] = result["FPU_COUNTER_pct"]+result["SFPU_COUNTER_pct"]-result["MATH_COUNTER_pct"]
        result["neither_pct"] = 100-result["MATH_COUNTER_pct"]
        result["sfpu_only_pct"] = result["SFPU_COUNTER_pct"]-result["both_pct"]
        results.append(result)
    assert results, "No lowp profile zones found"
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--q-repeats", type=int, default=8)
    parser.add_argument("--k-chunks", type=int, default=512)
    args = parser.parse_args()
    for record in analyze(args.csv, args.q_repeats, args.k_chunks):
        print(json.dumps(record))

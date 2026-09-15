# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read profile cycles separately from uninstrumented trace-replay throughput."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def profile(path):
    data = defaultdict(dict)
    with path.open() as source:
        header = source.readline()
        mhz = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header)[1])
        for raw in csv.DictReader(source):
            r = {k.strip(): v.strip() for k, v in raw.items()}
            key = (r["run host ID"], r["core_x"], r["core_y"])
            if r["RISC processor type"] == "TRISC_1" and r["zone name"] in ("SDPA_RESIDENT", "TRISC-KERNEL"):
                data[key][(r["zone name"], r["type"])] = int(r["time[cycles since reset]"])
            if r["timer_id"] == "9090":
                counter = json.loads(r["meta data"].replace(";", ","))
                data[key][counter["counter type"]] = counter
    for key, r in data.items():
        if ("SDPA_RESIDENT", "ZONE_START") not in r:
            continue
        cycles = r[("SDPA_RESIDENT", "ZONE_END")] - r[("SDPA_RESIDENT", "ZONE_START")]
        duration = r[("TRISC-KERNEL", "ZONE_END")] - r[("TRISC-KERNEL", "ZONE_START")]
        assert 0 < cycles <= duration < 2**32
        flops = 4 * 256 * 512 * 128 * 8 * 512
        result = dict(
            path=str(path),
            core=key,
            clock_mhz=mhz,
            resident_cycles=cycles,
            whole_kernel_cycles=duration,
            preload_and_other_cycles=duration - cycles,
            resident_ms=cycles / (mhz * 1000),
            useful_flops_per_cycle=flops / cycles,
            tflops_per_core=flops / cycles * mhz / 1e6,
        )
        for name in ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"):
            c = r[name]
            assert abs(duration - c["ref cnt"]) < 10000
            assert 0 <= c["value"] <= c["ref cnt"]
            result[name] = 100 * c["value"] / c["ref cnt"]
        result["both_pct"] = result["FPU_COUNTER"] + result["SFPU_COUNTER"] - result["MATH_COUNTER"]
        result["neither_pct"] = 100 - result["MATH_COUNTER"]
        print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for p in args.paths:
        profile(p)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Resident useful throughput and simultaneous math activity from raw device CSV."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--variant", choices=["C", "D"], required=True)
    parser.add_argument("--q-repeats", type=int, default=8)
    parser.add_argument("--k-chunks", type=int, default=512)
    args = parser.parse_args()
    for path in args.paths:
        records = defaultdict(dict)
        with path.open() as source:
            header = source.readline()
            mhz = float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header)[1])
            for raw in csv.DictReader(source):
                row = {k.strip(): v.strip() for k, v in raw.items()}
                key = (row["run host ID"], row["core_x"], row["core_y"])
                if row["RISC processor type"] == "TRISC_1" and row["zone name"] in (
                    "SDPA_FULLCHIP_LOFI",
                    "TRISC-KERNEL",
                ):
                    records[key][(row["zone name"], row["type"])] = int(row["time[cycles since reset]"])
                if row["timer_id"] == "9090":
                    counter = json.loads(row["meta data"].replace(";", ","))
                    records[key][counter["counter type"]] = counter
        for key, record in records.items():
            zone = "SDPA_FULLCHIP_LOFI"
            if (zone, "ZONE_START") not in record:
                continue
            cycles = record[(zone, "ZONE_END")] - record[(zone, "ZONE_START")]
            duration = record[("TRISC-KERNEL", "ZONE_END")] - record[("TRISC-KERNEL", "ZONE_START")]
            assert 0 < cycles <= duration < 2**32
            flops = 4 * 256 * 512 * 128 * args.q_repeats * args.k_chunks
            peak_flops_per_cycle = 1024 if args.variant == "D" else 4096 / 3
            result = dict(
                path=str(path),
                run_and_core=key,
                variant=args.variant,
                header_clock_mhz=mhz,
                resident_cycles=cycles,
                whole_kernel_cycles=duration,
                useful_flops_per_cycle=flops / cycles,
                own_matmul_peak_util_percent=100 * flops / cycles / peak_flops_per_cycle,
                tflops_per_core_at_header_clock=flops / cycles * mhz / 1e6,
            )
            for name in ("FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER"):
                counter = record[name]
                assert abs(duration - counter["ref cnt"]) < 10000
                assert 0 <= counter["value"] <= counter["ref cnt"]
                result[name] = 100 * counter["value"] / counter["ref cnt"]
            result["both_percent"] = result["FPU_COUNTER"] + result["SFPU_COUNTER"] - result["MATH_COUNTER"]
            result["neither_percent"] = 100 - result["MATH_COUNTER"]
            print(json.dumps(result))


if __name__ == "__main__":
    main()

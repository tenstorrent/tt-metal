#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import argparse

parser = argparse.ArgumentParser(description="Convert benchmark JSON to CSV")
parser.add_argument("json", help="JSON file to convert", type=argparse.FileType("r"))
args = parser.parse_args()

result = json.load(args.json)
columns = [
    "cores",
    "kernel_size",
    "fast_kernel_cycles",
    "slow_kernel_cycles",
    "kgs",
    "cbs",
    "rtas",
    "crtas",
    "sems",
    "brisc_enabled",
    "ncrisc_enabled",
    "trisc_enabled",
    "erisc_enabled",
    "cb_gs",
]
print("name,time (us)," + ",".join(columns))
for benchmark in result["benchmarks"]:
    print(
        f"{benchmark['name']},{benchmark['IterationTime']*1000000:.2f}"
        + ","
        + ",".join(str(int(benchmark.get(col, "0"))) for col in columns)
    )

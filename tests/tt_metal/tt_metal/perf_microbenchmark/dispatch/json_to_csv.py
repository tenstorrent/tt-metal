#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import argparse

parser = argparse.ArgumentParser(description="Convert benchmark JSON to CSV")
parser.add_argument("json", help="JSON file to convert", type=argparse.FileType("r"))
args = parser.parse_args()

result = json.load(args.json)
print("name,time (us)")
for benchmark in result["benchmarks"]:
    print(f"{benchmark['name']},{benchmark['IterationTime']*1000000:.2f}")

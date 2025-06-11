# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# python tests/didt/process_profiler_output.py --input-file <profiler-csv>
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from tt_metal.tools.profiler.process_device_log import import_device_profile_log

parser = argparse.ArgumentParser(description="Timing analysis.")
parser.add_argument("--input-file", type=str, help="Input file with profiler logs")
args = parser.parse_args()
file_name = Path(args.input_file).stem

devices_data = import_device_profile_log(args.input_file)

for device in devices_data["devices"].keys():
    first_core = list(devices_data["devices"][device]["cores"].keys())[0]
    first_core_data = devices_data["devices"][device]["cores"][first_core]["riscs"]["TRISC_0"]["timeseries"]
    first_core_math = [
        i for i in first_core_data if (i[0]["zone_name"] == "MATH-BLOCK" and i[0]["zone_phase"] == "begin")
    ]
    num_of_iter = sum(1 for i in first_core_data if (i[0]["zone_name"] == "TRISC-FW" and i[0]["zone_phase"] == "begin"))
    num_of_blocks = len(first_core_math) // num_of_iter
    num_of_cores = len(devices_data["devices"][device]["cores"].keys())
    logger.info(f"============ Device:{device} data ============")
    logger.info(f"num_of_iter: {num_of_iter}")
    logger.info(f"num_of_blocks: {num_of_blocks}")
    logger.info(f"num_of_cores: {num_of_cores}")

    per_core_diff = [list(range(num_of_blocks))]
    start_time_per_core = []

    cores = devices_data["devices"][device]["cores"].keys()

    for key in cores:
        second_core_data = devices_data["devices"][device]["cores"][key]["riscs"]["TRISC_0"]["timeseries"]
        second_core_math = [
            i for i in second_core_data if (i[0]["zone_name"] == "MATH-BLOCK" and i[0]["zone_phase"] == "begin")
        ]

        per_block_diff = []
        start_time_per_block = []

        for i in range(num_of_blocks):
            per_block_diff.append(first_core_math[i][1] - second_core_math[i][1])
            start_time_per_block.append(second_core_math[i][1])

        per_core_diff.append(per_block_diff)
        start_time_per_core.append(start_time_per_block)

    # Block start time diff from core (1, 1)
    per_core_diff = pd.DataFrame(per_core_diff)
    per_core_diff = per_core_diff.T
    header_row = pd.DataFrame([["Block ID", *[f"Core {core}" for core in cores]]])
    per_core_diff = pd.concat([header_row, per_core_diff])
    per_core_diff.to_csv(f"{file_name}-per-core-diff-{device}.csv", index=False)

    # Standard deviation of start time per block among all cores
    stddevs = pd.DataFrame(start_time_per_core).std()

    logger.info(f"Mean standard deviation is {stddevs.mean():.3f}")

    stddevs = stddevs.round(3)
    stddevs = pd.DataFrame({"Block ID": stddevs.index, "stddev": stddevs.values})
    stddevs.to_csv(f"{file_name}-stddevs-{device}.csv", index=False)

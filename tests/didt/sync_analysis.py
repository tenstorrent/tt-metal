# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# python tests/didt/process_profiler_output.py --input-file <profiler-csv>
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

from tt_metal.tools.profiler.process_device_log import import_device_profile_log

parser = argparse.ArgumentParser(description="Timing analysis.")
parser.add_argument("--input-file", type=str, help="Input file with profiler logs")
parser.add_argument("--output-file", type=str, help="Output files prefix")
parser.add_argument("--num-blocks", type=str, help="Number of blocks")

args = parser.parse_args()
file_name = Path(args.input_file).stem
output_file_name = args.output_file
num_blocks = int(args.num_blocks)

devices_data = import_device_profile_log(args.input_file)

for device in devices_data["devices"].keys():
    # Collect starting time of every core for every "MATH-BLOCK" in core data, remove cores with no block data
    start_time_per_core = []
    cores = list(devices_data["devices"][device]["cores"].keys())
    cores_to_remove = []
    for key in cores:
        # Use data for analysis only if data for analysis_core exist in devices_data
        analysis_core = "TRISC_0"
        if analysis_core in devices_data["devices"][device]["cores"][key]["riscs"].keys():
            core_data = devices_data["devices"][device]["cores"][key]["riscs"]["TRISC_0"]["timeseries"]
            core_math = [i for i in core_data if (i[0]["zone_name"] == "MATH-BLOCK" and i[0]["zone_phase"] == "begin")]

            start_time_per_block = []

            for i in range(num_blocks):
                start_time_per_block.append(core_math[i][1])

            start_time_per_core.append(start_time_per_block)
        else:
            cores_to_remove.append(key)

    for key in cores_to_remove:
        cores.remove(key)

    # Calculate sync between cores. Sync metric is based on number of cores that have start time diff less than abs(sync_thr) value
    # Since we have 64 cores [num_of_cores], we have (num_of_cores * (num_of_cores - 1) / 2) unique core pairs to check
    # Sync measure is expressed as percentual value of synced core pairs where 100% is equal to (num_of_cores * (num_of_cores - 1) / 2)
    # Overall sync is measured as mean value accross all blocks
    sync_thr = 300
    num_of_cores = len(start_time_per_core)
    num_of_blocks = len(start_time_per_core[0])
    num_of_sync_per_block = []
    for block in range(num_of_blocks):
        num_of_sync_cores = 0
        for core in range(num_of_cores):
            for compare in range(core + 1, num_of_cores):
                # Find block start time with smallest diff compared to current block start time (blocks are not always in sync)
                # min([abs(start_time_per_core[core][block] - x) for x in start_time_per_core[compare]])
                if min([abs(start_time_per_core[core][block] - x) for x in start_time_per_core[compare]]) < sync_thr:
                    num_of_sync_cores += 1
        num_of_sync_per_block.append(num_of_sync_cores)

    # Print relative sync numbers (in percents)
    sync_per_block_pct = [100 * (x / (num_of_cores * (num_of_cores - 1) / 2)) for x in num_of_sync_per_block]
    # formated_sync = [f"{i:2.2f}" for i in sync_per_block_pct]
    # print(formated_sync)
    print(np.mean(sync_per_block_pct))

    # Block start time diff matrix is calculated as diff between core start time for block and mean value of start time of all cores for that block
    # This matrix can be used as heatmap to show how latencies are disctributed over the cores
    block_start_diff = pd.DataFrame(start_time_per_core)
    block_start_diff -= block_start_diff.median()
    block_start_diff.index = [f"Core {str(core):>8}" for core in cores]
    block_start_diff.columns = [f"BlockID{block:03d}" for block in range(num_blocks)]
    block_start_diff.to_csv(
        f"{output_file_name}-block-start-diff-{device}.csv",
        index=True,
        index_label="Core ( X,  Y)",
        header=True,
        float_format="%10.1f",
    )

    sns.set(font_scale=0.2)
    sns.heatmap(block_start_diff, cmap="RdYlGn")
    plt.savefig(f"{output_file_name}-block-start-diff-{device}.pdf")
    plt.clf()

    # Standard deviation of start time per block among all cores
    stddevs = pd.DataFrame(start_time_per_core).std()

    print(f"Mean standard deviation is {stddevs.mean():.3f}")

    stddevs = stddevs.round(3)
    stddevs = pd.DataFrame({"Block ID": stddevs.index, "stddev": stddevs.values})
    stddevs.to_csv(f"{output_file_name}-stddevs-{device}.csv", index=False)

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import os
import glob
import csv
import click
import subprocess
from tests.tt_eager.profiling import ops_for_profiling
from pathlib import Path
from loguru import logger


@click.command()
@click.option(
    "-o", "--output_directory", default="host_overhead_profile/", type=str, help="Ouput folder path for csv's"
)
@click.option("-c", "--output_csv", default="final.csv", type=str, help="Ouput csv filename")
def main(output_directory, output_csv):
    currentEnvs = dict(os.environ)
    currentEnvs["TT_METAL_DEVICE_PROFILER"] = "1"

    all_ops = []

    for op in ops_for_profiling.all_binary_ops:
        all_ops.append(op["name"])

    for op in ops_for_profiling.all_unary_ops:
        all_ops.append(op["name"])

    for op in ops_for_profiling.all_ternary_ops:
        all_ops.append(op["name"])

    i = 0

    for op_name in all_ops:
        op_id = f"{i:03d}"
        command = f'python -m tracy -v -r -p -o {output_directory} -n {op_id}_{op_name} -m "pytest tests/tt_eager/profiling/profile_host_overhead.py --input-method cli --cli-input {output_directory}::{op_name}"'
        subprocess.run([command], shell=True, check=False, env=currentEnvs, timeout=1200)
        i += 1

    # Top level csv files
    top_level_files = Path(output_directory).glob("*.csv")
    # top_level_files = glob.glob('output_directory/*.csv')

    # Initialize the final dataframe
    final_df = None
    output_csv = os.path.join(output_directory, output_csv)

    # Iterate over top level csv files
    for file in top_level_files:
        # Do not parse results file if it exists
        if os.path.exists(output_csv):
            if os.path.samefile(file, output_csv):
                continue

        # Read the csv file
        df = pd.read_csv(file)

        if final_df is None:
            final_df = df
        else:
            # Append the dataframe to the final dataframe
            final_df = pd.concat([final_df, df], ignore_index=True)

    # Subfolder csv files
    subfolder_files = Path(output_directory).glob("**/*.csv")

    # Iterate over subfolder csv files
    for file in subfolder_files:
        basename = os.path.basename(file)

        if not basename.startswith("ops_perf_results_"):
            continue

        logger.info(f"Analyzing {file}")

        # Read the csv file
        df = pd.read_csv(file)

        # Iterate over the rows in the final dataframe
        for index, row in final_df.iterrows():
            # Find the rows in the subfolder csv file where 'OP CODE' is after 'start ' + op and before 'end ' + op
            start_index = df.loc[df["OP CODE"] == "start " + row["op"]].index.tolist()
            end_index = df.loc[df["OP CODE"] == "end " + row["op"]].index.tolist()

            if start_index and end_index:
                host_duration = 0

                for i in range(len(start_index)):
                    si = start_index[i]
                    ei = end_index[i]

                    # Sum the 'HOST DURATION [ns]' values
                    host_duration += df.loc[si + 1 : ei, "HOST DURATION [ns]"].sum()

                # Add the average value to the final dataframe
                op_count = row["count"]
                host_duration = round(host_duration / (op_count * 1000 * 1000), 2)
                final_df.loc[index, "HOST DURATION [ms]"] = host_duration

    logger.info(f"Total ops profiled {len(all_ops)}")

    # Sort output
    final_df = final_df.sort_values("op")

    # Write the final dataframe to a csv file
    final_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()

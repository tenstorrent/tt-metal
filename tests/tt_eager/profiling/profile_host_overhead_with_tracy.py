# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import click
import subprocess
import csv


@click.command()
@click.option("-o", "--ouput_directory", default="host_overhead_profile", type=str, help="Ouput folder path fro csv's")
def main(ouput_directory):
    command = f"python -m tracy -v -r -p -o {ouput_directory} -m 'pytest tests/tt_eager/profiling/profile_host_overhead.py --input-method cli --cli-input {ouput_directory}'"
    subprocess.run([command], shell=True, check=False)

    profiler_csv_file = os.path.join(output_directory, "host_overhead_profiler_output.csv")

    with open(profiler_csv_file, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            print(row)


if __name__ == "__main__":
    main()

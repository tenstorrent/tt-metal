# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import subprocess
import re
from tabulate import tabulate
from perf_csv import perf_report


def validate_results(df):
    reference_values = {
        "HOST DURATION [ns]": 10000,
        "Cycles Count": 120000,
        "OP TO OP LATENCY [ns]": 3500,
        "DEVICE FW DURATION [ns]": 115000,
        "DEVICE KERNEL DURATION [ns]": 110000,
        "PM BANDWIDTH [ns]": 8000,
    }

    averages = df.iloc[-1]

    for column, reference in reference_values.items():
        if averages[column] > reference:
            assert False, f"Output exceeds reference for {column}: {averages[column]} > {reference}"


def run_profile_and_extract_csv():
    command = [
        "./tt_metal/tools/profiler/profile_this.py",
        "-n",
        "all_gather_n300",
        "-c",
        "pytest /home/ubuntu/tt-metal/tests/ttnn/unit_tests/operations/ccl/perf/test_all_gather_N300_post_commit.py",
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    full_output = result.stdout + result.stderr

    csv_path_match = re.search(r"OPs csv generated at: (.+\.csv)", full_output)
    if csv_path_match:
        csv_path = csv_path_match.group(1)
        print(f"CSV path found: {csv_path}")

        average_values = perf_report(csv_path)

        print(tabulate(average_values, headers="keys", tablefmt="pretty"))
        validate_results(average_values)

    else:
        print("CSV path not found in the command output.")


run_profile_and_extract_csv()

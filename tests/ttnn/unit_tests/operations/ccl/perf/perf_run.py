# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import subprocess
import re
from tabulate import tabulate
from perf_csv import perf_report


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

        print("Min - Avg - Max by Common Runs:")
        print(tabulate(average_values, headers="keys", tablefmt="pretty"))

    else:
        print("CSV path not found in the command output.")


run_profile_and_extract_csv()

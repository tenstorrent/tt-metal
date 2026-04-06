# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tests.ttnn.profiling.profile_host_overhead_with_tracy import profile_host_overhead
from loguru import logger
import csv
import os
import time
import pytest


@pytest.mark.timeout(10000)
def test_host_overhead_ci():
    profile_output_folder = "host_overhead_profile/"
    profile_output_filename = "final.csv"
    measured_filename = os.path.join(profile_output_folder, profile_output_filename)
    reference_filename = "tests/ttnn/profiling/reference.txt"
    measuring_tolerance = 1.13

    profile_host_overhead(profile_output_folder, profile_output_filename)

    with open(measured_filename, mode="r") as infile:
        reader = csv.reader(infile)
        measured_durations = {rows[0]: rows[2] for rows in reader}

    with open(reference_filename, mode="r") as infile:
        reader = csv.reader(infile)
        reference_durations = {rows[0]: rows[2] for rows in reader}

    failed_ops = []
    failed_msg = ""

    for op in reference_durations:
        if op == "op":
            continue

        if op not in measured_durations:
            failed_ops.append(f"{op} duration measurement missing.)")

        elif float(measured_durations[op]) > measuring_tolerance * float(reference_durations[op]):
            failed_ops.append(
                f"{op}: measured duration {measured_durations[op]}ms, reference duration {reference_durations[op]}ms"
            )

    for failed_op in failed_ops:
        failed_msg += f"{failed_op}\n"

    assert len(failed_ops) == 0, failed_msg

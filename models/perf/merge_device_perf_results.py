# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.perf.device_perf_utils import check_device_perf_results
from models.perf.perf_utils import merge_perf_files, today

expected_cols = [
    "Model",
    "Setting",
    "Batch",
    "AVG DEVICE FW SAMPLES/S",
    "MIN DEVICE FW SAMPLES/S",
    "MAX DEVICE FW SAMPLES/S",
    "AVG DEVICE FW DURATION [ns]",
    "MIN DEVICE FW DURATION [ns]",
    "MAX DEVICE FW DURATION [ns]",
    "AVG DEVICE KERNEL SAMPLES/S",
    "Lower Threshold AVG DEVICE KERNEL SAMPLES/S",
    "Upper Threshold AVG DEVICE KERNEL SAMPLES/S",
    "MIN DEVICE KERNEL SAMPLES/S",
    "MAX DEVICE KERNEL SAMPLES/S",
    "AVG DEVICE KERNEL DURATION [ns]",
    "MIN DEVICE KERNEL DURATION [ns]",
    "MAX DEVICE KERNEL DURATION [ns]",
    "AVG DEVICE BRISC KERNEL SAMPLES/S",
    "MIN DEVICE BRISC KERNEL SAMPLES/S",
    "MAX DEVICE BRISC KERNEL SAMPLES/S",
    "AVG DEVICE BRISC KERNEL DURATION [ns]",
    "MIN DEVICE BRISC KERNEL DURATION [ns]",
    "MAX DEVICE BRISC KERNEL DURATION [ns]",
]

check_cols = ["AVG DEVICE KERNEL SAMPLES/S"]

if __name__ == "__main__":
    fname = f"Models_Device_Perf_{today}.csv"
    merge_perf_files(fname, "device_perf", expected_cols)
    check_device_perf_results(fname, expected_cols, check_cols)

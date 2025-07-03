# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.perf.perf_utils import check_perf_results, merge_perf_files, today

expected_cols = [
    "Model",
    "Setting",
    "Batch",
    "First Run (sec)",
    "Second Run (sec)",
    "Compile Time (sec)",
    "Expected Compile Time (sec)",
    "Inference Time (sec)",
    "Expected Inference Time (sec)",
    "Throughput (Batch*inf/sec)",
    "Inference Time CPU (sec)",
    "Throughput CPU (Batch*inf/sec)",
]

check_cols = ["Inference Time (sec)", "Compile Time (sec)"]

if __name__ == "__main__":
    fname = f"Models_Perf_{today}.csv"
    merge_perf_files(fname, "perf", expected_cols)
    check_perf_results(fname, expected_cols, check_cols)

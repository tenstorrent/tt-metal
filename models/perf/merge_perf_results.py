# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0



from models.perf.perf_utils import check_perf_results, merge_perf_files, today, prep_perf_report
import sys
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
    if len(sys.argv) > 1:
        my_arg = sys.argv[1]
        fname = f"perf_models_demos_{my_arg}_{today}.csv"
        merge_perf_files(fname, f"perf_models_demos_{my_arg}", expected_cols)
        check_perf_results(fname, expected_cols, check_cols)
    else:
        fname = f"Models_Perf_{today}.csv"
        merge_perf_files(fname, "perf", expected_cols)
        check_perf_results(fname, expected_cols, check_cols)

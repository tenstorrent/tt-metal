# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
import pandas as pd

from tt_metal.tools.profiler.common import PROFILER_OUTPUT_DIR, PROFILER_SCRIPTS_ROOT


def get_latest_ops_log_filename(output_logs_subdir):
    runDate = sorted(os.listdir(PROFILER_OUTPUT_DIR / output_logs_subdir))[-1]
    filename = PROFILER_OUTPUT_DIR / output_logs_subdir / runDate / f"ops_perf_results_{runDate}.csv"
    return filename


def post_process_ops_log(output_logs_subdir, columns, sum_vals=True, op_name="", has_signposts=False):
    filename = get_latest_ops_log_filename(output_logs_subdir)
    df = pd.read_csv(filename)

    if has_signposts:
        # there are explicit start and stop points in the model we want to measure between
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        start = markers[markers == "start"].index[0]
        stop = markers[markers == "stop"].index[0]
        df = df.iloc[start + 1 : stop]
    if op_name != "":
        df = df[df["OP CODE"] == op_name]

    results = {}
    for col in columns:
        df_filtered = df[df[col] != "-"]
        if sum_vals:
            results[col] = df_filtered[col].astype(float).sum()
        else:
            results[col] = df_filtered[col].astype(float).to_numpy()
    return results


def run_device_profiler(command, output_logs_subdir):
    output_logs_dir = PROFILER_OUTPUT_DIR / output_logs_subdir
    profiler_cmd = f"python -m tracy -p -r -o {output_logs_dir} -t 5000 -m {command}"
    subprocess.run([profiler_cmd], shell=True, check=True)


def get_samples_per_s(time_ns, num_samples):
    ns_to_s = 1e-9
    return 1 / (time_ns * ns_to_s) * num_samples

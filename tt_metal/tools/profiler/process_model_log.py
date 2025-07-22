# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path
import pandas as pd
from loguru import logger

from tt_metal.tools.profiler.common import PROFILER_ARTIFACTS_DIR, PROFILER_SCRIPTS_ROOT, generate_reports_folder


def get_profiler_folder(output_logs_subdir):
    return PROFILER_ARTIFACTS_DIR / output_logs_subdir


def get_latest_ops_log_filename(output_logs_subdir):
    output_report_dir = generate_reports_folder(get_profiler_folder(output_logs_subdir))
    runDate = sorted(os.listdir(output_report_dir))[-1]
    filename = output_report_dir / runDate / f"ops_perf_results_{runDate}.csv"
    return filename


def post_process_ops_log(output_logs_subdir, columns=None, sum_vals=True, op_name="", has_signposts=False):
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
    if columns:
        assert (
            type(columns) == list
        ), f"Bad columns name type, requested columns should be of type list but {type(columns)} was provided"
        for col in columns:
            df_filtered = df[df[col] != "-"]
            if sum_vals:
                results[col] = df_filtered[col].astype(float).sum()
            else:
                results[col] = df_filtered[col].astype(float).to_numpy()
    else:
        results = df
    return results


def run_device_profiler(command, output_logs_subdir, check_test_return_code=True, device_analysis_types=[]):
    output_profiler_dir = get_profiler_folder(output_logs_subdir)
    check_return_code = ""
    device_analysis_opt = ""
    if check_test_return_code:
        check_return_code = "--check-exit-code"
    if device_analysis_types:
        assert type(device_analysis_types) == list
        device_analysis_opt_list = [f" -a {analysis}" for analysis in device_analysis_types]
        device_analysis_opt = "".join(device_analysis_opt_list)
    profiler_cmd = f"python3 -m tracy -p -r -o {output_profiler_dir} {check_return_code} {device_analysis_opt} -t 5000 -m {command}"
    logger.info(profiler_cmd)
    subprocess.run([profiler_cmd], shell=True, check=True)


def get_samples_per_s(time_ns, num_samples):
    ns_to_s = 1e-9
    return 1 / (time_ns * ns_to_s) * num_samples

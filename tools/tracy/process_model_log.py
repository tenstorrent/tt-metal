# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import subprocess
from pathlib import Path
import pandas as pd
from loguru import logger

from tracy.common import PROFILER_ARTIFACTS_DIR, generate_reports_folder, PROFILER_DEFAULT_OP_SUPPORT_COUNT


def get_profiler_folder(output_logs_subdir):
    return PROFILER_ARTIFACTS_DIR / output_logs_subdir


def get_latest_ops_log_filename(output_logs_subdir=""):
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


def _build_profiler_cmd(
    command,
    output_profiler_dir,
    check_test_return_code,
    device_analysis_types,
    python_post_process,
    capture_perf_counters_groups,
    sum_profiling,
    op_support_count,
    is_command_binary_exe,
):
    check_return_code = ""
    device_analysis_opt = ""
    python_post_process_opt = ""
    capture_perf_counters_opt = ""
    sum_profiling_opt = ""
    op_support_count_opt = ""
    if python_post_process:
        python_post_process_opt = "-r"
    if sum_profiling:
        sum_profiling_opt = "--enable-sum-profiling"
    if op_support_count != PROFILER_DEFAULT_OP_SUPPORT_COUNT:
        op_support_count_opt = f"--op-support-count {op_support_count}"
    if check_test_return_code:
        check_return_code = "--check-exit-code"
    if device_analysis_types:
        assert type(device_analysis_types) == list
        device_analysis_opt_list = [f" -a {analysis}" for analysis in device_analysis_types]
        device_analysis_opt = "".join(device_analysis_opt_list)
    if capture_perf_counters_groups:
        assert type(capture_perf_counters_groups) == list
        capture_perf_counters_opt = "--profiler-capture-perf-counters=" + ",".join(capture_perf_counters_groups)

    cmd_call = "" if is_command_binary_exe else "-m"
    # Quote the embedded command so that arguments like `-k "expr with spaces"` survive through the outer shell
    quoted_command = command if is_command_binary_exe else shlex.quote(command)
    return f"python3 -m tracy -p {python_post_process_opt} -o {output_profiler_dir} {check_return_code} {device_analysis_opt} {sum_profiling_opt} {op_support_count_opt} {capture_perf_counters_opt} -t 5000 {cmd_call} {quoted_command}"


def _merge_l1_pass2_csv(pass1_csv_path, pass2_csv_path):
    """Merge L1 bank 1 columns from pass 2 CSV into the pass 1 CSV."""
    df1 = pd.read_csv(pass1_csv_path)
    df2 = pd.read_csv(pass2_csv_path)

    # Identify L1_1-specific columns (present in pass2 but not in pass1)
    l1_1_cols = [c for c in df2.columns if c not in df1.columns]
    if not l1_1_cols:
        logger.warning("No L1 bank 1 columns found in pass 2 CSV to merge")
        return

    merge_keys = ["GLOBAL CALL COUNT", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID"]
    merged = df1.merge(df2[merge_keys + l1_1_cols], on=merge_keys, how="left")
    merged.to_csv(pass1_csv_path, index=False)
    logger.info(f"Merged {len(l1_1_cols)} L1 bank 1 columns from pass 2 into {pass1_csv_path}")


def run_device_profiler(
    command,
    output_logs_subdir,
    check_test_return_code=True,
    device_analysis_types=[],
    python_post_process=True,
    capture_perf_counters_groups=[],
    sum_profiling=False,
    # Default op support count is multiplied by 1.333 because previously the profiler would reserve space
    # for approximately 33% more ops than the default. Several model tests call this function and rely on
    # this extra space to ensure that all ops are captured by the profiler. Now that the profiler doesn't
    # reserve this extra space, we multiply the default by 1.333 to ensure that these model tests continue
    # to capture all ops.
    op_support_count=int(PROFILER_DEFAULT_OP_SUPPORT_COUNT * 1.333),
    is_command_binary_exe=False,
):
    # Check if both L1 banks are requested (requires two-pass execution)
    needs_l1_two_pass = False
    if capture_perf_counters_groups:
        groups_lower = [g.lower() for g in capture_perf_counters_groups]
        has_l1_0 = "l1_0" in groups_lower or "all" in groups_lower
        has_l1_1 = "l1_1" in groups_lower
        needs_l1_two_pass = has_l1_0 and has_l1_1

    if needs_l1_two_pass:
        # Pass 1: all groups except l1_1
        groups_pass1 = [g for g in capture_perf_counters_groups if g.lower() != "l1_1"]
        output_profiler_dir = get_profiler_folder(output_logs_subdir)
        profiler_cmd = _build_profiler_cmd(
            command,
            output_profiler_dir,
            check_test_return_code,
            device_analysis_types,
            python_post_process,
            groups_pass1,
            sum_profiling,
            op_support_count,
            is_command_binary_exe,
        )
        logger.info(f"L1 two-pass: running pass 1 (L1 bank 0) — {profiler_cmd}")
        subprocess.run([profiler_cmd], shell=True, check=True)

        # Pass 2: replace l1_0/all with l1_1
        groups_pass2 = [g for g in capture_perf_counters_groups if g.lower() not in ("l1_0", "all")]
        if "all" in groups_lower:
            # "all" includes fpu,pack,unpack,l1_0,instrn — replace l1_0 with l1_1 and keep the rest
            groups_pass2 = ["fpu", "pack", "unpack", "l1_1", "instrn"]
        elif "l1_1" not in groups_pass2:
            groups_pass2.append("l1_1")

        pass2_subdir = f"{output_logs_subdir}_l1_pass2"
        output_profiler_dir_pass2 = get_profiler_folder(pass2_subdir)
        profiler_cmd_pass2 = _build_profiler_cmd(
            command,
            output_profiler_dir_pass2,
            check_test_return_code,
            device_analysis_types,
            python_post_process,
            groups_pass2,
            sum_profiling,
            op_support_count,
            is_command_binary_exe,
        )
        logger.info(f"L1 two-pass: running pass 2 (L1 bank 1) — {profiler_cmd_pass2}")
        subprocess.run([profiler_cmd_pass2], shell=True, check=True)

        # Merge L1_1 columns from pass 2 into pass 1 CSV
        if python_post_process:
            pass1_csv = get_latest_ops_log_filename(output_logs_subdir)
            pass2_csv = get_latest_ops_log_filename(pass2_subdir)
            _merge_l1_pass2_csv(pass1_csv, pass2_csv)
    else:
        output_profiler_dir = get_profiler_folder(output_logs_subdir)
        profiler_cmd = _build_profiler_cmd(
            command,
            output_profiler_dir,
            check_test_return_code,
            device_analysis_types,
            python_post_process,
            capture_perf_counters_groups,
            sum_profiling,
            op_support_count,
            is_command_binary_exe,
        )
        logger.info(profiler_cmd)
        subprocess.run([profiler_cmd], shell=True, check=True)


def get_samples_per_s(time_ns, num_samples):
    ns_to_s = 1e-9
    return 1 / (time_ns * ns_to_s) * num_samples

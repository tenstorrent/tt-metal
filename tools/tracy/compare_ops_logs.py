#! /usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
import pandas as pd
import click
from loguru import logger

from tracy.common import PROFILER_LOGS_DIR, PROFILER_CPP_DEVICE_PERF_REPORT
from tracy.process_model_log import get_latest_ops_log_filename


def compare_ops_logs(python_ops_perf_report=None, cpp_ops_perf_report=None, only_compare_op_ids=False):
    if not python_ops_perf_report:
        python_ops_perf_report = get_latest_ops_log_filename()
    if not cpp_ops_perf_report:
        cpp_ops_perf_report = PROFILER_LOGS_DIR / PROFILER_CPP_DEVICE_PERF_REPORT

    logger.info(f"Comparing {python_ops_perf_report} with {cpp_ops_perf_report}")

    try:
        python_df = pd.read_csv(python_ops_perf_report)
    except pd.errors.EmptyDataError:
        python_df = pd.DataFrame()

    try:
        cpp_df = pd.read_csv(cpp_ops_perf_report)
    except pd.errors.EmptyDataError:
        cpp_df = pd.DataFrame()

    if python_df.empty or cpp_df.empty:
        logger.info("Skipping comparison because at least one of the ops perf reports is empty")
        return

    common_columns = python_df.columns.intersection(cpp_df.columns)
    if common_columns.empty:
        logger.info("Skipping comparison because the two ops perf reports have no common columns")
        return

    assert common_columns.size >= 12, f"Only {common_columns.size} common columns found"

    python_df_filtered = python_df[common_columns].astype(float)
    assert (
        "DEVICE FW DURATION [ns]" in python_df_filtered.columns
    ), f"'DEVICE FW DURATION [ns]' column not found in {python_ops_perf_report}"
    python_df_filtered = python_df_filtered[python_df_filtered["DEVICE FW DURATION [ns]"].notna()]

    cpp_df_filtered = cpp_df[common_columns].astype(float)
    cpp_df_filtered = cpp_df_filtered[cpp_df_filtered["GLOBAL CALL COUNT"] != 0]

    sort_columns = ["GLOBAL CALL COUNT"]
    optional_sort_columns = ["METAL TRACE ID", "METAL TRACE REPLAY SESSION ID"]

    for col in optional_sort_columns:
        if col in common_columns:
            sort_columns.append(col)

    python_df_sorted = python_df_filtered.sort_values(by=sort_columns, na_position="first").reset_index(drop=True)
    cpp_df_sorted = cpp_df_filtered.sort_values(by=sort_columns, na_position="first").reset_index(drop=True)

    if only_compare_op_ids:
        python_df_sorted = python_df_sorted[sort_columns]
        cpp_df_sorted = cpp_df_sorted[sort_columns]

    are_equal = python_df_sorted.equals(cpp_df_sorted)
    if are_equal:
        logger.info("Ops perf reports are equal")
    else:
        logger.info("Ops perf reports are not equal")
        logger.info(f"{python_ops_perf_report}: {python_df_sorted}")
        logger.info(f"{cpp_ops_perf_report}: {cpp_df_sorted}")
        sys.exit(1)


@click.command()
@click.option("--only-compare-op-ids", is_flag=True, help="Only compare op ids")
@click.argument("python_ops_perf_report", type=click.Path(), required=False)
@click.argument("cpp_ops_perf_report", type=click.Path(), required=False)
def cli(only_compare_op_ids, python_ops_perf_report, cpp_ops_perf_report):
    compare_ops_logs(python_ops_perf_report, cpp_ops_perf_report, only_compare_op_ids)


if __name__ == "__main__":
    cli()

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

    python_df_filtered = python_df[common_columns].copy()
    assert (
        "DEVICE FW DURATION [ns]" in python_df_filtered.columns
    ), f"'DEVICE FW DURATION [ns]' column not found in {python_ops_perf_report}"
    python_df_filtered = python_df_filtered[python_df_filtered["DEVICE FW DURATION [ns]"].notna()]

    cpp_df_filtered = cpp_df[common_columns].copy()
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

    python_df_compare = python_df_sorted.copy()
    cpp_df_compare = cpp_df_sorted.copy()

    # Normalize dtypes for consistent comparison (only needed because pandas .equals() checks dtypes)
    # This handles cases where same values are stored as int vs float (e.g., 1024 vs 1024.0)
    for col in common_columns:
        if col in python_df_compare.columns and col in cpp_df_compare.columns:
            # Try to convert to numeric - if successful, normalize to float64 for both
            # If not numeric, ensure both are same dtype (object for strings)
            try:
                python_numeric = pd.to_numeric(python_df_compare[col], errors="raise")
                cpp_numeric = pd.to_numeric(cpp_df_compare[col], errors="raise")
                # Both are numeric - normalize to float64
                python_df_compare[col] = python_numeric.astype("float64")
                cpp_df_compare[col] = cpp_numeric.astype("float64")
            except (ValueError, TypeError):
                # Not numeric - ensure both are object type for consistent comparison
                python_df_compare[col] = python_df_compare[col].astype("object")
                cpp_df_compare[col] = cpp_df_compare[col].astype("object")

    ignored_zero_latency_counts: dict[str, int] = {}
    latency_columns = {
        "OP TO OP LATENCY [ns]",
        "OP TO OP LATENCY BR/NRISC START [ns]",
    }

    if not only_compare_op_ids:
        # Only do latency comparison if dataframes have the same shape (same indices)
        if python_df_compare.shape == cpp_df_compare.shape:
            for column in latency_columns:
                if column not in python_df_compare.columns or column not in cpp_df_compare.columns:
                    continue
                cpp_zero_mask = cpp_df_compare[column] == 0
                diff_mask = cpp_zero_mask & (python_df_compare[column] != cpp_df_compare[column])
                ignored_count = int(diff_mask.sum())
                if ignored_count > 0:
                    python_df_compare.loc[diff_mask, column] = cpp_df_compare.loc[diff_mask, column]
                    ignored_zero_latency_counts[column] = ignored_count

    if python_df_compare.equals(cpp_df_compare):
        if ignored_zero_latency_counts:
            logger.info(
                f"Ops perf reports only differ where CPP latency columns are zero; "
                f"ignored per column: {ignored_zero_latency_counts}"
            )
        else:
            logger.info("Ops perf reports are equal")
    else:
        if ignored_zero_latency_counts:
            logger.info(f"Ignored differences due to zero CPP latency values: {ignored_zero_latency_counts}")
        logger.error("Ops perf reports are not equal")

        if python_df_sorted.shape != cpp_df_sorted.shape:
            logger.error(f"Shape mismatch: python report {python_df_sorted.shape}, cpp report {cpp_df_sorted.shape}")
        else:
            diff_df = python_df_sorted.compare(cpp_df_sorted, align_axis=0, keep_shape=False, keep_equal=False)
            differing_rows = diff_df.index.nunique()
            differing_columns = sorted({col if isinstance(col, str) else col[0] for col in diff_df.columns})
            logger.error(
                f"Rows with differences: {differing_rows} / {python_df_sorted.shape[0]}. "
                f"Columns differing: {differing_columns}"
            )

            differing_cells = []
            for row_idx, row in diff_df.iterrows():
                for col in differing_columns:
                    if ("self", col) in row.index and ("other", col) in row.index:
                        val_a = row[("self", col)]
                        val_b = row[("other", col)]
                        if pd.notna(val_a) or pd.notna(val_b):
                            differing_cells.append((row_idx, col, val_a, val_b))
                if len(differing_cells) >= 10:
                    break
            logger.error(f"First differing cells (row, column, python, cpp): {differing_cells}")

        logger.error(f"First rows of python report:\n{python_df_sorted.head()}")
        logger.error(f"First rows of cpp report:\n{cpp_df_sorted.head()}")
        sys.exit(1)


@click.command()
@click.option("--only-compare-op-ids", is_flag=True, help="Only compare op ids")
@click.argument("python_ops_perf_report", type=click.Path(), required=False)
@click.argument("cpp_ops_perf_report", type=click.Path(), required=False)
def cli(only_compare_op_ids, python_ops_perf_report, cpp_ops_perf_report):
    compare_ops_logs(python_ops_perf_report, cpp_ops_perf_report, only_compare_op_ids)


if __name__ == "__main__":
    cli()

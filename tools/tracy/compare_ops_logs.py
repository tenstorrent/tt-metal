#! /usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import sys
import pandas as pd
import numpy as np
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

    # Historically, these reports only contained numeric columns, so the comparison
    # cast everything to float for stable equality. Newer reports include a few
    # string columns (e.g. DEVICE ARCH). We keep those as strings and only cast
    # truly-numeric columns to float.
    python_subset = python_df[common_columns].copy()
    cpp_subset = cpp_df[common_columns].copy()

    def _is_numeric_column(col: str) -> bool:
        # Consider "" and "-" as missing placeholders.
        a = python_subset[col].replace({"": np.nan, "-": np.nan})
        b = cpp_subset[col].replace({"": np.nan, "-": np.nan})
        a_num = pd.to_numeric(a, errors="coerce")
        b_num = pd.to_numeric(b, errors="coerce")
        # Numeric if every non-missing value is convertible on both sides.
        return bool((a.isna() | a_num.notna()).all() and (b.isna() | b_num.notna()).all())

    numeric_columns = [col for col in common_columns if _is_numeric_column(col)]
    non_numeric_columns = [col for col in common_columns if col not in set(numeric_columns)]

    python_df_filtered = python_subset.copy()
    cpp_df_filtered = cpp_subset.copy()

    for col in numeric_columns:
        python_df_filtered[col] = pd.to_numeric(python_df_filtered[col], errors="coerce").astype(float)
        cpp_df_filtered[col] = pd.to_numeric(cpp_df_filtered[col], errors="coerce").astype(float)

    for col in non_numeric_columns:
        python_df_filtered[col] = python_df_filtered[col].fillna("").astype(str)
        cpp_df_filtered[col] = cpp_df_filtered[col].fillna("").astype(str)
    assert (
        "DEVICE FW DURATION [ns]" in python_df_filtered.columns
    ), f"'DEVICE FW DURATION [ns]' column not found in {python_ops_perf_report}"
    python_df_filtered = python_df_filtered[python_df_filtered["DEVICE FW DURATION [ns]"].notna()]

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
        logger.error("Ops perf reports are not equal")
        logger.error(f"python report shape: {python_df_sorted.shape}, cpp report shape: {cpp_df_sorted.shape}")

        if python_df_sorted.shape != cpp_df_sorted.shape:
            logger.error(f"First rows of python report:\n{python_df_sorted.head()}")
            logger.error(f"First rows of cpp report:\n{cpp_df_sorted.head()}")
            sys.exit(1)

        # Show a compact diff (CI logs truncate full DataFrame dumps).
        diff_df = python_df_sorted.compare(cpp_df_sorted, keep_shape=False, keep_equal=False)
        differing_rows = diff_df.index.nunique()

        # pandas may return columns as MultiIndex in either of these shapes:
        #   (col, "self"/"other")  [pandas >= 2.0 default]
        #   ("self"/"other", col)  [older/alternate]
        # Detect which and extract the real column names.
        differing_columns: list[str] = []
        side_first = False
        if isinstance(diff_df.columns, pd.MultiIndex) and diff_df.columns.nlevels == 2:
            level0 = set(map(str, diff_df.columns.get_level_values(0)))
            level1 = set(map(str, diff_df.columns.get_level_values(1)))
            if level0.issubset({"self", "other"}):
                side_first = True
                differing_columns = sorted(set(map(str, diff_df.columns.get_level_values(1))))
            elif level1.issubset({"self", "other"}):
                side_first = False
                differing_columns = sorted(set(map(str, diff_df.columns.get_level_values(0))))
            else:
                # Unexpected MultiIndex layout; fall back to stringifying level 0.
                differing_columns = sorted(set(map(str, diff_df.columns.get_level_values(0))))
        else:
            differing_columns = sorted(map(str, diff_df.columns))
        logger.error(
            f"Rows with differences: {differing_rows} / {python_df_sorted.shape[0]}. "
            f"Columns differing: {differing_columns}"
        )

        differing_cells = []
        for row_idx in list(diff_df.index)[:10]:
            for col in differing_columns:
                if isinstance(diff_df.columns, pd.MultiIndex) and diff_df.columns.nlevels == 2:
                    if side_first:
                        key_self = ("self", col)
                        key_other = ("other", col)
                    else:
                        key_self = (col, "self")
                        key_other = (col, "other")
                    val_a = diff_df.loc[row_idx, key_self] if key_self in diff_df.columns else None
                    val_b = diff_df.loc[row_idx, key_other] if key_other in diff_df.columns else None
                else:
                    # Non-MultiIndex fallback: can't attribute to python/cpp side.
                    val_a = diff_df.loc[row_idx, col] if col in diff_df.columns else None
                    val_b = None

                differing_cells.append((int(row_idx), col, val_a, val_b))
                if len(differing_cells) >= 10:
                    break
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

import argparse
import json
import time
from collections import defaultdict

import pandas as pd
from loguru import logger

from tt_metal.tools.profiler.common import clear_profiler_runtime_artifacts
from tt_metal.tools.profiler.process_model_log import (
    get_latest_ops_log_filename,
    get_samples_per_s,
    post_process_ops_log,
    run_device_profiler,
)


def post_process_ops_log_detailed(
    filename,
    columns,
    sum_vals=True,
    op_name="",
    has_signposts=False,
    detailed=False,
    warmup_iters=0,
):
    # Load latest ops log
    df = pd.read_csv(filename)

    # Trim to region between 'signpost' markers if present
    if has_signposts:
        markers = df[df["OP TYPE"] == "signpost"]["OP CODE"]
        try:
            start = markers[markers == "start"].index[0]
            stop = markers[markers == "stop"].index[0]
        except IndexError:
            raise ValueError("Missing 'start' or 'stop' signpost in log.")
        df = df.iloc[start + 1 : stop]

    # Filter by specific op code if provided
    if op_name:
        df = df[df["OP CODE"] == op_name]

    # Group by DEVICE ID and sort groups by DEVICE FW START CYCLE
    grouped = df.groupby("DEVICE ID")
    sorted_groups = sorted(grouped, key=lambda x: x[1]["DEVICE FW START CYCLE"].iloc[0])

    print(f"Found {len(sorted_groups)} DEVICE ID groups in the log.")

    # Interleave rows from all DEVICE ID groups (round-robin merge)
    interleaved_rows = [group.iloc[[i]] for i in range(len(sorted_groups[0][1])) for _, group in sorted_groups]
    df = pd.concat(interleaved_rows, ignore_index=True)

    # Skip warmup iterations if specified
    if warmup_iters > 0:
        df = df.iloc[warmup_iters:]

    # Process specified columns
    results = {}
    for col in columns:
        df_filtered = df[df[col] != "-"]
        df_filtered[col] = df_filtered[col].astype(float)

        if sum_vals:
            results[col] = df_filtered[col].sum()
        else:
            results[col] = df_filtered[col].to_numpy()

        if detailed:
            results[f"AVG {col}"] = df_filtered[col].mean()
            results[f"MIN {col}"] = df_filtered[col].min()
            results[f"MAX {col}"] = df_filtered[col].max()
            results[f"STD {col}"] = df_filtered[col].std()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process operations log.")
    parser.add_argument("--filename", type=str, required=True, help="filename.")
    parser.add_argument("--op_name", type=str, default="AllReduceAsync", help="Operation name to filter by.")
    parser.add_argument("--warmup_iters", type=int, default=5 * 32, help="Number of warmup iterations to skip.")

    args = parser.parse_args()

    cols = ["DEVICE KERNEL"]
    duration_cols = [col + " DURATION [ns]" for col in cols]

    results = {}
    for d_col in duration_cols:
        results[f"AVG {d_col}"] = 0
        results[f"MIN {d_col}"] = float("inf")
        results[f"MAX {d_col}"] = -float("inf")
        results[f"STD {d_col}"] = 0

    r = post_process_ops_log_detailed(
        filename=args.filename,
        columns=duration_cols,
        sum_vals=True,
        op_name=args.op_name,
        has_signposts=False,
        detailed=True,
        warmup_iters=args.warmup_iters,
    )

    for d_col in duration_cols:
        results[f"AVG {d_col}"] = r[f"AVG {d_col}"]
        results[f"MIN {d_col}"] = r[f"MIN {d_col}"]
        results[f"MAX {d_col}"] = r[f"MAX {d_col}"]
        results[f"STD {d_col}"] = r[f"STD {d_col}"]

    post_processed_results = defaultdict(dict)
    for col, d_col in zip(cols, duration_cols):
        post_processed_results[col]["AVG"] = results[f"AVG {d_col}"]
        post_processed_results[col]["MIN"] = results[f"MIN {d_col}"]
        post_processed_results[col]["MAX"] = results[f"MAX {d_col}"]
        post_processed_results[col]["STD"] = results[f"STD {d_col}"]

    logger.info(f"\nPerformance statistics for op: {args.op_name}" f"\n{json.dumps(post_processed_results, indent=4)}")

    measured_min = post_processed_results[cols[0]]["MIN"]
    measured_max = post_processed_results[cols[0]]["MAX"]
    measured_avg = post_processed_results[cols[0]]["AVG"]
    measured_std = post_processed_results[cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000
    print(f"measured_avg_us = {measured_avg_us:.2f} us")

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import time
from collections import defaultdict

import pandas as pd
from loguru import logger

from models.perf.perf_utils import process_perf_results
from tt_metal.tools.profiler.common import clear_profiler_runtime_artifacts
from tt_metal.tools.profiler.process_model_log import (
    get_latest_ops_log_filename,
    get_samples_per_s,
    post_process_ops_log,
    run_device_profiler,
)


def run_device_perf(
    command,
    subdir,
    num_iterations,
    cols,
    batch_size,
    op_name="",
    has_signposts=False,
    device_analysis_types=["device_kernel_duration"],
):
    duration_cols = [col + " DURATION [ns]" for col in cols]
    samples_cols = [col + " SAMPLES/S" for col in cols]

    clear_profiler_runtime_artifacts()

    results = {}
    for d_col in duration_cols:
        results[f"AVG {d_col}"] = 0
        results[f"MIN {d_col}"] = float("inf")
        results[f"MAX {d_col}"] = -float("inf")

    for _ in range(num_iterations):
        run_device_profiler(command, subdir, device_analysis_types)
        r = post_process_ops_log(subdir, duration_cols, op_name=op_name, has_signposts=has_signposts)
        for d_col in duration_cols:
            results[f"AVG {d_col}"] += r[d_col]
            results[f"MIN {d_col}"] = min(results[f"MIN {d_col}"], r[d_col])
            results[f"MAX {d_col}"] = max(results[f"MAX {d_col}"], r[d_col])

    post_processed_results = {}
    for s_col, d_col in zip(samples_cols, duration_cols):
        post_processed_results[f"AVG {s_col}"] = get_samples_per_s(results[f"AVG {d_col}"] / num_iterations, batch_size)
        post_processed_results[f"MIN {s_col}"] = get_samples_per_s(results[f"MAX {d_col}"], batch_size)
        post_processed_results[f"MAX {s_col}"] = get_samples_per_s(results[f"MIN {d_col}"], batch_size)
        post_processed_results[f"AVG {d_col}"] = results[f"AVG {d_col}"] / num_iterations
        post_processed_results[f"MIN {d_col}"] = results[f"MIN {d_col}"]
        post_processed_results[f"MAX {d_col}"] = results[f"MAX {d_col}"]

    logger.info(
        f"\nTest: {command}"
        f"\nPerformance statistics over {num_iterations} iterations"
        f"\n{json.dumps(post_processed_results, indent=4)}"
    )
    return post_processed_results


# TODO: Move into process_model_log.py (#18698)
def post_process_ops_log_detailed(
    output_logs_subdir, columns, sum_vals=True, op_name="", has_signposts=False, detailed=False, warmup_iters=0
):
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

    # group by DEVICE ID
    df = df.groupby("DEVICE ID")
    # now sort the list of df by the DEVICE FW START CYCLE
    df = sorted(df, key=lambda x: x[1]["DEVICE FW START CYCLE"].iloc[0])

    # Convert list of tuples to list of dataframes
    dfs = [group for _, group in df]

    # concatenate the list of df into a single df by interleaving the rows
    df = pd.concat([df.iloc[[i]] for i in range(len(dfs[0])) for df in dfs], ignore_index=True)

    if warmup_iters > 0:
        df = df.iloc[warmup_iters:]

    results = {}
    for col in columns:
        df_filtered = df[df[col] != "-"]
        if sum_vals:
            results[col] = df_filtered[col].astype(float).sum()
        else:
            results[col] = df_filtered[col].astype(float).to_numpy()

        if detailed:
            results[f"AVG {col}"] = df_filtered[col].astype(float).mean()
            results[f"MIN {col}"] = df_filtered[col].astype(float).min()
            results[f"MAX {col}"] = df_filtered[col].astype(float).max()
            results[f"STD {col}"] = df_filtered[col].astype(float).std()

    return results


def run_device_perf_detailed(
    command, subdir, cols, op_name="", has_signposts=False, warmup_iters=0, device_analysis_types=None
):
    duration_cols = [col + " DURATION [ns]" for col in cols]

    clear_profiler_runtime_artifacts()

    results = {}
    for d_col in duration_cols:
        results[f"AVG {d_col}"] = 0
        results[f"MIN {d_col}"] = float("inf")
        results[f"MAX {d_col}"] = -float("inf")
        results[f"STD {d_col}"] = 0

    if device_analysis_types is None:
        device_analysis_types = ["device_kernel_duration"]

    run_device_profiler(command, subdir, device_analysis_types=device_analysis_types)
    r = post_process_ops_log_detailed(
        subdir, duration_cols, op_name=op_name, has_signposts=has_signposts, detailed=True, warmup_iters=warmup_iters
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

    logger.info(
        f"\nTest: {command}"
        f"\nPerformance statistics for op: {op_name}"
        f"\n{json.dumps(post_processed_results, indent=4)}"
    )
    return post_processed_results


def check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=False):
    expected_results = {}
    failed = False
    for col, expected_perf in expected_perf_cols.items():
        lower_threshold = (1 - margin) * expected_perf
        upper_threshold = (1 + margin) * expected_perf
        expected_results.update(
            {
                f"Lower Threshold {col}": lower_threshold,
                f"Upper Threshold {col}": upper_threshold,
            }
        )
        passing = lower_threshold <= post_processed_results[col] <= upper_threshold
        if not passing:
            failed = True
            logger.error(
                f"{col} {post_processed_results[col]} is outside of expected range ({lower_threshold}, {upper_threshold})"
            )
    if assert_on_fail:
        assert not failed, "Some performance metrics are outside of expected range, see above for details."
    return expected_results


def prep_device_perf_report(
    model_name: str,
    batch_size: int,
    post_processed_results: dict,
    expected_results: dict,
    comments: str,
):
    today = time.strftime("%Y_%m_%d")

    def write_dict_to_file(csv_path, dict_res):
        columns = ", ".join([str(d) for d in dict_res.keys()])
        values = ", ".join([d for d in dict_res.values()])

        with open(csv_path, "w") as csvfile:
            csvfile.write(columns)
            csvfile.write("\n")
            csvfile.write(values)

    formatted_results = {}
    for key, value in post_processed_results.items():
        formatted_results[key] = "{:.4f}".format(value)
        lower = expected_results.get(f"Lower Threshold {key}", None)
        if lower is not None:
            formatted_results[f"Lower Threshold {key}"] = "{:.4f}".format(lower)
        upper = expected_results.get(f"Upper Threshold {key}", None)
        if upper is not None:
            formatted_results[f"Upper Threshold {key}"] = "{:.4f}".format(upper)
    dict_res = {
        "Model": model_name,
        "Setting": comments,
        "Batch": str(batch_size),
        **formatted_results,
    }

    csv_file = f"device_perf_{model_name}_{comments}_{today}.csv"
    write_dict_to_file(csv_file, dict_res)


def check_device_perf_results(fname, expected_cols, check_cols):
    cols, merge_res = process_perf_results(fname, expected_cols)
    visited_models = []
    slow_measured = {col: [] for col in check_cols}
    fast_measured = {col: [] for col in check_cols}
    for models_info in merge_res:
        models_info = [item.strip() for item in models_info]
        dict_info = {name: value for name, value in zip(cols, models_info)}
        logger.info(dict_info)
        model_name = f"{dict_info['Model']}_{dict_info['Setting']}"
        visited_models.append(model_name)
        for col in check_cols:
            model_lower_threshold_col = float(dict_info[f"Lower Threshold {col}"])
            model_upper_threshold_col = float(dict_info[f"Upper Threshold {col}"])
            model_measured_col = float(dict_info[col])
            if model_measured_col > model_upper_threshold_col:
                fast_measured[col].append((model_name, model_measured_col, model_upper_threshold_col))
                logger.error(
                    f"{model_name} {col} is faster than expected with {model_measured_col}, max expected {model_upper_threshold_col}. Please update perf targets with latest expected perf"
                )
            if model_measured_col < model_lower_threshold_col:
                slow_measured[col].append((model_name, model_measured_col, model_lower_threshold_col))
                logger.error(
                    f"{model_name} {col} is too slow with {model_measured_col}, min expected {model_lower_threshold_col}."
                )

    assert any(
        len(slow) == 0 for slow in slow_measured.values()
    ), f"Some model(s) {', '.join(slow_measured.keys())} are too slow, see above for details. {slow_measured}"
    for col in slow_measured:
        assert (
            len(slow_measured[col]) == 0
        ), f"Some model(s) {col} are too slow, see above for details on slow models: {slow_measured[col]}"

    assert any(
        len(fast) == 0 for fast in fast_measured.values()
    ), f"Some model(s) {', '.join(slow_measured.keys())} are faster than expected, see above for details. {fast_measured}"
    for col in fast_measured:
        assert (
            len(fast_measured[col]) == 0
        ), f"Some model(s) {col} are faster than expected, see above for details on fast models: {slow_measured[col]}"

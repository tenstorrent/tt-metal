# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile

import pandas as pd
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.tt_transformers.tests.test_utils import merge_device_rows

# TP-collective ops: depend on TP=4 topology/bandwidth → take from 2x4
# Everything else: depends on SP=8 tokens-per-chip and 8 experts/device → take from 8x1
#
# Key insight for Matmul:
#   8x1: 64 experts, 8/device, dispatch_group_size=8, M=12800/expert, TP=1 (N=2048) → correct M+count, wrong N
#   2x4: 256 experts, 32/device, dispatch_group_size=2, M=800/expert,  TP=4 (N=512)  → correct N, wrong M+count
#   8x4: 256 experts,  8/device, dispatch_group_size=8, M=12800/expert, TP=4 (N=512) → target
#
#   8x1 M per expert (dispatch_group_size=8 × max_tokens=1600) = 12800 matches 8x4 exactly.
#   2x4 M per expert (dispatch_group_size=2 × max_tokens=400) = 800 — 16x smaller, only 32/8=4x
#   more experts, so 2x4 total work is 4x LESS than 8x4 (scaling by ÷4 makes it 16x too small).
#   Use 8x1 for matmuls: M and expert count match 8x4; only N differs (2048 vs 512).

TP_OPS = {
    "ReduceScatterDeviceOperation",  # TP collective
    "AllGatherDeviceOperation",  # TP collective
    "AllBroadcastDeviceOperation",  # gate AllReduce on TP axis
    "PostCombineReduceDeviceOperation",  # contains ReduceScatter on TP axis
    "DeepseekGroupedGateDeviceOperation",  # only exists with TP>1
    "ConcatDeviceOperation",  # gate routing, TP-structured (8x1~0ms vs 8x4=0.28ms)
    "CopyDeviceOperation",  # TP path op, absent in 8x1
    "ReshapeViewDeviceOperation",  # 2x4=8x4=0.039ms vs 8x1=0.013ms
    "MaskedBincountDeviceOperation",  # 2x4=8x4=0.026ms vs 8x1=0.011ms
    "InterleavedToShardedDeviceOperation",  # 2x4=8x4=0.002ms
}


def load_merged_durations(csv_path: str, use_avg: bool = False) -> pd.Series:
    df = pd.read_csv(csv_path)
    df = df[df["OP TYPE"] == "tt_dnn_device"]
    if use_avg:
        # avg across devices per invocation, then sum across invocations
        # = total duration / num_devices
        num_devices = df["DEVICE ID"].nunique()
        return df.groupby("OP CODE")["DEVICE KERNEL DURATION [ns]"].sum() / num_devices
    df_merged = merge_device_rows(df)
    return df_merged.groupby("OP CODE")["DEVICE KERNEL DURATION [ns]"].sum()


def approximate_8x4_perf(csv_8x1: str, csv_2x4: str, csv_8x4: str = None, use_avg: bool = False) -> pd.DataFrame:
    """
    Approximate 8x4 MOE performance from cheaper 8x1 + 2x4 runs.

    - TP collective ops (ReduceScatter / AllGather / AllReduce / PostCombineReduce etc.):
      taken from 2x4 — same TP=4 topology, near-identical bandwidth.
    - Everything else (Dispatch / Combine / element-wise): taken from 8x1 —
      same SP=8 tokens-per-chip and same 8 experts/device as 8x4.
    """
    ops_8x1 = load_merged_durations(csv_8x1, use_avg=use_avg)
    ops_2x4 = load_merged_durations(csv_2x4, use_avg=use_avg)
    ops_8x4 = load_merged_durations(csv_8x4, use_avg=use_avg) if csv_8x4 else None

    all_ops = set(ops_8x4.index) if ops_8x4 is not None else set(ops_8x1.index) | set(ops_2x4.index)

    rows = []
    for op in sorted(all_ops):
        if op in TP_OPS:
            src = "2x4"
            approx_ns = ops_2x4.get(op, 0)
        else:
            approx_ns = ops_8x1.get(op, 0)
            if approx_ns == 0 and ops_2x4.get(op, 0) > 0:
                src = "2x4 (fallback)"
                approx_ns = ops_2x4.get(op, 0)
            else:
                src = "8x1"

        row = {"OP CODE": op, "source": src, "approx [ms]": approx_ns / 1e6}
        if ops_8x4 is not None:
            row["actual 8x4 [ms]"] = ops_8x4.get(op, float("nan")) / 1e6
        rows.append(row)

    df_result = pd.DataFrame(rows).sort_values("approx [ms]", ascending=False).reset_index(drop=True)

    if ops_8x4 is not None:
        df_result["diff [ms]"] = (df_result["approx [ms]"] - df_result["actual 8x4 [ms]"]).round(3)
        df_result["err [%]"] = (
            df_result["diff [ms]"] / df_result["actual 8x4 [ms]"].replace(0, float("nan")) * 100
        ).round(1)

    approx_total = df_result["approx [ms]"].sum()
    actual_total = df_result["actual 8x4 [ms]"].sum() if ops_8x4 is not None else None

    print(f"{'Approximation total:':25s} {approx_total:.3f} ms")
    if ops_8x4 is not None:
        err = (approx_total - actual_total) / actual_total * 100
        print(f"{'8x4 actual total:':25s} {actual_total:.3f} ms")
        print(f"{'Error:':25s} {err:+.1f}%")

    return df_result


def run_model_device_perf_test_with_merge(
    command: str,
    expected_device_perf_ns_per_iteration: float,
    subdir: str,
    model_name: str,
    num_iterations: int = 1,
    batch_size: int = 1,
    margin: float = 0.015,
    comments: str = "",
):
    """
    Run device performance test with multi-device row merging.

    Extends run_model_device_perf_test by adding device row merging for accurate
    multi-chip performance measurement. In multi-chip scenarios:
    - Collective operations (AllGather, ReduceScatter, AllReduce) use AVERAGE duration
    - Non-collective operations use MAX duration (critical path)

    Args:
        command: Command to execute for running the model
        expected_device_perf_ns_per_iteration: Expected device kernel duration in nanoseconds
        subdir: Subdirectory where performance logs will be stored
        model_name: Name of the model being tested
        num_iterations: Number of iterations (default: 1)
        batch_size: Batch size for the model (default: 1)
        margin: Acceptable performance margin as percentage (default: 0.015 = 1.5%)
        comments: Additional settings description for the report
    """
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"

    post_processed_results = run_device_perf(
        command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
    )

    # Apply multi-device row merging
    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    total_rows = len(df)
    signpost_rows = len(df[df["OP TYPE"] == "tt_signpost"])
    device_rows = len(df[df["OP TYPE"] == "tt_dnn_device"])

    logger.debug(f"CSV total rows: {total_rows}, signposts: {signpost_rows}, device ops: {device_rows}")

    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]

    logger.debug(f"Device rows before merge: {len(df)}")
    df_merged = merge_device_rows(df)
    logger.debug(f"Device rows after merge: {len(df_merged)}")

    if not df_merged.empty:
        merged_kernel_durations = df_merged["DEVICE KERNEL DURATION [ns]"].dropna().tolist()
        if merged_kernel_durations:
            merged_sum_ns = sum(merged_kernel_durations)
            logger.debug(f"Merged operations count: {len(merged_kernel_durations)}")
            logger.debug(f"Merged sum (ns): {merged_sum_ns} ({merged_sum_ns / 1000:.1f} us)")
            logger.debug(f"Original {inference_time_key}: {post_processed_results.get(inference_time_key, 'N/A')}")
            post_processed_results[inference_time_key] = merged_sum_ns

        durations = df_merged["DEVICE KERNEL DURATION [ns]"].fillna(0)
        op_codes = df_merged["OP CODE"].astype(str)
        is_matmul = op_codes.str.contains("Matmul", case=False, na=False)
        is_ccl = op_codes.str.contains("AllGather|ReduceScatter|AllReduce", na=False)
        is_sdpa = op_codes.str.contains("SDPA|ScaledDotProductAttention", na=False)
        is_other = ~(is_matmul | is_ccl | is_sdpa)

        matmul_ns = durations[is_matmul].sum()
        ccl_ns = durations[is_ccl].sum()
        sdpa_ns = durations[is_sdpa].sum()
        other_ns = durations[is_other].sum()
        total_ns = matmul_ns + ccl_ns + sdpa_ns + other_ns

        logger.info(f"Matmul time: {matmul_ns:>15,.0f} ns ({matmul_ns / 1e3:>10,.1f} us)")
        logger.info(f"CCL    time: {ccl_ns:>15,.0f} ns ({ccl_ns / 1e3:>10,.1f} us)")
        logger.info(f"SDPA   time: {sdpa_ns:>15,.0f} ns ({sdpa_ns / 1e3:>10,.1f} us)")
        logger.info(f"Other  time: {other_ns:>15,.0f} ns ({other_ns / 1e3:>10,.1f} us)")
        logger.info(f"Total  time: {total_ns:>15,.0f} ns ({total_ns / 1e3:>10,.1f} us)")

        other_breakdown = (
            df_merged.loc[is_other]
            .groupby(op_codes[is_other])["DEVICE KERNEL DURATION [ns]"]
            .sum()
            .sort_values(ascending=False)
        )
        if not other_breakdown.empty:
            logger.info("Other ops breakdown:")
            for op_code, dur_ns in other_breakdown.items():
                logger.info(f"  {op_code:<40} {dur_ns:>15,.0f} ns ({dur_ns / 1e3:>10,.1f} us)")

    expected_perf_cols = {inference_time_key: expected_device_perf_ns_per_iteration}
    expected_results = check_device_perf(
        post_processed_results, margin=margin, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=model_name,
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=comments,
    )


def run_approx_galaxy_moe_perf(
    command_8x1: str,
    command_2x4: str,
    subdir: str,
    num_iterations: int = 1,
    batch_size: int = 1,
):
    """
    Approximate 8x4 galaxy MoE performance from cheaper 8x1 + 2x4 proxy runs.

    Runs both proxy commands sequentially, captures the profiler CSV from each,
    then applies the SP/TP op-selection logic from approx.py to log the estimated
    per-op breakdown and total. No perf assertion is performed.

    SP ops (Dispatch, Combine, expert FFN Matmul, ...) come from 8x1.
    TP ops (AllGather, ReduceScatter, AllBroadcast, gate, ...) come from 2x4.
    """
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    logger.info("Running 8x1 proxy (dispatch + combine + expert FFN)...")
    run_device_perf(command_8x1, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size)
    csv_8x1 = get_latest_ops_log_filename(subdir)
    logger.info(f"8x1 CSV: {csv_8x1}")

    # run_device_perf calls clear_profiler_runtime_artifacts() which deletes all of
    # generated/profiler/ before each run. Copy the 8x1 CSV to a temp file so it
    # survives the 2x4 run's cleanup.
    tmp_csv_8x1 = None
    try:
        tmp_csv_8x1 = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        shutil.copy(csv_8x1, tmp_csv_8x1)

        logger.info("Running 2x4 proxy (gate + TP collectives)...")
        run_device_perf(command_2x4, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size)
        csv_2x4 = get_latest_ops_log_filename(subdir)
        logger.info(f"2x4 CSV: {csv_2x4}")

        df_approx = approximate_8x4_perf(csv_8x1=tmp_csv_8x1, csv_2x4=csv_2x4)
        logger.info(f"\n{df_approx.to_string(index=False)}")
    finally:
        if tmp_csv_8x1:
            os.unlink(tmp_csv_8x1)

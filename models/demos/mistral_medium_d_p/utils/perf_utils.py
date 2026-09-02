# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-perf harness for the mistral_medium_d_p block perf tests (tests/perf/).

Ported from ``deepseek_v3_d_p/utils/perf_utils.py`` (the Pattern A house harness) so this model
carries no dependency on another team's utils. One behavioral difference: the tt-smi telemetry
read tolerates a missing/hung ``tt-smi`` binary and keeps the margin unchanged, instead of
erroring — dev boxes don't all have it installed.
"""

import json
import os
import subprocess

import pandas as pd
import pytest
from loguru import logger
from tracy.process_model_log import get_latest_ops_log_filename

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.tt_transformers.tests.test_utils import merge_device_rows


def get_ddr_speed() -> int | None:
    """DDR speed from tt-smi smbus telemetry, or None when unavailable (tt-smi missing/failed)."""
    try:
        result = subprocess.run(
            ["tt-smi", "-s", "--snapshot_no_tty"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    data = json.loads(result.stdout)
    if data and data.get("device_info"):
        smbus_telem = data["device_info"][0].get("smbus_telem", {})
        ddr_speed_hex = smbus_telem.get("DDR_SPEED") or smbus_telem.get("SMBUS_TX_DDR_SPEED")
        if ddr_speed_hex:
            return int(ddr_speed_hex, 16)
    return None


def adjust_margin_for_ddr_speed(margin: float, expected_speed: int = 16000) -> float:
    """Return *margin* adjusted for the actual DDR speed reported by tt-smi.

    - DDR speed < *expected_speed*  → double the margin (slower memory, looser threshold).
    - DDR speed > *expected_speed*  → warn that baselines may need updating, keep margin.
    - DDR speed == *expected_speed* or unavailable → keep margin unchanged.
    """
    ddr_speed = get_ddr_speed()
    if ddr_speed is not None and ddr_speed < expected_speed:
        logger.warning(
            f"DDR speed is {ddr_speed} (expected {expected_speed}), increasing margin from {margin} to {margin * 2}"
        )
        return margin * 2
    if ddr_speed is not None and ddr_speed > expected_speed:
        logger.warning(f"DDR speed is {ddr_speed} (above expected {expected_speed}), baselines may need updating")
    return margin


def is_galaxy_env() -> bool:
    """Galaxy detection without opening the cluster.

    `ttnn.cluster.get_cluster_type()` opens the chip cluster as a side effect. When used in a
    `@skipif` marker (evaluated at collection) or even in-test before `run_device_perf` spawns
    its tracy subprocess, the parent holds chip locks and the subprocess deadlocks waiting for
    them. CI sets `MESH_DEVICE=TG` for galaxy jobs.
    """
    return os.environ.get("MESH_DEVICE", "").upper() in ("TG", "GALAXY")


def run_model_device_perf_test_with_merge(
    command: str,
    expected_device_perf_ns_per_iteration: float,
    subdir: str,
    model_name: str,
    num_iterations: int = 1,
    batch_size: int = 1,
    margin: float = 0.015,
    comments: str = "",
    op_filter: str = "",
    between_signposts: tuple[str, str] | None = None,
    extra_env: dict | None = None,
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
        op_filter: If set, restricts the measurement to rows whose OP CODE
            contains the given substring — useful when the worker emits multiple
            ops and only one is under test.
        between_signposts: If set to (start_header, stop_header), restricts the
            measurement to device ops emitted between those two tracy signposts
            (e.g. ("MLP_START", "MLP_END")), excluding everything dispatched before
            the first start / after the last stop — such as one-time weight-load
            tilize/typecast at construction. Handles repeated/nested pairs (only ops
            inside an open region are kept).
        extra_env: If set, applied to os.environ for the duration of the subprocess
            invocation. Use for vars the worker reads directly — prefixing them into
            the command doesn't work because tracy's -m flag mis-parses leading
            KEY=VAL tokens as module names.
    """
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"

    saved_env = {k: os.environ.get(k) for k in (extra_env or {})}
    try:
        if extra_env:
            os.environ.update(extra_env)
        post_processed_results = run_device_perf(
            command, subdir=subdir, num_iterations=num_iterations, cols=cols, batch_size=batch_size
        )
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # Apply multi-device row merging
    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    total_rows = len(df)
    signpost_rows = len(df[df["OP TYPE"] == "tt_signpost"])
    device_rows = len(df[df["OP TYPE"] == "tt_dnn_device"])

    logger.debug(f"CSV total rows: {total_rows}, signposts: {signpost_rows}, device ops: {device_rows}")

    if between_signposts is not None:
        start_header, stop_header = between_signposts
        sp = df["OP TYPE"] == "signpost"
        is_start = sp & (df["OP CODE"] == start_header)
        is_stop = sp & (df["OP CODE"] == stop_header)
        if not is_start.any() or not is_stop.any():
            pytest.fail(
                f"between_signposts={between_signposts!r}: signpost(s) not found in {filename} "
                f"(found starts={int(is_start.sum())}, stops={int(is_stop.sum())})"
            )
        # CSV rows are in host-dispatch order; +1 at each start, -1 at each stop. A row is "inside"
        # an open region when the running depth is > 0. The start row itself raises depth to 1, so it
        # is excluded only by ~sp (signpost rows are never device ops); the stop row drops depth to 0.
        depth = (is_start.astype(int) - is_stop.astype(int)).cumsum()
        df = df[(depth > 0) & ~sp]
        if df.empty:
            pytest.fail(f"between_signposts={between_signposts!r} matched no device rows in {filename}")
        logger.debug(f"Rows between signposts {between_signposts}: {len(df)}")

    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]

    if op_filter:
        df = df[df["OP CODE"].str.contains(op_filter, na=False, regex=False)]
        if df.empty:
            pytest.fail(f"op_filter={op_filter!r} matched no rows in {filename}")
        logger.debug(f"Rows after op_filter={op_filter!r}: {len(df)}")

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

# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile

import pandas as pd
import pytest
from loguru import logger
from tracy.common import PROFILER_ARTIFACTS_DIR
from tracy.process_model_log import get_latest_ops_log_filename

from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import get_ddr_speed
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.tt_transformers.tests.test_utils import merge_device_rows


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

# SDPA op: scale by 4 when extrapolating from 2x4 to 8x4 (SP 2→8 = 4x, TP 4→4 = 1x)
SDPA_OP = "RingJointSDPADeviceOperation"


def _is_galaxy_env() -> bool:
    """Galaxy detection without opening the cluster.

    `conftest.is_galaxy()` calls `ttnn.cluster.get_cluster_type()` which opens the chip
    cluster as a side effect. When used in a `@skipif` marker (evaluated at collection)
    or even in-test before `run_device_perf` spawns its tracy subprocess, the parent
    holds chip locks and the subprocess deadlocks waiting for them.

    CI sets `MESH_DEVICE=TG` for galaxy jobs (see galaxy_deepseek_prefill_tests.yaml
    and demo_sp_release_tests.yaml).
    """
    return os.environ.get("MESH_DEVICE", "").upper() in ("TG", "GALAXY")


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


def approximate_8x4_perf(csv_8x1: str, csv_2x4: str, csv_8x4: str | None = None, use_avg: bool = False) -> pd.DataFrame:
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


def approximate_mla_galaxy_perf(csv_2x4: str, csv_8x4: str | None = None, use_avg: bool = False) -> pd.DataFrame:
    """
    Approximate 8x4 MLA performance from a cheaper 2x4 run.

    - SDPA ops: taken from 2x4 and scaled by 4 (SP 2→8 = 4x, TP 4→4 = 1x, total 4x).
    - Everything else: taken from 2x4 as-is (no scaling).
    """
    ops_2x4 = load_merged_durations(csv_2x4, use_avg=use_avg)
    ops_8x4 = load_merged_durations(csv_8x4, use_avg=use_avg) if csv_8x4 else None

    all_ops = set(ops_8x4.index) if ops_8x4 is not None else set(ops_2x4.index)

    rows = []
    for op in sorted(all_ops):
        base_ns = ops_2x4.get(op, 0)
        if SDPA_OP in op:
            src = "2x4 (×4)"
            approx_ns = base_ns * 4
        else:
            src = "2x4"
            approx_ns = base_ns

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
            (e.g. ("MLA_START", "MLA_END")), excluding everything dispatched before
            the first start / after the last stop — such as one-time weight-load
            tilize/typecast at construction. Handles repeated/nested pairs (only ops
            inside an open region are kept).
        extra_env: If set, applied to os.environ for the duration of the subprocess
            invocation. Use for vars the worker reads directly (e.g. TT_DS_CAPTURED_LAYER)
            — prefixing them into the command doesn't work because tracy's -m flag
            mis-parses leading KEY=VAL tokens as module names.
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


def run_model_device_perf_test_per_op(
    command: str,
    expected_per_op: dict,
    subdir: str,
    model_name: str,
    margin: float = 0.03,
    comments: str = "",
    extra_env: dict | None = None,
):
    """Run one worker subprocess and assert performance for multiple ops independently.

    Use when a single worker invocation produces multiple device ops in the Tracy CSV
    (e.g. dispatch + combine in one forward pass) and each needs its own baseline so
    a regression points to the responsible kernel rather than the combined total.

    Args:
        command: pytest command to spawn the worker.
        expected_per_op: dict mapping OP CODE substring → expected duration in ns.
            For each entry, the merged-device-rows DataFrame is filtered by substring,
            durations summed, and asserted against expected ± margin. Every entry must
            match at least one row in the CSV; missing matches `pytest.fail`.
        subdir: profiler artifacts subdir.
        model_name: name passed to `prep_device_perf_report`.
        margin: tolerance applied uniformly to all entries in `expected_per_op`.
        comments: report comments string.
        extra_env: optional env vars applied to `os.environ` for the duration of the
            subprocess invocation. Use this for vars the worker reads directly
            (e.g. TT_DS_CAPTURED_LAYER) — prefixing them into `command` doesn't work
            because tracy's `-m` flag mis-parses leading KEY=VAL tokens as module names.
    """
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL DURATION [ns]"

    saved_env = {k: os.environ.get(k) for k in (extra_env or {})}
    try:
        if extra_env:
            os.environ.update(extra_env)
        run_device_perf(command, subdir=subdir, num_iterations=1, cols=cols, batch_size=1)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)
    df = df[df["OP TYPE"] == "tt_dnn_device"]
    df_merged = merge_device_rows(df)
    logger.info(f"[per-op] CSV={filename}  device rows={len(df)}  merged ops={len(df_merged)}")

    measured_per_op = {}
    failures = []
    for op_substring, expected_ns in expected_per_op.items():
        rows = df_merged[df_merged["OP CODE"].str.contains(op_substring, na=False, regex=False)]
        if rows.empty:
            pytest.fail(
                f"No merged rows match op_substring={op_substring!r} in {filename}; "
                f"available op codes: {sorted(df_merged['OP CODE'].unique())}"
            )
        measured_ns = float(rows["DEVICE KERNEL DURATION [ns]"].sum())
        measured_per_op[op_substring] = measured_ns
        lo = (1 - margin) * expected_ns
        hi = (1 + margin) * expected_ns
        passing = lo <= measured_ns <= hi
        logger.info(
            f"[per-op] {op_substring}: measured={measured_ns:,.0f} ns  "
            f"expected={expected_ns:,.0f} ns  bounds=[{lo:,.0f}, {hi:,.0f}]  "
            f"{'PASS' if passing else 'FAIL'}"
        )
        if not passing:
            failures.append((op_substring, measured_ns, expected_ns, lo, hi))

    total_measured = sum(measured_per_op.values())
    post_processed_results = {inference_time_key: total_measured}
    expected_results = {}
    for op_substring, measured_ns in measured_per_op.items():
        key = f"{op_substring} DEVICE KERNEL DURATION [ns]"
        post_processed_results[key] = measured_ns
        expected_ns = expected_per_op[op_substring]
        expected_results[f"Lower Threshold {key}"] = (1 - margin) * expected_ns
        expected_results[f"Upper Threshold {key}"] = (1 + margin) * expected_ns

    prep_device_perf_report(
        model_name=model_name,
        batch_size=1,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=comments,
    )

    if failures:
        msg = "Per-op perf checks failed:\n  " + "\n  ".join(
            f"{op}: measured={m:,.0f} ns  expected={e:,.0f} ns (bounds [{lo:,.0f}, {hi:,.0f}], margin ±{margin*100:.0f}%)"
            for op, m, e, lo, hi in failures
        )
        pytest.fail(msg)


def run_moe_perf_with_approximation(
    command_8x1: str,
    expected_ns_8x1: float,
    model_name_8x1: str,
    command_2x4: str,
    expected_ns_2x4: float,
    model_name_2x4: str,
    subdir: str,
    num_iterations: int = 1,
    batch_size: int = 1,
    margin: float = 0.03,
    comments_8x1: str = "",
    comments_2x4: str = "",
):
    """
    Run 8x1 + 2x4 MoE proxies once each, perf-validate both against baselines,
    and compute the approximated 8x4 galaxy total from the same two CSVs.

    Replaces the earlier split across `test_deepseek_v3_moe_perf[8x1|2x4]` +
    `test_deepseek_v3_moe_perf_approx_galaxy` (4 runs total) with 2 runs: each
    proxy executes once and its CSV feeds both the per-proxy baseline check and
    the SP/TP op-selection approximation.

    SP ops (Dispatch, Combine, expert FFN Matmul, ...) come from 8x1.
    TP ops (AllGather, ReduceScatter, AllBroadcast, gate, ...) come from 2x4.
    """
    # Collect perf-check failures from each proxy so the entire pipeline
    # (8x1, 2x4, approximation) runs to completion regardless of which proxy
    # tripped its baseline. Re-raised as a single AssertionError at the end so
    # pytest still reports the test as FAILED with all offending proxies named.
    perf_failures: list[tuple[str, str]] = []

    logger.info("=== 8x1 proxy: dispatch + combine + expert FFN ===")
    try:
        run_model_device_perf_test_with_merge(
            command=command_8x1,
            expected_device_perf_ns_per_iteration=expected_ns_8x1,
            subdir=subdir,
            model_name=model_name_8x1,
            num_iterations=num_iterations,
            batch_size=batch_size,
            margin=margin,
            comments=comments_8x1,
        )
    except AssertionError as e:
        logger.warning(f"8x1 perf check FAILED but continuing to 2x4: {e}")
        perf_failures.append(("8x1", str(e)))
    csv_8x1 = get_latest_ops_log_filename(subdir)
    logger.info(f"8x1 CSV: {csv_8x1}")

    # run_device_perf (inside run_model_device_perf_test_with_merge) calls
    # clear_profiler_runtime_artifacts() which deletes generated/profiler/ before
    # each run. Copy 8x1 CSV to tmp so it survives the 2x4 run.
    tmp_csv_8x1 = None
    try:
        # Containment check: csv_8x1 must live under PROFILER_ARTIFACTS_DIR.
        # Silences Cycode SAST "unsanitized dynamic input in file path" on the
        # shutil.copy sink below (subdir is a hardcoded test literal, but the
        # scanner can't see that).
        profiler_root = os.path.abspath(str(PROFILER_ARTIFACTS_DIR))
        csv_8x1_abs = os.path.abspath(str(csv_8x1))
        if not csv_8x1_abs.startswith(profiler_root + os.sep):
            raise RuntimeError(f"Refusing to copy CSV outside profiler root: {csv_8x1}")
        tmp_csv_8x1 = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        shutil.copy(csv_8x1_abs, tmp_csv_8x1)

        logger.info("=== 2x4 proxy: gate + TP collectives ===")
        try:
            run_model_device_perf_test_with_merge(
                command=command_2x4,
                expected_device_perf_ns_per_iteration=expected_ns_2x4,
                subdir=subdir,
                model_name=model_name_2x4,
                num_iterations=num_iterations,
                batch_size=batch_size,
                margin=margin,
                comments=comments_2x4,
            )
        except AssertionError as e:
            logger.warning(f"2x4 perf check FAILED: {e}")
            perf_failures.append(("2x4", str(e)))
        csv_2x4 = get_latest_ops_log_filename(subdir)
        logger.info(f"2x4 CSV: {csv_2x4}")

        logger.info("=== Approximating 8x4 galaxy total from 8x1 + 2x4 ===")
        df_approx = approximate_8x4_perf(csv_8x1=tmp_csv_8x1, csv_2x4=csv_2x4)
        logger.info(f"\n{df_approx.to_string(index=False)}")
    finally:
        if tmp_csv_8x1:
            os.unlink(tmp_csv_8x1)

    if perf_failures:
        summary = "; ".join(f"{which}: {msg}" for which, msg in perf_failures)
        raise AssertionError(f"Perf check(s) outside expected range — {summary}")


def run_mla_perf_with_approximation(
    command_2x4: str,
    expected_ns_2x4: float,
    model_name_2x4: str,
    subdir: str,
    num_iterations: int = 1,
    batch_size: int = 1,
    margin: float = 0.03,
    comments_2x4: str = "",
):
    logger.info("=== 2x4 MLA perf test on LB ===")
    run_model_device_perf_test_with_merge(
        command=command_2x4,
        expected_device_perf_ns_per_iteration=expected_ns_2x4,
        subdir=subdir,
        model_name=model_name_2x4,
        num_iterations=num_iterations,
        batch_size=batch_size,
        margin=margin,
        comments=comments_2x4,
    )
    csv_2x4 = get_latest_ops_log_filename(subdir)
    logger.info(f"2x4 CSV: {csv_2x4}")

    logger.info("=== Approximating 8x4 Galaxy total from 2x4 ===")
    df_approx = approximate_mla_galaxy_perf(csv_2x4=csv_2x4)
    logger.info(f"\n{df_approx.to_string(index=False)}")

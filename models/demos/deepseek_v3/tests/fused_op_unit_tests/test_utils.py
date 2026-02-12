# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utility functions for DeepSeek V3 fused op unit tests.

This module contains common helper functions used across all fused op unit tests
to reduce code duplication and ensure consistent behavior.
"""

import math
import os
from collections import defaultdict
from typing import Callable

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from tools.tracy.process_model_log import get_profiler_folder


def get_int_env(name: str, default: int) -> int:
    """Get an integer environment variable with a default value.

    Args:
        name: Name of the environment variable.
        default: Default value if the environment variable is not set.

    Returns:
        The integer value of the environment variable, or the default.

    Raises:
        ValueError: If the environment variable is set but cannot be parsed as an integer.
    """
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError as e:
        raise ValueError(f"Env var {name} must be an int, got {val!r}") from e


def maybe_skip_long_seq(seq_len: int, long_seq_env_var: str, threshold: int = 8192):
    """Skip test if sequence length exceeds threshold and env var is not set.

    Args:
        seq_len: The sequence length being tested.
        long_seq_env_var: Name of the environment variable that enables long sequence tests.
        threshold: Maximum sequence length before requiring the env var (default: 8192).
    """
    if seq_len <= threshold:
        return
    if os.getenv(long_seq_env_var) is None:
        pytest.skip(f"Set {long_seq_env_var}=1 to enable seq_len={seq_len} coverage.")


def compare_with_reference(
    tt_output: torch.Tensor,
    ref_output: torch.Tensor,
    expected_pcc: float,
    atol: float,
    rtol: float,
    convert_to_float: bool = False,
    strict_assert: bool = True,
) -> tuple[float, float]:
    """Compare TTNN output with reference and return metrics.

    Args:
        tt_output: Output tensor from TTNN operation.
        ref_output: Reference output tensor.
        expected_pcc: Minimum expected Pearson correlation coefficient.
        atol: Absolute tolerance for torch.testing.assert_close.
        rtol: Relative tolerance for torch.testing.assert_close.
        convert_to_float: If True, convert tensors to float before comparison.
        strict_assert: If True, raise AssertionError on assert_close failure.
                      If False, log a warning instead.

    Returns:
        Tuple of (pcc_value, max_abs_error) for logging to superset.
    """
    if convert_to_float:
        tt_for_compare = tt_output.float()
        ref_for_compare = ref_output.float()
    else:
        tt_for_compare = tt_output
        ref_for_compare = ref_output

    passing, pcc = comp_pcc(ref_for_compare, tt_for_compare, expected_pcc)
    max_abs_error = (tt_for_compare - ref_for_compare).abs().max().item()
    logger.info(f"PCC: {pcc}")
    logger.info(f"Max absolute error: {max_abs_error}")
    assert passing, f"PCC {pcc} is below required {expected_pcc}"

    if strict_assert:
        torch.testing.assert_close(tt_for_compare, ref_for_compare, rtol=rtol, atol=atol)
    else:
        try:
            torch.testing.assert_close(tt_for_compare, ref_for_compare, rtol=rtol, atol=atol)
        except AssertionError as e:
            logger.warning(f"assert_close failed but PCC passed: {e}")

    return pcc, max_abs_error


def log_run_mode(
    mode: str,
    trace_mode: bool,
    program_cache_enabled: bool,
    seq_len: int,
    **extra_fields,
):
    """Log the test run configuration.

    Args:
        mode: Test mode ("decode" or "prefill").
        trace_mode: Whether trace mode is enabled.
        program_cache_enabled: Whether program cache is enabled.
        seq_len: Sequence length being tested.
        **extra_fields: Additional fields to log (e.g., use_real_weights, hf_config_size_attr).
    """
    logger.info("=== TEST RUN CONFIGURATION ===")
    logger.info(f"Mode: {mode}")
    logger.info(f"Sequence length: {seq_len}")
    for key, value in extra_fields.items():
        # Convert snake_case to readable format
        readable_key = key.replace("_", " ").title()
        logger.info(f"{readable_key}: {value}")
    logger.info(f"Trace mode: {trace_mode}")
    logger.info(f"Program cache enabled: {program_cache_enabled}")
    logger.info("===============================")


def deallocate_outputs(outputs):
    """Deallocate outputs which can be a single tensor or tuple of tensors.

    Args:
        outputs: A single ttnn.Tensor or a tuple of ttnn.Tensor objects.
    """
    if isinstance(outputs, tuple):
        for out in outputs:
            ttnn.deallocate(out)
    else:
        ttnn.deallocate(outputs)


def measure_perf_us(
    mesh_device: ttnn.MeshDevice,
    op_fn: Callable,
    warmup_iters: int,
    measure_iters: int,
    trace_mode: bool = False,
    profiler_name: str = "perf_measurement",
) -> float:
    """Measure operation performance in microseconds.

    Args:
        mesh_device: The mesh device to run operations on.
        op_fn: Function that executes the operation. Should return output tensor(s).
               For trace mode with persistent buffers, should accept optional
               `persistent_output_buffer` kwarg.
        warmup_iters: Number of warmup iterations before measurement.
        measure_iters: Number of iterations to measure.
        trace_mode: If True, use trace capture and replay for measurement.
        profiler_name: Name for the profiler measurement (default: "perf_measurement").

    Returns:
        Average operation time in microseconds.
    """
    ttnn.synchronize_device(mesh_device)

    if trace_mode:
        # Warmup in eager mode first
        for _ in range(warmup_iters):
            outputs = op_fn()
            ttnn.synchronize_device(mesh_device)
            deallocate_outputs(outputs)

        # Capture trace
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_outputs = op_fn()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        deallocate_outputs(traced_outputs)

        # Additional warmup with trace execution
        for _ in range(warmup_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)

        # Measure with trace
        profiler.clear()
        profiler.start(profiler_name)
        for _ in range(measure_iters):
            ttnn.execute_trace(mesh_device, trace_id, blocking=False)
            ttnn.synchronize_device(mesh_device)
        profiler.end(profiler_name, PERF_CNT=measure_iters)
        ttnn.release_trace(mesh_device, trace_id)
        return profiler.get(profiler_name) * 1e6

    # Eager mode
    for _ in range(warmup_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        deallocate_outputs(outputs)

    profiler.clear()
    profiler.start(profiler_name)
    for _ in range(measure_iters):
        outputs = op_fn()
        ttnn.synchronize_device(mesh_device)
        deallocate_outputs(outputs)
    profiler.end(profiler_name, PERF_CNT=measure_iters)
    return profiler.get(profiler_name) * 1e6


def merge_device_rows_for_perf(df: pd.DataFrame) -> pd.DataFrame:
    """Merge device performance rows across devices for collective operations.

    For collective operations (AllGather, ReduceScatter, etc.), computes average
    kernel duration across devices. For other operations, takes the maximum
    kernel duration.

    Args:
        df: DataFrame containing device performance data with columns:
            - "OP CODE": Operation name
            - "OP TYPE": Operation type (e.g., "tt_dnn_device")
            - "DEVICE ID": Device identifier
            - "DEVICE KERNEL DURATION [ns]": Kernel duration in nanoseconds

    Returns:
        DataFrame with merged performance rows.
    """
    block_by_device = defaultdict(list)

    # Device profiling CSV uses "OP NAME" instead of "OP CODE" and has no "OP TYPE"
    op_col = "OP NAME" if "OP NAME" in df.columns else "OP CODE"

    for _, row in df.iterrows():
        op_name = row[op_col]
        # Skip rows with NaN op names
        if pd.isna(op_name):
            continue
        op_name = str(op_name)  # Ensure it's a string
        device_id = int(row["DEVICE ID"])
        block_by_device[device_id].append((op_name, row.to_dict()))

    device_ids = sorted(block_by_device.keys())
    merged_blocks = []
    global_index = 0
    while max(len(block_by_device[device_id]) for device_id in device_ids) > 0:
        blocks = []
        op_name = None
        missing_devices = []
        for device_id in device_ids:
            if not len(block_by_device[device_id]):
                logger.warning(f"Warning: Device {device_id} is missing operation {op_name} at index {global_index}")
                continue
            if op_name is None:
                op_name = block_by_device[device_id][0][0]
            elif op_name != block_by_device[device_id][0][0]:
                missing_devices.append(device_id)
                continue

            blocks.append(block_by_device[device_id].pop(0))

        if missing_devices:
            logger.warning(
                f"Warning: {op_name} at index {global_index} not present in CSV for {len(missing_devices)} devices "
                f"{missing_devices} - do not trust data for this op or directly subsequent ops with the same name"
            )

        if not blocks:
            break

        is_collective = any(tag in op_name for tag in ("AllGather", "ReduceScatter", "AllReduce", "AllToAll"))
        if is_collective:
            device_kernel_durations = [
                d["DEVICE KERNEL DURATION [ns]"]
                for _, d in blocks
                if "DEVICE KERNEL DURATION [ns]" in d and not math.isnan(d["DEVICE KERNEL DURATION [ns]"])
            ]
            average_duration = (
                sum(device_kernel_durations) / len(device_kernel_durations) if device_kernel_durations else float("nan")
            )
            base_block = blocks[0][1].copy()
            base_block["DEVICE KERNEL DURATION [ns]"] = average_duration
            merged_blocks.append(base_block)
        else:
            max_duration_block = max(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(max_duration_block[1])

        global_index += 1

    return pd.DataFrame(merged_blocks)


def run_device_profiler_local(
    command: str,
    subdir: str,
    device_analysis_types: list[str],
    op_support_count: int = 10000,
) -> None:
    """Run device profiler with proper environment setup (self-contained version).

    This is a simplified version of tools.tracy.process_model_log.run_device_profiler
    that includes the environment variable fix for device profiling without -r flag.

    Args:
        command: The pytest command to profile.
        subdir: Subdirectory for profiler output.
        device_analysis_types: List of analysis types (e.g., ["device_kernel_duration"]).
        op_support_count: Maximum number of ops to profile.
    """
    import shlex
    import subprocess
    from pathlib import Path

    profiler_dir = Path(get_profiler_folder(subdir))

    # Build Tracy command arguments
    tracy_args = ["python3", "-m", "tracy", "-p"]  # -p only, no -r to avoid quote issues
    tracy_args.extend(["-o", str(profiler_dir)])
    tracy_args.append("--check-exit-code")

    for analysis in device_analysis_types:
        tracy_args.extend(["-a", analysis])

    tracy_args.extend(["--op-support-count", str(op_support_count)])
    tracy_args.extend(["-t", "5000", "-m"])

    # Parse command to extract pytest arguments
    cmd_parts = shlex.split(command)
    tracy_args.extend(cmd_parts)

    # Set environment variables for device profiling
    # These are normally set by Tracy's -r flag, but we set them manually
    # to avoid nested invocation quote issues
    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TTNN_OP_PROFILER"] = "1"
    env["TT_METAL_PROFILER_TRACE_TRACKING"] = "1"

    profiler_cmd = " ".join(shlex.quote(arg) for arg in tracy_args)
    logger.info(f"Running device profiler: {profiler_cmd}")

    result = subprocess.run(tracy_args, check=False, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        logger.error(f"Tracy profiler failed with return code {result.returncode}")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, profiler_cmd, result.stdout, result.stderr)


def collect_device_perf(
    command: str,
    subdir: str,
    warmup_iters: int,
    use_signposts: bool = False,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Collect device performance statistics by running a profiled command.

    Args:
        command: The pytest command to run with device profiling.
        subdir: Subdirectory for profiler output files.
        warmup_iters: Number of warmup iterations to exclude from statistics.
        use_signposts: If True, only analyze ops between "start" and "stop" signposts.

    Returns:
        Tuple of:
            - op_stats: Dict mapping op names to their avg kernel duration and op-to-op latency.
            - total_kernel_ns: Total kernel duration in nanoseconds.
            - total_op_to_op_ns: Total op-to-op latency in nanoseconds.
    """
    # Use local device profiler with environment fix
    device_analysis_types = ["device_kernel_duration"]
    run_device_profiler_local(
        command,
        subdir,
        device_analysis_types=device_analysis_types,
        op_support_count=10000,
    )

    # Read from cpp_device_perf_report.csv in .logs
    from pathlib import Path

    profiler_dir = Path(get_profiler_folder(subdir))
    filename = profiler_dir / ".logs" / "cpp_device_perf_report.csv"

    if not filename.exists():
        raise FileNotFoundError(f"Device perf CSV not found at {filename}")

    logger.info(f"Using CSV from .logs/: {filename}")
    df = pd.read_csv(filename)

    # Check if we have OP NAME populated (from -r mode) or if it's empty (from -p only mode)
    required_cols = ["DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    assert not missing_cols, f"Missing device perf columns: {missing_cols}"

    df["DEVICE KERNEL DURATION [ns]"] = pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce").fillna(0.0)
    df["OP TO OP LATENCY [ns]"] = pd.to_numeric(df["OP TO OP LATENCY [ns]"], errors="coerce").fillna(0.0)

    # Check if OP NAME has any valid (non-NaN) values
    has_op_names = False
    if "OP NAME" in df.columns:
        has_op_names = df["OP NAME"].notna().any()

    op_stats: dict[str, dict[str, float]] = {}

    if has_op_names:
        # We have operation names - provide per-op breakdown
        logger.info("Device perf CSV contains operation names - providing per-op breakdown")
        df = merge_device_rows_for_perf(df)

        for op_name, group in df.groupby("OP NAME"):
            if pd.isna(op_name):
                continue
            op_name = str(op_name)
            kernel_vals = group["DEVICE KERNEL DURATION [ns]"].tolist()
            op_to_op_vals = group["OP TO OP LATENCY [ns]"].tolist()
            if warmup_iters > 0:
                kernel_vals = kernel_vals[warmup_iters:]
                op_to_op_vals = op_to_op_vals[warmup_iters:]
            if kernel_vals and op_to_op_vals:
                op_stats[op_name] = {
                    "avg_kernel_duration_ns": sum(kernel_vals) / len(kernel_vals),
                    "avg_op_to_op_latency_ns": sum(op_to_op_vals) / len(op_to_op_vals),
                }

        total_kernel_ns = sum(entry["avg_kernel_duration_ns"] for entry in op_stats.values())
        total_op_to_op_ns = sum(entry["avg_op_to_op_latency_ns"] for entry in op_stats.values())
    else:
        # No operation names - can only provide aggregate totals
        logger.warning("Device perf CSV does not contain operation names - providing aggregate totals only")

        # CRITICAL FIX: Filter to ONLY traced operations to exclude compile/eager ops
        # Check if we have trace replay session IDs to filter by
        if "METAL TRACE REPLAY SESSION ID" in df.columns:
            # Filter to only rows with trace replay session ID (excludes compile/eager ops)
            traced_mask = df["METAL TRACE REPLAY SESSION ID"].notna() & (df["METAL TRACE REPLAY SESSION ID"] != "")
            traced_df = df[traced_mask].copy()

            if len(traced_df) > 0:
                # Count unique trace replay sessions (iterations)
                num_trace_sessions = traced_df["METAL TRACE REPLAY SESSION ID"].nunique()
                num_traced_ops = len(traced_df)
                ops_per_iteration = num_traced_ops / num_trace_sessions

                logger.info(
                    f"Filtered to {num_traced_ops} traced ops across {num_trace_sessions} iterations "
                    f"({ops_per_iteration:.0f} ops/iter, excluded {len(df) - num_traced_ops} compile/eager ops)"
                )

                # Group by iteration and device to calculate per-device metrics
                # Operations run in PARALLEL across devices, so we need max/avg, not sum!
                device_iter_metrics = []
                for session_id in traced_df["METAL TRACE REPLAY SESSION ID"].unique():
                    session_df = traced_df[traced_df["METAL TRACE REPLAY SESSION ID"] == session_id]
                    # For each device in this iteration
                    for device_id in session_df["DEVICE ID"].unique():
                        device_df = session_df[session_df["DEVICE ID"] == device_id]
                        device_kernel = float(device_df["DEVICE KERNEL DURATION [ns]"].sum())
                        device_op_to_op = float(device_df["OP TO OP LATENCY [ns]"].sum())
                        device_iter_metrics.append(
                            {
                                "session": session_id,
                                "device": device_id,
                                "kernel_ns": device_kernel,
                                "op_to_op_ns": device_op_to_op,
                            }
                        )

                # Calculate per-iteration metrics using MAX across devices (critical path)
                per_iter_metrics = []
                for session_id in traced_df["METAL TRACE REPLAY SESSION ID"].unique():
                    session_metrics = [m for m in device_iter_metrics if m["session"] == session_id]
                    max_kernel = max(m["kernel_ns"] for m in session_metrics)
                    max_op_to_op = max(m["op_to_op_ns"] for m in session_metrics)
                    per_iter_metrics.append({"kernel_ns": max_kernel, "op_to_op_ns": max_op_to_op})

                # Average across iterations
                per_iter_kernel_ns = sum(m["kernel_ns"] for m in per_iter_metrics) / len(per_iter_metrics)
                per_iter_op_to_op_ns = sum(m["op_to_op_ns"] for m in per_iter_metrics) / len(per_iter_metrics)

                # Calculate per-op average (for reference)
                per_op_kernel_ns = per_iter_kernel_ns / ops_per_iteration
                per_op_op_to_op_ns = per_iter_op_to_op_ns / ops_per_iteration

                logger.info(
                    f"Per-iteration totals: kernel={per_iter_kernel_ns/1e3:.2f} us, "
                    f"op-to-op={per_iter_op_to_op_ns/1e3:.2f} us, "
                    f"total={(per_iter_kernel_ns + per_iter_op_to_op_ns)/1e3:.2f} us"
                )
                logger.info(
                    f"Per-op averages: kernel={per_op_kernel_ns/1e3:.2f} us, "
                    f"op-to-op={per_op_op_to_op_ns/1e3:.2f} us, "
                    f"total={(per_op_kernel_ns + per_op_op_to_op_ns)/1e3:.2f} us"
                )

                # Return per-iteration totals (for comparison with e2e benchmark)
                total_kernel_ns = per_iter_kernel_ns
                total_op_to_op_ns = per_iter_op_to_op_ns
            else:
                # No traced operations - this is eager mode
                # Use GLOBAL CALL COUNT to filter out compile operations
                logger.info("No traced operations - running in eager mode, filtering by GLOBAL CALL COUNT")

                if "GLOBAL CALL COUNT" in df.columns:
                    # Filter to operations with call count >= 1024 to skip compile phase
                    # Compile ops typically have lower call counts
                    eager_mask = df["GLOBAL CALL COUNT"] >= 1024
                    eager_df = df[eager_mask].copy()

                    if len(eager_df) > 0:
                        # Group by device and calculate per-device totals
                        device_metrics = []
                        for device_id in eager_df["DEVICE ID"].unique():
                            device_df = eager_df[eager_df["DEVICE ID"] == device_id]
                            device_kernel = float(device_df["DEVICE KERNEL DURATION [ns]"].sum())
                            device_op_to_op = float(device_df["OP TO OP LATENCY [ns]"].sum())
                            device_metrics.append(
                                {"device": device_id, "kernel_ns": device_kernel, "op_to_op_ns": device_op_to_op}
                            )

                        # For parallel operations, use MAX across devices (critical path)
                        total_kernel_ns_all = max(m["kernel_ns"] for m in device_metrics)
                        total_op_to_op_ns_all = max(m["op_to_op_ns"] for m in device_metrics)

                        # In eager mode, tests run DEVICE_PERF_ITERS (typically 10) iterations
                        # We need to divide by this to get per-iteration time
                        DEVICE_PERF_ITERS = 10  # Standard across all device perf tests
                        total_kernel_ns = total_kernel_ns_all / DEVICE_PERF_ITERS
                        total_op_to_op_ns = total_op_to_op_ns_all / DEVICE_PERF_ITERS

                        logger.info(
                            f"Eager mode: Filtered {len(eager_df)} operations (call count >= 1024) "
                            f"across {len(device_metrics)} devices, {DEVICE_PERF_ITERS} iterations"
                        )
                        logger.info(
                            f"Total (all iterations): kernel={total_kernel_ns_all/1e3:.2f} us, "
                            f"op-to-op={total_op_to_op_ns_all/1e3:.2f} us, "
                            f"total={(total_kernel_ns_all + total_op_to_op_ns_all)/1e3:.2f} us"
                        )
                        logger.info(
                            f"Per-iteration average: kernel={total_kernel_ns/1e3:.2f} us, "
                            f"op-to-op={total_op_to_op_ns/1e3:.2f} us, "
                            f"total={(total_kernel_ns + total_op_to_op_ns)/1e3:.2f} us"
                        )
                    else:
                        logger.warning("No operations found with call count >= 1024")
                        total_kernel_ns = float(df["DEVICE KERNEL DURATION [ns]"].sum())
                        total_op_to_op_ns = float(df["OP TO OP LATENCY [ns]"].sum())
                else:
                    logger.warning("GLOBAL CALL COUNT column not found - using all rows (may include compile ops)")
                    total_kernel_ns = float(df["DEVICE KERNEL DURATION [ns]"].sum())
                    total_op_to_op_ns = float(df["OP TO OP LATENCY [ns]"].sum())
        else:
            # Fallback: No trace session column, use old behavior but warn
            logger.warning(
                "METAL TRACE REPLAY SESSION ID column not found - "
                "metrics may include compile/eager ops (will be inflated)"
            )
            total_kernel_ns = float(df["DEVICE KERNEL DURATION [ns]"].sum())
            total_op_to_op_ns = float(df["OP TO OP LATENCY [ns]"].sum())

        # Create a single aggregate entry (convert to Python float for JSON serialization)
        op_stats["<all_ops_aggregate>"] = {
            "avg_kernel_duration_ns": total_kernel_ns,
            "avg_op_to_op_latency_ns": total_op_to_op_ns,
        }

    return op_stats, total_kernel_ns, total_op_to_op_ns


def skip_single_device_ccl(op_name: str):
    """Skip test because it includes CCL operations that require multiple devices.

    Args:
        op_name: Name of the operation being tested.
    """
    pytest.skip(f"Single-device test is not applicable because {op_name} includes CCL ops.")


def skip_single_device_sharded(op_name: str):
    """Skip test because it relies on width-sharded matmuls across the mesh.

    Args:
        op_name: Name of the operation being tested.
    """
    pytest.skip(
        f"Single-device test is not applicable because {op_name} relies on width-sharded matmuls across the mesh."
    )

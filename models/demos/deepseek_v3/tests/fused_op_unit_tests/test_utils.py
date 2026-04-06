# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Shared utility functions for DeepSeek V3 fused op unit tests.

This module contains common helper functions used across all fused op unit tests
to reduce code duplication and ensure consistent behavior.
"""

import math
import os
import re
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, profiler
from tools.tracy.process_model_log import get_profiler_folder

TIMESTAMP_DIR_RE = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")


def _get_latest_timestamped_ops_report(subdir: str) -> tuple[Path, str]:
    """Return latest Tracy ops report from timestamped report directories only."""
    profiler_dir = Path(get_profiler_folder(subdir))
    reports_dir = profiler_dir / "reports"
    if not reports_dir.exists():
        raise FileNotFoundError(f"Profiler reports directory not found: {reports_dir}")

    run_dirs = sorted(
        [p for p in reports_dir.iterdir() if p.is_dir() and TIMESTAMP_DIR_RE.match(p.name)],
        key=lambda p: p.name,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No timestamped profiler report directories found in: {reports_dir}")

    latest_dir = run_dirs[-1]
    expected_name = latest_dir / f"ops_perf_results_{latest_dir.name}.csv"
    if expected_name.exists():
        return expected_name, latest_dir.name

    # Defensive fallback in case Tracy changes exact filename pattern.
    matches = sorted(latest_dir.glob("ops_perf_results_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No ops_perf_results_*.csv found in latest report directory: {latest_dir}")
    return matches[-1], latest_dir.name


def _build_report_tag_from_command(command: str) -> str:
    """Create a stable per-test tag from pytest command and -k expression."""
    normalized_command = command.strip()
    if (
        len(normalized_command) >= 2
        and normalized_command[0] == normalized_command[-1]
        and normalized_command[0] in {"'", '"'}
    ):
        normalized_command = normalized_command[1:-1]

    try:
        tokens = shlex.split(normalized_command)
    except ValueError:
        tokens = normalized_command.split()

    test_spec = next((t for t in tokens if "::" in t and t.endswith(".py") is False), None)
    if test_spec is None:
        # Try recovering from separate path + ::test style
        for i, token in enumerate(tokens[:-1]):
            if token.endswith(".py") and tokens[i + 1].startswith("::"):
                test_spec = f"{token}{tokens[i + 1]}"
                break

    op_part = "unknown_test"
    if test_spec:
        test_path = test_spec.split("::")[0]
        op_part = Path(test_path).stem

    expr = ""
    if "-k" in tokens:
        k_idx = tokens.index("-k")
        if k_idx + 1 < len(tokens):
            expr = tokens[k_idx + 1]

    expr_tokens = [t.strip() for t in re.split(r"\s+", expr) if t.strip()]
    mode = "decode" if "decode" in expr_tokens else ("prefill" if "prefill" in expr_tokens else "mode")
    runtime = "trace" if "trace" in expr_tokens else ("eager" if "eager" in expr_tokens else "runtime")
    seq = next((t for t in expr_tokens if t.isdigit()), "seq")
    program_cache = "pcache" if "program_cache" in expr_tokens else "nopcache"

    extras = []
    for token in expr_tokens:
        if token in {"and", "or", "not", "decode", "prefill", "trace", "eager", "program_cache", "no_program_cache"}:
            continue
        if token.isdigit():
            continue
        extras.append(token)
    extras = extras[:3]

    parts = [op_part, f"{mode}_seq{seq}", runtime, program_cache]
    if extras:
        parts.extend(extras)

    tag = "__".join(parts)
    # Keep the directory/file names portable.
    tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", tag).strip("._-")
    return tag or "unknown_test"


def _copy_ops_report_to_test_folder(subdir: str, report_tag: str, ops_report_path: Path, run_timestamp: str) -> None:
    """Copy ops report to reports/<test-tag>/<timestamp>_ops_perf_results_*.csv."""
    import shutil

    profiler_dir = Path(get_profiler_folder(subdir))
    destination_dir = profiler_dir / "reports" / report_tag
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_name = f"{run_timestamp}_{ops_report_path.name}"
    destination_path = destination_dir / destination_name
    shutil.copy2(ops_report_path, destination_path)
    logger.info(f"Saved test-specific ops report copy: {destination_path}")


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
    """Run device profiler with robust fallback behavior.

    Prefer Tracy post-processing mode (-r) for op-level reports. Some environments
    do not generate host capture artifacts reliably; in those cases, retry in
    device-only mode (-p) so perf collection still proceeds.

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

    # Set environment variables for device profiling.
    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TTNN_OP_PROFILER"] = "1"
    env["TT_METAL_PROFILER_TRACE_TRACKING"] = "1"

    normalized_command = command.strip()
    if (
        len(normalized_command) >= 2
        and normalized_command[0] == normalized_command[-1]
        and normalized_command[0] in {"'", '"'}
    ):
        # Guard against accidentally wrapping the full command in a single quote pair.
        normalized_command = normalized_command[1:-1]

    def run_tracy(python_post_process: bool):
        if python_post_process:
            tracy_args = [
                "python3",
                "-m",
                "tracy",
                "-p",
                "-r",
                "-o",
                str(profiler_dir),
                "--check-exit-code",
                "--op-support-count",
                str(op_support_count),
                "-t",
                "5000",
            ]
            for analysis in device_analysis_types:
                tracy_args.extend(["-a", analysis])
            tracy_args.append("-m")
            # Tracy expects a single command string after -m; tokenizing here
            # causes quoted pytest -k expressions to be split and misparsed.
            tracy_args.append(normalized_command)
            profiler_cmd = " ".join(shlex.quote(arg) for arg in tracy_args)
            logger.info(f"Running device profiler: {profiler_cmd}")
            result = subprocess.run(tracy_args, check=False, capture_output=True, text=True, env=env)
            return profiler_cmd, result

        # For -p fallback, pass argv tokens to avoid treating full command as a module name.
        tracy_args = [
            "python3",
            "-m",
            "tracy",
            "-p",
            "-o",
            str(profiler_dir),
            "--check-exit-code",
            "--op-support-count",
            str(op_support_count),
            "-t",
            "5000",
        ]
        for analysis in device_analysis_types:
            tracy_args.extend(["-a", analysis])
        tracy_args.append("-m")
        # Keep command as one string so pytest -k expressions remain intact.
        tracy_args.append(normalized_command)

        profiler_cmd = " ".join(shlex.quote(arg) for arg in tracy_args)
        logger.info(f"Running device profiler: {profiler_cmd}")
        result = subprocess.run(tracy_args, check=False, capture_output=True, text=True, env=env)
        return profiler_cmd, result

    # Attempt 1: -r mode for post-processed ops report
    profiler_cmd, result = run_tracy(python_post_process=True)
    if result.returncode == 0:
        return

    stderr = result.stderr or ""
    host_capture_missing = (
        "tracy_profile_log_host.tracy was not generated" in stderr or "tracy capture out not found" in stderr
    )

    if host_capture_missing:
        logger.error(
            "Tracy -r failed due to missing host capture artifact. "
            "Device perf collection requires reports/ops_perf_results_*.csv; "
            "-p fallback is disabled by policy."
        )
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, profiler_cmd, result.stdout, result.stderr)

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

    # Require Tracy post-processed ops report (tt-perf-report semantics).
    # We intentionally do not accept .logs/cpp_device_perf_report.csv as the source
    # for fused-op perf targets.
    try:
        # Use timestamped-run lookup so reports/<test-name>/ folders don't break discovery.
        ops_filename, run_timestamp = _get_latest_timestamped_ops_report(subdir)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Required Tracy ops report directory was not generated. "
            "Perf collection requires reports/ops_perf_results_*.csv output."
        ) from e

    if not ops_filename.exists():
        raise RuntimeError(
            f"Required Tracy ops report was not generated: {ops_filename}. "
            "Perf collection requires reports/ops_perf_results_*.csv output."
        )

    logger.info(f"Using post-processed ops report: {ops_filename}")
    ops_df = pd.read_csv(ops_filename)
    report_tag = _build_report_tag_from_command(command)
    _copy_ops_report_to_test_folder(subdir, report_tag, ops_filename, run_timestamp)

    required_ops_cols = ["OP TYPE", "OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]
    missing_ops_cols = [col for col in required_ops_cols if col not in ops_df.columns]
    if missing_ops_cols:
        raise RuntimeError(f"Ops report missing required columns: {missing_ops_cols}. " f"Report file: {ops_filename}")

    ops_df["DEVICE KERNEL DURATION [ns]"] = pd.to_numeric(
        ops_df["DEVICE KERNEL DURATION [ns]"], errors="coerce"
    ).fillna(0.0)
    ops_df["OP TO OP LATENCY [ns]"] = pd.to_numeric(ops_df["OP TO OP LATENCY [ns]"], errors="coerce").fillna(0.0)

    if use_signposts:
        signposts = ops_df[ops_df["OP TYPE"] == "signpost"]
        starts = signposts[signposts["OP CODE"] == "start"].index.tolist()
        stops = signposts[signposts["OP CODE"] == "stop"].index.tolist()
        if starts and stops:
            start_idx = starts[0]
            stop_idx = next((idx for idx in stops if idx > start_idx), None)
            if stop_idx is not None:
                ops_df = ops_df.iloc[start_idx + 1 : stop_idx]
                logger.info(
                    "Filtered ops report to signpost window: "
                    f"rows={len(ops_df)} between start={start_idx} and stop={stop_idx}"
                )

    ops_df = ops_df[ops_df["OP TYPE"] != "signpost"]

    if warmup_iters > 0 and not use_signposts and len(ops_df) > warmup_iters:
        ops_df = ops_df.iloc[warmup_iters:]

    if len(ops_df) == 0:
        raise RuntimeError(
            f"Ops report is empty after filtering for command: {command}. "
            "Cannot compute fused-op perf from empty reports/ops_perf_results_*.csv."
        )

    op_stats = {}
    for op_name, group in ops_df.groupby("OP CODE"):
        if pd.isna(op_name):
            continue
        op_name = str(op_name)
        op_stats[op_name] = {
            "avg_kernel_duration_ns": float(group["DEVICE KERNEL DURATION [ns]"].mean()),
            "avg_op_to_op_latency_ns": float(group["OP TO OP LATENCY [ns]"].mean()),
        }

    total_kernel_ns = float(sum(v["avg_kernel_duration_ns"] for v in op_stats.values()))
    total_op_to_op_ns = float(sum(v["avg_op_to_op_latency_ns"] for v in op_stats.values()))
    logger.info(
        f"Computed totals from ops report: kernel={total_kernel_ns/1e3:.2f} us, "
        f"op-to-op={total_op_to_op_ns/1e3:.2f} us"
    )
    return op_stats, total_kernel_ns, total_op_to_op_ns

    # Fallback: cpp_device_perf_report.csv in .logs
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
                        # Guardrail for unreliable fallback data:
                        # In some environments, fallback CSVs can contain a small number of
                        # massive op-to-op rows (seconds-scale) that dominate totals and
                        # produce meaningless fused-op latency.
                        op_to_op_vals = eager_df["OP TO OP LATENCY [ns]"]
                        outlier_mask = op_to_op_vals >= 1_000_000  # >= 1 ms per row
                        if outlier_mask.any():
                            total_op_to_op_ns_all_rows = float(op_to_op_vals.sum())
                            outlier_op_to_op_ns = float(op_to_op_vals[outlier_mask].sum())
                            outlier_contribution = (
                                outlier_op_to_op_ns / total_op_to_op_ns_all_rows
                                if total_op_to_op_ns_all_rows > 0
                                else 0.0
                            )
                            if outlier_contribution >= 0.95:
                                # Fallback robustness: remove pathological rows and continue.
                                # This prevents a handful of massive artifact rows from dominating
                                # eager-mode op-to-op totals when -r data is unavailable.
                                num_outliers = int(outlier_mask.sum())
                                eager_df = eager_df[~outlier_mask].copy()
                                if len(eager_df) == 0:
                                    raise RuntimeError(
                                        "Unreliable eager fallback device perf: all rows were outliers "
                                        "(>=1ms op-to-op) after filtering."
                                    )
                                logger.warning(
                                    "Outlier-dominated eager fallback detected: "
                                    f"{outlier_contribution:.2%} of op-to-op from {num_outliers} rows (>=1ms). "
                                    f"Dropping outliers and continuing with {len(eager_df)} rows."
                                )

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

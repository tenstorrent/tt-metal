#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run a test/command with Tracy profiling and extract per-iteration metrics.

This script wraps Tracy profiling to capture FPU/SFPU utilization, NOC bandwidth,
and L1 metrics per iteration per device, then outputs the results to CSV.

Usage:
    # Profile a fused op test:
    python -m tracy.profile_iterations \\
        --capture fpu,pack,unpack,l1_0 \\
        --output-csv sparse_layer_perf.csv \\
        -- pytest tests/blaze/backed/test_sparse_layer_backed.py::test_hot_cold_sparse_layer -k "4iter"

    # Profile with NOC traces:
    python -m tracy.profile_iterations \\
        --capture all \\
        --collect-noc-traces \\
        --output-csv perf_report.csv \\
        -- pytest tests/blaze/backed/test_sparse_layer_backed.py -k "100iter"

    # Just analyze existing profiler logs:
    python -m tracy.profile_iterations --analyze-only --output-csv report.csv

Environment Variables (set automatically):
    TT_METAL_DEVICE_PROFILER=1           - Enable device profiler
    TT_METAL_PROFILER_MID_RUN_DUMP=1     - Enable mid-run data dumps
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 - Enable C++ post-processing
    TT_METAL_PROFILE_PERF_COUNTERS=<N>   - Perf counter capture bitfield
    TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 - NOC event capture (optional)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

# Import from sibling modules
try:
    from tracy.common import PROFILER_LOGS_DIR, PROFILER_ARTIFACTS_DIR
    from tracy.per_iteration_analysis import analyze_per_iteration_metrics, write_csv, print_summary
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from common import PROFILER_LOGS_DIR, PROFILER_ARTIFACTS_DIR
    from per_iteration_analysis import analyze_per_iteration_metrics, write_csv, print_summary


# Perf counter group bit positions (from __main__.py)
COUNTER_GROUP_BITS = {
    "fpu": 0,
    "pack": 1,
    "unpack": 2,
    "l1_0": 3,
    "l1_1": 4,
    "instrn": 5,
    "l1_2": 6,  # BH only
    "l1_3": 7,  # BH only
    "l1_4": 8,  # BH only
}


def compute_perf_counter_bitfield(groups: list[str]) -> int:
    """Compute the bitfield for the given perf counter groups."""
    bitfield = 0
    for group in groups:
        group_lower = group.lower().strip()
        if group_lower == "all":
            # fpu | pack | unpack | l1_0 | instrn
            bitfield = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5)
            break
        elif group_lower in COUNTER_GROUP_BITS:
            bitfield |= 1 << COUNTER_GROUP_BITS[group_lower]
        else:
            logger.warning(f"Unknown counter group: {group}")
    return bitfield


def setup_profiler_environment(
    capture_groups: list[str],
    collect_noc_traces: bool = False,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Set up environment variables for profiling."""
    env = dict(os.environ)

    # Core profiler settings
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    env["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
    env["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"

    # Output directory
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        env["TT_METAL_PROFILER_DIR"] = str(output_dir)

    # Perf counter groups
    if capture_groups:
        bitfield = compute_perf_counter_bitfield(capture_groups)
        if bitfield > 0:
            env["TT_METAL_PROFILE_PERF_COUNTERS"] = str(bitfield)
            logger.info(f"Perf counter groups: {capture_groups} (bitfield={bitfield})")

    # NOC event traces
    if collect_noc_traces:
        env["TT_METAL_DEVICE_PROFILER_NOC_EVENTS"] = "1"
        noc_dir = (output_dir or PROFILER_ARTIFACTS_DIR) / "noc_traces"
        noc_dir.mkdir(parents=True, exist_ok=True)
        env["TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH"] = str(noc_dir)
        logger.info(f"NOC traces enabled, output: {noc_dir}")

    return env


def run_with_profiling(
    command: list[str],
    capture_groups: list[str],
    collect_noc_traces: bool = False,
    output_dir: Path | None = None,
) -> int:
    """Run a command with profiling environment set up."""
    env = setup_profiler_environment(capture_groups, collect_noc_traces, output_dir)

    logger.info(f"Running: {' '.join(command)}")
    result = subprocess.run(command, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run a test with Tracy profiling and extract per-iteration metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Profile a fused op test with FPU/SFPU counters:
    python -m tracy.profile_iterations \\
        --capture fpu,pack,unpack \\
        --output-csv sparse_layer.csv \\
        -- pytest tests/blaze/backed/test_sparse_layer_backed.py -k "4iter"

    # Profile with all counters and NOC traces:
    python -m tracy.profile_iterations \\
        --capture all \\
        --collect-noc-traces \\
        -- pytest tests/blaze/backed/test_sparse_layer_backed.py

    # Just analyze existing profiler logs:
    python -m tracy.profile_iterations --analyze-only --output-csv report.csv
""",
    )

    parser.add_argument(
        "--capture",
        type=str,
        default="fpu,pack,unpack,l1_0",
        help=(
            "Comma-separated list of perf counter groups to capture: "
            "fpu, pack, unpack, l1_0, l1_1, instrn, all (default: fpu,pack,unpack,l1_0)"
        ),
    )
    parser.add_argument(
        "--collect-noc-traces",
        action="store_true",
        help="Enable NOC event trace collection",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for profiler output (default: generated/profiler/)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("per_iteration_metrics.csv"),
        help="Output CSV file for per-iteration metrics (default: per_iteration_metrics.csv)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing profiler logs (skip running command)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing summary to stdout",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "command",
        nargs="*",
        help="Command to run with profiling (use -- to separate from script args)",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Parse capture groups
    capture_groups = [g.strip() for g in args.capture.split(",") if g.strip()]

    # Run the command if provided (and not analyze-only)
    if args.command and not args.analyze_only:
        returncode = run_with_profiling(
            args.command,
            capture_groups,
            args.collect_noc_traces,
            args.output_dir,
        )
        if returncode != 0:
            logger.error(f"Command failed with exit code {returncode}")
            return returncode

    # Analyze profiler logs
    log_dir = args.output_dir / "logs" if args.output_dir else PROFILER_LOGS_DIR
    if not log_dir.exists():
        # Try parent directory
        log_dir = args.output_dir or PROFILER_ARTIFACTS_DIR

    if not log_dir.exists():
        logger.error(f"Profiler log directory not found: {log_dir}")
        return 1

    # NoC traces were written here by setup_profiler_environment(); feed them to the analysis
    # so each ITERATION gets its own tt-npe simulation.
    noc_trace_dir = None
    if args.collect_noc_traces:
        noc_trace_dir = (args.output_dir or PROFILER_ARTIFACTS_DIR) / "noc_traces"
        if not noc_trace_dir.exists():
            logger.warning(f"NoC traces requested but {noc_trace_dir} does not exist; skipping NoC analysis")
            noc_trace_dir = None

    logger.info(f"Analyzing profiler logs in: {log_dir}")
    metrics = analyze_per_iteration_metrics(log_dir, noc_trace_dir=noc_trace_dir)

    if not metrics:
        logger.warning("No metrics extracted from profiler logs")
        # Not an error - the test might not have produced profiler data
        return 0

    if not args.no_summary:
        print_summary(metrics)

    write_csv(metrics, args.output_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())

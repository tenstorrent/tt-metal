# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from tracy.common import clear_profiler_runtime_artifacts
from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler


def collect_test_configs(base_command: str, shape_filter: str) -> list[dict]:
    """
    Collect test configurations by running pytest --collect-only.

    Args:
        base_command: Base pytest command (e.g., "pytest tests/.../test_file.py::test_func")
        shape_filter: Shape filter for -k flag (e.g., "spatial_activation")

    Returns:
        List of dicts with 'test_name' and 'hyperparam_string' keys, in execution order.
    """
    collect_cmd = f"{base_command} -k {shape_filter} --collect-only -q"
    logger.info(f"Collecting test configs: {collect_cmd}")

    result = subprocess.run(collect_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to collect tests: {result.stderr}")
        raise RuntimeError(f"pytest --collect-only failed: {result.stderr}")

    configs = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        # Match lines like:           <Function test_all_gather_wan[...-2buffers-1workers-10chunks-1links-fabric_ring-spatial_activation]>
        if "<Function " in line and "[" in line:
            test_name = line.split("<Function ")[-1]
            hyperparam_str = extract_hyperparam_string(test_name)
            configs.append({"test_name": test_name, "hyperparam_string": hyperparam_str})

    logger.info(f"Collected {len(configs)} test configurations")
    for i, cfg in enumerate(configs):
        logger.debug(f"  [{i}] {cfg['hyperparam_string']}")

    return configs


def extract_hyperparam_string(test_name: str) -> str:
    """
    Extract hyperparam string from test name.

    Example input: test_all_gather_wan[wormhole_b0-mesh_device0-75-10-2buffers-1workers-10chunks-1links-fabric_ring_1024B-spatial_activation]
    Example output: 2buffers-1workers-10chunks-1links-fabric_ring_1024B

    Args:
        test_name: Full test name with parameters

    Returns:
        Hyperparam string portion (e.g., "2buffers-1workers-10chunks-1links-fabric_ring_1024B")
    """
    # Extract the part inside brackets
    match = re.search(r"\[(.+)\]", test_name)
    if not match:
        return test_name

    params = match.group(1)

    # Find hyperparam parts: patterns like Nbuffers, Nworkers, Nchunks, Nlinks
    hyperparam_parts = []
    parts = params.split("-")
    for part in parts:
        if re.match(r"^\d+buffers$", part):
            hyperparam_parts.append(part)
        elif re.match(r"^\d+workers$", part):
            hyperparam_parts.append(part)
        elif part.endswith("chunks"):
            hyperparam_parts.append(part)
        elif re.match(r"^\d+links$", part):
            hyperparam_parts.append(part)
        elif re.match(r"^fabric_ring_\d+B$", part):
            hyperparam_parts.append(part)

    return "-".join(hyperparam_parts) if hyperparam_parts else params


def extract_num_workers(hyperparam_string: str) -> int | None:
    """
    Extract num_workers from a hyperparam string (e.g. "2buffers-1workers-10chunks-1links" -> 1).
    Returns None if no Nworkers pattern is found.
    """
    match = re.search(r"(\d+)workers", hyperparam_string)
    return int(match.group(1)) if match else None


def extract_short_op_name(full_op_name: str) -> str:
    """
    Extract short operation name from full OP CODE.

    Example: "MeshDeviceOperationAdapter<ttnn::prim::AllBroadcastDeviceOperation>" -> "AllBroadcast"
    """

    def get_op_name_helper() -> str:
        # Try to extract from template parameter (e.g., <...::OpName>)
        match = re.search(r"::(\w+)>", full_op_name)
        if match:
            return match.group(1)

        # Try to extract after last ::
        match = re.search(r"::(\w+)$", full_op_name)
        if match:
            return match.group(1)

        # Fallback: return as-is
        return full_op_name

    op_name = get_op_name_helper()

    # Remove "DeviceOperation" suffix if present
    if op_name.endswith("DeviceOperation"):
        op_name = op_name[: -len("DeviceOperation")]

    return op_name


def post_process_ops_log_by_config(
    output_logs_subdir: str,
    test_configs: list[dict],
    num_iters: int,
) -> dict:
    """
    Post-process the ops log CSV to extract per-config statistics.

    Each test config has its own start/stop signpost pair in the log.
    Warmup iterations are already excluded by the signposts.

    For each iteration, sums up durations per device across all ops in the sequence,
    then takes the worst (max) device sum as the perf number for that iteration.

    Args:
        output_logs_subdir: Subdirectory containing the ops log
        test_configs: List of test config dicts from collect_test_configs()
        num_iters: Number of iterations per test config

    Returns:
        Dict mapping hyperparam_string to stats dict with MIN, MAX, AVG, STD, OP_SEQUENCE, etc.
    """
    filename = get_latest_ops_log_filename(output_logs_subdir)
    logger.info(f"Processing ops log: {filename}")

    df = pd.read_csv(filename)

    # Find all start/stop signpost pairs
    signpost_df = df[df["OP TYPE"] == "signpost"]
    start_indices = signpost_df[signpost_df["OP CODE"] == "start"].index.tolist()
    stop_indices = signpost_df[signpost_df["OP CODE"] == "stop"].index.tolist()

    if not start_indices or not stop_indices:
        raise ValueError("No start/stop signposts found in the log")

    if len(start_indices) != len(stop_indices):
        raise ValueError(f"Mismatched signpost pairs: {len(start_indices)} starts, {len(stop_indices)} stops")

    if len(start_indices) != len(test_configs):
        logger.warning(
            f"Signpost pair count ({len(start_indices)}) != test config count ({len(test_configs)}). "
            f"Results may be incomplete."
        )

    logger.info(f"Found {len(start_indices)} start/stop signpost pairs for {len(test_configs)} test configs")

    duration_col = "DEVICE KERNEL DURATION [ns]"
    results = {}

    # Process each start/stop pair
    for i, cfg in enumerate(test_configs):
        if i >= len(start_indices):
            logger.warning(f"No signpost pair for config {i}: {cfg['hyperparam_string']}")
            break

        start = start_indices[i]
        stop = stop_indices[i]

        # Extract rows between this start/stop pair
        config_df = df.iloc[start + 1 : stop].copy()

        # Filter out rows with empty/invalid duration
        config_df = config_df[config_df[duration_col] != "-"]
        config_df = config_df[config_df[duration_col].notna()]

        if config_df.empty:
            logger.warning(f"No valid operations found for config: {cfg['hyperparam_string']}")
            continue

        config_df[duration_col] = config_df[duration_col].astype(float)

        # Sort by GLOBAL CALL COUNT to ensure correct execution order
        # (CSV may be sorted by DEVICE ID first, not by operation order)
        config_df = config_df.sort_values("GLOBAL CALL COUNT").reset_index(drop=True)

        # Get number of devices from max DEVICE ID
        num_devices = int(config_df["DEVICE ID"].max()) + 1
        total_rows = len(config_df)

        # Validate row count is divisible by num_iters
        if total_rows % num_iters != 0:
            raise ValueError(
                f"Config {cfg['hyperparam_string']}: total rows ({total_rows}) not divisible by "
                f"num_iters ({num_iters}). Check num_iters parameter."
            )

        rows_per_iter = total_rows // num_iters

        # Validate rows_per_iter is divisible by num_devices
        if rows_per_iter % num_devices != 0:
            raise ValueError(
                f"Config {cfg['hyperparam_string']}: rows_per_iter ({rows_per_iter}) not divisible by "
                f"num_devices ({num_devices}). Data structure mismatch."
            )

        ops_per_iter = rows_per_iter // num_devices

        # Extract op sequence from first iteration (one op name per num_devices rows)
        first_iter_df = config_df.iloc[:rows_per_iter]
        op_sequence = []
        for op_idx in range(ops_per_iter):
            op_start = op_idx * num_devices
            full_op_name = first_iter_df.iloc[op_start]["OP CODE"]
            short_name = extract_short_op_name(full_op_name)
            op_sequence.append(short_name)

        # For each iteration: sum durations per device, then take max (worst) across devices
        durations = config_df[duration_col].values
        device_ids = config_df["DEVICE ID"].values
        per_iter_worst = []

        for iter_idx in range(num_iters):
            iter_start = iter_idx * rows_per_iter
            iter_end = iter_start + rows_per_iter

            iter_durations = durations[iter_start:iter_end]
            iter_device_ids = device_ids[iter_start:iter_end]

            # Sum durations per device
            device_sums = {}
            for dur, dev_id in zip(iter_durations, iter_device_ids):
                device_sums[dev_id] = device_sums.get(dev_id, 0) + dur

            # Take max (worst) across devices
            worst_device_sum = max(device_sums.values())
            per_iter_worst.append(worst_device_sum)

        per_iter_worst = pd.Series(per_iter_worst)

        results[cfg["hyperparam_string"]] = {
            "MIN": per_iter_worst.min(),
            "MAX": per_iter_worst.max(),
            "AVG": per_iter_worst.mean(),
            "STD": per_iter_worst.std(),
            "NUM_ITERS": num_iters,
            "NUM_DEVICES": num_devices,
            "OPS_PER_ITER": ops_per_iter,
            "OP_SEQUENCE": op_sequence,
        }

        logger.debug(
            f"Config {cfg['hyperparam_string']}: {num_iters} iters x {num_devices} devices x {ops_per_iter} ops, "
            f"MIN={per_iter_worst.min()/1000:.2f}us, MAX={per_iter_worst.max()/1000:.2f}us"
        )

    return results


def post_process_cpp_log_by_config(
    csv_file: str,
    test_configs: list[dict],
    num_iters: int,
) -> dict:
    """
    Post-process the C++ profiler CSV log to extract per-config statistics.

    Each test config has two trace IDs: one for warmup and one for actual iterations.
    We identify the "actual" trace IDs by matching row count to num_iters * num_devices.

    Stats computation:
        - Per iteration:  max across devices  -> "worst device time for this iter"
        - Per config:     avg across iters    -> "average worst-device time"
        - Across configs: min of AVG          -> "best config"

    Args:
        csv_file: Path to the CSV file
        test_configs: List of test config dicts from collect_test_configs()
        num_iters: Number of actual iterations per test config (excluding warmup)

    Returns:
        Dict mapping hyperparam_string to stats dict with MIN, MAX, AVG, STD, etc.
    """
    logger.info(f"Processing C++ profiler log: {csv_file}")

    df = pd.read_csv(csv_file)

    # Filter out rows with empty METAL TRACE ID
    df = df[df["METAL TRACE ID"].notna()]

    # Infer num_devices from unique DEVICE IDs
    num_devices = df["DEVICE ID"].nunique()
    expected_rows = num_iters * num_devices

    # Find "actual" trace IDs (those with row count = num_iters * num_devices)
    trace_counts = df.groupby("METAL TRACE ID").size()
    actual_trace_ids = trace_counts[trace_counts == expected_rows].index.tolist()
    actual_trace_ids = sorted(actual_trace_ids)  # maintain execution order

    logger.info(
        f"Found {len(actual_trace_ids)} actual trace IDs (expected {len(test_configs)}), "
        f"{num_devices} devices, {num_iters} iters per config"
    )

    # Validate count matches test_configs
    if len(actual_trace_ids) != len(test_configs):
        raise ValueError(f"Found {len(actual_trace_ids)} actual trace IDs, expected {len(test_configs)} test configs")

    duration_col = "DEVICE KERNEL DURATION [ns]"
    results = {}

    for cfg, trace_id in zip(test_configs, actual_trace_ids):
        config_df = df[df["METAL TRACE ID"] == trace_id].copy()
        config_df = config_df.sort_values("GLOBAL CALL COUNT").reset_index(drop=True)

        # For each iteration: take max duration across devices
        per_iter_worst = []
        for iter_idx in range(num_iters):
            start = iter_idx * num_devices
            end = start + num_devices
            iter_df = config_df.iloc[start:end]
            worst = iter_df[duration_col].max()
            per_iter_worst.append(worst)

        per_iter_worst = pd.Series(per_iter_worst)

        results[cfg["hyperparam_string"]] = {
            "MIN": per_iter_worst.min(),
            "MAX": per_iter_worst.max(),
            "AVG": per_iter_worst.mean(),
            "STD": per_iter_worst.std(),
            "NUM_ITERS": num_iters,
            "NUM_DEVICES": num_devices,
            "OPS_PER_ITER": 1,
        }

        logger.debug(
            f"Config {cfg['hyperparam_string']}: {num_iters} iters x {num_devices} devices, "
            f"MIN={per_iter_worst.min()/1000:.2f}us, MAX={per_iter_worst.max()/1000:.2f}us, "
            f"AVG={per_iter_worst.mean()/1000:.2f}us"
        )

    return results


def identify_best_worst_configs(results_by_config: dict) -> dict:
    """
    Identify best and worst hyperparam configurations.

    Args:
        results_by_config: Dict from post_process_ops_log_by_config()

    Returns:
        Dict with "best" and "worst" keys, each containing (config_string, value) tuple
    """
    if not results_by_config:
        return {"best": (None, None), "worst": (None, None)}

    # Best: config with lowest AVG duration
    best_config = min(results_by_config.items(), key=lambda x: x[1]["AVG"])
    # Worst: config with highest AVG duration
    worst_config = max(results_by_config.items(), key=lambda x: x[1]["AVG"])

    return {
        "best": (best_config[0], best_config[1]["AVG"]),
        "worst": (worst_config[0], worst_config[1]["AVG"]),
    }


def generate_perf_chart(
    test_filter: str,
    results_by_config: dict,
    output_dir: str,
    op_sequence: list[str] = None,
    best_worst: dict = None,
) -> str:
    """
    Generate a matplotlib bar chart showing min/max performance for each config.
    Also saves results to CSV and a summary TXT file.

    Args:
        test_filter: Test filter name (e.g., "spatial_activation")
        results_by_config: Dict from post_process_ops_log_by_config()
        output_dir: Directory to save the chart
        op_sequence: List of op names in the sequence
        best_worst: Optional dict from identify_best_worst_configs() for annotations

    Returns:
        Path to the saved chart file
    """
    if not results_by_config:
        logger.warning("No results to plot")
        return None

    os.makedirs(output_dir, exist_ok=True)

    configs = list(results_by_config.keys())
    avgs = [results_by_config[c]["AVG"] for c in configs]

    # Determine unit based on magnitude (use microseconds if max >= 1000 ns)
    # max_val = max(avgs)
    # if max_val >= 1000:
    unit = "us"
    divisor = 1000
    plot_vals = [v / divisor for v in avgs]

    x = list(range(len(configs)))

    _fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.5), 6))
    ax.plot(x, plot_vals, marker="o", linestyle="-", color="blue", label="Avg", alpha=0.8)

    ax.set_title(f"Test filter: {test_filter}")
    ax.set_xlabel("Hyperparam Configuration")
    ax.set_ylabel(f"Device Kernel Duration ({unit})")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on data points
    for i, val in enumerate(plot_vals):
        ax.annotate(
            f"{val:.1f}",
            xy=(i, val),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="blue",
        )

    # Add op sequence in annotation
    op_seq_str = " -> ".join(op_sequence) if op_sequence else "N/A"
    annotation_text = f"Op sequence: {op_seq_str}"
    # Add best/worst annotations if provided
    if best_worst:
        best_cfg, best_val = best_worst["best"]
        worst_cfg, worst_val = best_worst["worst"]
        annotation_text += f"\nBest (avg): {best_cfg} = {best_val/divisor:.1f} {unit}\nWorst (avg): {worst_cfg} = {worst_val/divisor:.1f} {unit}"
    ax.text(
        0.02,
        0.98,
        annotation_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save chart
    chart_path = os.path.join(output_dir, f"{test_filter}_perf.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved performance chart to: {chart_path}")

    # Save CSV (in execution order)
    csv_path = os.path.join(output_dir, f"{test_filter}_results.csv")
    csv_rows = []
    for cfg in configs:
        stats = results_by_config[cfg]
        csv_rows.append(
            {
                "hyperparam_string": cfg,
                "AVG": stats["AVG"],
                "MIN": stats["MIN"],
                "MAX": stats["MAX"],
                "STD": stats["STD"],
                "NUM_ITERS": stats["NUM_ITERS"],
                "NUM_DEVICES": stats["NUM_DEVICES"],
                "OPS_PER_ITER": stats["OPS_PER_ITER"],
            }
        )
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"Saved results CSV to: {csv_path}")

    # Save summary TXT with top/bottom 3 and best per num_workers
    txt_path = os.path.join(output_dir, f"{test_filter}_summary.txt")
    sorted_configs = sorted(results_by_config.items(), key=lambda x: x[1]["AVG"])
    top_3 = sorted_configs[:3]
    bottom_3 = sorted_configs[-3:]

    # Best config per num_workers (configs with parseable Nworkers only)
    configs_by_workers: dict[int, list[tuple[str, dict]]] = {}
    for cfg, stats in results_by_config.items():
        nw = extract_num_workers(cfg)
        if nw is not None:
            configs_by_workers.setdefault(nw, []).append((cfg, stats))
    best_per_workers: list[tuple[int, tuple[str, dict]]] = []
    for nw in configs_by_workers.keys():
        cands = configs_by_workers[nw]
        best = min(cands, key=lambda x: x[1]["AVG"])
        best_per_workers.append((nw, best))
    best_per_workers.sort(key=lambda x: x[1][1]["AVG"])  # sort by perf (best first)

    with open(txt_path, "w") as f:
        f.write(f"Test filter: {test_filter}\n")
        f.write(f"Total configs: {len(configs)}\n")
        f.write(f"Op sequence: {op_seq_str}\n")
        f.write("\n")

        f.write("=" * 40 + "\n")
        f.write("BEST PER NUM_WORKERS\n")
        f.write("=" * 40 + "\n")
        for nw, (cfg, stats) in best_per_workers:
            f.write(f"\n{nw} worker(s) — best config: {cfg}\n")
            f.write(f"  AVG: {stats['AVG']/divisor:.2f} {unit}\n")
            f.write(f"  MIN: {stats['MIN']/divisor:.2f} {unit}\n")
            f.write(f"  MAX: {stats['MAX']/divisor:.2f} {unit}\n")
            f.write(f"  STD: {stats['STD']/divisor:.2f} {unit}\n")
        if not best_per_workers:
            f.write("  (no configs with parseable Nworkers in hyperparam string)\n")

        f.write("\n")
        f.write("=" * 40 + "\n")
        f.write("TOP 3 BEST CONFIGS (lowest AVG)\n")
        f.write("=" * 40 + "\n")
        for i, (cfg, stats) in enumerate(top_3, 1):
            f.write(f"\n#{i}: {cfg}\n")
            f.write(f"  AVG: {stats['AVG']/divisor:.2f} {unit}\n")
            f.write(f"  MIN: {stats['MIN']/divisor:.2f} {unit}\n")
            f.write(f"  MAX: {stats['MAX']/divisor:.2f} {unit}\n")
            f.write(f"  STD: {stats['STD']/divisor:.2f} {unit}\n")

        f.write("\n")
        f.write("=" * 40 + "\n")
        f.write("TOP 3 WORST CONFIGS (highest AVG)\n")
        f.write("=" * 40 + "\n")
        for i, (cfg, stats) in enumerate(reversed(bottom_3), 1):
            f.write(f"\n#{i}: {cfg}\n")
            f.write(f"  AVG: {stats['AVG']/divisor:.2f} {unit}\n")
            f.write(f"  MIN: {stats['MIN']/divisor:.2f} {unit}\n")
            f.write(f"  MAX: {stats['MAX']/divisor:.2f} {unit}\n")
            f.write(f"  STD: {stats['STD']/divisor:.2f} {unit}\n")

    logger.info(f"Saved summary TXT to: {txt_path}")

    return chart_path


def run_perf_test(base_command: str, test_filter: str, test_name: str, num_iters: int):
    """
    Run performance pytest for a specific test filter and analyze results by hyperparam config.

    Args:
        base_command: pytest command to run tests
        test_filter: Test filter for pytest -k (e.g., "spatial_activation", "layernorm_stats")
        test_name: used for generated dir name and benchmark step name
        num_iters: Number of iterations per test config (between signposts, excluding warmup)
    """
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = test_name
    subdir = test_name

    # Step 1: Collect test configurations
    logger.info(f"=== Collecting test configs for test filter: {test_filter} ===")
    test_configs = collect_test_configs(base_command, test_filter)

    if not test_configs:
        logger.error(f"No test configs found for test filter: {test_filter}")
        return

    # Step 2: Run the profiler
    logger.info(f"=== Running profiler for test filter: {test_filter} ===")
    command = f"{base_command} -k {test_filter}"
    profiler.start("run")
    profiler.start(step_name)
    run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
    profiler.end(step_name)
    profiler.end("run")

    # Step 3: Post-process logs by config
    logger.info(f"=== Post-processing logs for test filter: {test_filter} ===")
    # results_by_config = post_process_ops_log_by_config(
    #    subdir,
    #    test_configs,
    #    num_iters,
    # )
    results_by_config = post_process_cpp_log_by_config(
        os.path.join("generated", "profiler", subdir, ".logs", "cpp_device_perf_report.csv"),
        test_configs,
        num_iters,
    )

    # Step 4: Identify best/worst configs
    best_worst = identify_best_worst_configs(results_by_config)

    # Step 5: Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"Performance Results for {test_filter}")
    logger.info(f"{'='*60}")

    # Log op sequence from first config (should be same for all)
    first_cfg = next(iter(results_by_config.values()), None)
    if first_cfg and "OP_SEQUENCE" in first_cfg:
        logger.info(f"Op sequence: {' -> '.join(first_cfg['OP_SEQUENCE'])}")

    for cfg, stats in results_by_config.items():
        logger.info(
            f"  {cfg}: MIN={stats['MIN']/1000:.2f}us, MAX={stats['MAX']/1000:.2f}us, "
            f"AVG={stats['AVG']/1000:.2f}us, STD={stats['STD']/1000:.2f}us "
            f"({stats['NUM_ITERS']} iters x {stats['NUM_DEVICES']} devices x {stats['OPS_PER_ITER']} ops)"
        )

    logger.info(f"\nBest config (lowest AVG): {best_worst['best'][0]} = {best_worst['best'][1]/1000:.2f} us")
    logger.info(f"Worst config (highest AVG): {best_worst['worst'][0]} = {best_worst['worst'][1]/1000:.2f} us")

    # Step 6: Save to JSON
    for cfg, stats in results_by_config.items():
        benchmark_data.add_measurement(profiler, 0, step_name, f"{cfg}-min", stats["MIN"])
        benchmark_data.add_measurement(profiler, 0, step_name, f"{cfg}-max", stats["MAX"])
        benchmark_data.add_measurement(profiler, 0, step_name, f"{cfg}-avg", stats["AVG"])
        benchmark_data.add_measurement(profiler, 0, step_name, f"{cfg}-std", stats["STD"])
        benchmark_data.add_measurement(profiler, 0, step_name, f"{cfg}-num_iters", stats["NUM_ITERS"])
        benchmark_data.add_measurement(profiler, 0, step_name, f"{cfg}-num_devices", stats["NUM_DEVICES"])
        benchmark_data.add_measurement(profiler, 0, step_name, f"{cfg}-ops_per_iter", stats["OPS_PER_ITER"])

    benchmark_data.add_measurement(profiler, 0, step_name, "best_config", 0)  # Placeholder for string
    benchmark_data.add_measurement(profiler, 0, step_name, "best_avg_ns", best_worst["best"][1])
    benchmark_data.add_measurement(profiler, 0, step_name, "worst_config", 0)  # Placeholder for string
    benchmark_data.add_measurement(profiler, 0, step_name, "worst_avg_ns", best_worst["worst"][1])

    benchmark_data.save_partial_run_json(
        profiler,
        run_type="tg_wan_ops",
        ml_model_name=f"wan2_2-tg-{test_filter}",
    )

    # Step 7: Generate chart
    chart_dir = os.path.join("generated", "profiler", subdir, "charts")
    op_sequence = first_cfg.get("OP_SEQUENCE") if first_cfg else None
    generate_perf_chart(test_filter, results_by_config, chart_dir, op_sequence, best_worst)

    return results_by_config, best_worst


def test_all_gather_wan_perf():
    """
    Sweep hyperparams for AllGather used in Wan2.2 and obtain perf data.
    """
    # Delete existing tracy dirs
    clear_profiler_runtime_artifacts()

    # Run each test filter and collect stats
    base_command = "pytest tests/nightly/tg/ccl/test_minimal_all_gather_async.py::test_all_gather_wan"
    test_filters = ["spatial_activation", "layernorm_stats", "rmsnorm_stats_spatial", "rmsnorm_stats_prompt"]
    num_iters = 75
    for test_filter in test_filters:
        run_perf_test(
            base_command=base_command,
            test_filter=test_filter,
            test_name=f"all_gather_wan_{test_filter}",
            num_iters=num_iters,
        )


if __name__ == "__main__":
    test_all_gather_wan_perf()

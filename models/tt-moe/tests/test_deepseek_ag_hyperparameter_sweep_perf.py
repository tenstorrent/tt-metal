# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Master all-gather performance sweep test for DeepSeek-V3.
This file orchestrates the hyperparameter sweep and collects performance metrics.
"""

import math
import os
import time

import pandas as pd
import pytest
from loguru import logger
from sweep_deepseek_ag_hyperparameters import (
    CHUNKS_PER_SYNC,
    CHUNKS_PER_SYNC_IDS,
    CONFIGS,
    CONFIGS_IDS,
    NUM_LINKS,
    NUM_LINKS_IDS,
    TOPOLOGY,
    WORKERS_PER_LINK,
    WORKERS_PER_LINK_IDS,
    get_max_chunks_per_sync,
)

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed


def total_elems(ag_output_shape):
    """Calculate total number of elements in the tensor."""
    return math.prod(ag_output_shape)


def calculate_theoretical_bandwidth(num_links, topology):
    """Calculate theoretical bandwidth in GB/s for the given configuration."""
    # Assuming 16 GB/s per link for Wormhole
    bandwidth_per_link = 16.0  # GB/s
    total_bandwidth = bandwidth_per_link * num_links

    # Ring topology uses bidirectional communication
    if topology == "ring":
        total_bandwidth *= 2

    return total_bandwidth


@pytest.mark.models_device_performance_bare_metal
def test_all_gather_chunk_perf():
    """
    Main performance sweep test for DeepSeek-V3 all-gather operations.
    Tests various hyperparameter combinations and saves performance metrics to CSV.
    """
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    rows = []

    # Setup test environment
    subdir = "deepseek_ag_perf"
    file = f"pytest models/tt-moe/tests/sweep_deepseek_ag_hyperparameters.py"
    base_command = file + "::test_all_gather_chunks_per_sync"

    # Create results directory
    start_time = time.strftime("%Y_%m_%d_%H%M%S")
    results_subdir = f"DeepSeekV3_AllGather_{start_time}"
    os.makedirs(results_subdir, exist_ok=True)

    logger.info(f"Starting DeepSeek-V3 all-gather performance sweep")
    logger.info(f"Results will be saved to: {results_subdir}")

    # Test configurations
    num_links_list = NUM_LINKS
    chunks_per_sync_list = CHUNKS_PER_SYNC
    num_workers_per_link_list = WORKERS_PER_LINK
    topology_list = TOPOLOGY
    memory_configs = ["dram", "l1"]  # Test both DRAM and L1 configurations

    # Iterate through all test combinations
    for num_links_idx, num_links in enumerate(num_links_list):
        for topology in topology_list:
            theoretical_bw = calculate_theoretical_bandwidth(num_links, topology)

            for mem_config in memory_configs:
                for i, config in enumerate(CONFIGS):
                    ag_output_shape, cluster_axis, dim = config
                    elements = total_elems(ag_output_shape)
                    total_bytes = elements * 2  # bfloat16 = 2 bytes

                    # Skip L1 for large shapes
                    seq_len = ag_output_shape[2]
                    if mem_config == "l1" and seq_len >= 512:
                        continue

                    # Determine number of devices based on cluster axis
                    if cluster_axis == 0:
                        num_devices = 8  # Expert parallel
                    else:
                        num_devices = 4  # Tensor parallel

                    # Calculate data movement
                    total_bytes_moved = (
                        total_bytes * ((num_devices - 1) / num_devices) / num_links / (2 if topology == "ring" else 1)
                    )

                    data_size_bytes_gb = total_bytes / (10**9)
                    data_size_bytes_mb = total_bytes / (10**6)
                    data_moved_bytes_mb = total_bytes_moved / (10**6)

                    mode = "decode" if seq_len <= 32 else "prefill"
                    axis_type = "ep" if cluster_axis == 0 else "tp"

                    logger.info(f"\n{'='*60}")
                    logger.info(f"Testing {mode} mode, seq_len={seq_len}, {axis_type}, {mem_config}")
                    logger.info(f"Shape: {ag_output_shape}, Data size: {data_size_bytes_gb:.3f} GB")
                    logger.info(f"Num devices: {num_devices}, Num links: {num_links}, Topology: {topology}")

                    # Track best performance for this configuration
                    best_bandwidth_gbps = -float("inf")
                    best_chunks_per_sync = None
                    best_num_workers_per_link = None
                    best_latency_us = float("inf")

                    # Test different hyperparameter combinations
                    for j, chunks_per_sync in enumerate(chunks_per_sync_list):
                        for k, num_workers_per_link in enumerate(num_workers_per_link_list):
                            # Skip invalid combinations
                            if chunks_per_sync != None and num_workers_per_link == None:
                                continue
                            elif chunks_per_sync == None and num_workers_per_link != None:
                                continue

                            # Construct test command
                            cols = ["DEVICE KERNEL"]
                            op_name = "AllGatherAsync"
                            step_name = f"deepseek_ag_{mode}_{seq_len}_{axis_type}_{num_links}L_{chunks_per_sync}_{num_workers_per_link}_{topology}_{mem_config}"

                            # Build pytest command with filters
                            final_command = (
                                base_command
                                + f' -k "{CONFIGS_IDS[i]} and {CHUNKS_PER_SYNC_IDS[j]} and '
                                + f"{WORKERS_PER_LINK_IDS[k]} and {NUM_LINKS_IDS[num_links_idx]} and "
                                + f'{topology} and {mem_config}"'
                            )

                            results = None
                            try:
                                profiler.start("run")
                                profiler.start(step_name)
                                results = run_device_perf_detailed(
                                    final_command, subdir, cols, op_name, has_signposts=False, warmup_iters=10
                                )
                                profiler.end(step_name)
                                profiler.end("run")
                            except Exception as e:
                                logger.error(f"Error running command: {e}")
                                logger.debug(f"Failed command: {final_command}")
                                continue

                            if results is None:
                                continue

                            # Extract performance metrics
                            measured_min = results[cols[0]]["MIN"]
                            measured_max = results[cols[0]]["MAX"]
                            measured_avg = results[cols[0]]["AVG"]
                            measured_std = results[cols[0]]["STD"]

                            # Calculate actual chunks_per_sync if MAX was used
                            final_chunks_per_sync = (
                                get_max_chunks_per_sync(num_devices, ag_output_shape, num_links)
                                if chunks_per_sync == "MAX"
                                else chunks_per_sync
                            )

                            # Calculate bandwidth
                            current_bandwidth_gbps = total_bytes_moved / measured_avg
                            efficiency_pct = (current_bandwidth_gbps / theoretical_bw) * 100

                            logger.info(
                                f"  chunks={final_chunks_per_sync}, workers={num_workers_per_link}: "
                                f"{measured_avg/1000:.3f} us, {current_bandwidth_gbps:.3f} GB/s "
                                f"({efficiency_pct:.1f}% efficiency)"
                            )

                            # Append row for CSV output
                            rows.append(
                                {
                                    "Mode": mode,
                                    "Output Shape": str(ag_output_shape),
                                    "Seq Length": seq_len,
                                    "Dim": dim,
                                    "Cluster Axis": cluster_axis,
                                    "Axis Type": axis_type,
                                    "Num Devices": num_devices,
                                    "Num Links": num_links,
                                    "Topology": topology,
                                    "Memory Config": mem_config,
                                    "Chunks Per Sync": final_chunks_per_sync,
                                    "Num Workers Per Link": num_workers_per_link,
                                    "Data Size (MB)": data_size_bytes_mb,
                                    "Data Moved (MB)": data_moved_bytes_mb,
                                    "Latency Avg (us)": measured_avg / 1000.0,
                                    "Latency Max (us)": measured_max / 1000.0,
                                    "Latency StdDev (us)": measured_std / 1000.0,
                                    "Bandwidth (GB/s)": current_bandwidth_gbps,
                                    "Theoretical BW (GB/s)": theoretical_bw,
                                    "Efficiency (%)": efficiency_pct,
                                }
                            )

                            # Update best performance trackers
                            if current_bandwidth_gbps > best_bandwidth_gbps:
                                best_bandwidth_gbps = current_bandwidth_gbps
                                best_chunks_per_sync = final_chunks_per_sync
                                best_num_workers_per_link = num_workers_per_link
                                best_latency_us = measured_avg / 1000.0

                            # Save measurement to benchmark data
                            benchmark_data.add_measurement(
                                profiler,
                                0,
                                step_name,
                                f"{op_name}-{final_chunks_per_sync}-chunk-{num_workers_per_link}-workers-{topology}-min",
                                measured_min,
                            )
                            benchmark_data.add_measurement(
                                profiler,
                                0,
                                step_name,
                                f"{op_name}-{final_chunks_per_sync}-chunk-{num_workers_per_link}-workers-{topology}-max",
                                measured_max,
                            )
                            benchmark_data.add_measurement(
                                profiler,
                                0,
                                step_name,
                                f"{op_name}-{final_chunks_per_sync}-chunk-{num_workers_per_link}-workers-{topology}-avg",
                                measured_avg,
                            )
                            benchmark_data.add_measurement(
                                profiler,
                                0,
                                step_name,
                                f"{op_name}-{final_chunks_per_sync}-chunk-{num_workers_per_link}-workers-{topology}-std",
                                measured_std,
                            )

                    # Report best configuration for this shape
                    if best_chunks_per_sync is not None:
                        logger.info(f"\n✅ Best config for {mode} seq_len={seq_len} {axis_type} {mem_config}:")
                        logger.info(f"   Bandwidth: {best_bandwidth_gbps:.3f} GB/s")
                        logger.info(f"   Latency: {best_latency_us:.3f} us")
                        logger.info(f"   Chunks per sync: {best_chunks_per_sync}")
                        logger.info(f"   Workers per link: {best_num_workers_per_link}")
                        logger.info(f"   Efficiency: {(best_bandwidth_gbps/theoretical_bw)*100:.1f}%")

                # Save intermediate results per topology and memory config
                if len(rows) > 0:
                    curr_time = time.strftime("%Y_%m_%d_%H%M%S")
                    csv_path = (
                        f"{results_subdir}/DeepSeekV3_AG_{num_links}links_{topology}_{mem_config}_{curr_time}.csv"
                    )
                    df = pd.DataFrame(rows)
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved intermediate results to {csv_path}")

    # Save final aggregated CSV with all results
    if len(rows) > 0:
        curr_time = time.strftime("%Y_%m_%d_%H%M%S")
        csv_path = f"{results_subdir}/DeepSeekV3_AllGather_Complete_{curr_time}.csv"
        logger.info(f"\nSaving complete performance table to {csv_path}")
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Generate summary statistics
        summary_csv = f"{results_subdir}/DeepSeekV3_AG_Summary_{curr_time}.csv"
        summary = (
            df.groupby(["Mode", "Seq Length", "Axis Type", "Memory Config", "Num Links", "Topology"])
            .agg(
                {
                    "Bandwidth (GB/s)": ["max", "mean"],
                    "Latency Avg (us)": ["min", "mean"],
                    "Efficiency (%)": ["max", "mean"],
                }
            )
            .round(2)
        )
        summary.to_csv(summary_csv)
        logger.info(f"Saved summary statistics to {summary_csv}")

    # Save benchmark JSON data
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="deepseek_v3_all_gather_sweep",
        ml_model_name="deepseek_v3",
    )

    logger.info(f"\n{'='*60}")
    logger.info("DeepSeek-V3 All-Gather Performance Sweep Complete!")
    logger.info(f"Results saved to: {results_subdir}/")
    logger.info(f"{'='*60}")

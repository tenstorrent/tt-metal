# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import pytest
from loguru import logger
import os
import time
import pandas as pd
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed
from tests.ttnn.multidevice_perf_tests.sweep_all_gather_hyperparameters_t3000 import get_max_chunks_per_sync
from tests.ttnn.multidevice_perf_tests.sweep_all_gather_hyperparameters_galaxy import (
    CONFIGS,
    CHUNKS_PER_SYNC,
    WORKERS_PER_LINK,
    TOPOLOGY,
    CONFIGS_IDS,
    CHUNKS_PER_SYNC_IDS,
    WORKERS_PER_LINK_IDS,
)


def total_elems(ag_output_shape):
    return math.prod(ag_output_shape)


@pytest.mark.models_device_performance_bare_metal
def test_all_gather_chunk_perf():
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    rows = []

    num_links = 4
    subdir = "ag_perf"
    file = f"pytest tests/ttnn/multidevice_perf_tests/sweep_all_gather_hyperparameters_galaxy.py"

    base_command = file + "::test_all_gather_chunks_per_sync"

    chunks_per_sync_list = CHUNKS_PER_SYNC
    num_workers_per_link_list = WORKERS_PER_LINK
    topology_list = TOPOLOGY
    start_time = time.time()
    # create subdirectory for this run
    results_subdir = f"AllGather_{start_time}"
    os.makedirs(results_subdir, exist_ok=True)
    for topology in topology_list:
        for i, config in enumerate(CONFIGS):
            ag_output_shape, cluster_axis, dim = config
            elements = total_elems(ag_output_shape)
            total_bytes = elements * 2
            if cluster_axis == 0:
                num_devices = 8
            else:
                num_devices = 4
            total_bytes_moved = (
                total_bytes * ((num_devices - 1) / num_devices) / num_links / (2 if topology == "ring" else 1)
            )

            data_size_bytes_gb = total_bytes / (10**9)
            data_size_bytes_mb = total_bytes / (10**6)
            data_moved_bytes_mb = total_bytes_moved / (10**6)

            logger.info(f"Total elements: {elements}, Data size: {data_size_bytes_gb:.3f} GB")

            # Track best bandwidth for this shape
            best_bandwidth_gbps = -float("inf")
            best_chunks_per_sync = None
            best_num_workers_per_link = None
            for j, chunks_per_sync in enumerate(chunks_per_sync_list):
                for k, num_workers_per_link in enumerate(num_workers_per_link_list):
                    if chunks_per_sync != None and num_workers_per_link == None:
                        continue
                    elif chunks_per_sync == None and num_workers_per_link != None:
                        continue

                    cols = ["DEVICE KERNEL"]
                    op_name = "AllGatherAsync"
                    step_name = f"all_gather_chunk_perf_6U_{chunks_per_sync}_{num_workers_per_link}_{topology}_perf"

                    # Filter by both chunks_per_sync and shape
                    final_command = (
                        base_command
                        + f' -k "{CHUNKS_PER_SYNC_IDS[j]} and {CONFIGS_IDS[i]} and {WORKERS_PER_LINK_IDS[k]} and {topology}"'
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
                        logger.error(f"Error running command {final_command}: {e}")
                        continue

                    # Get the measured performance
                    measured_min = results[cols[0]]["MIN"]
                    measured_max = results[cols[0]]["MAX"]
                    measured_avg = results[cols[0]]["AVG"]
                    measured_std = results[cols[0]]["STD"]

                    final_chunks_per_sync = (
                        get_max_chunks_per_sync(num_devices, ag_output_shape, num_links)
                        if chunks_per_sync == "MAX"
                        else chunks_per_sync
                    )
                    logger.info(
                        f"Measured performance for topology {topology}, shape {ag_output_shape} with data size {data_size_bytes_mb:.3f} MB, chunks per sync {final_chunks_per_sync}, num workers per link {num_workers_per_link}: {measured_avg/1000:.3f} us at {total_bytes_moved/measured_avg:.6f} GB/s"
                    )

                    # Append row for CSV output
                    current_bandwidth_gbps = total_bytes_moved / measured_avg
                    rows.append(
                        {
                            "Output Shape": str(ag_output_shape),
                            "Dim": dim,
                            "Cluster Axis": cluster_axis,
                            "Num Devices": num_devices,
                            "Num Links": num_links,
                            "Topology": topology,
                            "Chunks Per Sync": final_chunks_per_sync,
                            "Num Workers Per Link": num_workers_per_link,
                            "Data Size in MB": data_size_bytes_mb,
                            "Data Moved in MB": data_moved_bytes_mb,
                            "Measured Average (us)": measured_avg / 1000.0,
                            "Measured Max (us)": measured_max / 1000.0,
                            "Standard deviation (us)": measured_std / 1000.0,
                            "Bandwidth (GB/s)": current_bandwidth_gbps,
                        }
                    )

                    # Update best bandwidth trackers for this shape
                    if current_bandwidth_gbps > best_bandwidth_gbps:
                        best_bandwidth_gbps = current_bandwidth_gbps
                        best_chunks_per_sync = final_chunks_per_sync
                        best_num_workers_per_link = num_workers_per_link

                    # Save the measurement
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
            curr_time = time.strftime("%Y_%m_%d_%H%M%S")
            csv_path = f"{results_subdir}/AllGatherPerformance_{curr_time}_{ag_output_shape}_{topology}.csv"
            logger.info(f"Saving performance table to {csv_path}")
            if len(rows) > 0:
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved performance table for {ag_output_shape} to {csv_path}")

            # After iterating over all chunks per sync, report best bandwidth for this shape
            if best_chunks_per_sync is not None:
                logger.info(
                    f"Best bandwidth for shape {ag_output_shape}: {best_bandwidth_gbps:.6f} GB/s at (chunks per sync: {best_chunks_per_sync}, num workers per link: {best_num_workers_per_link})"
                )

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"all_gather_chunk_perf",
        ml_model_name="ag",
    )

    # Save aggregated CSV
    curr_time = time.strftime("%Y_%m_%d_%H%M%S")
    csv_path = f"{results_subdir}/AllGatherPerformance_{curr_time}.csv"
    logger.info(f"Saving performance table to {csv_path}")
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved performance table to {csv_path}")

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
from tests.nightly.t3000.ccl.test_minimal_all_gather_async import get_max_chunks_per_sync


def total_elems(ag_output_shape):
    return math.prod(ag_output_shape)


@pytest.mark.parametrize("arch_type", ["T3K"])
@pytest.mark.models_device_performance_bare_metal
def test_all_gather_chunk_perf(
    arch_type,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    rows = []

    subdir = "ag_perf"
    num_links = 1
    if arch_type == "T3K":
        file = f"pytest tests/nightly/t3000/ccl/test_minimal_all_gather_async.py"
        num_links = 1
    else:
        raise ValueError(f"Invalid arch_type: {arch_type}")

    base_command = file + "::test_all_gather_chunks_per_sync"

    num_devices = 8
    output_shapes = [[1, 1, 352, 5120], [1, 1, 1024, 5120], [1, 1, 8192, 10240], [1, 1, 8192, 16384]]
    chunks_per_sync_list = ["MAX", 160, 80, 40, 20, 10]
    num_workers_per_link_list = [8, 4, 2, 1]
    topology_list = ["ring"]
    for i, ag_output_shape in enumerate(output_shapes):
        elements = total_elems(ag_output_shape)
        total_bytes = elements * 2
        total_bytes_moved = total_bytes * (7 / 8)
        data_size_bytes_gb = total_bytes / (10**9)
        data_size_bytes_mb = total_bytes / (10**6)
        data_moved_bytes_mb = total_bytes_moved / (10**6)

        logger.info(f"Total elements: {elements}, Data size: {data_size_bytes_gb:.3f} GB")

        for topology in topology_list:
            # Track best bandwidth for this shape
            best_bandwidth_gbps = -float("inf")
            best_chunks_per_sync = None
            best_num_workers_per_link = None
            for chunks_per_sync in chunks_per_sync_list:
                for num_workers_per_link in num_workers_per_link_list:
                    cols = ["DEVICE KERNEL"]
                    op_name = "AllGatherAsync"
                    step_name = (
                        f"all_gather_chunk_perf_{arch_type}_{chunks_per_sync}_{num_workers_per_link}_{topology}_perf"
                    )

                    # Filter by both chunks_per_sync and shape
                    shape_str = f"ag_output_shape{i}"
                    final_command = (
                        base_command
                        + f' -k "{chunks_per_sync}-chunks and {shape_str} and {num_workers_per_link}-workers and {topology}"'
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
                    rows.append(
                        {
                            "Output Shape": str(ag_output_shape),
                            "Topology": topology,
                            "Chunks Per Sync": final_chunks_per_sync,
                            "Num Workers Per Link": num_workers_per_link,
                            "Data Size in MB": data_size_bytes_mb,
                            "Data Moved in MB": data_moved_bytes_mb,
                            "Measured Average (us)": measured_avg / 1000.0,
                            "Measured Max (us)": measured_max / 1000.0,
                            "Standard deviation (us)": measured_std / 1000.0,
                            "Bandwidth (GB/s)": total_bytes_moved / measured_avg,
                        }
                    )

                    # Update best bandwidth trackers for this shape
                    current_bandwidth_gbps = total_bytes_moved / measured_avg
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
    csv_path = f"AllGatherPerformance_{curr_time}.csv"
    logger.info(f"Saving performance table to {csv_path}")
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved performance table to {csv_path}")

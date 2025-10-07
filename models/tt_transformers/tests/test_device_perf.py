# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import pandas as pd
import pytest
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf
from tools.tracy.process_model_log import get_latest_ops_log_filename

PREFILL_OP_START_INDEX = 0
PREFILL_OP_END_INDEX = 35
DECODER_OP_START_INDEX = 4
DECODER_OP_END_INDEX = -22


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "max_seq_len",
    (1024,),
)
@pytest.mark.parametrize(
    "run_prefill",
    (True,),
)
@pytest.mark.parametrize(
    "run_decode",
    (True,),
)
def test_decoder_device_perf_one_iter(mesh_device, device_params, batch_size, subdir, device_analysis_types, is_ci_env):
    # Run the existing unit test under the device profiler for a single iteration
    command = "models/tt_transformers/tests/simple_text_demo.py -k device-perf"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()

    profiler.start("run")
    profiler.start("decoder-perf-op-metrics")

    _ = run_device_perf(
        command,
        subdir,
        num_iterations=1,
        cols=cols,
        batch_size=batch_size,
        device_analysis_types=device_analysis_types,
    )

    # Parse the latest ops CSV and aggregate per-op metrics
    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows(df)
    # Compute per-op averages for kernel duration and op-to-op latency when available
    metrics = []
    if "DEVICE KERNEL DURATION [ns]" in df.columns:
        kernel_avg = df.groupby("OP CODE")["DEVICE KERNEL DURATION [ns]"].apply(
            lambda s: s[s != "-"].astype(float).mean()
        )
        metrics.append(("kernel", kernel_avg))
    if "OP TO OP LATENCY [ns]" in df.columns:
        op2op_avg = df.groupby("OP CODE")["OP TO OP LATENCY [ns]"].apply(lambda s: s[s != "-"].astype(float).mean())
        metrics.append(("op_to_op", op2op_avg))
    if "DEVICE KERNEL FIRST TO LAST START [ns]" in df.columns:
        first_last_avg = df.groupby("OP CODE")["DEVICE KERNEL FIRST TO LAST START [ns]"].apply(
            lambda s: s[s != "-"].astype(float).mean()
        )
        metrics.append(("first_to_last_start", first_last_avg))

    # Export per-op metrics
    for metric_name, series in metrics:
        for op_code, value in series.dropna().items():
            measurement_name = f"{op_code}-decoder-{metric_name}-avg"
            benchmark_data.add_measurement(
                profiler,
                0,
                "decoder-perf-op-metrics",
                measurement_name,
                float(value),
            )
            logger.info(f"{measurement_name}: {value} ns")

    profiler.end("decoder-perf-op-metrics")
    profiler.end("run")

    # Save partial run (no-op outside CI)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="ttnn_decoder_unit",
        ml_model_name="ttnn-decoder",
    )

    # No strict assertions on perf; test succeeds if profiling and export ran
    assert True

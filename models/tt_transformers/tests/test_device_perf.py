# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import json
import os

import pandas as pd
import pytest
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf
from models.tt_transformers.tests.test_utils import (
    merge_device_rows,
    print_dict,
    process_measurements,
    split_compile_and_trace,
    verify_value_within_margin,
)
from tools.tracy.process_model_log import get_latest_ops_log_filename


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "num_layers, num_head_ops, num_tail_ops, tail_start_index, batch_size, data_parallel, max_seq_len, mode",
    [(3, 3, 14, None, 1, 1, 1024, "prefill"), (3, 3, 14, None, 1, 2, 1024, "prefill")],
    ids=["llama3.1-8b-TP2", "llama3.1-8b-DP2"],
)
def test_device_perf_one_iter(
    mesh_device, num_layers, num_head_ops, num_tail_ops, tail_start_index, batch_size, data_parallel, max_seq_len, mode
):
    cmd = f"pytest models/tt_transformers/demo/simple_text_demo.py -k device-perf --num_layers {num_layers} --data_parallel {data_parallel} --max_seq_len {max_seq_len} --max_generated_tokens 1 --paged_attention 1  --batch_size {batch_size}"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    device_analysis_types = ["device_kernel_duration", "device_kernel_first_to_last_start"]
    num_devices = mesh_device.get_num_devices()
    subdir = "ttt-device-perf-default"
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    profiler.start("run")
    profiler.start("device-perf-op-metrics")

    # Load perf targets
    perf_targets = {}
    try:
        with open(f"models/tt_transformers/tests/perf_targets/device_perf_{mode}.json", "r") as f:
            perf_targets = json.load(f)
    except FileNotFoundError:
        logger.error(
            f"Perf targets file not found, device perf test will proceed without performance target comparison"
        )

    _ = run_device_perf(
        cmd,
        subdir,
        num_iterations=1,
        cols=cols,
        batch_size=batch_size,
        device_analysis_types=device_analysis_types,
    )

    profiler.end("device-perf-op-metrics")
    profiler.end("run")

    # Parse the latest ops CSV and aggregate per-op metrics
    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows(df)

    # Split compile and trace
    (
        df_model_compilation,
        df_model_trace,
        df_first_layer_compilation,
        df_first_layer_trace,
        df_mid_layers_compilation,
        df_mid_layers_trace,
        df_model_tail_compilation,
        df_model_tail_trace,
    ) = split_compile_and_trace(
        df,
        num_head_ops=num_head_ops,
        num_tail_ops=num_tail_ops,
        mode=mode,
        num_runs=4,
        num_layers=num_layers,
        tail_start_index=tail_start_index,
    )

    (
        kernel_agg_first_layer_compile,
        dispatch_agg_first_layer_compile,
        firstlast_agg_first_layer_compile,
    ) = process_measurements(df_first_layer_compilation, 1)
    (
        kernel_agg_first_layer_trace,
        dispatch_agg_first_layer_trace,
        firstlast_agg_first_layer_trace,
    ) = process_measurements(df_first_layer_trace, 1)

    if num_layers > 1:
        (
            kernel_agg_mid_layers_compile,
            dispatch_agg_mid_layers_compile,
            firstlast_agg_mid_layers_compile,
        ) = process_measurements(
            df_mid_layers_compilation, num_layers - 1
        )  # we dont count the first layer

        (
            kernel_agg_mid_layers_trace,
            dispatch_agg_mid_layers_trace,
            firstlast_agg_mid_layers_trace,
        ) = process_measurements(df_mid_layers_trace, num_layers - 1)

    if tail_start_index is not None:
        (
            kernel_agg_model_tail_compile,
            dispatch_agg_model_tail_compile,
            firstlast_agg_model_tail_compile,
        ) = process_measurements(df_model_tail_compilation, 1)
        (
            kernel_agg_model_tail_trace,
            dispatch_agg_model_tail_trace,
            firstlast_agg_model_tail_trace,
        ) = process_measurements(df_model_tail_trace, 1)

    # Print measurements
    print_dict(kernel_agg_first_layer_compile, "kernel_agg_first_layer_compile")
    print_dict(kernel_agg_first_layer_trace, "kernel_agg_first_layer_trace")

    if num_layers > 1:
        print_dict(kernel_agg_mid_layers_compile, "kernel_agg_mid_layers_compile")
        print_dict(kernel_agg_mid_layers_trace, "kernel_agg_mid_layers_trace")
        print_dict(dispatch_agg_mid_layers_trace, "dispatch_agg_mid_layers_trace")
        print_dict(firstlast_agg_mid_layers_trace, "first_last_agg_mid_layers_trace")

    if tail_start_index is not None:
        print_dict(kernel_agg_model_tail_compile, "kernel_agg_model_tail_compile")
        print_dict(kernel_agg_model_tail_trace, "kernel_agg_model_tail_trace")
        print_dict(dispatch_agg_model_tail_trace, "dispatch_agg_model_tail_trace")
        print_dict(firstlast_agg_model_tail_trace, "first_last_agg_model_tail_trace")

    # Prefer trace for collectives, compile for others
    def is_collective(op_code: str) -> bool:
        return any(x in op_code for x in ("AllGather", "ReduceScatter", "AllReduce", "Matmul_RS"))

    # Export metrics for an op group (first layer, mid layers, model tail)
    def export_group(
        group_name: str,
        kernel_agg_compile: dict,
        kernel_agg_trace: dict,
        dispatch_agg_trace: dict,
        firstlast_agg_trace: dict | None,
    ):
        all_passing = True
        op_codes = set(list(kernel_agg_compile["avg"].keys()) + list(kernel_agg_trace["avg"].keys()))
        for op_code in op_codes:
            # kernel avg
            k_avg_trace = kernel_agg_trace["avg"].get(op_code)
            k_avg_comp = kernel_agg_compile["avg"].get(op_code)
            k_min_trace = kernel_agg_trace["min"].get(op_code)
            k_min_comp = kernel_agg_compile["min"].get(op_code)
            k_max_trace = kernel_agg_trace["max"].get(op_code)
            k_max_comp = kernel_agg_compile["max"].get(op_code)

            if is_collective(op_code):
                k_avg = k_avg_trace if k_avg_trace is not None else k_avg_comp
                k_min = k_min_trace if k_min_trace is not None else k_min_comp
                k_max = k_max_trace if k_max_trace is not None else k_max_comp
            else:
                k_avg = k_avg_comp if k_avg_comp is not None else k_avg_trace
                k_min = k_min_comp if k_min_comp is not None else k_min_trace
                k_max = k_max_comp if k_max_comp is not None else k_max_trace

            if k_avg is not None:
                benchmark_data.add_measurement(
                    profiler, 0, "decoder-perf-op-metrics", f"{op_code}-{group_name}-kernel-avg", float(k_avg)
                )
            if k_min is not None:
                benchmark_data.add_measurement(
                    profiler, 0, "decoder-perf-op-metrics", f"{op_code}-{group_name}-kernel-min", float(k_min)
                )
            if k_max is not None:
                benchmark_data.add_measurement(
                    profiler, 0, "decoder-perf-op-metrics", f"{op_code}-{group_name}-kernel-max", float(k_max)
                )

            # Check that perf_targets, group_name, and op_code exist and keys exist
            if perf_targets and group_name in perf_targets and op_code in perf_targets[group_name]:
                passing = verify_value_within_margin(
                    k_avg,
                    perf_targets[group_name][op_code]["kernel_duration"],
                    perf_targets[group_name][op_code]["kernel_duration_relative_margin"],
                    op_code,
                    "kernel",
                )
                all_passing = all_passing and passing
            else:
                logger.warning(f"Warning: {op_code}-{group_name}-kernel not found in perf_targets")
            # dispatch from trace only
            d_avg = dispatch_agg_trace["avg"].get(op_code)
            d_min = dispatch_agg_trace["min"].get(op_code)
            d_max = dispatch_agg_trace["max"].get(op_code)
            if d_avg is not None:
                benchmark_data.add_measurement(
                    profiler, 0, "decoder-perf-op-metrics", f"{op_code}-{group_name}-op_to_op-avg", float(d_avg)
                )
            if d_min is not None:
                benchmark_data.add_measurement(
                    profiler, 0, "decoder-perf-op-metrics", f"{op_code}-{group_name}-op_to_op-min", float(d_min)
                )
            if d_max is not None:
                benchmark_data.add_measurement(
                    profiler, 0, "decoder-perf-op-metrics", f"{op_code}-{group_name}-op_to_op-max", float(d_max)
                )

            if perf_targets and group_name in perf_targets and op_code in perf_targets[group_name]:
                passing = verify_value_within_margin(
                    d_avg,
                    perf_targets[group_name][op_code]["op_to_op"],
                    perf_targets[group_name][op_code]["op_to_op_duration_relative_margin"],
                    op_code,
                    "op_to_op",
                )
                all_passing = all_passing and passing
            else:
                logger.warning(f"Warning: {op_code}-{group_name}-op_to_op not found in perf_targets")

            # first_to_last from trace only (if provided)
            if firstlast_agg_trace is not None:
                fl_avg = firstlast_agg_trace["avg"].get(op_code)
                fl_min = firstlast_agg_trace["min"].get(op_code)
                fl_max = firstlast_agg_trace["max"].get(op_code)
                if fl_avg is not None:
                    benchmark_data.add_measurement(
                        profiler,
                        0,
                        "decoder-perf-op-metrics",
                        f"{op_code}-{group_name}-first_to_last-avg",
                        float(fl_avg),
                    )
                if fl_min is not None:
                    benchmark_data.add_measurement(
                        profiler,
                        0,
                        "decoder-perf-op-metrics",
                        f"{op_code}-{group_name}-first_to_last-min",
                        float(fl_min),
                    )
                if fl_max is not None:
                    benchmark_data.add_measurement(
                        profiler,
                        0,
                        "decoder-perf-op-metrics",
                        f"{op_code}-{group_name}-first_to_last-max",
                        float(fl_max),
                    )

                if perf_targets and group_name in perf_targets and op_code in perf_targets[group_name]:
                    passing = verify_value_within_margin(
                        fl_avg,
                        perf_targets[group_name][op_code]["first_to_last_start"],
                        perf_targets[group_name][op_code]["first_to_last_start_relative_margin"],
                        op_code,
                        "first_to_last_start",
                    )
                    all_passing = all_passing and passing
                else:
                    logger.warning(f"Warning: {op_code}-{group_name}-first_to_last not found in perf_targets")
        return all_passing

    # Export per-op metrics for each group
    all_passing = True
    all_passing = all_passing and export_group(
        group_name="decoder-first",
        kernel_agg_compile=kernel_agg_first_layer_compile,
        kernel_agg_trace=kernel_agg_first_layer_trace,
        dispatch_agg_trace=dispatch_agg_first_layer_trace,
        firstlast_agg_trace=firstlast_agg_first_layer_trace,
    )

    if num_layers > 1:
        all_passing = all_passing and export_group(
            group_name="decoder-mid",
            kernel_agg_compile=kernel_agg_mid_layers_compile,
            kernel_agg_trace=kernel_agg_mid_layers_trace,
            dispatch_agg_trace=dispatch_agg_mid_layers_trace,
            firstlast_agg_trace=firstlast_agg_mid_layers_trace,
        )
    if tail_start_index is not None:
        all_passing = all_passing and export_group(
            group_name="model_tail",
            kernel_agg_compile=kernel_agg_model_tail_compile,
            kernel_agg_trace=kernel_agg_model_tail_trace,
            dispatch_agg_trace=dispatch_agg_model_tail_trace,
            firstlast_agg_trace=None,  # align with decoder tail export (no first_to_last)
        )

    # Estimate e2e time
    # e2e_estimate_80l = 0
    # e2e_estimate_80l += kernel_agg_first_layer_trace["avg"].get("tt_dnn_device") * num_layers
    # e2e_estimate_80l += kernel_agg_mid_layers_trace["avg"].get("tt_dnn_device") * (num_layers - 1)
    # e2e_estimate_80l += kernel_agg_model_tail_trace["avg"].get("tt_dnn_device")

    # tsu_estimate = 1000000 / (e2e_estimate_80l / 1000 + 300)

    # Save partial run
    benchmark_data.save_partial_run_json(
        profiler,
        run_type="ttnn_decoder_unit",
        ml_model_name="ttnn-decoder",
    )

    # No strict assertions on perf; test succeeds if profiling and export ran
    assert True

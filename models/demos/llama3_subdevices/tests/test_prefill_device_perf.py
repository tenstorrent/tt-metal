# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
from loguru import logger
import math
import pandas as pd
from collections import defaultdict
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf
from tt_metal.tools.profiler.process_model_log import (
    get_latest_ops_log_filename,
)

from termcolor import colored
import json

PREFILL_OP_START_INDEX = 0
PREFILL_OP_END_INDEX = 35


def merge_device_rows(df):
    block_by_device = defaultdict(list)

    for _, row in df.iterrows():
        op_name = row["OP CODE"]
        op_type = row["OP TYPE"]

        if op_type == "tt_dnn_device":
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
                print(
                    colored(
                        f"Warning: Device {device_id} is missing operation {op_name} at index {global_index}", "yellow"
                    )
                )
                continue
            if op_name is None:
                op_name = block_by_device[device_id][0][0]
            elif op_name != block_by_device[device_id][0][0]:
                missing_devices.append(device_id)
                continue

            blocks.append(block_by_device[device_id].pop(0))

        if missing_devices:
            print(
                colored(
                    f"Warning: {op_name} at index {global_index} not present in CSV for {len(missing_devices)} devices {missing_devices} - do not trust data for this op or directly subsequent ops with the same name",
                    "yellow",
                )
            )

        if not blocks:
            break

        if "AllGather" in op_name or "ReduceScatter" in op_name or "AllReduce" in op_name:
            # For collective ops, take the average duration over all rows within a block
            device_kernel_durations = [
                d["DEVICE KERNEL DURATION [ns]"]
                for _, d in blocks
                if "DEVICE KERNEL DURATION [ns]" in d and not math.isnan(d["DEVICE KERNEL DURATION [ns]"])
            ]

            average_duration = (
                sum(device_kernel_durations) / len(device_kernel_durations) if device_kernel_durations else float("nan")
            )
            # Use the first block's data but update its duration with the average
            base_block = blocks[0][1].copy()
            base_block["DEVICE KERNEL DURATION [ns]"] = average_duration
            merged_blocks.append(base_block)
        else:
            # For non-collective ops, take the row with maximum duration
            max_duration_block = max(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(max_duration_block[1])

        global_index += 1

    return pd.DataFrame(merged_blocks)


def build_duration_dict(raw_dict, column_name):
    op_code_dict = {}
    for entry in raw_dict:
        if column_name not in entry:
            print(f"Warning: {entry} does not have column {column_name}")
        op_code = entry["OP CODE"]
        duration = entry[column_name]
        if op_code not in op_code_dict:
            op_code_dict[op_code] = []
        op_code_dict[op_code].append(duration)
    return op_code_dict


def build_duration_per_instance_dict(input_dict, num_layers):
    per_instance_dict = {}
    for op_code in input_dict:
        num_ops_with_op_code = len(input_dict[op_code])
        num_instances = num_ops_with_op_code // num_layers
        if num_ops_with_op_code % num_layers != 0:
            print(f"Warning: {op_code} has {num_ops_with_op_code} ops, not a multiple of {num_layers} layers")
            print_dict(input_dict, "input_dict")
            assert num_ops_with_op_code % num_layers == 0
        for iteration_id in range(num_layers):
            for instance_id in range(num_instances):
                op_code_with_id = f"{op_code}_{instance_id}"
                if op_code_with_id not in per_instance_dict:
                    per_instance_dict[op_code_with_id] = []
                per_instance_dict[op_code_with_id].append(
                    input_dict[op_code][iteration_id * num_instances + instance_id]
                )
    return per_instance_dict


def average_per_instance_dict(input_dict):
    averaged_dict = {}
    for op_code_with_id in input_dict:
        averaged_dict[op_code_with_id] = sum(input_dict[op_code_with_id]) / len(input_dict[op_code_with_id])
    return averaged_dict


def min_per_instance_dict(input_dict):
    min_dict = {}
    for op_code_with_id in input_dict:
        min_dict[op_code_with_id] = min(input_dict[op_code_with_id])
    return min_dict


def max_per_instance_dict(input_dict):
    max_dict = {}
    for op_code_with_id in input_dict:
        max_dict[op_code_with_id] = max(input_dict[op_code_with_id])
    return max_dict


def print_dict(input_dict, dict_name):
    # print dict as a readable python dict
    print(f"\n{dict_name} = {{")
    for op_code_with_id in input_dict:
        print(f'"{op_code_with_id}": {input_dict[op_code_with_id]},')
    print("}")


def is_collective_op(op_code):
    return "AllGather" in op_code or "ReduceScatter" in op_code or "AllReduce" in op_code


@pytest.mark.models_device_performance_bare_metal
# To update:
# Run FAKE_DEVICE=TG pytest models/demos/llama3_subdevices/tests/test_prefill_device_perf.py::test_llama_TG_perf_device
# Copy the printed kernel_duration_per_instance_averaged_dict and dispatch_duration_per_instance_averaged_dict dictionaries
# Manually compare each entry between old-expected and the new average values
# - Any perf regressions? Everything as expected?
# If all looks good, update the expected_kernel_times_dict and expected_dispatch_times_dict with the new average values
# If the op list changed (new ops, less ops, fused ops), then update mapping_op_code_to_name and give the new ops meaningful names
# Run at least once again to verify the new expected values are correct and margins hold
@pytest.mark.parametrize("seqlen", [4096, 128])
def test_llama_TG_perf_device(
    seqlen,
    reset_seeds,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = f"tg-llama-prefill-device-perf-{seqlen}-noTrace"
    batch_size = 1
    subdir = f"tg-llama-prefill-device-perf-{seqlen}-noTrace"
    num_iterations = 1
    num_layers = 1

    # Load perf targets
    with open(f"models/demos/llama3_subdevices/tests/perf_targets/prefill_{seqlen}.json", "r") as f:
        perf_targets = json.load(f)

    command = f"pytest models/demos/llama3_subdevices/tests/unit_tests/test_llama_decoder_prefill.py -k {seqlen}"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    profiler.start("run")
    profiler.start(step_name)
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    profiler.end(step_name)
    profiler.end("run")

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows(df)
    # Excluding compile run and capture trace entries
    # df_model_compilation = df[: int(len(df) / 3)]
    # df_model_trace = df[int(len(df) / 3 * 2) :]
    df_model_compilation = df
    df_model_trace = df

    # Excluding model embeddings and lmhead+sampling ops
    df_layers_compilation = df_model_compilation[PREFILL_OP_START_INDEX:PREFILL_OP_END_INDEX]
    df_layers_trace = df_model_trace[PREFILL_OP_START_INDEX:PREFILL_OP_END_INDEX]
    # Use layers 2-9 for verifying against targets for more stability
    df_first_layer_compilation = df_layers_compilation[: int(len(df_layers_compilation) / num_layers)]
    df_first_layer_trace = df_layers_trace[: int(len(df_layers_trace) / num_layers)]

    # df_mid_layers_compilation = df_layers_compilation[int(len(df_layers_compilation) / num_layers) :]
    # df_mid_layers_trace = df_layers_trace[int(len(df_layers_trace) / num_layers) :]
    df_mid_layers_compilation = df_first_layer_compilation
    df_mid_layers_trace = df_first_layer_trace

    mid_layers_raw_dict_compilation = df_mid_layers_compilation[
        ["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]
    ].to_dict(orient="records")
    mid_layers_raw_dict_trace = df_mid_layers_trace[
        ["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]
    ].to_dict(orient="records")
    first_layer_raw_dict_compilation = df_first_layer_compilation[
        ["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]
    ].to_dict(orient="records")
    first_layer_raw_dict_trace = df_first_layer_trace[
        ["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]
    ].to_dict(orient="records")

    # Build dicts of op_code to list of durations
    kernel_duration_dict_compilation = build_duration_dict(
        mid_layers_raw_dict_compilation, "DEVICE KERNEL DURATION [ns]"
    )
    kernel_duration_dict_trace = build_duration_dict(mid_layers_raw_dict_trace, "DEVICE KERNEL DURATION [ns]")
    dispatch_duration_dict = build_duration_dict(mid_layers_raw_dict_trace, "OP TO OP LATENCY [ns]")

    # first layer
    kernel_duration_dict_compilation_first_layer = build_duration_dict(
        first_layer_raw_dict_compilation, "DEVICE KERNEL DURATION [ns]"
    )
    kernel_duration_dict_trace_first_layer = build_duration_dict(
        first_layer_raw_dict_trace, "DEVICE KERNEL DURATION [ns]"
    )
    dispatch_duration_dict_first_layer = build_duration_dict(first_layer_raw_dict_trace, "OP TO OP LATENCY [ns]")
    # Build dicts of op_code_with_id to list of durations - one list per op instance
    kernel_duration_per_instance_dict_compilation = build_duration_per_instance_dict(
        # kernel_duration_dict_compilation, num_layers - 1
        kernel_duration_dict_compilation,
        num_layers,
    )
    kernel_duration_per_instance_dict_trace = build_duration_per_instance_dict(
        # kernel_duration_dict_trace, num_layers - 1
        kernel_duration_dict_trace,
        num_layers,
    )
    # dispatch_duration_per_instance_dict = build_duration_per_instance_dict(dispatch_duration_dict, num_layers - 1)
    dispatch_duration_per_instance_dict = build_duration_per_instance_dict(dispatch_duration_dict, num_layers)

    # first layer
    kernel_duration_per_instance_dict_compilation_first_layer = build_duration_per_instance_dict(
        kernel_duration_dict_compilation_first_layer, 1
    )
    kernel_duration_per_instance_dict_trace_first_layer = build_duration_per_instance_dict(
        kernel_duration_dict_trace_first_layer, 1
    )
    dispatch_duration_per_instance_dict_first_layer = build_duration_per_instance_dict(
        dispatch_duration_dict_first_layer, 1
    )

    # Average over all iterations of each op instance
    kernel_duration_per_instance_averaged_dict_compilation = average_per_instance_dict(
        kernel_duration_per_instance_dict_compilation
    )
    kernel_duration_per_instance_averaged_dict_trace = average_per_instance_dict(
        kernel_duration_per_instance_dict_trace
    )
    dispatch_duration_per_instance_averaged_dict = average_per_instance_dict(dispatch_duration_per_instance_dict)

    # Min over all iterations of each op instance
    kernel_duration_per_instance_min_dict_compilation = min_per_instance_dict(
        kernel_duration_per_instance_dict_compilation
    )
    kernel_duration_per_instance_min_dict_trace = min_per_instance_dict(kernel_duration_per_instance_dict_trace)
    dispatch_duration_per_instance_min_dict = min_per_instance_dict(dispatch_duration_per_instance_dict)

    # Max over all iterations of each op instance
    kernel_duration_per_instance_max_dict_compilation = max_per_instance_dict(
        kernel_duration_per_instance_dict_compilation
    )
    kernel_duration_per_instance_max_dict_trace = max_per_instance_dict(kernel_duration_per_instance_dict_trace)
    dispatch_duration_per_instance_max_dict = max_per_instance_dict(dispatch_duration_per_instance_dict)

    if len(kernel_duration_per_instance_averaged_dict_compilation) != len(perf_targets):
        print(f"perf_targets: {perf_targets}")

    print_dict(
        kernel_duration_per_instance_averaged_dict_compilation, "kernel_duration_per_instance_averaged_dict_compilation"
    )
    print_dict(kernel_duration_per_instance_averaged_dict_trace, "kernel_duration_per_instance_averaged_dict_trace")
    print_dict(dispatch_duration_per_instance_averaged_dict, "dispatch_duration_per_instance_averaged_dict")

    assert len(kernel_duration_per_instance_averaged_dict_compilation) == len(
        perf_targets
    ), f"Expected {len(perf_targets)} operations, got {len(kernel_duration_per_instance_averaged_dict_compilation)}. If the number or type of operations changed, expected times must be updated."

    passing = True
    logger.info(
        "[TODO] NOTE: THE DEVICE TEST BEING MEASURED IS NOT RUN WITH TRACE MODE. OP-TO-OP TIMES ARE NOT ACCURATE"
    )
    for op_index, op_code_with_id in enumerate(kernel_duration_per_instance_averaged_dict_compilation.keys()):
        if op_code_with_id in perf_targets:
            op_name = perf_targets[op_code_with_id]["op_name"]

            # Dependent on the op_name we need to look at compile time or trace time for kernel duration
            if "AllGather" in op_code_with_id or "ReduceScatter" in op_code_with_id or "AllReduce" in op_code_with_id:
                avg_kernel_duration = kernel_duration_per_instance_averaged_dict_trace[op_code_with_id]
                min_kernel_duration = kernel_duration_per_instance_min_dict_trace[op_code_with_id]
                max_kernel_duration = kernel_duration_per_instance_max_dict_trace[op_code_with_id]
            else:
                avg_kernel_duration = kernel_duration_per_instance_averaged_dict_compilation[op_code_with_id]
                min_kernel_duration = kernel_duration_per_instance_min_dict_compilation[op_code_with_id]
                max_kernel_duration = kernel_duration_per_instance_max_dict_compilation[op_code_with_id]

            avg_dispatch_duration = dispatch_duration_per_instance_averaged_dict[op_code_with_id]
            # Workaround: We are storing the op index in the "iteration" field of the measurement, to facilitate the graph in superset

            # average
            benchmark_data.add_measurement(profiler, 0, step_name, op_name + "-model-kernel-avg", avg_kernel_duration)
            benchmark_data.add_measurement(
                profiler, 0, step_name, op_name + "-model-op_to_op-avg", avg_dispatch_duration
            )

            # min
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-kernel-min",
                min_kernel_duration,
            )
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-op_to_op-min",
                dispatch_duration_per_instance_min_dict[op_code_with_id],
            )

            # max
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-kernel-max",
                max_kernel_duration,
            )
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-op_to_op-max",
                dispatch_duration_per_instance_max_dict[op_code_with_id],
            )

            # Verify kernel duration is within tolerance
            upper_limit = (
                perf_targets[op_code_with_id]["kernel_duration"]
                + perf_targets[op_code_with_id]["kernel_duration_relative_margin"]
                * perf_targets[op_code_with_id]["kernel_duration"]
            )
            lower_limit = (
                perf_targets[op_code_with_id]["kernel_duration"]
                - perf_targets[op_code_with_id]["kernel_duration_relative_margin"]
                * perf_targets[op_code_with_id]["kernel_duration"]
            )

            if avg_kernel_duration > upper_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns is larger than target "
                    f"({perf_targets[op_code_with_id]['kernel_duration']}) ns, difference: "
                    f"{abs(avg_kernel_duration - upper_limit)} ns, margin: "
                    f"{perf_targets[op_code_with_id]['kernel_duration_relative_margin']}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id]['kernel_duration'] - avg_kernel_duration) / perf_targets[op_code_with_id]['kernel_duration']}"
                )
            elif avg_kernel_duration < lower_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns is smaller than target "
                    f"({perf_targets[op_code_with_id]['kernel_duration']}) ns, difference: "
                    f"{abs(lower_limit - avg_kernel_duration)} ns, margin: "
                    f"{perf_targets[op_code_with_id]['kernel_duration_relative_margin']}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id]['kernel_duration'] - avg_kernel_duration) / perf_targets[op_code_with_id]['kernel_duration']}"
                )
            # Verify op_to_op latency is within tolerance
            upper_limit = (
                perf_targets[op_code_with_id]["op_to_op"]
                + perf_targets[op_code_with_id]["op_to_op_duration_relative_margin"]
                * perf_targets[op_code_with_id]["op_to_op"]
            )
            lower_limit = (
                perf_targets[op_code_with_id]["op_to_op"]
                - perf_targets[op_code_with_id]["op_to_op_duration_relative_margin"]
                * perf_targets[op_code_with_id]["op_to_op"]
            )
            if avg_dispatch_duration > upper_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} op_to_op: {avg_dispatch_duration} ns is larger than target "
                    f"({perf_targets[op_code_with_id]['op_to_op']}) ns, difference: "
                    f"{abs(avg_dispatch_duration - upper_limit)} ns, margin: "
                    f"{perf_targets[op_code_with_id]['op_to_op_duration_relative_margin']}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id]['op_to_op'] - avg_dispatch_duration) / perf_targets[op_code_with_id]['op_to_op']}"
                )
            elif avg_dispatch_duration < lower_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} op_to_op: {avg_dispatch_duration} ns is smaller than target "
                    f"({perf_targets[op_code_with_id]['op_to_op']}) ns, difference: "
                    f"{abs(lower_limit - avg_dispatch_duration)} ns, margin: "
                    f"{perf_targets[op_code_with_id]['op_to_op_duration_relative_margin']}, "
                    f"relative margin to pass would be: "
                    f"{abs(perf_targets[op_code_with_id]['op_to_op'] - avg_dispatch_duration) / perf_targets[op_code_with_id]['op_to_op']}"
                )

        else:
            passing = False
            logger.info(f"Warning: {op_code_with_id} not found in perf_targets")

    # Calculate e2e performance
    ttft_estimate_80l = 0
    for op_id in kernel_duration_per_instance_dict_trace_first_layer.keys():  # first layer
        op_to_op_latency = dispatch_duration_per_instance_dict_first_layer[op_id][0]
        if is_collective_op(op_id):
            kernel_duration = kernel_duration_per_instance_dict_trace_first_layer[op_id][0]
        else:
            kernel_duration = kernel_duration_per_instance_dict_compilation_first_layer[op_id][0]

        if op_to_op_latency < 0:
            op_to_op_latency = 0

        print(f"op_id: {op_id}, kernel_duration: {kernel_duration}, op_to_op_latency: {op_to_op_latency}")

        # TODO add dispatch times
        ttft_estimate_80l += kernel_duration
    for op_id in kernel_duration_per_instance_averaged_dict_trace.keys():  # 79 layers based on average of layers 2-9
        if is_collective_op(op_id):
            avg_kernel_duration = kernel_duration_per_instance_averaged_dict_trace[op_id]
        else:
            avg_kernel_duration = kernel_duration_per_instance_averaged_dict_compilation[op_id]
        avg_dispatch_duration = dispatch_duration_per_instance_averaged_dict[op_id]
        ttft_estimate_80l += avg_kernel_duration * 79  # weighting avg for 79 layers

    # Estimated TTFT is not accounting for LMHead + python overhead #  1000000 / (80L-duration + ~2100 lmhead+sampling+embeddings + ~300 python-overhead
    # tsu_estimate = 1000000 / (e2e_estimate_80l / 1000 + 2100 + 300)
    ttft_estimate_80l = ttft_estimate_80l / 1000000
    print(f"80L TTFT time estimate: {ttft_estimate_80l}")

    benchmark_data.add_measurement(profiler, 0, step_name, "ttft_estimate_80l", ttft_estimate_80l)

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_llama_demo_prefill_{seqlen}",
        ml_model_name="llama70b-tg",
    )

    # assert passing

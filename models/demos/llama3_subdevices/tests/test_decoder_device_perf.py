# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
from loguru import logger
import os
import math
import ttnn
import pandas as pd
from collections import defaultdict
from models.demos.llama3_subdevices.tt.llama_common import (
    PagedAttentionConfig,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf
from tt_metal.tools.profiler.process_model_log import (
    get_latest_ops_log_filename,
)

from models.demos.llama3_subdevices.demo.demo_decode import run_llama3_demo
from models.demos.llama3_subdevices.demo.demo_decode import LlamaOptimizations

DECODE_OP_START_INDEX = 4
DECODE_OP_END_INDEX = -12

perf_targets = {
    "RMSAllGather_0": {
        "op_name": "PreRMS_0",
        "kernel_duration": 11694.534722222223,
        "op_to_op": 759.6666666666666,
        "non-overlapped-dispatch-time": 7260,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "RMSAllGather_1": {
        "op_name": "PostRMS_0",
        "kernel_duration": 9400.361111111111,
        "op_to_op": 634.3333333333334,
        "non-overlapped-dispatch-time": 6301.3,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "RMSAllGather_2": {
        "op_name": "PreRMS_1",
        "kernel_duration": 11078.270833333334,
        "op_to_op": 748.3333333333334,
        "non-overlapped-dispatch-time": 7326,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "RMSAllGather_3": {
        "op_name": "PostRMS_1",
        "kernel_duration": 9197.204861111111,
        "op_to_op": 641.0,
        "non-overlapped-dispatch-time": 6431.2,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "AllGatherConcat_0": {
        "op_name": "AllGatherConcat",
        "kernel_duration": 12419.194444444445,
        "op_to_op": 796.8888888888889,
        "non-overlapped-dispatch-time": 12541.7,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.5,
    },
    "AllGatherAsync_0": {
        "op_name": "AllGatherAsync_Binary_Mult",
        "kernel_duration": 10607.277777777777,
        "op_to_op": 959.5555,
        "non-overlapped-dispatch-time": 4351.1,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "Matmul_0": {
        "op_name": "QKV_MM",
        "kernel_duration": 10623,
        "op_to_op": 716.4444444444445,
        "non-overlapped-dispatch-time": 6102.0,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.1,
    },
    "Matmul_1": {
        "op_name": "DO_MM",
        "kernel_duration": 8902,
        "op_to_op": 723.0,
        "non-overlapped-dispatch-time": 6412.2,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.1,
    },
    "Matmul_2": {
        "op_name": "FF1_MM",
        "kernel_duration": 10829,
        "op_to_op": 711.8888888888889,
        "non-overlapped-dispatch-time": 6109.0,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.1,
    },
    "Matmul_3": {
        "op_name": "FF3_MM",
        "kernel_duration": 10848,
        "op_to_op": 688.7777777777778,
        "non-overlapped-dispatch-time": 6144.8,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.15,
    },
    "Matmul_4": {
        "op_name": "FF2_MM",
        "kernel_duration": 15891.0,
        "op_to_op": 658.7777777777778,
        "non-overlapped-dispatch-time": 6770.3,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.1,
    },
    "AllReduceCreateQkvHeads_0": {
        "op_name": "AllReduce_Fuse_Createheads",
        "kernel_duration": 14541.059027777777,
        "op_to_op": 667.0,
        "non-overlapped-dispatch-time": 7932.2,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.4,
        "dispatch_duration_relative_margin": 0.2,
    },
    "AllReduceAsync_0": {
        "op_name": "AllReduceAsync_DO",
        "kernel_duration": 21736.76736111111,
        "op_to_op": 626.5555555555555,
        "non-overlapped-dispatch-time": 8510.2,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.3,
    },
    "AllReduceAsync_1": {
        "op_name": "AllReduceAsync_FF2",
        "kernel_duration": 22341.23263888889,
        "op_to_op": 637.1111111111111,
        "non-overlapped-dispatch-time": 7252.4,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "LlamaReduceScatterDeviceOperation_0": {
        "op_name": "ReduceScatter_FF1",
        "kernel_duration": 9952.402777777777,
        "op_to_op": 708.1111111111111,
        "non-overlapped-dispatch-time": 8058.9,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.3,
    },
    "LlamaReduceScatterDeviceOperation_1": {
        "op_name": "ReduceScatter_FF3",
        "kernel_duration": 9817.020833333334,
        "op_to_op": 655.1111111111111,
        "non-overlapped-dispatch-time": 7359.9,
        "kernel_duration_relative_margin": 0.05,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.3,
    },
    "RotaryEmbeddingLlamaFusedQK_0": {
        "op_name": "RotaryEmbeddingLlamaFusedQK",
        "kernel_duration": 4296,
        "op_to_op": 606.1111111111111,
        "non-overlapped-dispatch-time": 2844.3,
        "kernel_duration_relative_margin": 0.1,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.35,
    },
    "PagedUpdateCacheDeviceOperation_0": {
        "op_name": "PagedUpdateCache",
        "kernel_duration": 5963,
        "op_to_op": 860.2222222222222,
        "non-overlapped-dispatch-time": 5890.0,
        "kernel_duration_relative_margin": 0.2,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "ScaledDotProductAttentionDecode_0": {
        "op_name": "SDPA",
        "kernel_duration": 9300,
        "op_to_op": 652.6666666666666,
        "non-overlapped-dispatch-time": 9741.5,
        "kernel_duration_relative_margin": 0.07,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.3,
    },
    "BinaryDeviceOperation_0": {
        "op_name": "Binary_Residual_0",
        "kernel_duration": 1059,
        "op_to_op": 725.6666666666666,
        "non-overlapped-dispatch-time": 6316.8,
        "kernel_duration_relative_margin": 0.25,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "BinaryDeviceOperation_1": {
        "op_name": "Binary_Mult_Silu",
        "kernel_duration": 2929,
        "op_to_op": 661.0,
        "non-overlapped-dispatch-time": 6111.2,
        "kernel_duration_relative_margin": 0.1,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "BinaryDeviceOperation_2": {
        "op_name": "Binary_Residual_1",
        "kernel_duration": 1052,
        "op_to_op": 731.4444444444445,
        "non-overlapped-dispatch-time": 6751.9,
        "kernel_duration_relative_margin": 0.1,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.2,
    },
    "Untilize_0": {
        "op_name": "Untilize",
        "kernel_duration": 1517.3333333333335,
        "op_to_op": 996.8888888888889,
        "non-overlapped-dispatch-time": 3113.6,
        "kernel_duration_relative_margin": 0.2,
        "op_to_op_duration_relative_margin": 0.2,
        "dispatch_duration_relative_margin": 0.5,
    },
}

DECODER_OP_START_IDX = 4
DECODER_OP_END_IDX = -11


@pytest.mark.parametrize(
    "weights, layers, input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stress_test, start_pos",
    [
        (  # 10 layers for devive perf measurements
            "random",
            10,
            "models/demos/llama3_subdevices/demo/input_data_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            1,  # max_generated_tokens
            True,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params  # TODO This will be serviced by vLLM
            {"top_k": 32, "top_p": 0.08, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
            127,  # start_pos
        ),
    ],
    ids=[
        "device-perf-measurement",
    ],
)
@pytest.mark.parametrize(
    "optimizations",
    [
        LlamaOptimizations.performance,
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 23887872,
            "worker_l1_size": 1344544,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_llama_demo(
    weights,
    layers,
    input_prompts,
    instruct,
    repeat_batches,
    max_seq_len,
    batch_size,
    max_generated_tokens,
    paged_attention,
    page_params,
    sampling_params,
    optimizations,
    mesh_device,
    use_program_cache,
    is_ci_env,
    reset_seeds,
    stress_test,
    start_pos,
):
    if is_ci_env and ("long" in input_prompts or optimizations == LlamaOptimizations.accuracy):
        pytest.skip("Do not run the 'long-context' or accuracy tests on CI to reduce load")

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("FAKE_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
    else:
        paged_attention_config = None

    return run_llama3_demo(
        user_input=input_prompts,
        mesh_device=mesh_device,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_batches=repeat_batches,
        paged_attention=paged_attention,
        paged_attention_config=paged_attention_config,
        max_generated_tokens=max_generated_tokens,
        optimizations=optimizations,
        sampling_params=sampling_params,
        instruct_mode=instruct,
        is_ci_env=is_ci_env,
        print_to_file=False,
        weights=weights,
        layers=layers,
        stress_test=stress_test,
        start_pos=start_pos,
    )


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


@pytest.mark.models_device_performance_bare_metal
# To update:
# Run FAKE_DEVICE=TG TT_METAL_ENABLE_ERISC_IRAM=1 pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device
# Copy the printed kernel_duration_per_instance_averaged_dict and dispatch_duration_per_instance_averaged_dict dictionaries
# Manually compare each entry between old-expected and the new average values
# - Any perf regressions? Everything as expected?
# If all looks good, update the expected_kernel_times_dict and expected_dispatch_times_dict with the new average values
# If the op list changed (new ops, less ops, fused ops), then update mapping_op_code_to_name and give the new ops meaningful names
# Run at least once again to verify the new expected values are correct and margins hold
def test_llama_TG_perf_device(
    reset_seeds,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = "tg-llama-demo-device-perf-default"
    batch_size = 32
    subdir = "tg-llama-demo-device-perf-default"
    num_iterations = 1
    num_layers = 10

    command = f"pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_demo"
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
    df_model_compilation = df[: int(len(df) / 3)]
    df_model_trace = df[int(len(df) / 3 * 2) :]

    # Excluding model embeddings and lmhead+sampling ops
    df_layers_compilation = df_model_compilation[DECODE_OP_START_INDEX:DECODE_OP_END_INDEX]
    df_layers_trace = df_model_trace[DECODE_OP_START_INDEX:DECODE_OP_END_INDEX]
    # Use layers 2-9 for verifying against targets for more stability
    df_first_layer_compilation = df_layers_compilation[: int(len(df_layers_compilation) / num_layers)]
    df_first_layer_trace = df_layers_trace[: int(len(df_layers_trace) / num_layers)]

    df_mid_layers_compilation = df_layers_compilation[int(len(df_layers_compilation) / num_layers) :]
    df_mid_layers_trace = df_layers_trace[int(len(df_layers_trace) / num_layers) :]

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

    # Build dicts of op_code_with_id to list of durations - one list per op instance
    kernel_duration_per_instance_dict_compilation = build_duration_per_instance_dict(
        kernel_duration_dict_compilation, num_layers - 1
    )
    kernel_duration_per_instance_dict_trace = build_duration_per_instance_dict(
        kernel_duration_dict_trace, num_layers - 1
    )
    dispatch_duration_per_instance_dict = build_duration_per_instance_dict(dispatch_duration_dict, num_layers - 1)

    # Average over all iterations of each op instance
    kernel_duration_per_instance_averaged_dict_compilation = average_per_instance_dict(
        kernel_duration_per_instance_dict_compilation
    )
    kernel_duration_per_instance_averaged_dict_trace = average_per_instance_dict(
        kernel_duration_per_instance_dict_trace
    )
    dispatch_duration_per_instance_averaged_dict = average_per_instance_dict(dispatch_duration_per_instance_dict)

    kernel_duration_per_instance_min_dict_compilation = min_per_instance_dict(
        kernel_duration_per_instance_dict_compilation
    )
    kernel_duration_per_instance_min_dict_trace = min_per_instance_dict(kernel_duration_per_instance_dict_trace)
    dispatch_duration_per_instance_min_dict = min_per_instance_dict(dispatch_duration_per_instance_dict)

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
    for op_code_with_id in kernel_duration_per_instance_averaged_dict_compilation.keys():
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
    e2e_estimate_80l = 0
    for entry in first_layer_raw_dict_compilation:
        kernel_duration = entry["DEVICE KERNEL DURATION [ns]"]
        e2e_estimate_80l += kernel_duration
    for entry in first_layer_raw_dict_trace:
        dispatch_duration = entry["OP TO OP LATENCY [ns]"]
        e2e_estimate_80l += dispatch_duration
    for op_code_with_id in kernel_duration_per_instance_averaged_dict_compilation.keys():
        if "AllGather" in op_code_with_id or "ReduceScatter" in op_code_with_id or "AllReduce" in op_code_with_id:
            avg_kernel_duration = kernel_duration_per_instance_averaged_dict_trace[op_code_with_id]
        else:
            avg_kernel_duration = kernel_duration_per_instance_averaged_dict_compilation[op_code_with_id]
        avg_dispatch_duration = dispatch_duration_per_instance_averaged_dict[op_code_with_id]
        e2e_estimate_80l += (avg_kernel_duration + avg_dispatch_duration) * 79  # weighting avg for 79 layers

    print(f"e2e estimate: {e2e_estimate_80l}")

    benchmark_data.add_measurement(profiler, 0, step_name, "e2e_estimate_80l", e2e_estimate_80l)
    # Estimated T/s/u is 1000000 / (80L-duration + ~2100 lmhead+sampling+embeddings + ~300 python-overhead
    benchmark_data.add_measurement(
        profiler, 0, step_name, "tsu_estimate", 1000000 / (e2e_estimate_80l / 1000 + 2100 + 300)
    )

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_llama_demo_decode",
        ml_model_name="llama70b-tg",
    )

    assert passing


@pytest.mark.models_device_performance_bare_metal
# To update:
# Run FAKE_DEVICE=TG TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_KERNELS_EARLY_RETURN=1  pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device_non_overlapped_dispatch
# Copy the printed dispatch_duration_per_instance_averaged_dict dictionary
# Manually compare each entry between old-expected and the new average values
# - Any perf regressions? Everything as expected?
# If all looks good, update the expected_dispatch_times_dict with the new average values
# If the op list changed (new ops, less ops, fused ops), and not done for the above test, then update mapping_op_code_to_name and give the new ops meaningful names
# Run at least once again to verify the new expected values are correct and margins hold
def test_llama_TG_perf_device_non_overlapped_dispatch(
    reset_seeds,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = "tg-llama-demo-device-perf-non-overlapped-dispatch"
    batch_size = 32
    subdir = "tg-llama-demo-device-perf-non-overlapped-dispatch"
    num_iterations = 1
    num_layers = 10

    command = f"pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_demo"
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
    # Exclude compilaton and capture trace runs
    df_model = df[int(len(df) / 3 * 2) :]
    df_layers = df_model[DECODE_OP_START_INDEX:DECODE_OP_END_INDEX]
    all_layers_raw_dict = df_layers[["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]].to_dict(
        orient="records"
    )

    # Build dicts of op_code to list of durations
    dispatch_duration_dict = build_duration_dict(all_layers_raw_dict, "OP TO OP LATENCY [ns]")

    # Build dicts of op_code_with_id to list of durations - one list per op instance
    dispatch_duration_per_instance_dict = build_duration_per_instance_dict(dispatch_duration_dict, num_layers)

    # Average over all iterations of each op instance
    dispatch_duration_per_instance_averaged_dict = average_per_instance_dict(dispatch_duration_per_instance_dict)
    dispatch_duration_per_instance_min_dict = min_per_instance_dict(dispatch_duration_per_instance_dict)
    dispatch_duration_per_instance_max_dict = max_per_instance_dict(dispatch_duration_per_instance_dict)

    print(f"dispatch_duration_per_instance_averaged_dict: {dispatch_duration_per_instance_averaged_dict}")

    assert len(dispatch_duration_per_instance_averaged_dict) == len(
        perf_targets
    ), f"Expected {len(perf_targets)} operations, got {len(dispatch_duration_per_instance_averaged_dict)}. If the number or type of operations changed, expected times must be updated."

    passing = True
    for op_code_with_id, avg_dispatch_duration in dispatch_duration_per_instance_averaged_dict.items():
        if op_code_with_id in perf_targets:
            expected_time = perf_targets[op_code_with_id]["non-overlapped-dispatch-time"]
            op_name = perf_targets[op_code_with_id]["op_name"]

            benchmark_data.add_measurement(
                profiler, 0, step_name, op_name + "-model-dispatch-avg", avg_dispatch_duration
            )
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-dispatch-min",
                dispatch_duration_per_instance_min_dict[op_code_with_id],
            )
            benchmark_data.add_measurement(
                profiler,
                0,
                step_name,
                op_name + "-model-dispatch-max",
                dispatch_duration_per_instance_max_dict[op_code_with_id],
            )

            upper_limit = (
                expected_time + perf_targets[op_code_with_id]["dispatch_duration_relative_margin"] * expected_time
            )
            lower_limit = (
                expected_time - perf_targets[op_code_with_id]["dispatch_duration_relative_margin"] * expected_time
            )
            if avg_dispatch_duration > upper_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} dispatch: {avg_dispatch_duration} ns is larger than target "
                    f"({expected_time}) ns, difference: "
                    f"{abs(avg_dispatch_duration - upper_limit)} ns, margin: "
                    f"{perf_targets[op_code_with_id]['dispatch_duration_relative_margin']}, "
                    f"relative margin to pass would be: "
                    f"{abs(expected_time - avg_dispatch_duration) / expected_time}"
                )
            elif avg_dispatch_duration < lower_limit:
                passing = False
                logger.info(
                    f"{op_code_with_id} dispatch: {avg_dispatch_duration} ns is smaller than target "
                    f"({expected_time}) ns, difference: "
                    f"{abs(lower_limit - avg_dispatch_duration)} ns, margin: "
                    f"{perf_targets[op_code_with_id]['dispatch_duration_relative_margin']}, "
                    f"relative margin to pass would be: "
                    f"{abs(expected_time - avg_dispatch_duration) / expected_time}"
                )
        else:
            passing = False
            logger.info(f"Warning: {op_code_with_id} not found in expected_times_dict")

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_llama_demo_decode",
        ml_model_name="llama70b-tg",
    )

    assert passing

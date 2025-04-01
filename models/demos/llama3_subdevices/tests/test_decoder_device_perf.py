# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
from loguru import logger
import os
import ttnn
import pandas as pd
from collections import defaultdict
from models.demos.llama3_subdevices.tt.llama_common import (
    PagedAttentionConfig,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler

from models.demos.llama3_subdevices.demo.demo_decode import run_llama3_demo
from models.demos.llama3_subdevices.demo.demo_decode import LlamaOptimizations


@pytest.mark.parametrize(
    "weights, layers, input_prompts, instruct, repeat_batches, max_seq_len, batch_size, max_generated_tokens, paged_attention, page_params, sampling_params, stress_test",
    [
        (  # Batch-32 run (Throughput) - 32 users, small prompt
            "instruct",  # weights
            10,  # layers
            "models/demos/llama3_subdevices/demo/input_data_prefill_128.json",  # input_prompts
            True,  # instruct mode
            1,  # repeat_batches
            1024,  # max_seq_len
            32,  # batch_size
            1,  # max_generated_tokens
            False,  # paged_attention
            {"page_block_size": 32, "page_max_num_blocks": 1024},  # page_params  # TODO This will be serviced by vLLM
            {"top_k": 32, "top_p": 0.08, "seed": 42},  # sampling_params (argmax)
            False,  # stress_test
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
    "device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}], indirect=True
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
):
    if is_ci_env and ("long" in input_prompts or optimizations == LlamaOptimizations.accuracy):
        pytest.skip("Do not run the 'long-context' or accuracy tests on CI to reduce load")

    # TODO: Remove this once all batch sizes are supported on TG
    if os.environ.get("FAKE_DEVICE") == "TG" and batch_size not in [1, 32]:
        pytest.skip("TG only supports batch 1 and 32")

    mesh_device.enable_async(True)

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

        if "AllGather" in op_name or "ReduceScatter" in op_name:
            # For collective ops, take the row with minimum duration
            min_duration_block = min(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(min_duration_block[1])
        else:
            # For non-collective ops, take the row with maximum duration
            max_duration_block = max(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(max_duration_block[1])

        global_index += 1

    return pd.DataFrame(merged_blocks)


def build_duration_dict(raw_dict, column_name):
    op_code_dict = {}
    for entry in raw_dict:
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
        averaged_dict[op_code_with_id] = sum(
            input_dict[op_code_with_id]
        ) / len(input_dict[op_code_with_id])
    return averaged_dict

@pytest.mark.parametrize(
    "abs_tolerance_ns",
    (500,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_reduce",
    (1200,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_gather",
    (500,),
)
@pytest.mark.models_device_performance_bare_metal
def test_llama_TG_perf_device(reset_seeds, abs_tolerance_ns, abs_tolerance_ns_all_reduce, abs_tolerance_ns_all_gather):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name_kernel_duration = "tg-llama-decoder-kernel-duration"
    step_name_overlapped_dispatch_duration = "tg-llama-decoder-overlapped-dispatch-duration"
    step_name_e2e_duration = "tg-llama-decoder-e2e-duration"
    batch_size = 32
    subdir = "tg-llama-decoder"
    num_iterations = 1
    num_layers = 10

    # command = f"pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_demo"
    # cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    # profiler.start("run")
    # profiler.start(step_name)
    # post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    # profiler.end(step_name)
    # profiler.end("run")

    # filename = get_latest_ops_log_filename(subdir)
    # filename = "/proj_sw/user_dev/jrock/tt-metal/generated/profiler/tg-llama-decoder/reports/2025_03_31_09_11_08/ops_perf_results_2025_03_31_09_11_08.csv"
    filename = "/proj_sw/user_dev/jrock/tt-metal/generated/profiler/tg-llama-decoder/reports/2025_03_31_10_07_34/ops_perf_results_2025_03_31_10_07_34.csv"

    df = pd.read_csv(filename)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows(df)
    # Excluding compile run and capture trace entries
    df_model = df[int(len(df) / 3 * 2) :]
    df_layers = df_model[4:-12]
    df_first_layer = df_layers[: int(len(df_layers) / num_layers)]
    df_mid_layers = df_layers[int(len(df_layers) / num_layers) :]
    mid_layers_raw_dict = df_mid_layers[["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]].to_dict(orient="records")
    first_layer_raw_dict = df_first_layer[["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]].to_dict(orient="records")

    # Build dicts of op_code to list of durations
    kernel_duration_dict = build_duration_dict(mid_layers_raw_dict, "DEVICE KERNEL DURATION [ns]")
    dispatch_duration_dict = build_duration_dict(mid_layers_raw_dict, "OP TO OP LATENCY [ns]")

    # Build dicts of op_code_with_id to list of durations - one list per op instance
    kernel_duration_per_instance_dict = build_duration_per_instance_dict(kernel_duration_dict, num_layers)
    dispatch_duration_per_instance_dict = build_duration_per_instance_dict(dispatch_duration_dict, num_layers)

    # Average over all iterations of each op instance
    kernel_duration_per_instance_averaged_dict = average_per_instance_dict(kernel_duration_per_instance_dict)
    dispatch_duration_per_instance_averaged_dict = average_per_instance_dict(dispatch_duration_per_instance_dict)

    print(kernel_duration_per_instance_averaged_dict)
    print(dispatch_duration_per_instance_averaged_dict)

    expected_kernel_times_dict = {
        "LayerNorm_0": 6443.3,
        "LayerNorm_1": 6491.0,
        "LayerNorm_2": 6148.1,
        "LayerNorm_3": 6390.6,
        "AllGatherAsync_0": 2710.3,
        "AllGatherAsync_1": 5837.7,
        "AllGatherAsync_2": 2486.7,
        "ReshardDeviceOperation_0": 1803.7,
        "ReshardDeviceOperation_1": 1851.5,
        "ReshardDeviceOperation_2": 1520.3,
        "Matmul_0": 8429.8,
        "Matmul_1": 8926.8,
        "Matmul_2": 9583.1,
        "Matmul_3": 9631.9,
        "Matmul_4": 17304.9,
        "AllReduceAsync_0": 515885.6,
        "AllReduceAsync_1": 350838.1,
        "AllReduceAsync_2": 3828933.1,
        "AllReduceAsync_3": 133560.4,
        "AllReduceAsync_4": 1007790.5,
        "NLPCreateHeadsDecodeDeviceOperation_0": 8496.5,
        "RotaryEmbeddingLlamaFusedQK_0": 5177.4,
        "PagedUpdateCacheDeviceOperation_0": 4613.0,
        "ScaledDotProductAttentionDecode_0": 19761.8,
        "NLPConcatHeadsDecodeDeviceOperation_0": 6234.9,
        "BinaryDeviceOperation_0": 871.3,
        "BinaryDeviceOperation_1": 13043.2,
        "BinaryDeviceOperation_2": 1860.9,
    }

    mapping_op_code_to_name = {
        "LayerNorm_0": "PreAllGatherLN_0",
        "LayerNorm_1": "PostAllGatherLN_0",
        "LayerNorm_2": "PreAllGatherLN_1",
        "LayerNorm_3": "PostAllGatherLN_1",
        "AllGatherAsync_0": "AllGatherAsync_LN_0",
        "AllGatherAsync_1": "AllGatherAsync_SDPA_0",
        "AllGatherAsync_2": "AllGatherAsync_LN_1",
        "ReshardDeviceOperation_0": "ReshardDeviceOperation_LN_0",
        "ReshardDeviceOperation_1": "ReshardDeviceOperation_CreateHeads",
        "ReshardDeviceOperation_2": "ReshardDeviceOperation_LN_1",
        "Matmul_0": "QKV_MM",
        "Matmul_1": "DO_MM",
        "Matmul_2": "FF1_MM",
        "Matmul_3": "FF3_MM",
        "Matmul_4": "FF2_MM",
        "AllReduceAsync_0": "AllReduceAsync_QKV",
        "AllReduceAsync_1": "AllReduceAsync_DO",
        "AllReduceAsync_2": "AllReduceAsync_FF1",
        "AllReduceAsync_3": "AllReduceAsync_FF3",
        "AllReduceAsync_4": "AllReduceAsync_FF2",
        "NLPCreateHeadsDecodeDeviceOperation_0": "CreateHeads",
        "RotaryEmbeddingLlamaFusedQK_0": "RotaryEmbeddingLlamaFusedQK",
        "PagedUpdateCacheDeviceOperation_0": "PagedUpdateCache",
        "ScaledDotProductAttentionDecode_0": "SDPA",
        "NLPConcatHeadsDecodeDeviceOperation_0": "ConcatHeads",
        "BinaryDeviceOperation_0": "Binary_Residual_0",
        "BinaryDeviceOperation_1": "Binary_Mult_Silu",
        "BinaryDeviceOperation_2": "Binary_Residual_1",
    }

    assert len(kernel_duration_per_instance_averaged_dict) == len(
        expected_kernel_times_dict
    ), f"Expected {len(expected_kernel_times_dict)} operations, got {len(kernel_duration_per_instance_averaged_dict)}. If the number or type of operations changed, expected times must be updated."

    passing = True
    for op_code_with_id, avg_kernel_duration in kernel_duration_per_instance_averaged_dict.items():
        if op_code_with_id in expected_kernel_times_dict:
            expected_time = expected_kernel_times_dict[op_code_with_id]
            op_name = mapping_op_code_to_name[op_code_with_id]
            avg_dispatch_duration = dispatch_duration_per_instance_averaged_dict[op_code_with_id]
            benchmark_data.add_measurement(profiler, 0, step_name_kernel_duration, op_name, avg_kernel_duration)
            benchmark_data.add_measurement(profiler, 0, step_name_overlapped_dispatch_duration, op_name, avg_dispatch_duration)
            if "AllReduceAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_reduce
            elif "AllGatherAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_gather
            else:
                tolerance = abs_tolerance_ns
            if avg_kernel_duration > expected_time + tolerance:
                passing = False
                logger.info(
                    f"{op_code_with_id}: {avg_kernel_duration} ns larger than expected {expected_time} ns by {abs(avg_kernel_duration - expected_time)} ns (tolerance {tolerance} ns)"
                )
            elif avg_kernel_duration < expected_time - tolerance:
                passing = False
                logger.info(
                    f"{op_code_with_id}: {avg_kernel_duration} ns smaller than expected {expected_time} ns by {abs(expected_time - avg_kernel_duration)} ns (tolerance {tolerance} ns)"
                )
        else:
            passing = False
            logger.info(f"Warning: {op_code_with_id} not found in expected_times_dict")

    # Calculate e2e performance
    e2e_duration_layer_avg = 0
    for entry in first_layer_raw_dict:
        kernel_duration = entry["DEVICE KERNEL DURATION [ns]"]
        dispatch_duration = entry["OP TO OP LATENCY [ns]"]
        e2e_duration_layer_avg += kernel_duration + dispatch_duration
    for op_code_with_id, avg_kernel_duration in kernel_duration_per_instance_averaged_dict.items():
        avg_dispatch_duration = dispatch_duration_per_instance_averaged_dict[op_code_with_id]
        e2e_duration_layer_avg += (avg_kernel_duration + avg_dispatch_duration) * 79 # weighting avg for 79 layers
    e2e_duration_layer_avg /= len(kernel_duration_per_instance_averaged_dict) * 79 + len(first_layer_raw_dict)

    benchmark_data.add_measurement(profiler, 0, step_name_e2e_duration, "e2e_duration", e2e_duration_layer_avg)


    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg-llama-decoder",
        ml_model_name="llama70b-tg-decoder",
    )

    assert passing


@pytest.mark.parametrize(
    "abs_tolerance_ns",
    (500,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_reduce",
    (500,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_gather",
    (500,),
)
@pytest.mark.models_device_performance_bare_metal
def test_llama_TG_perf_device_non_overlapped_dispatch(reset_seeds, abs_tolerance_ns, abs_tolerance_ns_all_reduce, abs_tolerance_ns_all_gather):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name_non_overlapped_dispatch_duration = "tg-llama-decoder-non-overlapped-dispatch-duration"
    batch_size = 32
    subdir = "tg-llama-decoder"
    num_iterations = 1
    num_layers = 10

    # command = f"TT_METAL_KERNELS_EARLY_RETURN=1 pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_decoder_inference"
    # cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    # profiler.start("run")
    # profiler.start(step_name)
    # post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    # profiler.end(step_name)
    # profiler.end("run")

    # filename = get_latest_ops_log_filename(subdir)
    filename = "/Users/jrock/repos/tt-metal/test_device_perf.csv"

    df = pd.read_csv(filename)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows(df)
    # Exclude compilaton and capture trace runs
    df_model = df[int(len(df) / 3 * 2) :]
    df_layers = df_model[4:-12]
    all_layers_raw_dict = df_layers[["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]].to_dict(orient="records")

    # Build dicts of op_code to list of durations
    dispatch_duration_dict = build_duratoin_dict(all_layers_raw_dict, "OP TO OP LATENCY [ns]")

    # Build dicts of op_code_with_id to list of durations - one list per op instance
    dispatch_duration_per_instance_dict = build_duration_per_instance_dict(dispatch_duration_dict, num_layers)

    # Average over all iterations of each op instance
    dispatch_duration_per_instance_averaged_dict = average_per_instance_dict(dispatch_duration_per_instance_dict)

    print(dispatch_duration_per_instance_averaged_dict)

    expected_non_overlapped_dispatch_times_dict = {
        "LayerNorm_0": 6443.3,
        "LayerNorm_1": 6491.0,
        "LayerNorm_2": 6148.1,
        "LayerNorm_3": 6390.6,
        "AllGatherAsync_0": 2710.3,
        "AllGatherAsync_1": 5837.7,
        "AllGatherAsync_2": 2486.7,
        "ReshardDeviceOperation_0": 1803.7,
        "ReshardDeviceOperation_1": 1851.5,
        "ReshardDeviceOperation_2": 1520.3,
        "Matmul_0": 8429.8,
        "Matmul_1": 8926.8,
        "Matmul_2": 9583.1,
        "Matmul_3": 9631.9,
        "Matmul_4": 17304.9,
        "AllReduceAsync_0": 515885.6,
        "AllReduceAsync_1": 350838.1,
        "AllReduceAsync_2": 3828933.1,
        "AllReduceAsync_3": 133560.4,
        "AllReduceAsync_4": 1007790.5,
        "NLPCreateHeadsDecodeDeviceOperation_0": 8496.5,
        "RotaryEmbeddingLlamaFusedQK_0": 5177.4,
        "PagedUpdateCacheDeviceOperation_0": 4613.0,
        "ScaledDotProductAttentionDecode_0": 19761.8,
        "NLPConcatHeadsDecodeDeviceOperation_0": 6234.9,
        "BinaryDeviceOperation_0": 871.3,
        "BinaryDeviceOperation_1": 13043.2,
        "BinaryDeviceOperation_2": 1860.9,
    }

    mapping_op_code_to_name = {
        "LayerNorm_0": "PreAllGatherLN_0",
        "LayerNorm_1": "PostAllGatherLN_0",
        "LayerNorm_2": "PreAllGatherLN_1",
        "LayerNorm_3": "PostAllGatherLN_1",
        "AllGatherAsync_0": "AllGatherAsync_LN_0",
        "AllGatherAsync_1": "AllGatherAsync_SDPA_0",
        "AllGatherAsync_2": "AllGatherAsync_LN_1",
        "ReshardDeviceOperation_0": "ReshardDeviceOperation_LN_0",
        "ReshardDeviceOperation_1": "ReshardDeviceOperation_CreateHeads",
        "ReshardDeviceOperation_2": "ReshardDeviceOperation_LN_1",
        "Matmul_0": "QKV_MM",
        "Matmul_1": "DO_MM",
        "Matmul_2": "FF1_MM",
        "Matmul_3": "FF3_MM",
        "Matmul_4": "FF2_MM",
        "AllReduceAsync_0": "AllReduceAsync_QKV",
        "AllReduceAsync_1": "AllReduceAsync_DO",
        "AllReduceAsync_2": "AllReduceAsync_FF1",
        "AllReduceAsync_3": "AllReduceAsync_FF3",
        "AllReduceAsync_4": "AllReduceAsync_FF2",
        "NLPCreateHeadsDecodeDeviceOperation_0": "CreateHeads",
        "RotaryEmbeddingLlamaFusedQK_0": "RotaryEmbeddingLlamaFusedQK",
        "PagedUpdateCacheDeviceOperation_0": "PagedUpdateCache",
        "ScaledDotProductAttentionDecode_0": "SDPA",
        "NLPConcatHeadsDecodeDeviceOperation_0": "ConcatHeads",
        "BinaryDeviceOperation_0": "Binary_Residual_0",
        "BinaryDeviceOperation_1": "Binary_Mult_Silu",
        "BinaryDeviceOperation_2": "Binary_Residual_1",
    }

    assert len(dispatch_duration_per_instance_averaged_dict) == len(
        expected_non_overlapped_dispatch_times_dict
    ), f"Expected {len(expected_non_overlapped_dispatch_times_dict)} operations, got {len(dispatch_duration_per_instance_averaged_dict)}. If the number or type of operations changed, expected times must be updated."

    passing = True
    for op_code_with_id, avg_dispatch_duration in dispatch_duration_per_instance_averaged_dict.items():
        if op_code_with_id in expected_non_overlapped_dispatch_times_dict:
            expected_time = expected_non_overlapped_dispatch_times_dict[op_code_with_id]
            op_name = mapping_op_code_to_name[op_code_with_id]
            benchmark_data.add_measurement(profiler, 0, step_name_non_overlapped_dispatch_duration, op_name, avg_dispatch_duration)
            if "AllReduceAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_reduce
            elif "AllGatherAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_gather
            else:
                tolerance = abs_tolerance_ns
            if avg_dispatch_duration > expected_time + tolerance:
                passing = False
                logger.info(
                    f"{op_code_with_id}: {avg_dispatch_duration} ns larger than expected {expected_time} ns by {abs(avg_dispatch_duration - expected_time)} ns (tolerance {tolerance} ns)"
                )
            elif avg_dispatch_duration < expected_time - tolerance:
                passing = False
                logger.info(
                    f"{op_code_with_id}: {avg_dispatch_duration} ns smaller than expected {expected_time} ns by {abs(expected_time - avg_dispatch_duration)} ns (tolerance {tolerance} ns)"
                )
        else:
            passing = False
            logger.info(f"Warning: {op_code_with_id} not found in expected_times_dict")

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg-llama-decoder",
        ml_model_name="llama70b-tg-decoder",
    )

    assert passing

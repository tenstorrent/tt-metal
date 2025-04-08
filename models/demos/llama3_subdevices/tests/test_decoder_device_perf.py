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
from models.perf.device_perf_utils import run_device_perf
from tt_metal.tools.profiler.process_model_log import (
    get_latest_ops_log_filename,
)

from models.demos.llama3_subdevices.demo.demo_decode import run_llama3_demo
from models.demos.llama3_subdevices.demo.demo_decode import LlamaOptimizations

mapping_op_code_to_name = {
    "LayerNorm_0": "PreAllGatherLN_0",
    "LayerNorm_1": "PostAllGatherLN_0",
    "LayerNorm_2": "PreAllGatherLN_1",
    "LayerNorm_3": "PostAllGatherLN_1",
    "AllGatherAsync_0": "AllGatherAsync_LN_0",
    "AllGatherAsync_1": "AllGatherAsync_SDPA_0",
    "AllGatherAsync_2": "AllGatherAsync_LN_1",
    "ShardedToInterleavedDeviceOperation_0": "ShardedToInterleavedDeviceOperation_LN_0",
    "ShardedToInterleavedDeviceOperation_1": "ShardedToInterleavedDeviceOperation_LN_1",
    "InterleavedToShardedDeviceOperation_0": "InterleavedToShardedDeviceOperation_LN_0",
    "InterleavedToShardedDeviceOperation_1": "InterleavedToShardedDeviceOperation_LN_1",
    "ReshardDeviceOperation_0": "ReshardDeviceOperation_LN_0",
    "ReshardDeviceOperation_1": "ReshardDeviceOperation_CreateHeads",
    "ReshardDeviceOperation_2": "ReshardDeviceOperation_LN_1",
    "ReshardDeviceOperation_3": "ReshardDeviceOperation_BinaryMultSilu",
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
            False,  # paged_attention
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
    start_pos,
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
        averaged_dict[op_code_with_id] = sum(input_dict[op_code_with_id]) / len(input_dict[op_code_with_id])
    return averaged_dict


@pytest.mark.parametrize(
    "abs_tolerance_ns",
    (2000,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_reduce",
    (3000,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_gather",
    (1500,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_sdpa",
    (3000,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_op_to_op",
    (800,),
)
@pytest.mark.models_device_performance_bare_metal
# To update:
# Run FAKE_DEVICE=TG TT_METAL_WORKER_RINGBUFFER_SIZE=122880 TT_METAL_ENABLE_ERISC_IRAM=1 pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device
# Copy the printed kernel_duration_per_instance_averaged_dict and dispatch_duration_per_instance_averaged_dict dictionaries
# Manually compare each entry between old-expected and the new average values
# - Any perf regressions? Everything as expected?
# If all looks good, update the expected_kernel_times_dict and expected_dispatch_times_dict with the new average values
# If the op list changed (new ops, less ops, fused ops), then update mapping_op_code_to_name and give the new ops meaningful names
# Run at least once again to verify the new expected values are correct and margins hold
def test_llama_TG_perf_device(
    reset_seeds,
    abs_tolerance_ns,
    abs_tolerance_ns_all_reduce,
    abs_tolerance_ns_all_gather,
    abs_tolerance_ns_sdpa,
    abs_tolerance_ns_op_to_op,
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = "tg-llama-decoder"
    batch_size = 32
    subdir = "tg-llama-decoder"
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
    df_model = df[int(len(df) / 3 * 2) :]

    # Excluding model embeddings and lmhead+sampling ops
    df_layers = df_model[4:-12]
    # Use layers 2-9 for verifying against targets for more stability
    df_first_layer = df_layers[: int(len(df_layers) / num_layers)]
    df_mid_layers = df_layers[int(len(df_layers) / num_layers) :]
    mid_layers_raw_dict = df_mid_layers[["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]].to_dict(
        orient="records"
    )
    first_layer_raw_dict = df_first_layer[["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]].to_dict(
        orient="records"
    )

    # Build dicts of op_code to list of durations
    kernel_duration_dict = build_duration_dict(mid_layers_raw_dict, "DEVICE KERNEL DURATION [ns]")
    dispatch_duration_dict = build_duration_dict(mid_layers_raw_dict, "OP TO OP LATENCY [ns]")

    # Build dicts of op_code_with_id to list of durations - one list per op instance
    kernel_duration_per_instance_dict = build_duration_per_instance_dict(kernel_duration_dict, num_layers - 1)
    dispatch_duration_per_instance_dict = build_duration_per_instance_dict(dispatch_duration_dict, num_layers - 1)

    # Average over all iterations of each op instance
    kernel_duration_per_instance_averaged_dict = average_per_instance_dict(kernel_duration_per_instance_dict)
    dispatch_duration_per_instance_averaged_dict = average_per_instance_dict(dispatch_duration_per_instance_dict)

    print(kernel_duration_per_instance_averaged_dict)
    print(dispatch_duration_per_instance_averaged_dict)

    expected_kernel_times_dict = {
        "LayerNorm_0": 6954.111111111111,
        "LayerNorm_1": 6530.777777777777,
        "LayerNorm_2": 6752.333333333333,
        "LayerNorm_3": 6543.111111111111,
        "AllGatherAsync_0": 4205.888888888889,
        "AllGatherAsync_1": 9638.666666666666,
        "AllGatherAsync_2": 4095.5555555555557,
        "ShardedToInterleavedDeviceOperation_0": 3947.222222222222,
        "ShardedToInterleavedDeviceOperation_1": 3388.5555555555557,
        "InterleavedToShardedDeviceOperation_0": 2546.5555555555557,
        "InterleavedToShardedDeviceOperation_1": 2036.111111111111,
        "Matmul_0": 10696.555555555555,
        "Matmul_1": 8584.111111111111,
        "Matmul_2": 10554.555555555555,
        "Matmul_3": 12118.888888888889,
        "Matmul_4": 16474.222222222223,
        "AllReduceAsync_0": 15395.555555555555,
        "AllReduceAsync_1": 21078.333333333332,
        "AllReduceAsync_2": 20948.777777777777,
        "AllReduceAsync_3": 21195.444444444445,
        "AllReduceAsync_4": 21145.777777777777,
        "NLPCreateHeadsDecodeDeviceOperation_0": 8568.222222222223,
        "RotaryEmbeddingLlamaFusedQK_0": 5023.888888888889,
        "PagedUpdateCacheDeviceOperation_0": 5606.444444444444,
        "ScaledDotProductAttentionDecode_0": 20472.11111111111,
        "NLPConcatHeadsDecodeDeviceOperation_0": 6665.0,
        "ReshardDeviceOperation_0": 1752.7777777777778,
        "BinaryDeviceOperation_0": 2408.8888888888887,
        "BinaryDeviceOperation_1": 11838.444444444445,
        "BinaryDeviceOperation_2": 2435.5555555555557,
    }

    expected_dispatch_times_dict = {
        "LayerNorm_0": 657.6666666666666,
        "LayerNorm_1": 634.3333333333334,
        "LayerNorm_2": 661.4444444444445,
        "LayerNorm_3": 641.0,
        "AllGatherAsync_0": 2417.5555555555557,
        "AllGatherAsync_1": 676.4444444444445,
        "AllGatherAsync_2": 2403.4444444444443,
        "ShardedToInterleavedDeviceOperation_0": 2211.6666666666665,
        "ShardedToInterleavedDeviceOperation_1": 2212.222222222222,
        "InterleavedToShardedDeviceOperation_0": 631.8888888888889,
        "InterleavedToShardedDeviceOperation_1": 637.7777777777778,
        "Matmul_0": 610.7777777777778,
        "Matmul_1": 653.3333333333334,
        "Matmul_2": 630.4444444444445,
        "Matmul_3": 737.5555555555555,
        "Matmul_4": 658.7777777777778,
        "AllReduceAsync_0": 609.0,
        "AllReduceAsync_1": 626.5555555555555,
        "AllReduceAsync_2": 641.2222222222222,
        "AllReduceAsync_3": 629.7777777777778,
        "AllReduceAsync_4": 637.1111111111111,
        "NLPCreateHeadsDecodeDeviceOperation_0": 761.1111111111111,
        "RotaryEmbeddingLlamaFusedQK_0": 540.7777777777778,
        "PagedUpdateCacheDeviceOperation_0": 773.8888888888889,
        "ScaledDotProductAttentionDecode_0": 585.5555555555555,
        "NLPConcatHeadsDecodeDeviceOperation_0": 620.5555555555555,
        "ReshardDeviceOperation_0": 655.2222222222222,
        "BinaryDeviceOperation_0": 653.6666666666666,
        "BinaryDeviceOperation_1": 717.4444444444445,
        "BinaryDeviceOperation_2": 647.7777777777778,
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
            benchmark_data.add_measurement(profiler, 0, step_name, op_name, avg_kernel_duration)
            benchmark_data.add_measurement(profiler, 0, step_name, op_name + "_op_to_op", avg_dispatch_duration)

            # Verify kernel duration is within tolerance
            if "AllReduceAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_reduce
            elif "AllGatherAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_gather
            elif "ScaledDotProductAttentionDecode" in op_code_with_id:
                tolerance = abs_tolerance_ns_sdpa
            else:
                tolerance = abs_tolerance_ns
            if avg_kernel_duration > expected_time + tolerance:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns larger than expected {expected_time} ns by {abs(avg_kernel_duration - expected_time)} ns (tolerance {tolerance} ns)"
                )
            elif avg_kernel_duration < expected_time - tolerance:
                passing = False
                logger.info(
                    f"{op_code_with_id} kernel: {avg_kernel_duration} ns smaller than expected {expected_time} ns by {abs(expected_time - avg_kernel_duration)} ns (tolerance {tolerance} ns)"
                )
            # Verify op_to_op latency is within tolerance
            expected_time = expected_dispatch_times_dict[op_code_with_id]
            if avg_dispatch_duration > expected_time + abs_tolerance_ns_op_to_op:
                passing = False
                logger.info(
                    f"{op_code_with_id} dispatch: {avg_dispatch_duration} ns larger than expected {expected_time} ns by {abs(avg_dispatch_duration - expected_time)} ns (tolerance {abs_tolerance_ns_op_to_op} ns)"
                )
            elif avg_dispatch_duration < expected_time - abs_tolerance_ns_op_to_op:
                passing = False
                logger.info(
                    f"{op_code_with_id} dispatch: {avg_dispatch_duration} ns smaller than expected {expected_time} ns by {abs(expected_time - avg_dispatch_duration)} ns (tolerance {abs_tolerance_ns_op_to_op} ns)"
                )

        else:
            passing = False
            logger.info(f"Warning: {op_code_with_id} not found in expected_times_dict")

    # Calculate e2e performance
    e2e_estimate_80l = 0
    for entry in first_layer_raw_dict:
        kernel_duration = entry["DEVICE KERNEL DURATION [ns]"]
        dispatch_duration = entry["OP TO OP LATENCY [ns]"]
        e2e_estimate_80l += kernel_duration + dispatch_duration
    for op_code_with_id, avg_kernel_duration in kernel_duration_per_instance_averaged_dict.items():
        avg_dispatch_duration = dispatch_duration_per_instance_averaged_dict[op_code_with_id]
        e2e_estimate_80l += (avg_kernel_duration + avg_dispatch_duration) * 79  # weighting avg for 79 layers

    print(f"e2e estimate: {e2e_estimate_80l}")

    benchmark_data.add_measurement(profiler, 0, step_name, "e2e_estimate_80l", e2e_estimate_80l)
    # Estimated T/s/u is 1000000 / (80L-duration + ~1240 lmhead+sampling+embeddings + ~300 python-overhead
    benchmark_data.add_measurement(profiler, 0, step_name, "tsu_estimate", 1000000 / (e2e_estimate_80l + 1240 + 300))

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg-llama-decoder",
        ml_model_name="llama70b-tg-decoder",
    )

    assert passing


@pytest.mark.parametrize(
    "abs_tolerance_ns",
    (1500,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_reduce",
    (1500,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_gather",
    (1500,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_sdpa",
    (2000,),
)
@pytest.mark.models_device_performance_bare_metal
# To update:
# Run FAKE_DEVICE=TG TT_METAL_WORKER_RINGBUFFER_SIZE=122880 TT_METAL_ENABLE_ERISC_IRAM=1 TT_METAL_KERNELS_EARLY_RETURN=1  pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_TG_perf_device_non_overlapped_dispatch
# Copy the printed dispatch_duration_per_instance_averaged_dict dictionary
# Manually compare each entry between old-expected and the new average values
# - Any perf regressions? Everything as expected?
# If all looks good, update the expected_dispatch_times_dict with the new average values
# If the op list changed (new ops, less ops, fused ops), and not done for the above test, then update mapping_op_code_to_name and give the new ops meaningful names
# Run at least once again to verify the new expected values are correct and margins hold
def test_llama_TG_perf_device_non_overlapped_dispatch(
    reset_seeds, abs_tolerance_ns, abs_tolerance_ns_all_reduce, abs_tolerance_ns_all_gather, abs_tolerance_ns_sdpa
):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = "tg-llama-decoder"
    batch_size = 32
    subdir = "tg-llama-decoder"
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
    df_layers = df_model[4:-12]
    all_layers_raw_dict = df_layers[["OP CODE", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"]].to_dict(
        orient="records"
    )

    # Build dicts of op_code to list of durations
    dispatch_duration_dict = build_duration_dict(all_layers_raw_dict, "OP TO OP LATENCY [ns]")

    # Build dicts of op_code_with_id to list of durations - one list per op instance
    dispatch_duration_per_instance_dict = build_duration_per_instance_dict(dispatch_duration_dict, num_layers)

    # Average over all iterations of each op instance
    dispatch_duration_per_instance_averaged_dict = average_per_instance_dict(dispatch_duration_per_instance_dict)

    print(dispatch_duration_per_instance_averaged_dict)

    expected_non_overlapped_dispatch_times_dict = {
        "LayerNorm_0": 6493.1,
        "LayerNorm_1": 6190.6,
        "LayerNorm_2": 6376.6,
        "LayerNorm_3": 6431.2,
        "AllGatherAsync_0": 2273.2,
        "AllGatherAsync_1": 2983.2,
        "AllGatherAsync_2": 2272.1,
        "ShardedToInterleavedDeviceOperation_0": 1919.0,
        "ShardedToInterleavedDeviceOperation_1": 1910.2,
        "InterleavedToShardedDeviceOperation_0": 10347.3,
        "InterleavedToShardedDeviceOperation_1": 10736.0,
        "Matmul_0": 6102.0,
        "Matmul_1": 5641.0,
        "Matmul_2": 6109.0,
        "Matmul_3": 6144.8,
        "Matmul_4": 6029.3,
        "AllReduceAsync_0": 7349.4,
        "AllReduceAsync_1": 6484.7,
        "AllReduceAsync_2": 11900.0,
        "AllReduceAsync_3": 11874.1,
        "AllReduceAsync_4": 6475.9,
        "NLPCreateHeadsDecodeDeviceOperation_0": 8156.7,
        "RotaryEmbeddingLlamaFusedQK_0": 2844.3,
        "PagedUpdateCacheDeviceOperation_0": 4670.5,
        "ScaledDotProductAttentionDecode_0": 9741.5,
        "NLPConcatHeadsDecodeDeviceOperation_0": 3463.5,
        "ReshardDeviceOperation_0": 10074.9,
        "BinaryDeviceOperation_0": 5907.6,
        "BinaryDeviceOperation_1": 6111.2,
        "BinaryDeviceOperation_2": 6751.9,
    }

    assert len(dispatch_duration_per_instance_averaged_dict) == len(
        expected_non_overlapped_dispatch_times_dict
    ), f"Expected {len(expected_non_overlapped_dispatch_times_dict)} operations, got {len(dispatch_duration_per_instance_averaged_dict)}. If the number or type of operations changed, expected times must be updated."

    passing = True
    for op_code_with_id, avg_dispatch_duration in dispatch_duration_per_instance_averaged_dict.items():
        if op_code_with_id in expected_non_overlapped_dispatch_times_dict:
            expected_time = expected_non_overlapped_dispatch_times_dict[op_code_with_id]
            op_name = mapping_op_code_to_name[op_code_with_id]
            benchmark_data.add_measurement(profiler, 0, step_name, op_name + "_dispatch", avg_dispatch_duration)
            if "AllReduceAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_reduce
            elif "AllGatherAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_gather
            elif "ScaledDotProductAttentionDecode" in op_code_with_id:
                tolerance = abs_tolerance_ns_sdpa
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

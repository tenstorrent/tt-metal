# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler
from models.perf.device_perf_utils import run_device_perf_detailed

os.environ.setdefault("MESH_DEVICE", "TG")

THRESHOLD = 0.5
THRESHOLD_PERCENTAGE = 0.03

"""
old_moe_gate_prefill_dict = {
    "SliceDeviceOperation": 13,
    "MatmulDeviceOperation": 12,
    "BinaryNgDeviceOperation": 12,
    "RepeatDeviceOperation": 7,
    "AllGatherAsyncDeviceOperation": 6,
    "ReshapeViewDeviceOperation": 6,
    "FillPadDeviceOperation": 6,
    "TilizeDeviceOperation": 4,
    "UntilizeWithUnpaddingDeviceOperation": 4,
    "TransposeDeviceOperation": 3,
    "TilizeWithValPaddingDeviceOperation": 3,
    "TopKDeviceOperation": 3,
    "LayerNormDeviceOperation": 2,
    "FastReduceNCDeviceOperation": 2,
    "LayerNormPostAllGatherDeviceOperation": 2,
    "LayerNormPreAllGatherDeviceOperation": 2,
    "UntilizeDeviceOperation": 2,
    "ConcatDeviceOperation": 2,
    "ReduceDeviceOperation": 2,
    "ReduceScatterMinimalAsyncDeviceOperation": 2,
    "RotaryEmbeddingLlamaDeviceOperation": 2,
    "PadDeviceOperation": 2,
    "TypecastDeviceOperation": 1,
    "SDPAOperation": 1,
    "PagedFillCacheDeviceOperation": 1,
    "UnaryDeviceOperation": 1,
    "GatherDeviceOperation": 1,
    "ScatterDeviceOperation": 1,
    "PermuteDeviceOperation": 1,
    "AllToAllDispatchDeviceOperation": 1,
    "FullDeviceOperation": 1,
    "AllToAllCombineDeviceOperation": 1,
}
"""

moe_gate_prefill_dict = {
    "MatmulDeviceOperation": 12,
    "SliceDeviceOperation": 7,
    "BinaryNgDeviceOperation": 7,
    "RepeatDeviceOperation": 6,
    "AllGatherAsyncDeviceOperation": 6,
    "UntilizeDeviceOperation": 6,
    "InterleavedToShardedDeviceOperation": 5,
    "TilizeDeviceOperation": 5,
    "ReshapeViewDeviceOperation": 4,
    "TransposeDeviceOperation": 3,
    "FastReduceNCDeviceOperation": 2,
    "LayerNormPreAllGatherDeviceOperation": 2,
    "LayerNormPostAllGatherDeviceOperation": 2,
    "LayerNormDeviceOperation": 2,
    "ReduceScatterMinimalAsyncDeviceOperation": 2,
    "ConcatDeviceOperation": 2,
    "RotaryEmbeddingLlamaDeviceOperation": 2,
    "ReshardDeviceOperation": 2,
    "ShardedToInterleavedDeviceOperation": 2,
    "PagedFillCacheDeviceOperation": 1,
    "TypecastDeviceOperation": 1,
    "UntilizeWithUnpaddingDeviceOperation": 1,
    "SDPAOperation": 1,
    "GenericOpDeviceOperation": 1,
    "TilizeWithValPaddingDeviceOperation": 1,
    "PermuteDeviceOperation": 1,
    "AllToAllDispatchDeviceOperation": 1,
    "FullDeviceOperation": 1,
    "AllToAllCombineDeviceOperation": 1,
}

"""
old_moe_gate_decode_dict = {
    'BinaryNgDeviceOperation': 14,
    'SliceDeviceOperation': 13,
    'MatmulDeviceOperation': 12,
    'TransposeDeviceOperation': 8,
    'RepeatDeviceOperation': 7,
    'FillPadDeviceOperation': 6,
    'ShardedToInterleavedDeviceOperation': 6,
    'TilizeDeviceOperation': 5,
    'InterleavedToShardedDeviceOperation': 5,
    'ReshapeViewDeviceOperation': 4,
    'AllGatherAsyncDeviceOperation': 4,
    'LayerNormDeviceOperation': 4,
    'UntilizeDeviceOperation': 4,
    'TilizeWithValPaddingDeviceOperation': 4,
    'UnaryDeviceOperation': 4,
    'UntilizeWithUnpaddingDeviceOperation': 4,
    'TopKDeviceOperation': 3,
    'ConcatDeviceOperation': 3,
    'FastReduceNCDeviceOperation': 2,
    'ReshardDeviceOperation': 2,
    'RotaryEmbeddingLlamaDeviceOperation': 2,
    'PadDeviceOperation': 2,
    'ReduceDeviceOperation': 2,
    'AllBroadcastDeviceOperation': 1,
    'SdpaDecodeDeviceOperation': 1,
    'MeshPartitionDeviceOperation': 1,
    'GenericOpDeviceOperation': 1,
    'AllToAllAsyncGenericDeviceOperation': 1,
    'PagedUpdateCacheDeviceOperation': 1,
    'ScatterDeviceOperation': 1,
    'GatherDeviceOperation': 1,
    'PermuteDeviceOperation': 1,
    'AllToAllDispatchDeviceOperation': 1,
    'FullDeviceOperation': 1,
    'AllToAllCombineDeviceOperation': 1,
    'ReduceScatterMinimalAsyncDeviceOperation': 1
}
"""

moe_gate_decode_dict = {
    "MatmulDeviceOperation": 12,
    "InterleavedToShardedDeviceOperation": 10,
    "BinaryNgDeviceOperation": 9,
    "TransposeDeviceOperation": 8,
    "ShardedToInterleavedDeviceOperation": 8,
    "UntilizeDeviceOperation": 8,
    "SliceDeviceOperation": 7,
    "TilizeDeviceOperation": 6,
    "RepeatDeviceOperation": 6,
    "LayerNormDeviceOperation": 4,
    "ReshardDeviceOperation": 4,
    "AllGatherAsyncDeviceOperation": 4,
    "ConcatDeviceOperation": 3,
    "UnaryDeviceOperation": 3,
    "FastReduceNCDeviceOperation": 2,
    "TilizeWithValPaddingDeviceOperation": 2,
    "GenericOpDeviceOperation": 2,
    "RotaryEmbeddingLlamaDeviceOperation": 2,
    "ReshapeViewDeviceOperation": 2,
    "MeshPartitionDeviceOperation": 1,
    "PagedUpdateCacheDeviceOperation": 1,
    "AllBroadcastDeviceOperation": 1,
    "SdpaDecodeDeviceOperation": 1,
    "AllToAllAsyncGenericDeviceOperation": 1,
    "PermuteDeviceOperation": 1,
    "UntilizeWithUnpaddingDeviceOperation": 1,
    "AllToAllDispatchDeviceOperation": 1,
    "FullDeviceOperation": 1,
    "AllToAllCombineDeviceOperation": 1,
    "ReduceScatterMinimalAsyncDeviceOperation": 1,
}


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip device initialization since this test spawns child pytest processes."""


# Override galaxy_type fixture to avoid calling ttnn.cluster.get_cluster_type()
# which might acquire device locks
@pytest.fixture(scope="function")
def galaxy_type():
    """Return galaxy type without initializing devices."""

    # Use environment variable if set, otherwise default to "TG"
    return os.environ.get("GALAXY_TYPE", "TG")


@pytest.mark.parametrize(
    "step_name, mode, warmup_iters, num_iters, perf_target_us",
    [
        ("moe_gate_decode", "decode", 3, 5, 7924.49),
        ("moe_gate_prefill", "prefill", 3, 5, 11815.22),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_moe_gate_perf(
    step_name,
    mode,
    warmup_iters,
    num_iters,
    perf_target_us,
    galaxy_type,
):
    subdir = "deepseek_moe_gate_perf"
    if mode == "decode":
        command = f"pytest models/demos/deepseek_v3/tests/test_decoder_block_trace_mode.py::test_forward_pass[{warmup_iters}-{num_iters}-mode_decode_seq_1_batch_32_pos_random-MoEDecoderBlock2D-model.layers.3-3-run_test_forward_pass_decoder2d-device_params0] --recalculate-weights"
    else:
        command = f"pytest models/demos/deepseek_v3/tests/test_decoder_block_trace_mode.py::test_forward_pass[{warmup_iters}-{num_iters}-mode_prefill_seq_128_batch_1-MoEDecoderBlock2D-model.layers.3-3-run_test_forward_pass_decoder2d-device_params0] --recalculate-weights"
    cols = ["DEVICE KERNEL"]

    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    op_name = ""

    profiler.start("run")
    profiler.start(step_name)
    results = run_device_perf_detailed(
        command, subdir, cols, op_name, has_signposts=True, warmup_iters=warmup_iters, per_op=True
    )
    profiler.end(step_name)
    profiler.end("run")
    # Get the measured performance
    measured_min = 0
    measured_max = 0
    measured_avg = 0
    measured_std = 0
    if mode == "decode":
        current_value_dict = moe_gate_decode_dict.copy()
    else:
        current_value_dict = moe_gate_prefill_dict.copy()
    for op in results.keys():
        assert op in current_value_dict, f"Operation {op} not found in current_value_dict"
        measured_min += current_value_dict[op] * results[op][cols[0]]["MIN"]
        measured_max += current_value_dict[op] * results[op][cols[0]]["MAX"]
        measured_avg += current_value_dict[op] * results[op][cols[0]]["AVG"]
        measured_std += current_value_dict[op] * results[op][cols[0]]["STD"]
    measured_avg_us = measured_avg / 1000

    logger.info(f"Measured performance: {measured_avg_us:.3f} us vs. target: {perf_target_us} us")

    # Save the measurement
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-min", measured_min)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-max", measured_max)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-avg", measured_avg)
    benchmark_data.add_measurement(profiler, 0, step_name, f"{step_name}-std", measured_std)
    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg_deepseek_moe_perf",
        ml_model_name="deepseek-v3-tg",
    )

    threshold = max(THRESHOLD, perf_target_us * THRESHOLD_PERCENTAGE)

    assert (
        measured_avg_us < perf_target_us + threshold
    ), f"Performance is worse than target: {measured_avg_us} us > {perf_target_us} us, the threshold was {threshold} us"
    assert (
        measured_avg_us > perf_target_us - threshold
    ), f"Performance is more than {threshold} us better than target, update the target: {measured_avg_us} us < {perf_target_us} us"

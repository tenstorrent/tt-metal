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
new_moe_gate_prefill_dict = {}


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
    "step_name, warmup_iters, perf_target_us",
    [
        ("moe_gate", 10, 31.5),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_moe_gate_perf(
    step_name,
    warmup_iters,
    perf_target_us,
    galaxy_type,
):
    subdir = "deepseek_moe_gate_perf"
    command = "pytest models/demos/deepseek_v3/tests/test_decoder_block.py::test_forward_pass[mode_prefill_seq_128_batch_1-MoEDecoderBlock2D-model.layers.3-3-run_test_forward_pass_decoder2d-device_params0]"
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
    current_value_dict = old_moe_gate_prefill_dict.copy()
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

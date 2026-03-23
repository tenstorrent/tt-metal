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

unoptimized_moe_gate_decode_dict = {
    "FillPadDeviceOperation": 6,
    "SliceDeviceOperation": 6,
    "BinaryNgDeviceOperation": 5,
    "RepeatDeviceOperation": 5,
    "ReshapeViewDeviceOperation": 4,
    "TopKDeviceOperation": 3,
    "TilizeWithValPaddingDeviceOperation": 3,
    "UntilizeWithUnpaddingDeviceOperation": 2,
    "ReduceDeviceOperation": 2,
    "PadDeviceOperation": 2,
    "TilizeDeviceOperation": 1,
    "UnaryDeviceOperation": 1,
    "ScatterDeviceOperation": 1,
    "GatherDeviceOperation": 1,
    "MatmulDeviceOperation": 1,
}

unoptimized_moe_gate_prefill_dict = {
    "FillPadDeviceOperation": 6,
    "SliceDeviceOperation": 6,
    "BinaryNgDeviceOperation": 5,
    "RepeatDeviceOperation": 5,
    "ReshapeViewDeviceOperation": 4,
    "TopKDeviceOperation": 3,
    "TilizeWithValPaddingDeviceOperation": 3,
    "UntilizeWithUnpaddingDeviceOperation": 2,
    "ReduceDeviceOperation": 2,
    "PadDeviceOperation": 2,
    "TilizeDeviceOperation": 1,
    "UnaryDeviceOperation": 1,
    "ScatterDeviceOperation": 1,
    "GatherDeviceOperation": 1,
    "MatmulDeviceOperation": 1,
}

moe_gate_decode_dict = {
    "InterleavedToShardedDeviceOperation": 5,
    "UntilizeDeviceOperation": 5,
    "RepeatDeviceOperation": 4,
    "CopyDeviceOperation": 4,
    "TilizeDeviceOperation": 3,
    "SliceDeviceOperation": 3,
    "TypecastDeviceOperation": 2,
    "TilizeWithValPaddingDeviceOperation": 2,
    "ReshapeViewDeviceOperation": 2,
    "ShardedToInterleavedDeviceOperation": 2,
    "PermuteDeviceOperation": 1,
    "UntilizeWithUnpaddingDeviceOperation": 1,
    "GenericOpDeviceOperation": 1,
    "MatmulDeviceOperation": 1,
}

moe_gate_prefill_dict = {
    "InterleavedToShardedDeviceOperation": 10,
    "RepeatDeviceOperation": 8,
    "UntilizeDeviceOperation": 8,
    "TilizeWithValPaddingDeviceOperation": 5,
    "TilizeDeviceOperation": 5,
    "SliceDeviceOperation": 5,
    "ShardedToInterleavedDeviceOperation": 4,
    "ReshapeViewDeviceOperation": 4,
    "TypecastDeviceOperation": 3,
    "PermuteDeviceOperation": 2,
    "GenericOpDeviceOperation": 2,
    "UntilizeWithUnpaddingDeviceOperation": 2,
    "ConcatDeviceOperation": 2,
    "MatmulDeviceOperation": 1,
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
    "use_unoptimized_moe_gate, mode, warmup_iters, num_iters, perf_target_us",
    [
        (False, "decode", 5, 10, 357.00),
        (False, "prefill", 5, 10, 338.89),
        (True, "decode", 5, 10, 812.04),
        (True, "prefill", 5, 10, 832.85),
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_moe_gate_perf(
    use_unoptimized_moe_gate,
    mode,
    warmup_iters,
    num_iters,
    perf_target_us,
    galaxy_type,
):
    subdir = f"deepseek_moe_gate_perf_{'unoptimized' if use_unoptimized_moe_gate else 'optimized'}"
    step_name = f"moe_gate_{mode}"
    if use_unoptimized_moe_gate:
        if mode == "decode":
            command = "pytest models/demos/deepseek_v3/tests/test_unoptimized_moe_gate.py::test_forward_pass[real-5-10-True-True-decode-32-1-device_params0]"
        else:
            command = "pytest models/demos/deepseek_v3/tests/test_unoptimized_moe_gate.py::test_forward_pass[real-5-10-True-True-prefill-1-512-device_params0]"
    else:
        if mode == "decode":
            command = "pytest models/demos/deepseek_v3/tests/test_moe_gate.py::test_forward_pass[5-10-decode-32-1-device_params0]"
        else:
            command = "pytest models/demos/deepseek_v3/tests/test_moe_gate.py::test_forward_pass[5-10-prefill-1-512-device_params0]"
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
        if use_unoptimized_moe_gate:
            current_value_dict = unoptimized_moe_gate_decode_dict.copy()
        else:
            current_value_dict = moe_gate_decode_dict.copy()
    else:
        if use_unoptimized_moe_gate:
            current_value_dict = unoptimized_moe_gate_prefill_dict.copy()
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

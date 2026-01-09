# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_wormhole_b0, profiler
from models.demos.blackhole.vit.tests.vit_hiRes_test_infra import create_test_infra
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def run_trace_2cq_model(device, test_infra, num_warmup_iterations, num_measurement_iterations):
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

    def model_wrapper(l1_input_tensor):
        test_infra.input_tensor = l1_input_tensor
        return test_infra.run()

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2),
        model=model_wrapper,
        device=device,
        dram_input_memory_config=sharded_mem_config_DRAM,
        l1_input_memory_config=input_mem_config,
    )

    logger.info("Compiling pipeline...")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")
    ttnn.ReadDeviceProfiler(device)

    host_inputs = [tt_inputs_host] * num_measurement_iterations
    pipeline.preallocate_output_tensors_on_host(num_measurement_iterations)

    logger.info(f"Running warmup for {num_warmup_iterations} iterations...")
    warmup_inputs = [tt_inputs_host] * num_warmup_iterations
    pipeline.preallocate_output_tensors_on_host(num_warmup_iterations)
    _ = pipeline.enqueue(warmup_inputs).pop_all()

    logger.info(f"Running measurement for {num_measurement_iterations} iterations...")
    pipeline.preallocate_output_tensors_on_host(num_measurement_iterations)

    ttnn.synchronize_device(device)
    if use_signpost:
        signpost(header="start")

    profiler.start("run")
    pipeline.enqueue(host_inputs).pop_all()
    profiler.end("run")

    if use_signpost:
        signpost(header="stop")
    ttnn.ReadDeviceProfiler(device)

    pipeline.cleanup()


@pytest.mark.skipif(is_wormhole_b0(), reason="Unsupported on WH")
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 8000000}],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [1024, 2048, 3072])
@pytest.mark.parametrize("hidden_size", [512, 1024, 1536, 2304])
def test_vit_hiRes(device, batch_size, sequence_size, hidden_size):
    expected_samples_per_sec_map = {
        # (batch_size, sequence_size, hidden_size): expected_samples/sec
        # Measured on Blackhole (batch_size=1, 12 encoder layers)
        (1, 1024, 512): 400,
        (1, 1024, 1024): 242,
        (1, 1024, 1536): 167,
        (1, 1024, 2304): 108,
        (1, 2048, 512): 164,
        (1, 2048, 1024): 105,
        (1, 2048, 1536): 76,
        (1, 2048, 2304): 55,
        (1, 3072, 512): 114,
        (1, 3072, 1024): 69,
        (1, 3072, 1536): 48,
        (1, 3072, 2304): 31,
    }

    key = (batch_size, sequence_size, hidden_size)
    expected_samples_per_sec = expected_samples_per_sec_map.get(key, 100)

    torch.manual_seed(0)
    profiler.clear()

    num_attention_heads = 16
    if hidden_size >= 2048:
        num_attention_heads = 12

    test_infra = create_test_infra(
        device,
        batch_size,
        sequence_size,
        hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=12,
        use_random_input_tensor=True,
    )

    ttnn.synchronize_device(device)

    if sequence_size >= 2048 or hidden_size >= 1536:
        num_warmup_iterations = 20
        num_measurement_iterations = 100
    else:
        num_warmup_iterations = 50
        num_measurement_iterations = 500

    run_trace_2cq_model(device, test_infra, num_warmup_iterations, num_measurement_iterations)

    inference_time_avg = profiler.get("run") / num_measurement_iterations
    expected_inference_time_avg = batch_size / expected_samples_per_sec

    prep_perf_report(
        model_name=f"ttnn_vit_hiRes_trace_2cq_batch_{batch_size}_seq_{sequence_size}_hidden_{hidden_size}",
        batch_size=batch_size,
        inference_and_compile_time=0,
        inference_time=inference_time_avg,
        expected_compile_time=0,
        expected_inference_time=expected_inference_time_avg,
        comments=f"sequence_size={sequence_size}, hidden_size={hidden_size}",
        inference_time_cpu=0,
    )

    model_name = f"ttnn_vit_hiRes_batch_{batch_size}_seq_{sequence_size}_hidden_{hidden_size}"
    logger.info(f"{model_name} inference time (avg): {inference_time_avg}")
    samples_per_sec = 1 / inference_time_avg * batch_size
    logger.info(f"Samples per second: {samples_per_sec}")

    margin = 0.10
    min_range = expected_samples_per_sec * (1 - margin)
    max_range = expected_samples_per_sec * (1 + margin)

    if samples_per_sec < min_range or samples_per_sec > max_range:
        logger.warning(
            f"Samples per second {samples_per_sec} outside expected range [{min_range}, {max_range}]. "
            f"Consider updating expected_samples_per_sec to {int(samples_per_sec)}"
        )

    assert (
        samples_per_sec > min_range and samples_per_sec < max_range
    ), f"Samples per second {samples_per_sec} is either too low or high, expected at to be in range of: [{min_range}, {max_range}]"

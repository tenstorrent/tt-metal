# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import profiler, run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.experimental.atss_swin_l_dyhead.runner.performant_runner_infra import ATSSPerformanceRunnerInfra
from models.perf.perf_utils import prep_perf_report


def run_model_pipeline(device, test_infra, num_measurement_iterations, use_trace, num_command_queues):
    torch_input_nhwc = test_infra.torch_input_tensor
    tt_inputs_host = ttnn.from_torch(
        torch_input_nhwc.permute(0, 3, 1, 2),
        dtype=ttnn.bfloat16,
    )

    def run_single():
        input_on_device = ttnn.from_torch(
            torch_input_nhwc.permute(0, 3, 1, 2),
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        test_infra.input_tensor = input_on_device
        test_infra.run()

    logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")

    profiler.start("compile")
    run_single()
    ttnn.synchronize_device(device)
    profiler.end("compile")

    run_profiler_key = f"run_model_pipeline_{num_command_queues}cqs"
    profiler.start(run_profiler_key)
    for _ in range(num_measurement_iterations):
        run_single()
    ttnn.synchronize_device(device)
    profiler.end(run_profiler_key)

    logger.info("Performance measurement complete (PCC validation skipped for perf test)")

    return run_profiler_key


def run_perf_e2e_atss_swinl_dyhead(
    device,
    batch_size_per_device,
    model_location_generator,
    resolution,
    expected_inference_throughput,
    use_trace=False,
    num_command_queues=1,
):
    profiler.clear()

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    test_infra = ATSSPerformanceRunnerInfra(
        device,
        batch_size,
        model_location_generator=model_location_generator,
        resolution=resolution,
        inputs_mesh_mapper=inputs_mesh_mapper,
        outputs_mesh_composer=output_mesh_composer,
        input_path=".models/experimental/atss_swin_l_dyhead/demo/horse_dog.jpg",
    )

    num_measurement_iterations = 32
    run_profiler_key = run_model_pipeline(
        device,
        test_infra,
        num_measurement_iterations,
        use_trace=use_trace,
        num_command_queues=num_command_queues,
    )

    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get(run_profiler_key) / num_measurement_iterations
    expected_inference_time = batch_size / expected_inference_throughput

    prep_perf_report(
        model_name=f"ttnn_atss_swinl_dyhead_{'trace' if use_trace else 'notrace'}_{num_command_queues}cqs_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time_avg,
        expected_compile_time=240,
        expected_inference_time=expected_inference_time,
        comments=f"{resolution[0]}x{resolution[1]}_batchsize{batch_size}",
        inference_time_cpu=0.0,
    )

    logger.info(
        f"AtssSwinlDyhead{resolution[0]}x{resolution[1]} batch_size: {batch_size}, inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"AtssSwinlDyhead compile time: {compile_time} s")


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((640, 640), 2)],
)
def test_atss_swinl_dyhead_perf_single_device(
    device,
    batch_size_per_device,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    run_perf_e2e_atss_swinl_dyhead(
        device,
        batch_size_per_device,
        model_location_generator,
        resolution,
        expected_inference_throughput,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((640, 640), 2)],
)
def test_atss_swinl_dyhead_perf_multi_device(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    run_perf_e2e_atss_swinl_dyhead(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
        resolution,
        expected_inference_throughput,
    )

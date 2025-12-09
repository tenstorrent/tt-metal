# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import profiler, run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.experimental.mobileNetV3.runner.performant_runner_infra import MobileNetV3PerformanceRunnerInfra
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


def run_model_pipeline(device, test_infra, num_measurement_iterations):
    tt_inputs_host, dram_input_mem_config, l1_input_mem_config, channels = test_infra.setup_sharded_input(device)

    original_batch = test_infra.batch_size
    original_height = test_infra.resolution[0]
    original_width = test_infra.resolution[1]

    def model_wrapper(input_tensor):
        reshaped_input = ttnn.reshape(input_tensor, (original_batch, original_height, original_width, channels))
        test_infra.input_tensor = reshaped_input
        test_infra.run()
        return test_infra.tt_output

    pipeline = create_pipeline_from_config(
        device=device,
        model=model_wrapper,
        config=PipelineConfig(
            use_trace=True,
            num_command_queues=2,
            all_transfers_on_separate_command_queue=False,
        ),
        dram_input_memory_config=dram_input_mem_config,
        l1_input_memory_config=l1_input_mem_config,
    )

    logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")

    host_inputs = [tt_inputs_host] * num_measurement_iterations
    pipeline.preallocate_output_tensors_on_host(num_measurement_iterations)

    logger.info(
        f"Starting performance pipeline for {num_measurement_iterations} iterations with batch_size={test_infra.batch_size} and num_devices={test_infra.num_devices}"
    )
    profiler.start("run_model_pipeline_2cqs")
    outputs = pipeline.enqueue(host_inputs).pop_all()
    profiler.end("run_model_pipeline_2cqs")
    for i, output in enumerate(outputs):
        test_infra.validate(output)
        logger.info(f"Output {i} validation passed")

    pipeline.cleanup()

    return outputs


def run_perf_e2e_mobilenetV3(
    device,
    batch_size_per_device,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    profiler.clear()

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    test_infra = MobileNetV3PerformanceRunnerInfra(
        device,
        batch_size,
        model_location_generator=model_location_generator,
        resolution=resolution,
        inputs_mesh_mapper=inputs_mesh_mapper,
        outputs_mesh_composer=output_mesh_composer,
        input_path=".models/experimental/mobileNetV3/resources/dog.jpeg",
    )

    num_measurement_iterations = 32
    run_model_pipeline(device, test_infra, num_measurement_iterations)

    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get("run_model_pipeline_2cqs") / num_measurement_iterations
    expected_inference_time = batch_size / expected_inference_throughput

    prep_perf_report(
        model_name=f"ttnn_mobilenetV3_trace_2cqs_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time_avg,
        expected_compile_time=240,
        expected_inference_time=expected_inference_time,
        comments=f"{resolution[0]}x{resolution[1]}_batchsize{batch_size}",
        inference_time_cpu=0.0,
    )

    logger.info(
        f"MobileNetV3{resolution[0]}x{resolution[1]} batch_size: {batch_size}, inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"MobileNetV3 compile time: {compile_time} s")


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1702912, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((224, 224), 250)],
)
def test_e2e_performant(
    device,
    batch_size_per_device,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    run_perf_e2e_mobilenetV3(
        device,
        batch_size_per_device,
        model_location_generator,
        resolution,
        expected_inference_throughput,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1702912, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("batch_size_per_device", (1,))
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((224, 224), 250)],
)
def test_e2e_performant_dp(
    mesh_device,
    batch_size_per_device,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    run_perf_e2e_mobilenetV3(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
        resolution,
        expected_inference_throughput,
    )

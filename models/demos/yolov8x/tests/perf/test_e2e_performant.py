# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

from models.common.utility_functions import profiler, run_for_wormhole_b0
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8x.common import YOLOV8X_L1_SMALL_SIZE
from models.demos.yolov8x.runner.performant_runner_infra import YOLOv8xPerformanceRunnerInfra
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


def _run_model_pipeline(device, test_infra, num_measurement_iterations, num_command_queues, trace):
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

    def model_wrapper(l1_input_tensor):
        test_infra.input_tensor = l1_input_tensor
        test_infra.run()
        return test_infra.output_tensor

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=trace, num_command_queues=num_command_queues, all_transfers_on_separate_command_queue=False
        ),
        model=model_wrapper,
        device=device,
        dram_input_memory_config=sharded_mem_config_DRAM,
        l1_input_memory_config=input_mem_config,
    )

    logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")

    host_inputs = [tt_inputs_host] * num_measurement_iterations

    pipeline.preallocate_output_tensors_on_host(
        num_measurement_iterations,
    )

    logger.info(
        f"Starting performance pipeline for {num_measurement_iterations} iterations with batch_size={test_infra.batch_size} and num_devices={test_infra.num_devices}"
    )
    profiler.start(f"run_model_pipeline_{num_command_queues}cqs")
    outputs = pipeline.enqueue(host_inputs).pop_all()
    profiler.end(f"run_model_pipeline_{num_command_queues}cqs")

    for i, output in enumerate(outputs):
        test_infra.validate(output)
        logger.info(f"Output {i} validation passed")

    pipeline.cleanup()


def run_trace_2cq_model_pipeline(device, test_infra, num_measurement_iterations):
    _run_model_pipeline(
        device,
        test_infra,
        num_measurement_iterations,
        num_command_queues=2,
        trace=True,
    )


def run_perf_e2e_yolov8x(
    device,
    batch_size_per_device,
    model_location_generator,
    expected_inference_throughput,
):
    profiler.clear()

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    test_infra = YOLOv8xPerformanceRunnerInfra(
        device,
        batch_size,
        model_location_generator=model_location_generator,
        inputs_mesh_mapper=inputs_mesh_mapper,
        outputs_mesh_composer=output_mesh_composer,
    )

    num_measurement_iterations = 15
    run_trace_2cq_model_pipeline(device, test_infra, num_measurement_iterations)

    first_iter_time = profiler.get("compile")
    inference_time_avg = profiler.get(f"run_model_pipeline_2cqs") / num_measurement_iterations
    compile_time = first_iter_time - 2 * inference_time_avg
    expected_inference_time = batch_size / expected_inference_throughput

    prep_perf_report(
        model_name=f"ttnn_yolov8x_trace_2cqs_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=120,
        expected_inference_time=expected_inference_time,
        comments=f"640x640_batchsize{batch_size}",
        inference_time_cpu=0.0,
    )

    logger.info(
        f"YoloV8x 640x640 batch_size{batch_size} inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"YoloV8x compile time: {compile_time}")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8X_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8x_performant(
    model_location_generator,
    device,
    batch_size_per_device,
):
    run_perf_e2e_yolov8x(
        device,
        batch_size_per_device,
        model_location_generator,
        expected_inference_throughput=62,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8X_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((1),),
)
def test_run_yolov8x_performant_dp(
    model_location_generator,
    mesh_device,
    batch_size_per_device,
):
    run_perf_e2e_yolov8x(
        mesh_device,
        batch_size_per_device,
        model_location_generator,
        expected_inference_throughput=121,
    )

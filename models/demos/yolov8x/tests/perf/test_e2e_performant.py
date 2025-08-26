# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov8x.common import YOLOV8X_L1_SMALL_SIZE
from models.demos.yolov8x.runner.performant_runner_infra import YOLOv8xPerformanceRunnerInfra
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from models.utility_functions import profiler, run_for_wormhole_b0


def _run_model_pipeline(
    device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations, num_command_queues, trace
):
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


def run_model_pipeline(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    _run_model_pipeline(
        device,
        tt_inputs,
        test_infra,
        num_warmup_iterations,
        num_measurement_iterations,
        num_command_queues=1,
        trace=False,
    )


def run_trace_model_pipeline(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    _run_model_pipeline(
        device,
        tt_inputs,
        test_infra,
        num_warmup_iterations,
        num_measurement_iterations,
        num_command_queues=1,
        trace=True,
    )


def run_2cq_model_pipeline(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    _run_model_pipeline(
        device,
        tt_inputs,
        test_infra,
        num_warmup_iterations,
        num_measurement_iterations,
        num_command_queues=2,
        trace=False,
    )


def run_trace_2cq_model_pipeline(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    _run_model_pipeline(
        device,
        tt_inputs,
        test_infra,
        num_warmup_iterations,
        num_measurement_iterations,
        num_command_queues=2,
        trace=True,
    )


def run_perf_e2e_yolov8x(
    device,
    batch_size_per_device,
    model_location_generator,
    expected_inference_throughput,
    model_version="yolov8x_trace_2cqs",
):
    profiler.clear()

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    test_infra = YOLOv8xPerformanceRunnerInfra(
        device,
        batch_size_per_device,
        model_location_generator=model_location_generator,
        inputs_mesh_mapper=inputs_mesh_mapper,
        outputs_mesh_composer=output_mesh_composer,
    )

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    input_shape = (batch_size_per_device * num_devices, 3, 640, 640)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    if "yolov8x_trace_2cqs" in model_version:
        run_trace_2cq_model_pipeline(
            device, ttnn_input_tensor, test_infra, num_warmup_iterations, num_measurement_iterations
        )
    elif "yolov8x_trace" in model_version:
        run_trace_model_pipeline(
            device, ttnn_input_tensor, test_infra, num_warmup_iterations, num_measurement_iterations
        )
    elif "yolov8x_2cqs" in model_version:
        run_2cq_model_pipeline(device, ttnn_input_tensor, test_infra, num_warmup_iterations, num_measurement_iterations)
    elif "yolov8x" in model_version:
        run_model_pipeline(device, ttnn_input_tensor, test_infra, num_warmup_iterations, num_measurement_iterations)
    else:
        assert False, f"Model version to run {model_version} not found"

    first_iter_time = profiler.get("compile")

    # ensuring inference time fluctuations is not noise
    num_cqs = 2 if "2cqs" in model_version else 1
    inference_time_avg = profiler.get(f"run_model_pipeline_{num_cqs}cqs") / num_measurement_iterations

    compile_time = first_iter_time - 2 * inference_time_avg
    expected_inference_time = batch_size / expected_inference_throughput

    prep_perf_report(
        model_name=f"ttnn_{model_version}_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=1,
        expected_inference_time=expected_inference_time,
        comments=f"640x640_batchsize{batch_size}",
        inference_time_cpu=0.0,
    )

    logger.info(
        f"YoloV8x 640x640 batch_size{batch_size} inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"YoloV8x compile time: {compile_time}")


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
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
        expected_inference_throughput=50,
        model_version="yolov8x_trace_2cqs",
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
        expected_inference_throughput=100,
        model_version="yolov8x_trace_2cqs",
    )

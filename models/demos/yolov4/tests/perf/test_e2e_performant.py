# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE
from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra
from models.demos.yolov4.runner.pipeline_runner import YoloV4PipelineRunner
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from models.utility_functions import profiler, run_for_wormhole_b0


def _run_model_pipeline(
    device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations, num_command_queues, trace
):
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(
            use_trace=trace, num_command_queues=num_command_queues, all_transfers_on_separate_command_queue=False
        ),
        model=YoloV4PipelineRunner(test_infra),
        device=device,
        dram_input_memory_config=sharded_mem_config_DRAM,
        l1_input_memory_config=input_mem_config,
    )

    logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")

    host_inputs = [tt_inputs_host] * num_measurement_iterations
    logger.info(
        f"Starting performance pipeline for {num_measurement_iterations} iterations with batch_size={test_infra.batch_size} and num_devices={test_infra.num_devices}"
    )
    outputs = []
    profiler.start(f"run_model_pipeline_{num_command_queues}cqs")
    for host_input in host_inputs:
        outputs.append(*pipeline.enqueue([host_input]).pop_all())
    profiler.end(f"run_model_pipeline_{num_command_queues}cqs")

    for i, output in enumerate(outputs):
        test_infra.validate(output)
        logger.info(f"Output {i} validation passed")

    pipeline.cleanup()


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


def run_perf_e2e_yolov4(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    profiler.clear()

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    test_infra = YOLOv4PerformanceRunnerInfra(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator=model_location_generator,
        resolution=resolution,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    input_shape = (batch_size_per_device * num_devices, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    run_trace_2cq_model_pipeline(
        device, ttnn_input_tensor, test_infra, num_warmup_iterations, num_measurement_iterations
    )

    compile_time = profiler.get("compile")
    inference_time_avg = profiler.get(f"run_model_pipeline_2cqs") / num_measurement_iterations
    expected_inference_time = batch_size / expected_inference_throughput

    prep_perf_report(
        model_name=f"ttnn_yolov4_trace_2cqs_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=compile_time,
        inference_time=inference_time_avg,
        expected_compile_time=1,
        expected_inference_time=expected_inference_time,
        comments=f"{resolution[0]}x{resolution[1]}_batchsize{batch_size}",
        inference_time_cpu=0.0,
    )

    logger.info(
        f"YoloV4 {resolution[0]}x{resolution[1]} batch_size{batch_size} inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"YoloV4 compile time: {compile_time}")

    throughput_avg = batch_size / inference_time_avg
    assert (
        throughput_avg >= expected_inference_throughput
    ), f"Expected end-to-end performance to exceed {expected_inference_throughput} fps but was {throughput_avg} fps"


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV4_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((320, 320), 130), ((640, 640), 65)],
)
def test_e2e_performant(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    pytest.skip("https://github.com/tenstorrent/tt-metal/issues/28113")
    run_perf_e2e_yolov4(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
        expected_inference_throughput,
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV4_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((320, 320), 235), ((640, 640), 120)],
)
def test_e2e_performant_dp(
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    expected_inference_throughput,
):
    pytest.skip("https://github.com/tenstorrent/tt-metal/issues/28113")
    run_perf_e2e_yolov4(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
        expected_inference_throughput,
    )

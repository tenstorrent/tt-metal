# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import sys

import pytest
import torch
from loguru import logger

import models.demos.yolov7.reference.yolov7_model as yolov7_model
import models.demos.yolov7.reference.yolov7_utils as yolov7_utils
import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov7.common import YOLOV7_L1_SMALL_SIZE
from models.demos.yolov7.runner.performant_runner_infra import YOLOv7PerformanceRunnerInfra
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from models.utility_functions import profiler, run_for_wormhole_b0

sys.modules["models.common"] = yolov7_utils
sys.modules["models.yolo"] = yolov7_model


def _run_model_pipeline(device, tt_inputs, test_infra, num_warmup_iterations, num_measurement_iterations):
    """Run the model using TT-CNN Pipeline and Executor APIs."""

    # Get memory configs from the infrastructure
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(
        device, mesh_mapper=test_infra.inputs_mesh_mapper, mesh_composer=test_infra.outputs_mesh_composer
    )

    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        use_trace=True,
        num_command_queues=2,
        all_transfers_on_separate_command_queue=False,
    )

    # Create a model wrapper that handles the input tensor properly
    def model_wrapper(l1_input_tensor):
        # Set up the input tensor for the model
        test_infra.input_tensor = l1_input_tensor
        test_infra.run()
        return test_infra.output_tensor

    # Create pipeline from config
    pipeline = create_pipeline_from_config(
        device=device,
        model=model_wrapper,
        config=pipeline_config,
        dram_input_memory_config=sharded_mem_config_DRAM,
        l1_input_memory_config=input_mem_config,
    )

    # Compile the pipeline
    logger.info(f"Running model warmup with input shape {list(tt_inputs_host.shape)}")
    profiler.start("compile")
    pipeline.compile(tt_inputs_host)
    profiler.end("compile")

    # Prepare inputs for measurement
    host_inputs = [tt_inputs_host] * num_measurement_iterations

    # Preallocate output tensors
    pipeline.preallocate_output_tensors_on_host(num_measurement_iterations)

    # Run measurement iterations
    logger.info(
        f"Starting performance pipeline for {num_measurement_iterations} iterations with batch_size={test_infra.batch_size} and num_devices={test_infra.num_devices}"
    )

    profiler.start("run_model_pipeline_2cqs")
    outputs = pipeline.enqueue(host_inputs).pop_all()
    profiler.end("run_model_pipeline_2cqs")

    # Validate outputs
    for i, output in enumerate(outputs):
        test_infra.validate(output)
        logger.info(f"Output {i} validation passed")

    pipeline.cleanup()

    return outputs


def run_perf_e2e_yolov7(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    expected_inference_throughput,
    model_version="yolov7_trace_2cqs",
):
    profiler.clear()

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    test_infra = YOLOv7PerformanceRunnerInfra(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator=model_location_generator,
        resolution=resolution,
        inputs_mesh_mapper=inputs_mesh_mapper,
        outputs_mesh_composer=output_mesh_composer,
    )

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    input_shape = (batch_size_per_device * num_devices, 3, *resolution)
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)

    num_warmup_iterations = 5
    num_measurement_iterations = 15

    if "yolov7_trace_2cqs" in model_version:
        _run_model_pipeline(device, ttnn_input_tensor, test_infra, num_warmup_iterations, num_measurement_iterations)
    else:
        assert False, f"Model version to run {model_version} not found"

    first_iter_time = profiler.get("compile")

    # ensuring inference time fluctuations is not noise
    num_cqs = 2 if "2cqs" in model_version else 1
    inference_time_avg = profiler.get(f"run_model_pipeline_{num_cqs}cqs") / num_measurement_iterations

    compile_time = first_iter_time

    logger.info(
        f"YoloV7 {resolution[0]}x{resolution[1]} batch_size{batch_size} inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"YoloV7 compile time: {compile_time}")


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV7_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((640, 640), 46)],
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
    run_perf_e2e_yolov7(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
        expected_inference_throughput,
        model_version="yolov7_trace_2cqs",
    )


@run_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV7_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution, expected_inference_throughput",
    [((640, 640), 46)],
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
    run_perf_e2e_yolov7(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
        expected_inference_throughput,
        model_version="yolov7_trace_2cqs",
    )

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger
from ttnn.device import is_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.vanilla_unet_new.tt.common import (
    VANILLA_UNET_L1_SMALL_SIZE,
    VANILLA_UNET_PCC_WH,
    VANILLA_UNET_TRACE_SIZE,
    create_unet_preprocessor,
    load_reference_model,
)
from models.demos.vanilla_unet_new.tt.config import create_unet_configs_from_parameters
from models.demos.vanilla_unet_new.tt.model import create_unet_from_configs
from models.experimental.functional_unet.tt.model_preprocessing import create_unet_input_tensors
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.tt_cnn.tt.pipeline import (
    PipelineConfig,
    create_pipeline_from_config,
    get_memory_config_for_persistent_dram_tensor,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch, expected_device_perf_fps",
    ((1, 200.0) if is_wormhole_b0() else (1, 400.0),),
)
def test_vanilla_unet_perf_device(batch: int, expected_device_perf_fps: float):
    command = f"pytest models/demos/vanilla_unet_new/tests/test_unet_model.py::test_vanilla_unet_model"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    post_processed_results = run_device_perf(
        command, subdir="unet_vanilla", num_iterations=1, cols=cols, batch_size=batch
    )
    expected_perf_cols = {inference_time_key: expected_device_perf_fps}
    expected_results = check_device_perf(
        post_processed_results, margin=0.01, expected_perf_cols=expected_perf_cols, assert_on_fail=True
    )
    prep_device_perf_report(
        model_name=f"ttnn_vanilla_unet{batch}",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )


def create_unet_pipeline_model(model):
    def run(l1_input_tensor):
        assert l1_input_tensor.storage_type() == ttnn.StorageType.DEVICE, "Model expects input tensor to be on device"
        assert (
            l1_input_tensor.memory_config().buffer_type == ttnn.BufferType.L1
        ), "Model expects input tensor to be in L1"
        return model(l1_input_tensor, deallocate_input_activation=True)

    return run


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": VANILLA_UNET_L1_SMALL_SIZE,
            "trace_region_size": VANILLA_UNET_TRACE_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch, input_channels, input_height, input_width, expected_compile_time, expected_throughput_fps",
    [(1, 3, 480, 640, 30.0, 180)],
)
def test_vanilla_unet_perf_e2e(
    num_iterations,
    batch,
    input_channels,
    input_height,
    input_width,
    expected_compile_time,
    expected_throughput_fps,
    device,
    reset_seeds,
    model_location_generator,
):
    reference_model = load_reference_model(model_location_generator)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_unet_preprocessor(device), device=None
    )

    configs = create_unet_configs_from_parameters(
        parameters=parameters, input_height=input_height, input_width=input_width, batch_size=batch
    )

    model = create_unet_from_configs(configs, device)
    run_model = create_unet_pipeline_model(model)

    torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(
        batch=batch,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        groups=1,
        channel_order="first",
        pad=False,
        fold=True,
    )
    torch_output_tensor = reference_model(torch_input_tensor)

    dram_input_memory_config = get_memory_config_for_persistent_dram_tensor(
        ttnn_input_tensor.shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED, device.dram_grid_size()
    )

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2),
        model=run_model,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=configs.l1_input_memory_config,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()

    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={num_iterations * batch / (end-start) : .2f} fps")

    total_num_samples = batch
    prep_perf_report(
        model_name="unet_vanilla-trace-2cq",
        batch_size=total_num_samples,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=total_num_samples / expected_throughput_fps,
        comments=f"batch_{batch}-dp_1",
    )

    logger.info(f"Running sanity check against reference model output")
    output_tensor = ttnn.to_torch(outputs[0]).reshape(torch_output_tensor.shape)
    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc=VANILLA_UNET_PCC_WH)
    logger.info(f"PCC check was successful ({pcc_message:.5f} > {VANILLA_UNET_PCC_WH:.5f})")


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": VANILLA_UNET_L1_SMALL_SIZE,
            "trace_region_size": VANILLA_UNET_TRACE_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("num_iterations", [32])
@pytest.mark.parametrize(
    "batch, input_channels, input_height, input_width, expected_compile_time, expected_throughput_fps",
    [(1, 3, 480, 640, 30.0, 350)],
)
def test_vanilla_unet_perf_e2e_multi_device(
    num_iterations,
    batch,
    input_channels,
    input_height,
    input_width,
    expected_compile_time,
    expected_throughput_fps,
    mesh_device,
    reset_seeds,
    model_location_generator,
):
    reference_model = load_reference_model(model_location_generator)

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    num_devices = len(mesh_device.get_device_ids())
    total_batch = num_devices * batch
    logger.info(f"Using {num_devices} devices for this test")

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_unet_preprocessor(mesh_device, mesh_mapper=weights_mesh_mapper),
        device=None,
    )

    configs = create_unet_configs_from_parameters(
        parameters=parameters, input_height=input_height, input_width=input_width, batch_size=batch
    )

    model = create_unet_from_configs(configs, mesh_device)
    run_model = create_unet_pipeline_model(model)

    torch_input_tensor, ttnn_input_tensor = create_unet_input_tensors(
        batch=total_batch,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        groups=1,
        channel_order="first",
        pad=False,
        fold=True,
        mesh_mapper=inputs_mesh_mapper,
    )
    torch_output_tensor = reference_model(torch_input_tensor)

    dram_input_memory_config = get_memory_config_for_persistent_dram_tensor(
        ttnn_input_tensor.shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED, mesh_device.dram_grid_size()
    )

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2),
        model=run_model,
        device=mesh_device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=configs.l1_input_memory_config,
    )

    input_tensors = [ttnn_input_tensor] * num_iterations

    start = time.time()
    pipeline.compile(ttnn_input_tensor)
    end = time.time()

    compile_time = end - start

    pipeline.preallocate_output_tensors_on_host(num_iterations)

    start = time.time()
    outputs = pipeline.enqueue(input_tensors).pop_all()
    end = time.time()

    pipeline.cleanup()

    inference_time = (end - start) / num_iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={num_iterations * total_batch / (end-start) : .2f} fps")

    prep_perf_report(
        model_name="unet_vanilla-trace-2cq-multi_device",
        batch_size=total_batch,
        inference_and_compile_time=compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=total_batch / expected_throughput_fps,
        comments=f"batch_{batch}-dp_{num_devices}",
    )

    logger.info(f"Running sanity check against reference model output")
    output_tensor = ttnn.to_torch(outputs[0], mesh_composer=output_mesh_composer).reshape(torch_output_tensor.shape)
    pcc_passed, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc=VANILLA_UNET_PCC_WH)
    logger.info(f"PCC check was successful ({pcc_message:.5f} > {VANILLA_UNET_PCC_WH:.5f})")

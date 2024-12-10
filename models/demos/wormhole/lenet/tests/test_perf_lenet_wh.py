# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time

from loguru import logger
import ttnn
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.demos.wormhole.lenet.tt import tt_lenet
from models.demos.wormhole.lenet import lenet_utils
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import is_wormhole_b0, skip_for_grayskull, is_grayskull
from ttnn.model_preprocessing import preprocess_model_parameters


def get_expected_times(tt_lenet):
    if is_wormhole_b0():
        return {
            tt_lenet: (17.64, 0.1026),
        }[tt_lenet]


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size",
    [64],
)
@pytest.mark.parametrize(
    "tt_lenet",
    [tt_lenet],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
def test_perf_lenet(mesh_device, batch_size, tt_lenet, model_location_generator, reset_seeds):
    disable_persistent_kernel_cache()
    num_classes = 10
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 2 * batch_size if mesh_device_flag else batch_size
    test_input, images, outputs = lenet_utils.get_test_data(batch_size)

    pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
    torch_lenet, state_dict = lenet_utils.load_torch_lenet(pt_model_path, num_classes)
    model = torch_lenet.float()
    inputs_mesh_mapper = None
    output_mesh_composer = None

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model, custom_preprocessor=lenet_utils.custom_preprocessor
        )

    parameters = lenet_utils.custom_preprocessor_device(parameters, device=mesh_device)
    x = test_input.permute(0, 2, 3, 1)
    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)
    durations = []
    for _ in range(100):
        start = time.time()

        ttnn_output = tt_lenet.lenet(
            mesh_device=mesh_device,
            input_tensor=x,
            parameters=parameters,
            mesh_mapper=inputs_mesh_mapper,
            mesh_composer=output_mesh_composer,
        )
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, *inference_times = durations
    inference_time = sum(inference_times) / len(inference_times)
    expected_compile_time, expected_inference_time = get_expected_times(tt_lenet)

    prep_perf_report(
        model_name="tt_lenet",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Inference times: {inference_times}")
    logger.info(f"Sample(s) per second: {1 / inference_time * batch_size}")
    assert (
        inference_time < expected_inference_time
    ), f"Expected inference time: {expected_inference_time} Actual inference time: {inference_time}"
    logger.info("Exit Lenet perf test")


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size",
    [64],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal(batch_size, reset_seeds):
    subdir = "tt_lenet"
    num_iterations = 1
    margin = 0.03

    if is_wormhole_b0():
        expected_perf = 62689.81422

    command = f"pytest tests/ttnn/integration_tests/lenet/test_lenet_wh.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 2 * batch_size if mesh_device_flag else batch_size
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"tt_lenet{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

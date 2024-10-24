# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import time
import pytest
import torch
from loguru import logger
from torchvision import transforms, datasets
from models.perf.perf_utils import prep_perf_report
from models.demos.wormhole.mnist.tt import tt_mnist
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.mnist.reference.mnist import MnistModel
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.utility_functions import is_grayskull, is_wormhole_b0, skip_for_grayskull
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, transform=None, download=True)


@skip_for_grayskull()
def get_expected_times(tt_mnist):
    if is_grayskull():
        return {
            tt_mnist: (2.3, 0.0041),
        }[tt_mnist]
    elif is_wormhole_b0():
        return {
            tt_mnist: (6.44, 0.0073),
        }[tt_mnist]


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "batch_size",
    [4],
)
@pytest.mark.parametrize(
    "tt_mnist",
    [tt_mnist],
)
def test_performance_mnist(mesh_device, batch_size, tt_mnist, model_location_generator):
    state_dict = torch.load(model_location_generator("mnist_model.pt", model_subdir="mnist"))
    model = MnistModel(state_dict)
    model = model.eval()
    disable_persistent_kernel_cache()
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    x, labels = next(iter(dataloader))

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=mesh_device,
        )

    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper, device=mesh_device)
    durations = []

    for _ in range(2):
        start = time.time()

        ttnn_output = tt_mnist.mnist(mesh_device, batch_size, x, parameters)
        end = time.time()
        durations.append(end - start)
        # enable_persistent_kernel_cache()

    inference_and_compile_time, *inference_times = durations
    average_inference_time = sum(inference_times) / len(inference_times)
    expected_compile_time, expected_inference_time = get_expected_times(tt_mnist)

    prep_perf_report(
        model_name="MNIST",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=average_inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - average_inference_time}")
    logger.info(f"Inference time: {average_inference_time}")
    logger.info(f"Inference times: {inference_times}")
    logger.info(f"Sample(s) per second: {1 / average_inference_time * batch_size}")


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [4, 14074.34],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal(batch_size, expected_perf):
    subdir = "ttnn_mnist"
    num_iterations = 1
    margin = 0.03

    command = f"pytest tests/ttnn/integration_tests/mnist/test_mnist.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"tt_mnist{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.mnist.reference.mnist import MnistModel
from models.demos.mnist.tt import tt_mnist
from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf
from models.perf.perf_utils import prep_perf_report
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    is_grayskull,
    is_wormhole_b0,
)

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, transform=None, download=True)


def get_expected_times(tt_mnist):
    if is_grayskull():
        return {
            tt_mnist: (3.54, 0.006),
        }[tt_mnist]
    elif is_wormhole_b0():
        return {
            tt_mnist: (3.89, 0.006),
        }[tt_mnist]


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "batch_size",
    [128],
)
@pytest.mark.parametrize(
    "tt_mnist",
    [tt_mnist],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_performance_mnist(device, batch_size, tt_mnist, model_location_generator, reset_seeds):
    state_dict = torch.load(model_location_generator("mnist_model.pt", model_subdir="mnist"))
    model = MnistModel(state_dict)
    model = model.eval()
    disable_persistent_kernel_cache()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: True,
        device=device,
    )
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    x, labels = next(iter(dataloader))
    test_input = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device)
    durations = []
    for _ in range(100):
        start = time.time()

        ttnn_output = tt_mnist.mnist(
            device=device,
            x=test_input,
            batch_size=batch_size,
            parameters=parameters,
        )
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()
    inference_and_compile_time, *inference_times = durations
    inference_time = sum(inference_times) / len(inference_times)
    expected_compile_time, expected_inference_time = get_expected_times(tt_mnist)

    prep_perf_report(
        model_name="MNIST",
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
    logger.info("Exit MNIST perf test")


@pytest.mark.parametrize(
    "batch_size",
    [128],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal(batch_size, reset_seeds):
    subdir = "ttnn_mnist"
    num_iterations = 1
    margin = 0.03
    expected_perf = 880000.0

    command = f"pytest tests/ttnn/integration_tests/mnist/test_mnist.py::test_mnist"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols, assert_on_fail=True)
    prep_device_perf_report(
        model_name=f"tt_mnist{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

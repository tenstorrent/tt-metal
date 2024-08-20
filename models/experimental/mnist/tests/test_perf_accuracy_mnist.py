# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loguru import logger
import ttnn
import pytest
import numpy as np
import evaluate
from torch import Generator

from models.experimental.mnist.tt.mnist_model import mnist_model
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)

from models.utility_functions import profiler
from models.perf.perf_utils import prep_perf_report


@pytest.mark.parametrize(
    "iterations",
    ((1000),),
)
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            0.1,
            9.0,
        ),
    ),
)
def test_perf(device, expected_inference_time, expected_compile_time, iterations, model_location_generator):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_iter"
    cpu_key = "ref_key"

    # Data preprocessing/loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(213)
    dataloader = DataLoader(test_dataset, batch_size=1, generator=g_cpu, shuffle=True)

    # Load model
    tt_model, pt_model = mnist_model(device, model_location_generator)

    with torch.no_grad():
        test_input, label = next(iter(dataloader))
        tt_input = torch_to_tt_tensor_rm(test_input, device)

        profiler.start(cpu_key)
        pt_out = pt_model(test_input)
        ttnn.synchronize_device(device)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_out = tt_model(tt_input)
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_out

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_out = tt_model(tt_input)
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_out

        calculated_label = []
        golden_label = []
        profiler.start(third_key)
        accuracy_metric = evaluate.load("accuracy")
        ttnn.synchronize_device(device)
        for idx, (image, label) in enumerate(dataloader):
            if idx == iterations:
                break

            tt_input = torch_to_tt_tensor_rm(image, device)
            tt_out = tt_model(tt_input)
            py_tt_out = tt_to_torch_tensor(tt_out)

            tt_prediction = torch.argmax(py_tt_out)
            calculated_label.append(tt_prediction.item())
            golden_label.append(label.item())

        profiler.end(third_key)
        accuracy = accuracy_metric.compute(references=golden_label, predictions=calculated_label)
        logger.info(f"Accuracy: {accuracy}")

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    # TODO: expected compile time (100 s) and expected inference time (100 s) are not real values
    # update to real time and add to CI pipeline
    prep_perf_report("mnist", 1, first_iter_time, second_iter_time, 100, 100, "", cpu_time)

    logger.info(f"mnist inference time: {second_iter_time}")
    logger.info(f"mnist compile time: {compile_time}")
    logger.info(f"mnist inference for {iterations} samples: {third_iter_time}")

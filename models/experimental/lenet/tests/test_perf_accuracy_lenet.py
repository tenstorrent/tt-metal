# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from loguru import logger
import pytest
import ttnn
import evaluate
from torch import Generator

from models.utility_functions import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from models.experimental.lenet.tt.lenet import lenet5
from models.experimental.lenet.lenet_utils import load_torch_lenet
from models.perf.perf_utils import prep_perf_report


def run_perf_inference(device, pcc, iterations, model_location_generator, reset_seeds):
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "third_iter"
    cpu_key = "ref_key"

    # Data preprocessing/loading
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=(0.1325,), std=(0.3105,))]
    )
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(213)
    dataloader = DataLoader(test_dataset, batch_size=1, generator=g_cpu, shuffle=True)

    disable_persistent_kernel_cache()
    num_classes = 10

    with torch.no_grad():
        # Initialize Torch model
        pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
        torch_LeNet, _ = load_torch_lenet(pt_model_path, num_classes)

        # Initialize TT model
        tt_lenet = lenet5(num_classes, device, model_location_generator)

        test_input, label = next(iter(dataloader))
        tt_input = torch_to_tt_tensor_rm(test_input, device)

        profiler.start(cpu_key)
        torch_output = torch_LeNet(test_input)
        ttnn.synchronize_device(device)
        profiler.end(cpu_key)

        tt_image = torch_to_tt_tensor_rm(test_input, device, put_on_device=False)

        profiler.start(first_key)
        tt_output = tt_lenet(tt_image)
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_lenet(tt_image)
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_output

        calculated_label = []
        golden_label = []
        profiler.start(third_key)
        accuracy_metric = evaluate.load("accuracy")

        ttnn.synchronize_device(device)
        for idx, (image, label) in enumerate(dataloader):
            if idx == iterations:
                break
            tt_input = torch_to_tt_tensor_rm(image, device)
            tt_out = tt_lenet(tt_input)
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

    prep_perf_report("lenet", 1, first_iter_time, second_iter_time, 100, 100, "", cpu_time)
    logger.info(f"lenet inference time: {second_iter_time}")
    logger.info(f"lenet compile time: {compile_time}")
    logger.info(f"lenet inference for {iterations} samples: {third_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "pcc, iterations",
    ((0.99, 1000),),
)
def test_perf_bare_metal(device, use_program_cache, pcc, iterations, model_location_generator, reset_seeds):
    run_perf_inference(device, pcc, iterations, model_location_generator, reset_seeds)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "pcc, iterations",
    ((0.99, 1000),),
)
def test_perf_virtual_machine(device, use_program_cache, pcc, iterations, model_location_generator, reset_seeds):
    run_perf_inference(device, pcc, iterations, model_location_generator, reset_seeds)

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from loguru import logger

from models.utility_functions import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    torch2tt_tensor,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.convnet_mnist.tt.convnet_mnist import convnet_mnist
from models.experimental.convnet_mnist.convnet_mnist_utils import get_test_data


@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            1.3,
            6.5,
        ),
    ),
)
def test_perf(device, use_program_cache, expected_inference_time, expected_compile_time):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    tt_model, pt_model = convnet_mnist(device)
    test_input, images = get_test_data(64)

    tt_input = torch2tt_tensor(test_input, device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

    with torch.no_grad():
        profiler.start(cpu_key)
        pt_model(test_input)
        ttnn.synchronize_device(device)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_model_outputs = tt_model(tt_input)
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_model_outputs

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_model_outputs = tt_model(tt_input)
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_model_outputs

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_perf_report("convnet_mnist", 1, first_iter_time, second_iter_time, "", cpu_time)

    logger.info(f"ConvNet Mnist inference time: {second_iter_time}")
    logger.info(f"ConvNet Mnist compile time: {compile_time}")

    assert second_iter_time < expected_inference_time, "ConvNet Mnist is too slow"
    assert compile_time < expected_compile_time, "ConvNet Mnist compile time is too slow"

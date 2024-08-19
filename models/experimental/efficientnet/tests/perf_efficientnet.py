# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
import torchvision
import pytest

from models.utility_functions import (
    torch2tt_tensor,
    profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report
from models.experimental.efficientnet.tt.efficientnet_model import efficientnet_b0


def make_input_tensor(imagenet_sample_input, resize=256, crop=224):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(resize),
            torchvision.transforms.CenterCrop(crop),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform(imagenet_sample_input)


@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            6.0,
            16.5,
        ),
    ),
)
def test_perf_efficientnet_b0(
    device,
    use_program_cache,
    imagenet_sample_input,
    expected_inference_time,
    expected_compile_time,
):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    test_input = make_input_tensor(imagenet_sample_input)

    hf_model = torchvision.models.efficientnet_b0()
    tt_input = torch2tt_tensor(test_input, tt_device=device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_model = efficientnet_b0(device)

    with torch.no_grad():
        profiler.start(cpu_key)
        hf_model(test_input)
        ttnn.synchronize_device(device)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_input)
        ttnn.synchronize_device(device)
        profiler.end(first_key)
        del tt_output

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_input)
        ttnn.synchronize_device(device)
        profiler.end(second_key)
        del tt_output

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    # TODO: expected compile time (100 s) and expected inference time (100 s) are not real values
    # update to real time and add to CI pipeline
    prep_perf_report("EfficientNet", 1, first_iter_time, second_iter_time, 100, 100, "b0", cpu_time)

    logger.info(f"efficientnet_b0 inference time: {second_iter_time}")
    logger.info(f"efficientnet_b0 compile time: {compile_time}")

    assert second_iter_time < expected_inference_time, "efficientnet_b0 is too slow"
    assert compile_time < expected_compile_time, "efficientnet_b0 compile time is too slow"

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import timm

import pytest
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    Profiler,
)
from models.perf.perf_utils import prep_perf_report

from models.experimental.hrnet.tt.hrnet_model import hrnet_w18_small

BATCH_SIZE = 1


@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            11.3,
            22,
        ),
    ),
)
def test_perf(device, expected_inference_time, expected_compile_time, imagenet_sample_input):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    HF_model = timm.create_model("hrnet_w18_small", pretrained=True)

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=False)

    tt_model = hrnet_w18_small(device, multi_scale_output=True)

    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = HF_model(imagenet_sample_input)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_input)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_input)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    prep_perf_report("hrnet", BATCH_SIZE, first_iter_time, second_iter_time, "w18_small", cpu_time)
    compile_time = first_iter_time - second_iter_time
    logger.info(f"hrnet inference time: {second_iter_time}")
    logger.info(f"hrnet compile time: {compile_time}")
    assert second_iter_time < expected_inference_time, "hrnet is too slow"
    assert compile_time < expected_compile_time, "hrnet compile time is too slow"

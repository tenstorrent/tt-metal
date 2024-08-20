# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
)
from models.utility_functions import Profiler
from models.perf.perf_utils import prep_perf_report
from models.experimental.ssd.tt.ssd_lite import ssd_for_object_detection
import pytest


from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large as pretrained,
)

BATCH_SIZE = 1


@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (
            13.48890358,
            0.5188875198,
        ),
    ),
)
def test_perf(
    device,
    expected_inference_time,
    expected_compile_time,
    imagenet_sample_input,
):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    torch_model = pretrained(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    torch_model.eval()

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=True)

    tt_model = ssd_for_object_detection(device)
    tt_model.eval()
    with torch.no_grad():
        profiler.start(cpu_key)
        torch_output = torch_model(imagenet_sample_input)
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

    # TODO: expected compile time (100 s) and expected inference time (100 s) are not real values
    # update to real time and add to CI pipeline
    prep_perf_report("SSD", BATCH_SIZE, first_iter_time, second_iter_time, 100, 100, "SSD_lite", cpu_time)

    compile_time = first_iter_time - second_iter_time

    logger.info(f"SSD inference time: {second_iter_time}")
    logger.info(f"SSD compile time: {compile_time}")
    assert second_iter_time < expected_inference_time, "SSD is too slow"
    assert compile_time < expected_compile_time, "SSD compile time is too slow"

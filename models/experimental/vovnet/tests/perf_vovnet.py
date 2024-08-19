# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import timm
import torch

from loguru import logger

from models.experimental.vovnet.tt.vovnet import vovnet_for_image_classification
from models.utility_functions import (
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.perf.perf_utils import prep_perf_report


BATCH_SIZE = 1


def test_perf(device, imagenet_sample_input):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "Execution time of vovnet first run"
    second_key = "Execution time of vovnet second run"
    cpu_key = "Execution time of reference model"

    torch_model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    tt_model = vovnet_for_image_classification(
        device=device,
    )

    input = imagenet_sample_input
    tt_input = torch_to_tt_tensor_rm(input, device)

    with torch.no_grad():
        profiler.start(cpu_key)
        model_output = torch_model(input)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_model(tt_input)
        tt_output_torch = tt_to_torch_tensor(tt_output)
        tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_model(tt_input)
        tt_output_torch = tt_to_torch_tensor(tt_output)
        tt_output_torch = tt_output_torch.squeeze(0).squeeze(0)
        profiler.end(second_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    cpu_time = profiler.get(cpu_key)

    logger.info("cpu time", cpu_time)
    profiler.print()

    # TODO: expected compile time (100 s) and expected inference time (100 s) are not real values
    # update to real time and add to CI pipeline
    prep_perf_report("vovnet", BATCH_SIZE, first_iter_time, second_iter_time, 100, 100, "vovnet", cpu_time)

import sys
from pathlib import Path

import timm
import torch

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../")

import tt_lib
from loguru import logger
from models.utility_functions_new import (
    Profiler,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    prep_report,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.vovnet.tt.vovnet import vovnet_for_image_classification

BATCH_SIZE = 1


def test_perf(imagenet_sample_input):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "Execution time of vovnet first run"
    second_key = "Execution time of vovnet second run"
    cpu_key = "Execution time of reference model"

    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    torch_model = timm.create_model(
        "hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True
    )

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

    prep_report(
        "vovnet", BATCH_SIZE, first_iter_time, second_iter_time, "vovnet", cpu_time
    )
    tt_lib.device.CloseDevice(device)

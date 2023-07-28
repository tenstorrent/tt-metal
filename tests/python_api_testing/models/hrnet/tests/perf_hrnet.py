from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")

import torch
from loguru import logger
import timm

import pytest
import tt_lib
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor, disable_persistent_kernel_cache, enable_persistent_kernel_cache
from utility_functions_new import Profiler, prep_report

from hrnet.tt.hrnet_model import hrnet_w18_small

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
def test_perf(expected_inference_time, expected_compile_time, imagenet_sample_input):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    HF_model = timm.create_model("hrnet_w18_small", pretrained=True)

    tt_input = torch_to_tt_tensor_rm(imagenet_sample_input, device, put_on_device=False)

    tt_model = hrnet_w18_small(device, host, multi_scale_output=True)

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

    prep_report(
        "hrnet", BATCH_SIZE, first_iter_time, second_iter_time, "w18_small", cpu_time
    )
    compile_time = first_iter_time - second_iter_time
    logger.info(f"hrnet inference time: {second_iter_time}")
    logger.info(f"hrnet compile time: {compile_time}")
    tt_lib.device.CloseDevice(device)
    assert second_iter_time < expected_inference_time, "hrnet is too slow"
    assert compile_time < expected_compile_time, "hrnet compile time is too slow"

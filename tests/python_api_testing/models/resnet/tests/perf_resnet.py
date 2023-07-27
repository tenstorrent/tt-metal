from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../tt")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
import pytest
import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, profiler
from utility_functions_new import disable_persistent_kernel_cache, enable_persistent_kernel_cache
from utility_functions_new import prep_report

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from loguru import logger
from tests.python_api_testing.models.resnet.metalResnetBlock import ResNet, Bottleneck

BATCH_SIZE = 1


@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time",
    (
        (5,
         58,
        ),
    ),
)

def test_perf(use_program_cache, expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    disable_persistent_kernel_cache()
    first_key = "first_iter"
    second_key = "second_iter"
    cpu_key = "ref_key"
    model_name = "microsoft/resnet-50"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)


    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}"

    torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()

    tt_resnet50 = ResNet(Bottleneck, [3, 4, 6, 3],
                    device=device,
                    state_dict=state_dict,
                    base_address="",
                    fold_batchnorm=True)


    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_resnet50(inputs)
        tt_lib.device.Synchronize()
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_resnet50(inputs)
        tt_lib.device.Synchronize()
        profiler.end(second_key)


    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    tt_lib.device.CloseDevice(device)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_report("resnet50", BATCH_SIZE, first_iter_time, second_iter_time, comments, cpu_time)

    logger.info(f"resnet50 {comments} inference time: {second_iter_time}")
    logger.info(f"resnet50 compile time: {compile_time}")

    assert second_iter_time < expected_inference_time, f"resnet50 {comments} is too slow"
    assert compile_time < expected_compile_time, "resnet50 compile time is too slow"

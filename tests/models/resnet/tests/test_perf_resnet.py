"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

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
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor, profiler
from models.utility_functions import disable_persistent_kernel_cache, enable_persistent_kernel_cache
from models.utility_functions import prep_report

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from loguru import logger
from tests.models.resnet.metalResnetBlock50 import ResNet, Bottleneck


def run_perf_resnet(batch_size, expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    disable_persistent_kernel_cache()
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "microsoft/resnet-50"

    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    inputs1 = inputs
    for i in range(batch_size-1):
        inputs = torch.cat((inputs, inputs1), dim=0)

    torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()

    tt_resnet50 = ResNet(Bottleneck, [3, 4, 6, 3],
                    device=device,
                    state_dict=state_dict,
                    base_address="",
                    fold_batchnorm=True,
                    storage_in_dram=False,
                    batch_size=batch_size)


    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        profiler.start(first_key)
        tt_output = tt_resnet50(inputs)
        tt_lib.device.Synchronize()
        profiler.end(first_key)

        #enable_persistent_kernel_cache()

        profiler.start(second_key)
        tt_output = tt_resnet50(inputs)
        tt_lib.device.Synchronize()
        profiler.end(second_key)


    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    tt_lib.device.CloseDevice(device)

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time
    prep_report(
        model_name=f"resnet50_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time
    )

    logger.info(f"resnet50 {comments} inference time: {second_iter_time}")
    logger.info(f"resnet50 compile time: {compile_time}")

    #assert second_iter_time < expected_inference_time, f"resnet50 {comments} is too slow"
    assert compile_time < expected_compile_time, "resnet50 compile time is too slow"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (1, 0.225, 33),
        (2, 0.225, 33),
        (8, 0.225, 33),
    ),
)
@pytest.mark.skip(reason="Conv disabled in main.")
def test_perf_bare_metal(use_program_cache, batch_size, expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    run_perf_resnet(batch_size, expected_inference_time, expected_compile_time, hf_cat_image_sample_input)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (1, 0.3, 36),
        (2, 0.3, 36),
        (8, 0.3, 36),
    ),
)
@pytest.mark.skip(reason="Conv disabled in main.")
def test_perf_virtual_machine(use_program_cache, batch_size, expected_inference_time, expected_compile_time, hf_cat_image_sample_input):
    run_perf_resnet(batch_size, expected_inference_time, expected_compile_time, hf_cat_image_sample_input)

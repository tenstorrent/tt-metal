# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
import pytest
import tt_lib

from models.utility_functions import is_e75
from models.utility_functions import profiler
from models.utility_functions import disable_persistent_kernel_cache
from models.perf.perf_utils import prep_perf_report

from loguru import logger
from models.demos.resnet.tt.metalResnetBlock50 import ResNet, Bottleneck

model_config = {
    "MATH_FIDELITY": tt_lib.tensor.MathFidelity.HiFi2,
    "WEIGHTS_DTYPE": tt_lib.tensor.DataType.BFLOAT8_B,
    "ACTIVATIONS_DTYPE": tt_lib.tensor.DataType.BFLOAT8_B,
}


def run_perf_resnet(
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
    device,
):
    disable_persistent_kernel_cache()
    if batch_size <= 2:
        pytest.skip("Batch size 1 and 2 are not supported with sharded data")
    first_key = f"first_iter_batchsize{batch_size}"
    second_key = f"second_iter_batchsize{batch_size}"
    cpu_key = f"ref_key_batchsize{batch_size}"
    model_name = "microsoft/resnet-50"

    image = hf_cat_image_sample_input
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    inputs = image_processor(image, return_tensors="pt")

    inputs = inputs["pixel_values"]
    comments = f"{list(inputs.shape)[-2]}x{list(inputs.shape)[-1]}_batchsize{batch_size}"

    inputs1 = inputs
    for i in range(batch_size - 1):
        inputs = torch.cat((inputs, inputs1), dim=0)

    torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()
    sharded = False
    if batch_size >= 8:
        sharded = True
    tt_resnet50 = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        device=device,
        state_dict=state_dict,
        base_address="",
        fold_batchnorm=True,
        storage_in_dram=False,
        batch_size=batch_size,
        model_config=model_config,
        sharded=sharded,
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        logits = torch_resnet50(inputs)
        profiler.end(cpu_key)

        tt_inputs = tt_resnet50.preprocessing(inputs)
        warmup_end = 5
        for iter in range(0, warmup_end):
            profiler.start(f"{iter}_key")
            _ = tt_resnet50(tt_inputs).cpu(blocking=True)
            profiler.end(f"{iter}_key")

        num_warm_iterations = 15
        warm_start = warmup_end
        warm_end = warm_start + num_warm_iterations

        outputs = []
        profiler.start(f"run")
        for iter in range(warm_start, warm_end):
            outputs.append(tt_resnet50(tt_inputs).cpu(blocking=False))

        tt_lib.device.Synchronize(device)
        profiler.end(f"run")

        # enable_persistent_kernel_cache()

    first_iter_time = profiler.get(f"{0}_key")

    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / num_warm_iterations

    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - inference_time_avg
    prep_perf_report(
        model_name=f"resnet50_batch_size{batch_size}",
        batch_size=batch_size,
        inference_and_compile_time=first_iter_time,
        inference_time=inference_time_avg,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"resnet50 {comments} inference time (avg): {inference_time_avg}")
    logger.info(f"resnet50 compile time: {compile_time}")

    assert inference_time_avg < expected_inference_time, f"resnet50 {comments} inference is too slow"
    assert compile_time < expected_compile_time, f"resnet50 {comments} compilation is too slow"


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (1, 0.015, 25),
        (2, 0.015, 25),
        (8, 0.015, 25),
        (16, 0.015, 25),
        (20, 0.0185, 25),
    ),
)
def test_perf_bare_metal(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")

    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "batch_size, expected_inference_time, expected_compile_time",
    (
        (1, 0.015, 30),
        (2, 0.02, 30),
        (8, 0.02, 30),
        (16, 0.04, 30),
        (20, 0.04, 30),
    ),
)
def test_perf_virtual_machine(
    device,
    use_program_cache,
    batch_size,
    expected_inference_time,
    expected_compile_time,
    hf_cat_image_sample_input,
):
    if is_e75(device):
        pytest.skip("Resnet is not supported on E75")

    run_perf_resnet(
        batch_size,
        expected_inference_time,
        expected_compile_time,
        hf_cat_image_sample_input,
        device,
    )

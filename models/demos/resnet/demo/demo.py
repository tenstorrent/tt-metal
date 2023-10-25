# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
import pytest
import tt_lib

from models.utility_functions import prep_report

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler
)

from loguru import logger
from models.demos.resnet.tt.metalResnetBlock50 import ResNet, Bottleneck


def run_resnet_inference(
    batch_size,
    hf_cat_image_sample_input,
    imagenet_label_dict,
    device,
    model_version = "microsoft/resnet-50",
):

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    # set up huggingface model - TT model will use weights from this model
    torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()
    # set up image processor
    image_processor = AutoImageProcessor.from_pretrained(model_version)

    # load inputs
    image = hf_cat_image_sample_input
    profiler.start(f"processing_inputs")
    inputs = image_processor(image, return_tensors="pt")
    inputs = inputs["pixel_values"]
    profiler.end(f"processing_inputs")
    inputs1 = inputs
    # TODO: make sure to load 8 different images!
    for i in range(batch_size - 1):
        inputs = torch.cat((inputs, inputs1), dim=0)

    # Create TT Model Start
    # this will move weights to device
    profiler.start(f"move_weights")
    sharded = False
    if batch_size == 8:
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
        sharded=sharded,
    )
    profiler.end(f"move_weights")

    profiler.start(f"preprocessing")

    tt_inputs = tt_resnet50.preprocessing(inputs)
    profiler.end(f"preprocessing")

    profiler.disable()
    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    tt_out = tt_resnet50(tt_inputs)
    tt_out = tt_out.to_torch().to(torch.float)
    prediction = tt_out[0][0][0].argmax()
    prediction = prediction.item()
    prediction = imagenet_label_dict[prediction]
    # breakpoint()
    tt_lib.device.Synchronize()
    profiler.end("first_model_run_with_compile", force_enable=True)
    # tt_out.deallocate()
    del tt_out

    profiler.enable()
    enable_persistent_kernel_cache()

    ##### Run Forward on TT Model Start
    profiler.start(f"model_run_for_inference")
    tt_out = tt_resnet50(tt_inputs)
    tt_out = tt_out.to_torch().to(torch.float)
    tt_lib.device.Synchronize()
    profiler.end(f"model_run_for_inference")

    profiler.start(f"post_processing")
    prediction = tt_out[0][0][0].argmax()
    prediction = prediction.item()
    profiler.end(f"post_processing")

    SINGLE_RUN = 1
    measurements = {
        "preprocessing": profiler.get('processing_input_one') + profiler.get('processing_input_two'),
        "moving_weights_to_device": profiler.get('move_weights'),
        "compile": profiler.get('first_model_run_with_compile') - (profiler.get('model_run_for_inference') / SINGLE_RUN),
        f"inference_for_single_run_batch_{batch_size}_without_cache": profiler.get('first_model_run_with_compile'),
        f"inference_for_{SINGLE_RUN}_run_batch_{batch_size}_without_cache": profiler.get('model_run_for_inference'),
        "inference_throughput": (SINGLE_RUN * batch_size) / profiler.get('model_run_for_inference'),
        "post_processing": profiler.get("processing_output_to_string")

    }

@pytest.mark.parametrize(
    "batch_size",
    (
        (8),
    ),
)
def test_demo(
    use_program_cache,
    batch_size,
    hf_cat_image_sample_input,
    imagenet_label_dict,
    device,
):
    run_resnet_inference(
        batch_size,
        hf_cat_image_sample_input,
        imagenet_label_dict,
        device,
    )

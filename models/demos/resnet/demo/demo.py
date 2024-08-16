# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
import pytest
import ttnn


from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
)

from models.demos.resnet.tests.demo_utils import get_data, get_data_loader, get_batch, load_resnet50_model

from loguru import logger
from models.demos.resnet.tt.metalResnetBlock50 import ResNet, Bottleneck

resnet_model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


def run_resnet_imagenet_inference(
    batch_size,
    iterations,
    imagenet_label_dict,
    model_location_generator,
    device,
    model_config=resnet_model_config,
    model_version="microsoft/resnet-50",
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    profiler.clear()

    # set up huggingface model - TT model will use weights from this model
    torch_resnet50 = load_resnet50_model(model_location_generator)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()
    # set up image processor
    image_processor = AutoImageProcessor.from_pretrained(model_version)

    # load inputs
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    # Create TT Model Start
    # this will move weights to device
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

    # load ImageNet batch by batch
    # and run inference
    correct = 0
    for iter in range(iterations):
        predictions = []
        inputs, labels = get_batch(data_loader, image_processor)
        tt_inputs = tt_resnet50.preprocessing(inputs)
        tt_output = tt_resnet50(tt_inputs)
        tt_output = tt_output.cpu().to_torch().to(torch.float)
        prediction = tt_output[:, 0, 0, :].argmax(dim=-1)
        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1
        del tt_output, tt_inputs, inputs, labels, predictions
    accuracy = correct / (batch_size * iterations)
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")


def run_resnet_inference(
    batch_size,
    input_loc,
    imagenet_label_dict,
    device,
    model_location_generator,
    model_config=resnet_model_config,
    model_version="microsoft/resnet-50",
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    # set up huggingface model - TT model will use weights from this model
    torch_resnet50 = load_resnet50_model(model_location_generator)
    torch_resnet50.eval()

    state_dict = torch_resnet50.state_dict()
    # set up image processor
    image_processor = AutoImageProcessor.from_pretrained(model_version)

    # load inputs
    images = get_data(input_loc)

    profiler.start(f"processing_inputs")
    inputs = None
    for i in range(batch_size):
        input_image = images[i].image
        if input_image.mode == "L":
            input_image = input_image.convert(mode="RGB")
        input = image_processor(input_image, return_tensors="pt")
        input = input["pixel_values"]
        if inputs == None:
            inputs = input
        else:
            inputs = torch.cat((inputs, input), dim=0)
    profiler.end(f"processing_inputs")

    # Create TT Model Start
    # this will move weights to device
    profiler.start(f"move_weights")
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
    profiler.end(f"move_weights")

    profiler.start(f"preprocessing")
    tt_inputs = tt_resnet50.preprocessing(inputs)
    profiler.end(f"preprocessing")

    profiler.disable()
    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    tt_out = tt_resnet50(tt_inputs)
    ttnn.synchronize_device(device)
    profiler.end("first_model_run_with_compile", force_enable=True)
    tt_out.deallocate()
    del tt_out

    profiler.enable()
    enable_persistent_kernel_cache()

    ##### Run Forward on TT Model Start
    profiler.start(f"model_run_for_inference")
    tt_out = tt_resnet50(tt_inputs)
    ttnn.synchronize_device(device)
    profiler.end(f"model_run_for_inference")

    profiler.start(f"post_processing")
    predictions = []
    tt_out = tt_out.cpu().to_torch().to(torch.float)

    prediction = tt_out[:, 0, 0, :].argmax(dim=-1)
    for i in range(batch_size):
        predictions.append(imagenet_label_dict[prediction[i].item()])
    profiler.end(f"post_processing")

    for pr, image in zip(predictions, images):
        logger.info(f"Expected Label: {imagenet_label_dict[image.label]}, Predicted Label: {pr}")

    SINGLE_RUN = 1
    measurements = {
        "preprocessing": profiler.get("preprocessing"),
        "moving_weights_to_device": profiler.get("move_weights"),
        "compile": profiler.get("first_model_run_with_compile")
        - (profiler.get("model_run_for_inference") / SINGLE_RUN),
        f"inference_for_single_run_batch_{batch_size}_without_cache": profiler.get("first_model_run_with_compile"),
        f"inference_for_{SINGLE_RUN}_run_batch_{batch_size}_without_cache": profiler.get("model_run_for_inference"),
        "inference_throughput": (SINGLE_RUN * batch_size) / profiler.get("model_run_for_inference"),
        "post_processing": profiler.get("post_processing"),
    }

    logger.info(f"pre processing duration: {measurements['preprocessing']} s")
    logger.info(f"moving weights to device duration: {measurements['moving_weights_to_device']} s")
    logger.info(f"compile time: {measurements['compile']} s")
    logger.info(
        f"inference time for single run of model with batch size {batch_size} without using cache: {measurements[f'inference_for_single_run_batch_{batch_size}_without_cache']} s"
    )
    logger.info(
        f"inference time for {SINGLE_RUN} run(s) of model with batch size {batch_size} and using cache: {measurements[f'inference_for_{SINGLE_RUN}_run_batch_{batch_size}_without_cache']} s"
    )
    logger.info(f"inference throughput: {measurements['inference_throughput'] } inputs/s")
    logger.info(f"post processing time: {measurements['post_processing']} s")

    del tt_out
    return measurements, predictions


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    (
        (16, 200),
        (20, 160),
    ),
)
def test_demo_imagenet(batch_size, iterations, imagenet_label_dict, model_location_generator, device):
    run_resnet_imagenet_inference(batch_size, iterations, imagenet_label_dict, model_location_generator, device)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_loc",
    ((20, "models/demos/resnet/demo/images/"),),
)
def test_demo_sample(device, use_program_cache, batch_size, input_loc, imagenet_label_dict, model_location_generator):
    run_resnet_inference(batch_size, input_loc, imagenet_label_dict, device, model_location_generator)

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from loguru import logger
from torchvision import models
from transformers import AutoImageProcessor
import pytest
import tt_lib


from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
)
import ttnn
from models.experimental.functional_vgg.tests.demo_utils import get_data, get_data_loader, get_batch, preprocess
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_vgg.tt import ttnn_vgg

vgg_model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
}


def run_vgg_inference(batch_size, input_loc, imagenet_label_dict, device, model_config=vgg_model_config):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    profiler.clear()

    # Setup model
    torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_model.to(torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
        custom_preprocessor=ttnn_vgg.custom_preprocessor,
    )

    # load inputs
    images = get_data(input_loc)

    profiler.start(f"processing_inputs")
    inputs = None
    for i in range(batch_size):
        input_image = images[i].image
        if input_image.mode == "L":
            input_image = input_image.convert(mode="RGB")
        input = preprocess(input_image)
        input = input.to(torch.bfloat16)
        input = input.unsqueeze(0)
        if inputs == None:
            inputs = input
        else:
            inputs = torch.cat((inputs, input), dim=0)

    permuted_inputs = torch.permute(inputs, (0, 2, 3, 1))
    tt_batched_input_tensor = ttnn.from_torch(permuted_inputs, ttnn.bfloat16)
    profiler.end(f"processing_inputs")

    profiler.disable()
    # Use force enable to only record this profiler call while others are disabled

    profiler.start("first_model_run_with_compile", force_enable=True)
    tt_out = ttnn_vgg.ttnn_vgg16(device, tt_batched_input_tensor, parameters, batch_size, model_config)
    tt_lib.device.Synchronize(device)
    profiler.end("first_model_run_with_compile", force_enable=True)
    tt_out.deallocate()
    del tt_out

    profiler.enable()
    enable_persistent_kernel_cache()

    ##### Run Forward on TT Model Start
    profiler.start(f"model_run_for_inference")
    tt_out = ttnn_vgg.ttnn_vgg16(device, tt_batched_input_tensor, parameters, batch_size, model_config)
    tt_lib.device.Synchronize(device)
    profiler.end(f"model_run_for_inference")

    profiler.start(f"post_processing")
    predictions = []
    tt_out = ttnn.to_torch(tt_out)

    prediction = tt_out[:, 0, 0, :].argmax(dim=-1)
    for i in range(batch_size):
        predictions.append(imagenet_label_dict[prediction[i].item()])
    profiler.end(f"post_processing")

    for pr, image in zip(predictions, images):
        logger.info(f"Expected Label: {imagenet_label_dict[image.label]}, Predicted Label: {pr}")

    SINGLE_RUN = 1
    measurements = {
        "preprocessing": profiler.get("preprocessing"),
        # "moving_weights_to_device": profiler.get("move_weights"),
        "compile": profiler.get("first_model_run_with_compile")
        - (profiler.get("model_run_for_inference") / SINGLE_RUN),
        f"inference_for_single_run_batch_{batch_size}_without_cache": profiler.get("first_model_run_with_compile"),
        f"inference_for_{SINGLE_RUN}_run_batch_{batch_size}_without_cache": profiler.get("model_run_for_inference"),
        "inference_throughput": (SINGLE_RUN * batch_size) / profiler.get("model_run_for_inference"),
        "post_processing": profiler.get("post_processing"),
    }

    logger.info(f"pre processing duration: {measurements['preprocessing']} s")
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
    "batch_size, input_loc",
    ((1, "models/experimental/functional_vgg/demo/images/"),),
)
def test_demo_sample(device, use_program_cache, batch_size, input_loc, imagenet_label_dict, model_location_generator):
    run_vgg_inference(batch_size, input_loc, imagenet_label_dict, device)


def run_vgg_imagenet_inference(
    batch_size, iterations, imagenet_label_dict, model_location_generator, device, model_config=vgg_model_config
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    profiler.clear()

    # Setup model
    torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_model.to(torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        device=device,
        custom_preprocessor=ttnn_vgg.custom_preprocessor,
    )

    # load inputs
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    # load ImageNet batch by batch
    # and run inference
    correct = 0
    for iter in range(iterations):
        predictions = []
        torch_predictions = []
        inputs, labels = get_batch(data_loader)
        permuted_inputs = torch.permute(inputs, (0, 2, 3, 1))
        tt_batched_input_tensor = ttnn.from_torch(permuted_inputs, ttnn.bfloat16)
        tt_output = ttnn_vgg.ttnn_vgg16(device, tt_batched_input_tensor, parameters, batch_size, model_config)
        tt_output = ttnn.to_torch(tt_output)
        prediction = tt_output[:, 0, 0, :].argmax(dim=-1)
        torch_prediction = torch_outputs[:, :].argmax(dim=-1)
        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            torch_predictions.append(imagenet_label_dict[torch_prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- \n Torch Predicted label:{predictions[-1]} \tPredicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1
        del tt_output, tt_batched_input_tensor, inputs, labels, predictions
    accuracy = correct / (batch_size * iterations)
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((1, 2),),
)
def test_demo_imagenet(batch_size, iterations, imagenet_label_dict, model_location_generator, device):
    run_vgg_imagenet_inference(batch_size, iterations, imagenet_label_dict, model_location_generator, device)

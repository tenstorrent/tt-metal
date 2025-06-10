# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger
from transformers import AutoImageProcessor

import ttnn
from models.demos.ttnn_resnet.tests.demo_utils import get_batch, get_data, get_data_loader
from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra
from models.utility_functions import profiler

resnet_model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}

ops_parallel_config = {}


def run_resnet_imagenet_inference(
    batch_size_per_device,
    iterations,
    imagenet_label_dict,
    model_location_generator,
    device,
    model_config=resnet_model_config,
    model_version="microsoft/resnet-50",
):
    profiler.clear()

    # set up image processor
    image_processor = AutoImageProcessor.from_pretrained(model_version)

    batch_size = batch_size_per_device * device.get_num_devices()
    iterations = iterations // device.get_num_devices()

    # load inputs
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)

    # Create TT Model Start
    # this will move weights to device
    profiler.start(f"compile")

    test_infra = create_test_infra(
        device,
        batch_size_per_device,
        model_config["ACTIVATIONS_DTYPE"],
        model_config["WEIGHTS_DTYPE"],
        model_config["MATH_FIDELITY"],
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )
    ttnn.synchronize_device(device)
    profiler.end(f"compile")

    # load ImageNet batch by batch
    # and run inference
    input_tensors_all = []
    input_labels_all = []
    for iter in range(iterations):
        inputs, labels = get_batch(data_loader, image_processor)
        input_tensors_all.append(inputs)
        input_labels_all.append(labels)
    logger.info("Processed ImageNet-1k validation Dataset")

    correct = 0
    profiler.start(f"run")
    is_first_run = True
    logger.info("Starting inference")
    for iter in range(iterations):
        predictions = []
        inputs = input_tensors_all[iter]
        labels = input_labels_all[iter]
        tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, inputs)
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        if is_first_run:
            profiler.start("compile")
        tt_output = test_infra.run()
        if is_first_run:
            profiler.end("compile")
            is_first_run = False
        tt_output = ttnn.from_device(tt_output, blocking=True)
        tt_output = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)).to(torch.float)
        prediction = tt_output[:, 0, 0, :].argmax(dim=-1)
        for i in range(batch_size):
            predictions.append(imagenet_label_dict[prediction[i].item()])
            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1
        del tt_output, inputs, labels, predictions
    profiler.end(f"run")
    accuracy = correct / (batch_size * iterations)
    logger.info(f"Accuracy for {batch_size}x{iterations} inputs: {accuracy}")

    compile_time = profiler.get(f"compile")
    # ensuring inference time fluctuations is not noise
    inference_time_avg = profiler.get("run") / (iterations)

    logger.info(
        f"ttnn_{model_version}_batch_size{batch_size} tests inference time (avg): {inference_time_avg}, FPS: {batch_size/inference_time_avg}"
    )
    logger.info(f"ttnn_{model_version}_batch_size{batch_size} compile time: {compile_time}")


def run_resnet_inference(
    batch_size_per_device,
    input_loc,
    imagenet_label_dict,
    device,
    model_location_generator,
    model_config=resnet_model_config,
    model_version="microsoft/resnet-50",
):
    # set up image processor
    image_processor = AutoImageProcessor.from_pretrained(model_version)

    # load inputs
    images = get_data(input_loc)
    batch_size = batch_size_per_device * device.get_num_devices()

    profiler.start(f"processing_inputs")
    inputs = None
    num_images = len(images)
    for i in range(batch_size):
        input_image = images[i % num_images].image
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

    test_infra = create_test_infra(
        device,
        batch_size_per_device,
        model_config["ACTIVATIONS_DTYPE"],
        model_config["WEIGHTS_DTYPE"],
        model_config["MATH_FIDELITY"],
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )
    ttnn.synchronize_device(device)

    profiler.end(f"move_weights")

    profiler.start(f"preprocessing")
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, inputs)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    ttnn.synchronize_device(device)
    profiler.end(f"preprocessing")

    profiler.disable()
    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run", force_enable=True)
    tt_out = test_infra.run()
    ttnn.synchronize_device(device)
    profiler.end("first_model_run", force_enable=True)
    tt_out.deallocate()
    del tt_out

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    ttnn.synchronize_device(device)

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("second_model_run_with_compile", force_enable=True)
    tt_out = test_infra.run()
    ttnn.synchronize_device(device)
    profiler.end("second_model_run_with_compile", force_enable=True)
    tt_out.deallocate()
    del tt_out

    profiler.enable()

    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    ttnn.synchronize_device(device)

    ##### Run Forward on TT Model Start
    profiler.start(f"model_run_for_inference")
    tt_out = test_infra.run()
    ttnn.synchronize_device(device)
    profiler.end(f"model_run_for_inference")

    profiler.start(f"post_processing")
    predictions = []
    tt_out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)).to(torch.float)

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
        "compile": profiler.get("second_model_run_with_compile")
        - (profiler.get("model_run_for_inference") / SINGLE_RUN),
        f"inference_for_single_run_batch_{batch_size}_without_cache": profiler.get("second_model_run_with_compile"),
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
    ((16, 100),),
)
def test_demo_imagenet(batch_size, iterations, imagenet_label_dict, model_location_generator, device):
    run_resnet_imagenet_inference(batch_size, iterations, imagenet_label_dict, model_location_generator, device)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_loc",
    ((16, "models/demos/ttnn_resnet/demo/images/"),),
)
def test_demo_sample(device, use_program_cache, batch_size, input_loc, imagenet_label_dict, model_location_generator):
    run_resnet_inference(batch_size, input_loc, imagenet_label_dict, device, model_location_generator)

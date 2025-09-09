# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from tqdm import tqdm
from models.experimental.vovnet.runner.performant_runner import VovnetPerformantRunner
from models.demos.utils.common_demo_utils import get_batch, get_data_loader, load_imagenet_dataset
from models.utility_functions import profiler, run_for_wormhole_b0
from models.experimental.vovnet.common import VOVNET_L1_SMALL_SIZE

NUM_VALIDATION_IMAGES_IMAGENET = 49920


def run_vovnet_imagenet_demo(
    device,
    device_batch_size,
    iterations,
    imagenet_label_dict,
    act_dtype,
    weight_dtype,
    model_location_generator,
    entire_imagenet_dataset=False,
    resolution=224,
):
    batch_size = device_batch_size * device.get_num_devices()
    if entire_imagenet_dataset:
        iterations = NUM_VALIDATION_IMAGES_IMAGENET // batch_size
    else:
        iterations = iterations // batch_size
    profiler.clear()
    with torch.no_grad():
        vovnet_trace_2cq = VovnetPerformantRunner(
            device,
            device_batch_size,
            act_dtype,
            weight_dtype,
            resolution=(resolution, resolution),
            model_location_generator=model_location_generator,
        )

        profiler.start(f"compile")
        vovnet_trace_2cq._capture_vovnet_trace_2cqs()
        profiler.end(f"compile")
        logger.info("ImageNet-1k validation Dataset")
        input_loc = load_imagenet_dataset(model_location_generator)
        data_loader = get_data_loader(input_loc, batch_size, iterations, entire_imagenet_dataset)

        input_tensors_all = []
        input_labels_all = []
        for iter in tqdm(range(iterations), desc="Preparing images"):
            inputs, labels = get_batch(data_loader, resolution)
            input_tensors_all.append(inputs)
            input_labels_all.append(labels)
        logger.info("Processed ImageNet-1k validation Dataset")

        logger.info("Starting inference")
        correct = 0
        total_inference_time = 0
        for iter in range(iterations):
            predictions = []
            torch_input_tensor = input_tensors_all[iter]
            labels = input_labels_all[iter]
            profiler.start(f"run")
            output = vovnet_trace_2cq.run(torch_input_tensor)
            output = ttnn.to_torch(output, mesh_composer=vovnet_trace_2cq.runner_infra.output_mesh_composer)
            prediction = output.argmax(dim=-1)

            profiler.end(f"run")
            total_inference_time += profiler.get(f"run")
            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1
        vovnet_trace_2cq.release()
        accuracy = correct / (batch_size * iterations)
        logger.info(f"=============")
        logger.info(f"Accuracy for  batch size: {batch_size} over {iterations} iterations is: {accuracy}")
        if entire_imagenet_dataset:
            assert (
                accuracy < expected_accuracy
            ), f"Accuracy {accuracy} does not match expected accuracy {expected_accuracy}"

        first_iter_time = profiler.get(f"compile")
        inference_time_avg = total_inference_time / (iterations)

        compile_time = first_iter_time - 2 * inference_time_avg


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VOVNET_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_batch_size, iterations, act_dtype, weight_dtype",
    ((1, 100, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_vovnet_imagenet_demo(
    device, device_batch_size, iterations, imagenet_label_dict, act_dtype, weight_dtype, model_location_generator
):
    return run_vovnet_imagenet_demo(
        device, device_batch_size, iterations, imagenet_label_dict, act_dtype, weight_dtype, model_location_generator
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VOVNET_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_batch_size, iterations, act_dtype, weight_dtype",
    ((1, 100, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_vovnet_imagenet_demo_dp(
    mesh_device, device_batch_size, iterations, imagenet_label_dict, act_dtype, weight_dtype, model_location_generator
):
    return run_vovnet_imagenet_demo(
        mesh_device,
        device_batch_size,
        iterations,
        imagenet_label_dict,
        act_dtype,
        weight_dtype,
        model_location_generator,
    )

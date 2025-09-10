# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from models.demos.utils.common_demo_utils import get_batch, get_data_loader, load_imagenet_dataset
from models.experimental.swin_v2.runner.performant_runner import SwinV2PerformantRunner
from tqdm import tqdm
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0
from models.experimental.swin_v2.common import SWIN_V2_L1_SMALL_SIZE
from loguru import logger


def run_swin_v2_trace_2cqs_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    imagenet_label_dict,
    iterations=100,
):
    disable_persistent_kernel_cache()
    batch_size = device.get_num_devices() * device_batch_size
    iterations = iterations // batch_size
    with torch.no_grad():
        swin_v2_trace_2cq = SwinV2PerformantRunner(
            device,
            device_batch_size,
            act_dtype,
            weight_dtype,
            resolution=resolution,
            model_location_generator=model_location_generator,
        )
        logger.info("ImageNet-1k validation Dataset")
        input_loc = load_imagenet_dataset(model_location_generator)
        data_loader = get_data_loader(input_loc, batch_size, iterations)

        input_tensors_all = []
        input_labels_all = []
        for iter in tqdm(range(iterations), desc="Preparing images"):
            inputs, labels = get_batch(data_loader, resolution[0])
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
            output = swin_v2_trace_2cq.run(torch_input_tensor)
            output = ttnn.to_torch(output, mesh_composer=swin_v2_trace_2cq.runner_infra.output_composer).to(torch.float)
            prediction = output.argmax(dim=-1)
            for i in range(batch_size):
                predictions.append(imagenet_label_dict[prediction[i].item()])
                logger.info(
                    f"Iter: {iter}  - Expected Label: {imagenet_label_dict[labels[i]]} -- Predicted Label: {predictions[-1]}"
                )
                if imagenet_label_dict[labels[i]] == predictions[-1]:
                    correct += 1
        swin_v2_trace_2cq.release()
        accuracy = correct / (batch_size * iterations)
        logger.info(f"=============")
        logger.info(f"Accuracy for  batch size: {batch_size} over {iterations} iterations is: {accuracy}")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SWIN_V2_L1_SMALL_SIZE, "trace_region_size": 16998400, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "iterations,batch_size, act_dtype, weight_dtype",
    ((100, 1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (512, 512),
    ],
)
def test_swin_v2_trace_2cqs_inference(
    device,
    batch_size,
    iterations,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    imagenet_label_dict,
):
    return run_swin_v2_trace_2cqs_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
        imagenet_label_dict,
        iterations,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SWIN_V2_L1_SMALL_SIZE, "trace_region_size": 16998400, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "iterations,device_batch_size, act_dtype, weight_dtype",
    ((100, 1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (512, 512),
    ],
)
def test_swin_v2_trace_2cqs_inference_dp(
    mesh_device,
    device_batch_size,
    iterations,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
    imagenet_label_dict,
):
    return run_swin_v2_trace_2cqs_inference(
        mesh_device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
        imagenet_label_dict,
        iterations,
    )

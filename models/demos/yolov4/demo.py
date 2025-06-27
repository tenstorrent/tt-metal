# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os
import random

import cv2
import fiftyone
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov4.post_processing import load_class_names, plot_boxes_cv2, post_processing
from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
from models.experimental.yolo_evaluation.yolo_evaluation_utils import LoadImages
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0

yolov4_model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}


def preprocess_image(im, resolution):
    sized = cv2.resize(im, resolution)
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0)
    return tensor


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, output_mesh_composer


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 40960, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
def test_yolov4(
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    disable_persistent_kernel_cache()

    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    logger.info(f"Running with batch_size={batch_size} across {mesh_device.get_num_devices()} devices")

    inputs_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    runner = YOLOv4PerformantRunner(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
    )

    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=10)
    sampled = random.sample(list(dataset), batch_size)
    data_set = LoadImages([sample["filepath"] for sample in sampled])

    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json", "r") as file:
        coco_info = json.load(file)
        class_names = coco_info["classes"]

    # Preprocess batch of images to [BS, C, H, W]
    torch_images = []
    orig_images = []
    paths_images = []
    for batch in data_set:
        paths, im0s, s = batch
        paths_images.append(paths[0])
        orig_images.append(im0s[0])
        tensor = preprocess_image(im0s[0], resolution)
        torch_images.append(tensor.unsqueeze(0))
        if len(torch_images) == batch_size:
            break

    torch_input_tensor = torch.cat(torch_images, dim=0)
    tt_output = runner.run(torch_input_tensor)

    conf_thresh = 0.3
    nms_thresh = 0.4
    boxes = post_processing(torch_images, conf_thresh, nms_thresh, tt_output)

    namesfile = "models/demos/yolov4/resources/coco.names"
    class_names = load_class_names(namesfile)

    img_cv = cv2.imread(paths_images[0])

    # Create a unique output files using image name and resolution
    output_dir = "yolov4_predictions"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(batch_size):
        img_base = os.path.splitext(os.path.basename(paths_images[i]))[0]
        output_filename = f"ttnn_yolov4_{img_base}_{resolution[0]}x{resolution[1]}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plot_boxes_cv2(orig_images[i], boxes[i], output_path, class_names)

    runner.release()

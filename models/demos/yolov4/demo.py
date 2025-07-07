# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import cv2
import fiftyone
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov4.common import get_mesh_mappers
from models.demos.yolov4.post_processing import load_class_names, plot_boxes_cv2, post_processing
from models.demos.yolov4.runner.performant_runner import YOLOv4PerformantRunner
from models.experimental.yolo_eval.utils import LoadImages
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


def preprocess_image(im, resolution):
    sized = cv2.resize(im, resolution)
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0)
    return tensor


def run_yolov4(
    device,
    input_loc,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    disable_persistent_kernel_cache()

    batch_size = batch_size_per_device * device.get_num_devices()
    logger.info(f"Running with batch_size={batch_size} across {device.get_num_devices()} devices")

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
    runner = YOLOv4PerformantRunner(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )

    image_files = [
        os.path.join(input_loc, f) for f in os.listdir(input_loc) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    assert len(image_files) >= batch_size, "Not enough input images for the batch"

    torch_images = []
    orig_images = []
    paths_images = []

    for i in range(batch_size):
        image_path = image_files[i]
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or unreadable: {image_path}")

        orig_images.append(image)
        paths_images.append(image_path)
        tensor = preprocess_image(image, resolution)
        torch_images.append(tensor.unsqueeze(0))

    torch_input_tensor = torch.cat(torch_images, dim=0)
    tt_output = runner.run(torch_input_tensor)

    conf_thresh = 0.3
    nms_thresh = 0.4
    boxes = post_processing(torch_images, conf_thresh, nms_thresh, tt_output)

    namesfile = "models/demos/yolov4/resources/coco.names"
    class_names = load_class_names(namesfile)
    output_dir = "yolov4_predictions"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(batch_size):
        img_base = os.path.splitext(os.path.basename(paths_images[i]))[0]
        output_filename = f"ttnn_yolov4_{img_base}_{resolution[0]}x{resolution[1]}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plot_boxes_cv2(orig_images[i], boxes[i], output_path, class_names)

    runner.release()


def run_yolov4_coco(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    disable_persistent_kernel_cache()

    batch_size = batch_size_per_device * device.get_num_devices()
    logger.info(f"Running with batch_size={batch_size} across {device.get_num_devices()} devices")

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
    runner = YOLOv4PerformantRunner(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=model_location_generator,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )

    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    data_set = LoadImages([sample["filepath"] for sample in dataset])

    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json", "r") as file:
        coco_info = json.load(file)
        class_names = coco_info["classes"]

    # Preprocess batch of images to [BS, C, H, W]
    torch_images = []
    orig_images = []
    paths_images = []
    for batch in data_set:
        paths, im0s, _ = batch
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
    output_dir = "yolov4_predictions"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(batch_size):
        img_base = os.path.splitext(os.path.basename(paths_images[i]))[0]
        output_filename = f"ttnn_yolov4_{img_base}_{resolution[0]}x{resolution[1]}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plot_boxes_cv2(orig_images[i], boxes[i], output_path, class_names)

    runner.release()


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 40960, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "input_loc",
    [
        "models/demos/yolov4/resources",
    ],
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
    device,
    input_loc,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov4(
        device,
        input_loc,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 40960, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "input_loc",
    [
        "models/demos/yolov4/resources",
    ],
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
def test_yolov4_dp(
    mesh_device,
    input_loc,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov4(
        mesh_device,
        input_loc,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )


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
def test_yolov4_coco(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov4_coco(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )


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
def test_yolov4_coco_dp(
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov4_coco(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )

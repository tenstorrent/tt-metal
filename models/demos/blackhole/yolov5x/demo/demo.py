# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import fiftyone
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import disable_persistent_kernel_cache, run_for_blackhole
from models.demos.utils.common_demo_utils import (
    LoadImages,
    load_coco_class_names,
    postprocess,
    preprocess,
    save_yolo_predictions_by_model,
)
from models.demos.yolov5x.common import YOLOV5X_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov5x.runner.performant_runner import YOLOv5xPerformantRunner
from models.demos.yolov5x.tt.common import get_mesh_mappers


def init_model_and_runner(model_location_generator, device, model_type, batch_size_per_device):
    disable_persistent_kernel_cache()
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")
    inputs_mapper, weights_mapper, outputs_composer = get_mesh_mappers(device)
    torch_model = load_torch_model(model_location_generator)
    runner = None
    if model_type == "tt_model":
        runner = YOLOv5xPerformantRunner(
            device,
            batch_size_per_device,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            resolution=(640, 640),
            model_location_generator=model_location_generator,
            mesh_mapper=inputs_mapper,
            weights_mesh_mapper=weights_mapper,
            mesh_composer=outputs_composer,
        )
        runner._capture_yolov5x_trace_2cqs()

    return torch_model, runner, outputs_composer, batch_size


def process_images(dataset, res, batch_size):
    torch_images, orig_images, paths_images = [], [], []

    for paths, im0s, _ in dataset:
        assert len(im0s) == batch_size, f"Expected batch of size {batch_size}, but got {len(im0s)}"

        paths_images.extend(paths)
        orig_images.extend(im0s)

        for idx, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[idx]}")
            tensor = preprocess([img], res=res)
            torch_images.append(tensor)

        if len(torch_images) >= batch_size:
            break

    torch_input_tensor = torch.cat(torch_images, dim=0)
    return torch_input_tensor, orig_images, paths_images


def run_inference_and_save(
    model, runner, model_type, outputs_mesh_composer, im_tensor, orig_images, paths_images, save_dir, names
):
    if model_type == "torch_model":
        preds = model(im_tensor)
    else:
        preds = runner.run(im_tensor)
        preds = ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=outputs_mesh_composer)

    results = postprocess(preds, im_tensor, orig_images, paths_images, names)

    for result, image_path in zip(results, paths_images):
        save_yolo_predictions_by_model(result, save_dir, image_path, model_type)


def run_yolov5x_demo(model_location_generator, device, model_type, input_loc, batch_size_per_device):
    model, runner, mesh_composer, batch_size = init_model_and_runner(
        model_location_generator, device, model_type, batch_size_per_device
    )

    dataset = LoadImages(path=os.path.abspath(input_loc), batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(dataset, (640, 640), batch_size)
    names = load_coco_class_names()
    save_dir = "models/demos/blackhole/yolov5x/demo/runs"

    run_inference_and_save(
        model, runner, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir, names
    )

    if runner:
        runner.release()
    logger.info("Inference done")


def run_yolov5x_demo_dataset(model_location_generator, device, model_type, batch_size_per_device):
    model, runner, mesh_composer, batch_size = init_model_and_runner(
        model_location_generator, device, model_type, batch_size_per_device
    )

    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    filepaths = [sample["filepath"] for sample in dataset]
    image_loader = LoadImages(filepaths, batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(image_loader, (640, 640), batch_size)

    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json") as f:
        names = json.load(f)["classes"]

    save_dir = "models/demos/blackhole/yolov5x/demo/runs"
    run_inference_and_save(
        model, runner, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir, names
    )

    if runner:
        runner.release()
    logger.info("Inference done")


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/blackhole/yolov5x/demo/images",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model", # Uncomment to run the demo with torch model.
    ],
)
def test_demo(model_location_generator, device, batch_size_per_device, input_loc, model_type):
    run_yolov5x_demo(
        model_location_generator,
        device,
        model_type,
        input_loc,
        batch_size_per_device,
    )


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/blackhole/yolov5x/demo/images",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model", # Uncomment to run the demo with torch model.
    ],
)
def test_demo_dp(model_location_generator, mesh_device, batch_size_per_device, input_loc, model_type):
    run_yolov5x_demo(
        model_location_generator,
        mesh_device,
        model_type,
        input_loc,
        batch_size_per_device,
    )


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model", # Uncomment to run the demo with torch model.
    ],
)
def test_demo_dataset(model_location_generator, device, model_type):
    run_yolov5x_demo_dataset(
        model_location_generator,
        device,
        model_type,
        batch_size_per_device=1,
    )


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model", # Uncomment to run the demo with torch model.
    ],
)
def test_demo_dataset_dp(model_location_generator, mesh_device, model_type):
    run_yolov5x_demo_dataset(
        model_location_generator,
        mesh_device,
        model_type,
        batch_size_per_device=1,
    )

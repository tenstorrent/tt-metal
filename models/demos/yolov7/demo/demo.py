# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys

import fiftyone
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0
from models.demos.utils.common_demo_utils import LoadImages, get_mesh_mappers, load_coco_class_names, preprocess
from models.demos.yolov7.common import YOLOV7_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov7.demo.demo_utils import postprocess
from models.demos.yolov7.reference import yolov7_model, yolov7_utils
from models.demos.yolov7.runner.performant_runner import YOLOv7PerformantRunner

sys.modules["models.common"] = yolov7_utils
sys.modules["models.yolo"] = yolov7_model


def init_model_and_runner(model_location_generator, device, model_type, batch_size_per_device):
    disable_persistent_kernel_cache()
    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")
    inputs_mapper, weights_mapper, outputs_composer = get_mesh_mappers(device)
    torch_model = load_torch_model(model_location_generator)
    runner = None
    if model_type == "tt_model":
        runner = YOLOv7PerformantRunner(
            device,
            batch_size,
            ttnn.bfloat16,
            ttnn.bfloat16,
            resolution=(640, 640),
            model_location_generator=model_location_generator,
            inputs_mesh_mapper=inputs_mapper,
            weights_mesh_mapper=weights_mapper,
            outputs_mesh_composer=outputs_composer,
        )

    return torch_model, runner, outputs_composer, batch_size


def process_images(dataset, res, batch_size):
    torch_images, orig_images, paths_images = [], [], []

    for batch in dataset:
        paths, im0s, _ = batch
        assert len(im0s) == batch_size, f"Expected batch size {batch_size}, got {len(im0s)}"
        orig_images.extend(im0s)
        paths_images.extend(paths)
        for img in im0s:
            tensor = preprocess([img], res=res)
            torch_images.append(tensor)

        if len(torch_images) >= batch_size:
            break

    torch_input_tensor = torch.cat(torch_images, dim=0)
    return torch_input_tensor, orig_images, paths_images, batch


def run_inference_and_save(
    model, runner, model_type, outputs_composer, im_tensor, orig_images, paths_images, save_dir, names, batch, dataset
):
    if model_type == "torch_model":
        preds = model(im_tensor)[0]
    else:
        preds = runner.run(im_tensor)
        preds = ttnn.to_torch(preds, mesh_composer=outputs_composer)

    results = postprocess(
        preds,
        im_tensor,
        orig_images,
        batch,
        names,
        paths_images,
        dataset,
        save_dir,
    )

    logger.info(f"Saved {len(results)} outputs to {save_dir}")


def run_yolov7_demo(model_location_generator, device, model_type, input_loc, batch_size_per_device):
    model, runner, mesh_composer, batch_size = init_model_and_runner(
        model_location_generator, device, model_type, batch_size_per_device
    )

    dataset = LoadImages(path=input_loc, batch=batch_size, img_size=640, vid_stride=32)
    im_tensor, orig_images, paths_images, batch = process_images(dataset, (640, 640), batch_size)
    names = load_coco_class_names()
    save_dir = "models/demos/yolov7/demo/runs"

    run_inference_and_save(
        model, runner, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir, names, batch, dataset
    )

    if runner:
        runner.release()

    logger.info("YOLOv7 demo completed")


def run_yolov7_demo_dataset(model_location_generator, device, model_type, batch_size_per_device):
    model, runner, mesh_composer, batch_size = init_model_and_runner(
        model_location_generator, device, model_type, batch_size_per_device
    )
    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    filepaths = [sample["filepath"] for sample in dataset]
    image_loader = LoadImages(filepaths, batch=batch_size)
    im_tensor, orig_images, paths_images, batch = process_images(image_loader, (640, 640), batch_size)

    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json") as f:
        names = json.load(f)["classes"]

    save_dir = "models/demos/yolov7/demo/runs"
    run_inference_and_save(
        model,
        runner,
        model_type,
        mesh_composer,
        im_tensor,
        orig_images,
        paths_images,
        save_dir,
        names,
        batch,
        image_loader,
    )

    if runner:
        runner.release()
    logger.info("Inference done")


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV7_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov7/demo/images",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
def test_demo(model_location_generator, device, batch_size_per_device, input_loc, model_type):
    run_yolov7_demo(
        model_location_generator,
        device,
        model_type,
        input_loc,
        batch_size_per_device,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV7_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov7/demo/images",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
def test_demo_dp(model_location_generator, mesh_device, batch_size_per_device, input_loc, model_type):
    run_yolov7_demo(
        model_location_generator,
        mesh_device,
        model_type,
        input_loc,
        batch_size_per_device,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV7_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
def test_demo_dataset(model_location_generator, device, model_type):
    run_yolov7_demo_dataset(
        model_location_generator,
        device,
        model_type,
        batch_size_per_device=1,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV7_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        # "torch_model", # Uncomment to run the demo with torch model
        "tt_model",
    ),
)
def test_demo_dataset_dp(model_location_generator, mesh_device, model_type):
    run_yolov7_demo_dataset(
        model_location_generator,
        mesh_device,
        model_type,
        batch_size_per_device=1,
    )

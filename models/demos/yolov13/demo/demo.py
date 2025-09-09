# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

from models.demos.utils.common_demo_utils import LoadImages, get_mesh_mappers, load_coco_class_names, preprocess
from models.demos.yolov13.common import YOLOV13_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov13.reference import yolov13
from models.utility_functions import disable_persistent_kernel_cache


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


def init_model_and_runner(
    device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_location_generator
):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices

    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")

    inputs_mesh_mapper, weights_mesh_mapper, outputs_mesh_composer = get_mesh_mappers(device)

    if use_weights_from_ultralytics:
        torch_model = load_torch_model(model_location_generator)
        model = torch_model.eval()
    else:
        model = yolov13.YoloV13()

    performant_runner = None
    # if model_type == "tt_model":
    # performant_runner = YOLOv11PerformantRunner(
    #    device,
    #    batch_size_per_device,
    #    model_location_generator=model_location_generator,
    #    inputs_mesh_mapper=inputs_mesh_mapper,
    #    weights_mesh_mapper=weights_mesh_mapper,
    #    outputs_mesh_composer=outputs_mesh_composer,
    # )

    return model, performant_runner, outputs_mesh_composer, batch_size


def run_yolov13_demo(
    device, model_type, use_weights_from_ultralytics, res, input_loc, batch_size_per_device, model_location_generator
):
    model, runner, mesh_composer, batch_size = init_model_and_runner(
        device, model_type, use_weights_from_ultralytics, batch_size_per_device, model_location_generator
    )

    dataset = LoadImages(path=os.path.abspath(input_loc), batch=batch_size)
    im_tensor, orig_images, paths_images = process_images(dataset, res, batch_size)
    names = load_coco_class_names()
    save_dir = "models/demos/yolov13/demo/runs"

    run_inference_and_save(
        model, runner, model_type, mesh_composer, im_tensor, orig_images, paths_images, save_dir, names
    )

    if runner:
        runner.release()
    logger.info("Inference done")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV13_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_type",
    (
        "torch_model",  # Uncomment to run the demo with torch model
        # "tt_model",
    ),
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
@pytest.mark.parametrize(
    "input_loc, batch_size_per_device",
    [
        (
            "models/demos/yolov13/demo/images",
            1,
        ),
    ],
)
def test_demo(
    device, model_type, use_weights_from_ultralytics, res, input_loc, batch_size_per_device, model_location_generator
):
    run_yolov13_demo(
        device,
        model_type,
        use_weights_from_ultralytics,
        res,
        input_loc,
        batch_size_per_device,
        model_location_generator,
    )

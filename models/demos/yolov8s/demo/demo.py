# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import disable_persistent_kernel_cache
from models.demos.utils.common_demo_utils import (
    LoadImages,
    get_mesh_mappers,
    load_coco_class_names,
    postprocess,
    preprocess,
    save_yolo_predictions_by_model,
)
from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner


def run_yolov8s(
    device,
    batch_size_per_device,
    input_loc,
    model_type,
    use_pretrained_weights,
    res,
    model_location_generator=None,
):
    disable_persistent_kernel_cache()

    num_devices = device.get_num_devices()
    batch_size = batch_size_per_device * num_devices
    logger.info(f"Running with batch_size={batch_size} across {num_devices} devices")
    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    if use_pretrained_weights:
        model = load_torch_model(model_location_generator)

    if model_type == "tt_model":
        performant_runner = YOLOv8sPerformantRunner(
            device,
            device_batch_size=batch_size,
            mesh_mapper=inputs_mesh_mapper,
            mesh_composer=output_mesh_composer,
            model_location_generator=model_location_generator,
        )

    save_dir = "models/demos/yolov8s/demo/runs"

    input_loc = os.path.abspath(input_loc)
    dataset = LoadImages(path=input_loc, batch=batch_size)

    names = load_coco_class_names()
    for batch in dataset:
        paths, im0s, _ = batch
        assert len(im0s) == batch_size, f"Expected batch of size {batch_size}, but got {len(im0s)}"

        preprocessed_im = []
        for i, img in enumerate(im0s):
            if img is None:
                raise ValueError(f"Could not read image: {paths[i]}")
            processed = preprocess([img], res=res)
            preprocessed_im.append(processed)

        im = torch.cat(preprocessed_im, dim=0)

        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = performant_runner.run(im)
            preds = ttnn.to_torch(preds[0], dtype=torch.float32, mesh_composer=output_mesh_composer)
        results = postprocess(preds, im, im0s, paths, names)

        for result, image_path in zip(results, paths):
            save_yolo_predictions_by_model(result, save_dir, image_path, model_type)

    if model_type == "tt_model":
        performant_runner.release()

    logger.info("Inference done")


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov8s/demo/images",
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
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo(
    device, batch_size_per_device, input_loc, model_type, use_weights_from_ultralytics, res, model_location_generator
):
    run_yolov8s(
        device,
        batch_size_per_device,
        input_loc,
        model_type,
        use_weights_from_ultralytics,
        res,
        model_location_generator,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, input_loc",
    [
        (
            1,
            "models/demos/yolov8s/demo/images",
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
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_demo_dp(
    mesh_device,
    batch_size_per_device,
    input_loc,
    model_type,
    use_weights_from_ultralytics,
    res,
    model_location_generator,
):
    run_yolov8s(
        mesh_device,
        batch_size_per_device,
        input_loc,
        model_type,
        use_weights_from_ultralytics,
        res,
        model_location_generator,
    )

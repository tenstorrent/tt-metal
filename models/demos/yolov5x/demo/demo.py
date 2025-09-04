# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.utils.common_demo_utils import (
    LoadImages,
    load_coco_class_names,
    postprocess,
    preprocess,
    save_yolo_predictions_by_model,
)
from models.demos.yolov5x.common import YOLOV5X_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov5x.reference.yolov5x import YOLOv5
from models.demos.yolov5x.runner.performant_runner import YOLOv5xPerformantRunner
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "source",
    [
        "models/demos/yolov5x/demo/images/bus.jpg",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model", # Uncomment to run the demo with torch model.
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        # "False", # Uncomment to run the demo with random weights.
        "True",
    ],
)
def test_demo(device, source, model_type, use_weights_from_ultralytics, reset_seeds, model_location_generator):
    disable_persistent_kernel_cache()

    if model_type == "torch_model":
        model = YOLOv5()
        if use_weights_from_ultralytics:
            model = load_torch_model(model_location_generator)
        logger.info("Inferencing [Torch] Model")
    else:
        model = YOLOv5xPerformantRunner(
            device,
            device_batch_size=1,
            act_dtype=ttnn.bfloat8_b,
            weight_dtype=ttnn.bfloat8_b,
            model_location_generator=model_location_generator,
        )
        model._capture_yolov5x_trace_2cqs()
        logger.info("Inferencing [TTNN] Model")

    save_dir = "models/demos/yolov5x/demo/runs"
    dataset = LoadImages(path=source)
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, res=640)
        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = model.run(im)
            preds = ttnn.to_torch(preds, dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch, names)[0]
        save_yolo_predictions_by_model(results, save_dir, source, model_type)
    if model_type == "tt_model":
        model.release()

    logger.info("Inference done")

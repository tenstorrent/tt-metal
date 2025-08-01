# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from loguru import logger

import ttnn
from models.demos.yolov7.common import load_torch_model
from models.demos.yolov7.demo.demo_utils import LoadImages, load_coco_class_names, postprocess, preprocess
from models.demos.yolov7.reference import yolov7_model, yolov7_utils
from models.demos.yolov7.runner.performant_runner import YOLOv7PerformantRunner
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0

sys.modules["models.common"] = yolov7_utils
sys.modules["models.yolo"] = yolov7_model


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 16384, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "source",
    [
        "models/demos/yolov7/demo/horses.jpg",
    ],
)
@pytest.mark.parametrize("model_type", ["torch_model", "tt_model"])
def test_demo(device, reset_seeds, model_type, source, model_location_generator):
    disable_persistent_kernel_cache()

    names = load_coco_class_names()
    dataset = LoadImages(path=source, img_size=640, stride=32)
    save_dir = "models/demos/yolov7/demo/runs/detect"

    if model_type == "torch_model":
        torch_model = load_torch_model(model_location_generator)

        logger.info("Inferencing [Torch] Model")
        for batch in dataset:
            path, im0s, s = batch
            im = preprocess(im0s)
            preds = torch_model(im)[0]
            postprocess(preds, im, im0s, batch, names, path, s, dataset, save_dir=save_dir)

    else:
        for batch in dataset:
            path, im0s, s = batch
            im = preprocess(im0s)
            performant_runner = YOLOv7PerformantRunner(
                device,
                1,
                ttnn.bfloat16,
                ttnn.bfloat16,
                resolution=(640, 640),
                model_location_generator=None,
                torch_input_tensor=im,
            )
            performant_runner._capture_yolov7_trace_2cqs()
            logger.info("Inferencing [TTNN] Model")
            preds = performant_runner.run(im)
            preds = ttnn.to_torch(preds)
            postprocess(preds, im, im0s, batch, names, path, s, dataset, save_dir=save_dir)

    logger.info("Yolov7 Demo completed")

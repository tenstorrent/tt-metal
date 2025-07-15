# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov9c.demo.demo_utils import load_coco_class_names
from models.experimental.yolov12x.reference import yolov12x
from models.experimental.yolo_eval.evaluate import save_yolo_predictions_by_model
from models.experimental.yolo_eval.utils import LoadImages, postprocess, preprocess
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0
from models.experimental.yolov12x.runner.performant_runner import YOLOv12xPerformantRunner


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "source",
    [
        "models/experimental/yolov12x/demo/input_images/bus.jpg",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model",  # Uncomment to run the demo with torch model.
    ],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        # "False", # Uncomment to run the demo with random weights.
        "True",
    ],
)
def test_demo(device, source, model_type, use_weights_from_ultralytics, reset_seeds, batch_size=1):
    disable_persistent_kernel_cache()

    if model_type == "torch_model":
        state_dict = None
        if use_weights_from_ultralytics:
            torch_model = YOLO("yolo12x.pt")
            state_dict = torch_model.state_dict()

        model = yolov12x.YoloV12x()
        state_dict = model.state_dict() if state_dict is None else state_dict

        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("Inferencing [Torch] Model")
    else:
        logger.info("Inferencing [TTNN] Model")
        model = YOLOv12xPerformantRunner(
            device,
            batch_size,
            act_dtype=ttnn.bfloat8_b,
            weight_dtype=ttnn.bfloat8_b,
            model_location_generator=None,
        )
        model._capture_yolov12x_trace_2cqs()

    save_dir = "models/experimental/yolov12x/demo"
    dataset = LoadImages(path=source)
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, res=(640, 640))
        if model_type == "torch_model":
            preds = model(im)
        else:
            preds = model.run(torch_input_tensor=im)
            print("run_finished")
            preds = ttnn.to_torch(preds, dtype=torch.float32)
        results = postprocess(preds, im, im0s, batch, names)[0]
        save_yolo_predictions_by_model(results, save_dir, source, model_type)

    logger.info("Inference done")

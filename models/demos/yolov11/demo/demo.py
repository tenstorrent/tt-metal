# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov11.demo.demo_utils import load_coco_class_names
from models.demos.yolov11.reference import yolov11
from models.demos.yolov11.runner.performant_runner import YOLOv11PerformantRunner
from models.demos.yolov11.tt.model_preprocessing import create_yolov11_input_tensors, create_yolov11_model_parameters
from models.experimental.yolo_eval.evaluate import save_yolo_predictions_by_model
from models.experimental.yolo_eval.utils import LoadImages, postprocess, preprocess
from models.utility_functions import disable_persistent_kernel_cache


@pytest.mark.parametrize(
    "source, model_type,resolution",
    [
        ("models/demos/yolov11/demo/images/cycle_girl.jpg", "torch_model", [3, 640, 640]),
        ("models/demos/yolov11/demo/images/cycle_girl.jpg", "tt_model", [3, 640, 640]),
        # ("models/demos/yolov11/demo/images/dog.jpg","torch_model",[3, 640, 640]),  # Uncomment this to run for different image
        # ("models/demos/yolov11/demo/images/dog.jpg", "tt_model", [3, 640, 640]),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
        #  False
    ],
    ids=[
        "pretrained_weight_true",
        # "pretrained_weight_false",
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
def test_demo(device, source, model_type, resolution, use_pretrained_weight):
    disable_persistent_kernel_cache()
    weights = "yolo11n.pt"
    if use_pretrained_weight:
        torch_model = YOLO(weights)
        state_dict = {k.replace("model.", "", 1): v for k, v in torch_model.state_dict().items()}

    model = yolov11.YoloV11()
    model.eval()
    if use_pretrained_weight:
        model.load_state_dict(state_dict)
    if model_type == "torch_model":
        logger.info("Inferencing using Torch Model")
    else:
        torch_input, ttnn_input = create_yolov11_input_tensors(
            device,
            input_channels=resolution[0],
            input_height=resolution[1],
            input_width=resolution[2],
            is_sub_module=False,
        )
        parameters = create_yolov11_model_parameters(model, torch_input, device=device)
        logger.info("Inferencing using ttnn Model")

    save_dir = "models/demos/yolov11/demo/runs"
    dataset = LoadImages(path=source)
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    ttnn_module = None
    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s, (resolution[1], resolution[2]))
        if model_type == "torch_model":
            t0 = time.time()
            preds = model(im)
            t1 = time.time()
        else:
            if ttnn_module is None:
                performant_runner = YOLOv11PerformantRunner(
                    device, act_dtype=ttnn.bfloat8_b, weight_dtype=ttnn.bfloat8_b, torch_input_tensor=im
                )
                performant_runner._capture_yolov11_trace_2cqs()
            t0 = time.time()
            preds = performant_runner.run(im)
            t1 = time.time()
            preds = ttnn.to_torch(preds, dtype=torch.float32)
        results = postprocess(preds, im, im0s, batch, load_coco_class_names())[0]
        save_yolo_predictions_by_model(results, save_dir, source, model_type)

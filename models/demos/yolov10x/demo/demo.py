# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from loguru import logger
from ultralytics import YOLO

import ttnn
from models.demos.yolov10x.demo.demo_utils import (
    LoadImages,
    load_coco_class_names,
    postprocess,
    preprocess,
    save_yolo_predictions_by_model,
)
from models.demos.yolov10x.reference.yolov10x import YOLOv10
from models.demos.yolov10x.runner.performant_runner import YOLOv10PerformantRunner
from models.utility_functions import disable_persistent_kernel_cache, run_for_wormhole_b0


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 10 * 1024, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "source",
    [
        "models/demos/yolov10x/demo/images/dog.jpg",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        "tt_model",
        # "torch_model",
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        "False",
        "True",
    ],
)
def test_demo_ttnn(device, source, model_type, use_pretrained_weight):
    disable_persistent_kernel_cache()

    if model_type == "torch_model":
        state_dict = None
        if use_pretrained_weight:
            torch_model = YOLO("yolov10x.pt")
            state_dict = torch_model.state_dict()

            torch_model = YOLOv10()
            state_dict = torch_model.state_dict() if state_dict is None else state_dict
            ds_state_dict = {k: v for k, v in state_dict.items()}
            new_state_dict = {}
            for (name1, parameter1), (name2, parameter2) in zip(
                torch_model.state_dict().items(), ds_state_dict.items()
            ):
                if isinstance(parameter2, torch.FloatTensor):
                    new_state_dict[name1] = parameter2
            torch_model.load_state_dict(new_state_dict)
            torch_model.eval()
            model = torch_model
            logger.info("Inferencing [Torch] Model")
        save_dir = "models/demos/yolov10x/demo/runs"
        dataset = LoadImages(path=source)
        model_save_dir = os.path.join(save_dir, model_type)
        os.makedirs(model_save_dir, exist_ok=True)
        names = load_coco_class_names()

        for batch in dataset:
            paths, im0s, s = batch
            im = preprocess(im0s)
            preds = model(im)
            results = postprocess(preds, im, im0s, batch, names)[0]
            save_yolo_predictions_by_model(results, save_dir, source, model_type)
    else:
        save_dir = "models/demos/yolov10x/demo/runs"
        dataset = LoadImages(path=source)
        model_save_dir = os.path.join(save_dir, model_type)
        os.makedirs(model_save_dir, exist_ok=True)
        names = load_coco_class_names()

        for batch in dataset:
            paths, im0s, s = batch
            im = preprocess(im0s)
            img = im
            model = YOLOv10PerformantRunner(
                device,
                act_dtype=ttnn.bfloat8_b,
                weight_dtype=ttnn.bfloat8_b,
                torch_input_tensor=im,
                use_pretrained_weight=use_pretrained_weight,
            )
            preds = model.run(img)
            preds = ttnn.to_torch(preds, dtype=torch.float32)

            results = postprocess(preds, im, im0s, batch, names)[0]
            save_yolo_predictions_by_model(results, save_dir, source, model_type)

    logger.info("Inference done")

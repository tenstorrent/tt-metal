# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import ttnn
import torch
import pytest
import torch.nn as nn
from loguru import logger
from models.utility_functions import run_for_wormhole_b0
from models.experimental.functional_yolov9c.tt import ttnn_yolov9c
from models.experimental.functional_yolov9c.reference import yolov9c
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_yolov9c.tt.model_preprocessing import (
    create_yolov9c_input_tensors,
    create_yolov9c_model_parameters,
    create_yolov9c_model_parameters_detect,
)
from models.experimental.functional_yolov9c.demo.demo_utils import (
    LoadImages,
    preprocess,
    postprocess,
    attempt_load,
    load_coco_class_names,
    save_yolo_predictions_by_model,
)

try:
    sys.modules["ultralytics"] = yolov9c
    sys.modules["ultralytics.nn.tasks"] = yolov9c
    sys.modules["ultralytics.nn.modules.conv"] = yolov9c
    sys.modules["ultralytics.nn.modules.block"] = yolov9c
    sys.modules["ultralytics.nn.modules.head"] = yolov9c

except KeyError:
    logger.error("models.experimental.functional_yolov9c.reference.yolov9c not found.")


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source",
    [
        "models/sample_data/huggingface_cat_image.jpg",
        # "models/experimental/functional_yolov9c/demo/images/dog.jpg",
        # "models/experimental/functional_yolov9c/demo/images/cycle_girl.jpg",
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
        # "True",
    ],
)
def test_demo(device, source, model_type, use_pretrained_weight):
    disable_persistent_kernel_cache()

    if model_type == "torch_model":
        state_dict = None
        if use_pretrained_weight:
            torch_model = attempt_load("yolov9c.pt", map_location="cpu")
            state_dict = torch_model.state_dict()

        model = yolov9c.YoloV9()
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
        torch_input, ttnn_input = create_yolov9c_input_tensors(device)
        state_dict = None
        if use_pretrained_weight:
            torch_model = attempt_load("yolov9c.pt", map_location="cpu")
            state_dict = torch_model.state_dict()

        torch_model = yolov9c.YoloV9()
        state_dict = torch_model.state_dict() if state_dict is None else state_dict
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {}
        for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
            if isinstance(parameter2, torch.FloatTensor):
                new_state_dict[name1] = parameter2

        torch_model.load_state_dict(new_state_dict)
        torch_model.eval()
        parameters = create_yolov9c_model_parameters(torch_model, torch_input, device=device)
        model = ttnn_yolov9c.YoloV9(device, parameters)
        logger.info("Inferencing [TTNN] Model")

    save_dir = "models/experimental/functional_yolov9c/demo/runs"
    dataset = LoadImages(path=source)
    model_save_dir = os.path.join(save_dir, model_type)
    os.makedirs(model_save_dir, exist_ok=True)
    names = load_coco_class_names()

    for batch in dataset:
        paths, im0s, s = batch
        im = preprocess(im0s)
        if model_type == "torch_model":
            preds = model(im)
        else:
            img = torch.permute(im, (0, 2, 3, 1))
            img = img.reshape(
                1,
                1,
                img.shape[0] * img.shape[1] * img.shape[2],
                img.shape[3],
            )
            ttnn_im = ttnn.from_torch(img, dtype=ttnn.bfloat16)
            preds = model(ttnn_im)
            preds = ttnn.to_torch(preds, dtype=torch.float32)

        results = postprocess(preds, im, im0s, batch, names)[0]
        save_yolo_predictions_by_model(results, save_dir, source, model_type)

    logger.info("Inference done")

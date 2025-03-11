# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
import torch
import pytest
from loguru import logger
from ultralytics import YOLO
from models.utility_functions import run_for_wormhole_b0
from models.experimental.functional_yolov9c.tt import ttnn_yolov9c
from models.experimental.functional_yolov9c.reference import yolov9c
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_yolov9c.tt.model_preprocessing import (
    create_yolov9c_input_tensors,
    create_yolov9c_model_parameters,
)
from models.experimental.functional_yolov9c.demo.demo_utils import load_coco_class_names
from models.experimental.yolo_evaluation.yolo_common_evaluation import save_yolo_predictions_by_model
from models.experimental.yolo_evaluation.yolo_evaluation_utils import LoadImages, preprocess, postprocess


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "source",
    [
        "models/sample_data/huggingface_cat_image.jpg",
        # "models/demos/yolov4/demo/giraffe_320.jpg", # Uncomment to run the demo with another image.
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
    "use_weights_from_ultralytics",
    [
        # "False",
        "True",
    ],
)
def test_demo(device, source, model_type, use_weights_from_ultralytics):
    disable_persistent_kernel_cache()
    model = yolov9c.YoloV9()

    if model_type == "torch_model":
        if use_weights_from_ultralytics:
            pretrained_model = YOLO("yolov9c.pt")
            model.load_state_dict(pretrained_model.state_dict(), strict=False)

        new_state_dict = {
            name: param for name, param in model.state_dict().items() if isinstance(param, torch.FloatTensor)
        }
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info("Inferencing [Torch] Model")
    else:
        torch_input, ttnn_input = create_yolov9c_input_tensors(device, model=True)
        state_dict = None
        if use_weights_from_ultralytics:
            pretrained_model = YOLO("yolov9c.pt")
            model.load_state_dict(pretrained_model.state_dict(), strict=False)

        new_state_dict = {
            name: param for name, param in model.state_dict().items() if isinstance(param, torch.FloatTensor)
        }
        model.load_state_dict(new_state_dict)
        model.eval()
        parameters = create_yolov9c_model_parameters(model, torch_input, device=device)
        model = ttnn_yolov9c.YoloV9(device, parameters)
        logger.info("Inferencing [TTNN] Model")

    save_dir = "models/experimental/functional_yolov9c/demo/runs"
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

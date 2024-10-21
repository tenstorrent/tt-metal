# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import numpy as np
from loguru import logger
import sys
from models.experimental.functional_yolov7.reference.model import Yolov7_model
from models.experimental.functional_yolov7.ttnn.tt_yolov7 import ttnn_yolov7
import models.experimental.functional_yolov7.reference.yolov7_utils
import models.experimental.functional_yolov7.reference.yolov7_model
from ttnn.model_preprocessing import preprocess_model_parameters, fold_batch_norm2d_into_conv2d, infer_ttnn_module_args
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_yolov7.reference.yolov7_detect import detect, parse_opt
from models.experimental.functional_yolov7.reference.yolov7_model import Model
from models.experimental.functional_yolov7.reference.yolov7_utils import *

from tests.ttnn.integration_tests.yolov7.test_ttnn_yolov7 import create_custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_demo(device, reset_seeds):
    sys.modules["models.yolo"] = sys.modules["models.experimental.functional_yolov7.reference.yolov7_model"]
    sys.modules["models.common"] = sys.modules["models.experimental.functional_yolov7.reference.yolov7_utils"]

    def load_weights(model, weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model"].float().state_dict()
        model.load_state_dict(state_dict, strict=False)

    args = [
        "--weights",
        "yolov7.pt",
        "--source",
        "models/experimental/functional_yolov7/demo/horses.jpg",
        "--conf-thres",
        "0.50",
        "--img-size",
        "640",
    ]

    opt = parse_opt(args)
    opt.device = device

    torch_model = Yolov7_model()

    new_state_dict = {}
    keys = [name for name, parameter in torch_model.state_dict().items()]
    ds_state_dict = {k: v for k, v in torch_model.state_dict().items()}
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    weights_path = "tests/ttnn/integration_tests/yolov7/yolov7.pt"
    load_weights(torch_model, weights_path)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )

    ttnn_model = ttnn_yolov7(device, parameters)

    ttnn_output = detect(opt, ttnn_model)

    logger.info("Yolov7 Demo completed")

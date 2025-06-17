# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import ttnn
import torch
import pytest
from pathlib import Path
import torch.nn as nn
from loguru import logger
from ultralytics import YOLO
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.yolov8s.tt.tt_yolov8s_utils import (
    ttnn_decode_bboxes,
    custom_preprocessor,
)
from models.experimental.yolov8s.tt.ttnn_yolov8s import TtYolov8sModel, TtConv, TtC2f, TtSppf, TtDFL


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True, ids=["0"])
@pytest.mark.parametrize(
    "input_tensor",
    [torch.rand((1, 3, 640, 640))],
    ids=["input_tensor1"],
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [True],
)
def test_yolov8s_640(device, input_tensor, use_weights_from_ultralytics):
    disable_persistent_kernel_cache()

    inp_h, inp_w = input_tensor.shape[2], input_tensor.shape[3]
    if use_weights_from_ultralytics:
        torch_model = YOLO("yolov8s.pt")
        torch_model = torch_model.model
        torch_model.eval()
        state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict, inp_h=inp_h, inp_w=inp_w)
    ttnn_model = TtYolov8sModel(device=device, parameters=parameters, res=(inp_h, inp_w))

    n, c, h, w = input_tensor.shape
    if c == 3:
        c = 8
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn_input.to(device, input_mem_config)

    with torch.inference_mode():
        ttnn_model_output = ttnn_model(ttnn_input)[0]
        ttnn_model_output = ttnn.to_torch(ttnn_model_output)

    with torch.inference_mode():
        torch_model_output = torch_model(input_tensor)[0]

    passing, pcc = assert_with_pcc(ttnn_model_output, torch_model_output, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")

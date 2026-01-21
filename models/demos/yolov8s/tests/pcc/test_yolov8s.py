# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import disable_persistent_kernel_cache
from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov8s.tt.tt_yolov8s_utils import custom_preprocessor
from models.demos.yolov8s.tt.ttnn_yolov8s import TtYolov8sModel
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV8S_L1_SMALL_SIZE}], indirect=True, ids=["0"])
@pytest.mark.parametrize(
    "input_tensor",
    [torch.rand((1, 3, 640, 640))],
    ids=["input_tensor1"],
)
@pytest.mark.parametrize(
    "use_pretrained_weights",
    [True],
)
def test_yolov8s_640(device, input_tensor, use_pretrained_weights, model_location_generator):
    disable_persistent_kernel_cache()

    inp_h, inp_w = input_tensor.shape[2], input_tensor.shape[3]
    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)
        state_dict = torch_model.state_dict()

    parameters = custom_preprocessor(device, state_dict, inp_h=inp_h, inp_w=inp_w)
    ttnn_model = TtYolov8sModel(device=device, parameters=parameters, res=(inp_h, inp_w))

    n, c, h, w = input_tensor.shape
    if c == 3:
        c = 16
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

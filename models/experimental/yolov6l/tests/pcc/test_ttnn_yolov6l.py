# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters
from models.experimental.yolov6l.tt.ttnn_yolov6l import TtYolov6l
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov6l.common import load_torch_model


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov6l(device, reset_seeds, model_location_generator):
    model = load_torch_model(model_location_generator)

    torch_input = torch.randn(1, 3, 640, 640)

    n, c, h, w = torch_input.shape
    if c == 3:
        c = 16
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_x = ttnn_x.to(device, input_mem_config)

    parameters = create_yolov6l_model_parameters(model, torch_input, device)

    ttnn_model = TtYolov6l(device, parameters, parameters.model_args)

    output = ttnn_model(ttnn_x)

    torch_output = model(torch_input)

    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output[0], output, pcc=0.999)

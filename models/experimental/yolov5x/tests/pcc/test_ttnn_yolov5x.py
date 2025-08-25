# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov5x.tt.yolov5x import Yolov5x
from models.experimental.yolov5x.tt.model_preprocessing import (
    create_yolov5x_input_tensors,
    create_yolov5x_model_parameters,
)
from models.experimental.yolov5x.common import load_torch_model, YOLOV5X_L1_SMALL_SIZE


@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE}], indirect=True)
def test_yolov5x(device, reset_seeds, model_location_generator):
    torch_input, ttnn_input = create_yolov5x_input_tensors(device)
    n, c, h, w = torch_input.shape
    padded_c = 16 if c < 16 else c  # If the channels < 16, pad the channels to 16 to run the Conv layer
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, padded_c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn_input.to(device, input_mem_config)

    torch_model = load_torch_model(model_location_generator)
    torch_model = torch_model.model

    parameters = create_yolov5x_model_parameters(torch_model, torch_input, device)

    torch_model_output = torch_model(torch_input)[0]
    ttnn_module = Yolov5x(
        device=device,
        parameters=parameters,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)

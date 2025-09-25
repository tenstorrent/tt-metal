# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov5x.tt.c3 import TtnnC3
from models.experimental.yolov5x.tt.model_preprocessing import (
    create_yolov5x_input_tensors,
    create_yolov5x_model_parameters,
)
from models.experimental.yolov5x.common import load_torch_model, YOLOV5X_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "index, fwd_input_shape , num_layers, shortcut",
    [
        (
            2,
            (1, 160, 160, 160),
            4,
            True,
        ),
        (
            4,
            (1, 320, 80, 80),
            8,
            True,
        ),
        (
            17,
            (1, 640, 80, 80),
            4,
            False,
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE}], indirect=True)
def test_yolov5x_C3(
    device,
    reset_seeds,
    index,
    fwd_input_shape,
    num_layers,
    shortcut,
    model_location_generator,
):
    torch_input, ttnn_input = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[0],
        input_channels=fwd_input_shape[1],
        input_height=fwd_input_shape[2],
        input_width=fwd_input_shape[3],
    )

    ttnn_input = ttnn.to_device(ttnn_input, device=device)
    ttnn_input = ttnn.to_layout(ttnn_input, layout=ttnn.TILE_LAYOUT)

    torch_model = load_torch_model(model_location_generator)
    torch_model = torch_model.model.model[index]

    parameters = create_yolov5x_model_parameters(torch_model, torch_input, device=device)

    torch_model_output = torch_model(torch_input)[0]

    ttnn_module = TtnnC3(
        shortcut=shortcut, n=num_layers, device=device, parameters=parameters.conv_args, conv_pt=parameters
    )
    ttnn_output = ttnn_module(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)
    ttnn_output = ttnn_output.reshape(torch_model_output.shape)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)

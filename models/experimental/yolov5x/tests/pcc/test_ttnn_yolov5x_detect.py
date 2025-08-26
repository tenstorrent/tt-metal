# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov5x.tt.detect import TtnnDetect
from models.experimental.yolov5x.tt.model_preprocessing import (
    create_yolov5x_input_tensors,
    create_yolov5x_model_parameters_detect,
)
from models.experimental.yolov5x.common import load_torch_model, YOLOV5X_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "fwd_input_shape",
    [
        ([1, 320, 80, 80], [1, 640, 40, 40], [1, 1280, 20, 20]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV5X_L1_SMALL_SIZE}], indirect=True)
def test_yolov5x_Detect(
    device,
    reset_seeds,
    fwd_input_shape,
    model_location_generator,
):
    torch_input_1, ttnn_input_1 = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[0][0],
        input_channels=fwd_input_shape[0][1],
        input_height=fwd_input_shape[0][2],
        input_width=fwd_input_shape[0][3],
    )
    torch_input_2, ttnn_input_2 = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[1][0],
        input_channels=fwd_input_shape[1][1],
        input_height=fwd_input_shape[1][2],
        input_width=fwd_input_shape[1][3],
    )
    torch_input_3, ttnn_input_3 = create_yolov5x_input_tensors(
        device,
        batch_size=fwd_input_shape[2][0],
        input_channels=fwd_input_shape[2][1],
        input_height=fwd_input_shape[2][2],
        input_width=fwd_input_shape[2][3],
    )

    torch_input = [torch_input_1, torch_input_2, torch_input_3]

    torch_model = load_torch_model(model_location_generator)
    torch_model = torch_model.model.model[24]

    parameters = create_yolov5x_model_parameters_detect(
        torch_model, torch_input[0], torch_input[1], torch_input[2], device=device
    )

    torch_model_output = torch_model(torch_input)[0]

    ttnn_module = TtnnDetect(
        device=device,
        parameters=parameters.model_args,
        conv_pt=parameters,
    )
    ttnn_output = ttnn_module(ttnn_input_1, ttnn_input_2, ttnn_input_3)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_model_output, ttnn_output, 0.99)

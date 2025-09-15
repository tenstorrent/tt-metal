# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.yolov13.common import YOLOV13_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov13.reference import yolov13


# mock for now
def create_yolov13_input_tensors(device, batch_size=1, input_channels=3, input_height=640, input_width=640):
    torch_input_tensor = torch.randn(batch * device.get_num_devices(), input_channels, input_height, input_width)
    return torch_input_tensor


@pytest.mark.parametrize(
    "resolution",
    [
        ([1, 3, 640, 640]),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weights",
    [
        True,
        # False
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": YOLOV13_L1_SMALL_SIZE}], indirect=True)
def test_yolov13(device, reset_seeds, resolution, use_pretrained_weights, model_location_generator, min_channels=8):
    torch_model = yolov13.YoloV13()
    torch_model.eval()

    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)

    torch_input, ttnn_input = create_yolov13_input_tensors(
        device,
        batch=resolution[0],
        input_channels=resolution[1],
        input_height=resolution[2],
        input_width=resolution[3],
        is_sub_module=False,
    )

    # ttnn_input = ttnn_input.to(device, input_mem_config)
    torch_output = torch_model(torch_input)
    # parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    # ttnn_model = ttnn_yolov11.TtnnYoloV11(device, parameters)
    # ttnn_output = ttnn_model(ttnn_input)
    # ttnn_output = ttnn.to_torch(ttnn_output)
    # assert_with_pcc(torch_output, ttnn_output, 0.99)

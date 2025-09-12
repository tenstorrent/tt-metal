# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from models.demos.yolov11m.common import YOLOV11_L1_SMALL_SIZE, load_torch_model
from models.demos.yolov11m.reference import yolov11


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
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV11_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
def test_yolov11(device, reset_seeds, resolution, use_pretrained_weights, model_location_generator, min_channels=8):
    torch_model = yolov11.YoloV11()
    torch_model.eval()

    if use_pretrained_weights:
        torch_model = load_torch_model(model_location_generator)
    torch_input = torch.randn(resolution[0], resolution[1], resolution[2], resolution[3])
    
    torch_output = torch_model(torch_input)
    print(torch_output.shape)
    import pdb; pdb.set_trace()

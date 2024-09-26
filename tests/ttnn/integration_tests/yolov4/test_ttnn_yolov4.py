# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import skip_for_grayskull
from models.experimental.yolov4.reference.yolov4 import Yolov4
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov4.ttnn.yolov4 import TtYOLOv4
import pytest


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov4(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolo")

    if model_path == "models":
        pytest.skip(
            "Requires weights file to be downloaded from https://drive.google.com/file/d/1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ/view"
        )
    else:
        weights_pth = str(model_path / "yolov4.pth")

    ttnn_model = TtYOLOv4(weights_pth)

    torch_input = torch.randn((1, 320, 320, 3), dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    torch_input = torch_input.permute(0, 3, 1, 2).float()
    torch_model = Yolov4()

    new_state_dict = {}
    ds_state_dict = {k: v for k, v in ttnn_model.torch_model.items()}

    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]

    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    result_1, result_2, result_3 = ttnn_model(device, ttnn_input)
    result_1 = ttnn.to_torch(result_1)
    result_2 = ttnn.to_torch(result_2)
    result_3 = ttnn.to_torch(result_3)

    ref1, ref2, ref3 = torch_model(torch_input)

    result_1 = result_1.reshape(1, ref1.shape[2], ref1.shape[3], 255)
    result_1 = result_1.permute(0, 3, 1, 2)

    result_2 = result_2.reshape(1, ref2.shape[2], ref2.shape[3], 255)
    result_2 = result_2.permute(0, 3, 1, 2)

    result_3 = result_3.reshape(1, ref3.shape[2], ref3.shape[3], 255)
    result_3 = result_3.permute(0, 3, 1, 2)

    # Output is sliced because ttnn.conv returns 256 channels instead of 255.
    result_1 = result_1[:, :255, :, :]
    result_2 = result_2[:, :255, :, :]
    result_3 = result_3[:, :255, :, :]

    assert_with_pcc(result_1, ref1, 0.99)
    assert_with_pcc(result_2, ref2, 0.99)
    assert_with_pcc(result_3, ref3, 0.99)

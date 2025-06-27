# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import os
import torch
import pytest
import ttnn
from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.mobilenetv2.tests.mobilenetv2_common import MOBILENETV2_BATCH_SIZE, MOBILENETV2_L1_SMALL_SIZE
from models.demos.mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_tensors,
    create_mobilenetv2_model_parameters,
)
from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": MOBILENETV2_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
    ids=[
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [
        MOBILENETV2_BATCH_SIZE,
    ],
)
def test_mobilenetv2(device, use_pretrained_weight, batch_size, reset_seeds):
    # Check if weights file exists, if not, download them
    weights_path = "models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"
    if not os.path.exists(weights_path):
        os.system("bash models/demos/mobilenetv2/weights_download.sh")
    if use_pretrained_weight:
        state_dict = torch.load(weights_path)
        ds_state_dict = {k: v for k, v in state_dict.items()}

        torch_model = Mobilenetv2()
        new_state_dict = {
            name1: parameter2
            for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items())
            if isinstance(parameter2, torch.FloatTensor)
        }
        torch_model.load_state_dict(new_state_dict)
    else:
        torch_model = Mobilenetv2()
        state_dict = torch_model.state_dict()

    torch_model.eval()

    torch_input_tensor, ttnn_input_tensor = create_mobilenetv2_input_tensors(
        batch=batch_size, input_height=224, input_width=224
    )
    torch_output_tensor = torch_model(torch_input_tensor)

    model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)

    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size)
    output_tensor = ttnn_model(ttnn_input_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.944 if use_pretrained_weight else 0.999)

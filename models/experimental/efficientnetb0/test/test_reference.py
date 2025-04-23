# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.efficientnetb0.reference import efficientnetb0
from efficientnet_pytorch import EfficientNet


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_mobilenetv2(device, reset_seeds):
    model = EfficientNet.from_pretrained("efficientnet-b0").eval()

    state_dict = model.state_dict()

    ds_state_dict = {k: v for k, v in state_dict.items()}

    torch_model = efficientnetb0.Efficientnetb0()
    new_state_dict = {}
    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    input_tensor = torch.randn((1, 3, 256, 256), dtype=torch.float32)

    model_output = model(input_tensor)
    torch_model_output = torch_model(input_tensor)

    assert_with_pcc(torch_model_output, model_output, 1)

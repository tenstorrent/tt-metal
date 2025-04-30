# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.efficientnetb0.reference import efficientnetb0
from models.experimental.efficientnetb0.tt.model_preprocessing import (
    create_efficientnetb0_input_tensors,
    create_efficientnetb0_model_parameters,
)
from models.experimental.efficientnetb0.tt import efficientnetb0 as ttnn_efficientnetb0
from efficientnet_pytorch import EfficientNet


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_efficientnetb0_model(device, reset_seeds):
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

    torch_input, ttnn_input = create_efficientnetb0_input_tensors(device)
    torch_output = torch_model(torch_input)
    conv_params, parameters = create_efficientnetb0_model_parameters(torch_model, torch_input, device=device)

    ttnn_model = ttnn_efficientnetb0.Efficientnetb0(device, parameters, conv_params)

    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_output, ttnn_output, 0.95)

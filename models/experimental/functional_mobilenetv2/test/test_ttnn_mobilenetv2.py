# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.experimental.functional_mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_tensors,
    create_mobilenetv2_model_parameters,
)
from models.experimental.functional_mobilenetv2.tt import ttnn_mobilenetv2
import os


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_mobilenetv2(device, reset_seeds):
    if not os.path.exists("models/experimental/functional_mobilenetv2/mobilenet_v2-b0353104.pth"):
        os.system(
            "bash models/experimental/functional_mobilenetv2/weights_download.sh"
        )  # execute the weights_download.sh file

    state_dict = torch.load("models/experimental/functional_mobilenetv2/mobilenet_v2-b0353104.pth")
    ds_state_dict = {k: v for k, v in state_dict.items()}
    torch_model = Mobilenetv2()

    new_state_dict = {}

    for (name1, parameter1), (name2, parameter2) in zip(torch_model.state_dict().items(), ds_state_dict.items()):
        if isinstance(parameter2, torch.FloatTensor):
            new_state_dict[name1] = parameter2

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    torch_input_tensor, ttnn_input_tensor = create_mobilenetv2_input_tensors()
    torch_output_tensor = torch_model(torch_input_tensor)

    parameters = create_mobilenetv2_model_parameters(torch_model, torch_input_tensor, device=device)

    ttnn_model = ttnn_mobilenetv2.MobileNetV2(parameters, device, torch_model)
    output_tensor = ttnn_model(device, ttnn_input_tensor)

    #
    # Tensor Postprocessing
    #
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = output_tensor.reshape(torch_output_tensor.shape)
    output_tensor = output_tensor.to(torch_input_tensor.dtype)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.95)

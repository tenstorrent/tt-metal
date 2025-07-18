# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

import ttnn
from models.experimental.functional_vovnet.tt.conv_norm_act import TtConvNormAct
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_vovnet.tt.model_preprocessing import custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vovnet_conv_norm_act_inference(device, reset_seeds):
    base_address = f"stem.{0}"
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()
    model.eval()
    parameters = custom_preprocessor(device, model.state_dict())

    torch_model = model.stem[0]
    tt_model = TtConvNormAct(
        stride=2,
        base_address=base_address,
        device=device,
        parameters=parameters,
    )

    input = torch.rand(1, 3, 224, 224)
    model_output = torch_model(input)

    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output[0])
    assert_with_pcc(model_output, tt_output_torch, 0.99)

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger
import ttnn
from models.experimental.functional_vovnet.tt.model_preprocessing import create_vovnet_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


from models.experimental.functional_vovnet.tt.separable_conv_norm_act import TtSeparableConvNormAct


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_separable_conv_norm_act_inference(device, pcc, reset_seeds):
    base_address = f"stem.1"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.stem[1].conv_dw
    torch_model2 = model.stem[1].conv_pw

    tt_model = TtSeparableConvNormAct(
        stride=1,
        padding=1,
        bias=False,
        # groups=64,
        # channel_multiplier=1.0,
        torch_model=model.state_dict(),
        base_address=base_address,
        device=device,
    )

    input = torch.randn(1, 64, 112, 112)

    # run torch model
    model_output = torch_model(input)
    model_output = torch_model2(model_output)

    # run tt model

    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)
    # tt_input = ttnn.permute(tt_input, (0, 2, 3, 1))
    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output[0])
    # tt_output_torch = torch.permute(tt_output_torch, (0, 3, 1, 2))
    # tt_output_torch = torch.reshape(tt_output_torch, model_output.shape)
    assert_with_pcc(model_output, tt_output_torch, 0.99)
    # compare output

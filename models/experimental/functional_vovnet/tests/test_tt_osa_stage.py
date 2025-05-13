# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger
import ttnn

# from models.experimental.functional_vovnet.tt.conv_norm_act import TtConvNormAct
from models.experimental.functional_vovnet.tt.model_preprocessing import create_vovnet_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_vovnet.tt.osa_stage import TtOsaStage


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_osa_stage_inference(device, pcc, reset_seeds):
    STAGE_INDEX = 3

    base_address = f"stages.{STAGE_INDEX}"
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.stages[STAGE_INDEX]
    print(torch_model)

    downsample = False
    if STAGE_INDEX > 0:
        downsample = True
    tt_model = TtOsaStage(
        base_address=base_address,
        torch_model=model.state_dict(),
        device=device,
        downsample=downsample,
    )

    # run torch model
    input = torch.randn(1, 768, 14, 14)
    model_output = torch_model(input)

    # run tt model
    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)
    # tt_output_torch = torch.permute(tt_output_torch, (0, 3, 1, 2))
    # tt_output_torch = torch.reshape(tt_output_torch, model_output.shape)
    print("Shape of maxpool :", model_output.shape, "  ", tt_output_torch.shape)
    assert_with_pcc(model_output, tt_output_torch, 0.99)
    # compare output


"""
Stage shape of x : torch.Size([1, 64, 56, 56])
Shape of xw : torch.Size([1, 256, 1, 1])
Stage shape of x : torch.Size([1, 256, 56, 56])
Shape of xw : torch.Size([1, 512, 1, 1])
Stage shape of x : torch.Size([1, 512, 28, 28])
Shape of xw : torch.Size([1, 768, 1, 1])
Stage shape of x : torch.Size([1, 768, 14, 14])
Shape of xw : torch.Size([1, 1024, 1, 1])
"""

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_vovnet.tt.osa_stage import TtOsaStage
from models.experimental.functional_vovnet.tt.model_preprocessing import custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_osa_stage_inference(device, reset_seeds):
    STAGE_INDEX = 3

    base_address = f"stages.{STAGE_INDEX}"
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()
    parameters = custom_preprocessor(device, model.state_dict())
    torch_model = model.stages[STAGE_INDEX]

    downsample = False
    if STAGE_INDEX > 0:
        downsample = True
    tt_model = TtOsaStage(
        base_address=base_address,
        parameters=parameters,
        device=device,
        downsample=downsample,
    )

    input = torch.randn(1, 768, 14, 14)
    model_output = torch_model(input)

    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(model_output, tt_output_torch, 0.99)


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

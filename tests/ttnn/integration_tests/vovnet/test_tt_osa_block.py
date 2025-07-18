# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_vovnet.tt.osa_block import TtOsaBlock
from models.experimental.functional_vovnet.tt.model_preprocessing import custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_osa_block_inference(device, reset_seeds):
    STAGE_INDEX = 0
    BLOCK_INDEX = 0
    base_address = f"stages.{STAGE_INDEX}.blocks.{BLOCK_INDEX}"
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()

    torch_model = model.stages[STAGE_INDEX].blocks[BLOCK_INDEX]
    parameters = custom_preprocessor(device, model.state_dict())
    tt_model = TtOsaBlock(
        base_address=base_address,
        parameters=parameters,
        device=device,
    )
    torch_model = model.stages[STAGE_INDEX].blocks[BLOCK_INDEX]

    input = torch.randn(1, 64, 56, 56)

    out_list = [input]
    model_output = torch_model(input)

    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16)

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)
    assert_with_pcc(model_output, tt_output_torch, 0.99)

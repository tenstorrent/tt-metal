# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

from loguru import logger
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_vovnet.tt.effective_se_module import TtEffectiveSEModule


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_effective_se_module_inference(device, pcc, reset_seeds):
    base_address = f"stages.0.blocks.0.attn"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)

    torch_model = model.stages[0].blocks[0].attn
    print(torch_model)

    tt_model = TtEffectiveSEModule(
        stride=1,
        dilation=1,
        padding=0,
        bias=None,
        torch_model=model.state_dict(),
        base_address=base_address,
        device=device,
        temp_model=torch_model,
    )

    # run torch model
    input = torch.randn(1, 256, 56, 56)
    model_output = torch_model(input)

    # run tt model
    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = ttnn.to_torch(tt_output)
    # tt_output_torch = torch.permute(tt_output_torch, (0, 3, 1, 2))
    # tt_output_torch = torch.reshape(tt_output_torch, model_output.shape)
    print("Shape of models :", model_output.shape, " ", tt_output_torch.shape)
    assert_with_pcc(model_output, tt_output_torch, 0.99)
    # compare output

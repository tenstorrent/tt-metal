# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import timm

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.functional_vovnet.tt.sequential_append_list import TtSequentialAppendList
from models.experimental.functional_vovnet.tt.model_preprocessing import custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_sequential_append_list_inference(device, reset_seeds):
    STAGE_INDEX = 0
    BLOCK_INDEX = 0

    base_address = f"stages.{STAGE_INDEX}.blocks.{BLOCK_INDEX}"

    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True).eval()

    torch_model = model.stages[STAGE_INDEX].blocks[BLOCK_INDEX].conv_mid
    parameters = custom_preprocessor(device, model.state_dict())
    tt_model = TtSequentialAppendList(
        layer_per_block=3,
        base_address=f"{base_address}",
        device=device,
        parameters=parameters,
    )

    input = torch.randn(1, 128, 56, 56)
    model_output = torch_model(input, [input])

    tt_input = ttnn.from_torch(input, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_output = tt_model.forward(tt_input, [tt_input])
    tt_output_torch = ttnn.to_torch(tt_output)

    assert_with_pcc(model_output, tt_output_torch, 0.99)

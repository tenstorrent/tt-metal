# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.vovnet.tt.osa_block import TtOsaBlock
from models.experimental.vovnet.tt.model_preprocessing import custom_preprocessor
from models.experimental.vovnet.common import load_torch_model, VOVNET_L1_SMALL_SIZE


@pytest.mark.parametrize("device_params", [{"l1_small_size": VOVNET_L1_SMALL_SIZE}], indirect=True)
def test_osa_block_inference(device, reset_seeds, model_location_generator):
    STAGE_INDEX = 0
    BLOCK_INDEX = 0
    base_address = f"stages.{STAGE_INDEX}.blocks.{BLOCK_INDEX}"
    model = load_torch_model(model_location_generator)

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

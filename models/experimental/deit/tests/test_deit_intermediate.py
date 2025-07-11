# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers import DeiTModel


from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_intermediate import TtDeiTIntermediate
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose_and_pcc,
)


def test_deit_intermediate_inference(device, pcc=0.99):
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address = "encoder.layer.0.intermediate"
    torch_intermediate = model.encoder.layer[0].intermediate

    input_shape = torch.Size([1, 198, 768])
    hidden_state = torch.randn(input_shape)

    torch_output = torch_intermediate(hidden_state)

    # setup tt model
    tt_intermediate = TtDeiTIntermediate(DeiTConfig(), device, state_dict, base_address)

    tt_input = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)
    tt_out = tt_intermediate(tt_input)
    tt_output = tt_to_torch_tensor(tt_out).squeeze(0)

    pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
    logger.info(f"Output {pcc_output}")
    assert pcc_passing, f"Failed! Low pcc: {pcc}."

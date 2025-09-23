# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers import DeiTModel


from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_self_output import TtDeiTSelfOutput
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose_and_pcc,
)


def test_deit_self_output_inference(device, pcc=0.99):
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address = "encoder.layer.0.attention.output"
    torch_self_output = model.encoder.layer[0].attention.output

    input_shape = torch.Size([1, 198, 768])
    hidden_state = torch.randn(input_shape)
    input_tensor = None

    torch_output = torch_self_output(hidden_state, None)

    # setup tt model
    tt_self_output = TtDeiTSelfOutput(DeiTConfig(), device, state_dict, base_address)

    tt_input = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)
    tt_out = tt_self_output(tt_input, input_tensor)
    tt_output = tt_to_torch_tensor(tt_out).squeeze(0)

    pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
    logger.info(f"Output {pcc_output}")
    assert pcc_passing, f"Failed! Low pcc: {pcc}."

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers import DeiTModel


from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_output import TtDeiTOutput
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose_and_pcc,
)


def test_deit_output_inference(device, pcc=0.99):
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address = "encoder.layer.0.output"
    torch_model = model.encoder.layer[0].output

    hidden_state_shape = torch.Size([1, 198, 3072])
    hidden_state = torch.randn(hidden_state_shape)

    input_tensor_shape = torch.Size([1, 198, 768])
    input_tensor = torch.randn(input_tensor_shape)

    torch_output = torch_model(hidden_state, input_tensor)

    # setup tt model
    tt_output = TtDeiTOutput(DeiTConfig(), device, state_dict, base_address)

    tt_hidden_state = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)
    tt_input_tensor = torch_to_tt_tensor_rm(input_tensor, device, put_on_device=False)

    tt_output = tt_output(tt_hidden_state, tt_input_tensor)
    tt_output = tt_to_torch_tensor(tt_output).squeeze(0)

    pcc_passing, _ = comp_pcc(torch_output, tt_output, pcc)
    _, pcc_output = comp_allclose_and_pcc(torch_output, tt_output, pcc)
    logger.info(f"Output {pcc_output}")
    assert pcc_passing, f"Failed! Low pcc: {pcc}."

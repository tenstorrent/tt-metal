# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers import DeiTModel


from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
    comp_pcc,
    comp_allclose_and_pcc,
)


from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_self_attention import TtDeiTSelfAttention


def test_deit_self_attention_inference(device, pcc=0.99):
    # setup pytorch model
    model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model.eval()
    state_dict = model.state_dict()

    # synthesize the input
    base_address = "encoder.layer.0.attention.attention"
    torch_self_attention = model.encoder.layer[0].attention.attention
    head_mask = None
    output_attentions = False
    input_shape = torch.Size([1, 1, 198, 768])
    hidden_state = torch.randn(input_shape)

    torch_output = torch_self_attention(hidden_state.squeeze(0), head_mask, output_attentions)[0]

    # setup tt model
    tt_self_attention = TtDeiTSelfAttention(DeiTConfig(), device, state_dict, base_address)

    tt_input = torch_to_tt_tensor_rm(hidden_state, device, put_on_device=False)
    tt_out = tt_self_attention(tt_input, head_mask, output_attentions)
    tt_output = tt_to_torch_tensor(tt_out[0]).squeeze(0)

    passing = comp_pcc(torch_output, tt_output, pcc)
    logger.info(comp_allclose_and_pcc(tt_output, torch_output, pcc))
    assert passing[0], passing[1:]
    logger.info(f"PASSED {passing[1]}")

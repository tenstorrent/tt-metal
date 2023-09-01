# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import pytest

from transformers import GPT2LMHeadModel


from loguru import logger
import models.nanogpt.tt.nanogpt_attention as nanogpt_attention

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose,
    comp_pcc,
)


@pytest.mark.parametrize(
    "pcc",
    ((0.99,),),
)

def test_nanogpt_attn(device, pcc, reset_seeds):

    # Prepare input
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = model_hf.state_dict()
    config = model_hf.config
    model_hf.eval()
    block = 0
    base_address = f"transformer.h.{block}.attn"

    test_in = torch.rand(1, 60, 768)
    pt_attn = model_hf.transformer.h[block].attn
    pt_out = pt_attn.forward(test_in)

    tt_test_in = torch_to_tt_tensor_rm(test_in, device)
    tt_attn = nanogpt_attention.TtCausalSelfAttention(config, sd, base_address, device)

    tt_out = tt_attn.forward(tt_test_in)

    tt_out_converted = tt_to_torch_tensor(tt_out).squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out[0], tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_attention: Passed!")
    else:
        logger.warning("nanogpt_attention: Failed!")

    assert does_pass

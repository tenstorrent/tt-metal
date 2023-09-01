# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import pytest

from transformers import GPT2LMHeadModel

from loguru import logger
import models.nanogpt.tt.nanogpt_mlp as nanogpt_mlp


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
def test_nanogpt_mlp(device, pcc, reset_seeds):

    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd = model_hf.state_dict()
    config = model_hf.config
    model_hf.eval()
    block = 0
    base_address = f"transformer.h.{block}.mlp"

    test_in = torch.rand(1, 43, 768)
    tt_test_in = torch_to_tt_tensor_rm(test_in, device)
    tt_mlp = nanogpt_mlp.TtMLP(base_address, config, sd, device)

    tt_out = tt_mlp.forward(tt_test_in)

    pt_mlp = model_hf.transformer.h[block].mlp
    pt_out = pt_mlp.forward(test_in)

    tt_out_converted = tt_to_torch_tensor(tt_out).squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out, tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_mlp: Passed!")
    else:
        logger.warning("nanogpt_mlp: Failed!")

    assert does_pass

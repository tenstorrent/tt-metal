# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import pytest

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from loguru import logger
import models.nanogpt.tt.nanogpt_model as nanogpt_model

from models.utility_functions import tt_to_torch_tensor, comp_allclose, comp_pcc



@pytest.mark.parametrize(
    "pcc, prompt",
    ((0.99, "Hello, my dog is a little"),),
)
def test_nanogpt_model_real(device, pcc, prompt, reset_seeds):

    # Prepare input
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    sd = model_hf.state_dict()
    model_hf.eval()

    inputs = tokenizer(prompt, return_tensors="pt", padding=False)

    pt_model = model_hf
    pt_out = pt_model.forward(inputs.input_ids)

    config = model_hf.config

    tt_model = nanogpt_model.TtGPT(config, sd, device)

    tt_out = tt_model.forward(inputs.input_ids)

    tt_out_converted = tt_to_torch_tensor(tt_out).squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out[0], tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out[0], tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("nanogpt_model_real: Passed!")
    else:
        logger.warning("nanogpt_model_real: Failed!")

    assert does_pass

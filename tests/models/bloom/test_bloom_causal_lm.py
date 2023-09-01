# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import tt_lib

from transformers import BloomForCausalLM, BloomTokenizerFast
from models.utility_functions import comp_pcc, tt_to_torch_tensor, comp_allclose

from loguru import logger
from models.bloom.tt.bloom_causal_lm import TtBloomForCausalLM


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_bloom_causal_lm(pcc, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-560m"
    )
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()

    tt_bloom_causal_lm = TtBloomForCausalLM(config, state_dict, device)
    pt_bloom_causal_lm = hugging_bloom_reference_model

    # Prepare input
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    input_sentance = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    tokenized = tokenizer(input_sentance, return_tensors="pt")
    input_ids = tokenized.input_ids

    with torch.no_grad():
        # pytorch output
        pt_out = pt_bloom_causal_lm(input_ids)
        # tt output
        tt_out = tt_bloom_causal_lm(input_ids)

    pt_out = pt_out[0].unsqueeze(0)
    tt_out = tt_out[0]
    tt_out = tt_to_torch_tensor(tt_out)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)

    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    tt_lib.device.CloseDevice(device)

    if does_pass:
        logger.info("bloom_causal_lm: Passed!")
    else:
        logger.warning("bloom_causal_lm: Failed!")

    assert does_pass, f"bloom_causal_lm output does not meet PCC requirement {pcc}."

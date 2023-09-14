# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
import pytest

from transformers import BloomForCausalLM, BloomTokenizerFast
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    tt_to_torch_tensor,
)
from loguru import logger

from models.bloom.tt.bloom_model import TtBloomModel


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_bloom_model(pcc, reset_seeds, device):
    tt_lib.device.SetDefaultDevice(device)

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained(
        "bigscience/bloom-560m"
    )
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config

    state_dict = hugging_bloom_reference_model.state_dict()
    base_address = "transformer"

    tt_bloom_model = TtBloomModel(config, state_dict, base_address, device)
    pt_bloom_model = hugging_bloom_reference_model.transformer

    # Prepare input
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    input_sentance = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    tokenized = tokenizer(input_sentance, return_tensors="pt")
    input_ids = tokenized.input_ids

    with torch.no_grad():
        pt_out = pt_bloom_model(input_ids)[0]
        tt_out = tt_bloom_model(input_ids)[0]

    tt_out_converted = tt_to_torch_tensor(tt_out)
    tt_out_converted = tt_out_converted.squeeze(0)

    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, pcc)

    logger.info(comp_allclose(pt_out, tt_out_converted))
    logger.info(pcc_message)

    if does_pass:
        logger.info("bloom_model: Passed!")
    else:
        logger.warning("bloom_model: Failed!")

    assert does_pass, f"bloom_model output does not meet PCC requirement {pcc}."

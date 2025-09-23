# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from transformers import BloomForCausalLM, BloomTokenizerFast
from models.utility_functions import print_diff_argmax
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from loguru import logger
import models.experimental.bloom_old.tt.bloom_causal_lm as bloom_causal_lm


def pad_input_32(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def run_bloom_causal_lm_test(device):
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()

    tt_bloom_causal_lm = bloom_causal_lm.TtBloomForCausalLM(config, state_dict, device)
    pt_bloom_causal_lm = hugging_bloom_reference_model

    # Prepare input
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    input_sentance = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    tokenized = tokenizer(input_sentance, return_tensors="pt")
    input_ids = pad_input_32(tokenized.input_ids, config.pad_token_id)

    pt_out = pt_bloom_causal_lm.forward(input_ids)
    print("PT finished")

    tt_out = tt_bloom_causal_lm.forward(device, input_ids)
    print("TT finished")

    pt_out = pt_out[0]
    tt_out = tt_out[0]
    tt_out = tt_out.squeeze(0)
    tt_out = tt_out.squeeze(0)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.50)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_causal_lm: Passed!")
    else:
        logger.warning("bloom_causal_lm: Failed!")

    assert does_pass


def test_bloom_causal_lm(device):
    run_bloom_causal_lm_test(device)

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

import models.experimental.bloom_old.bloom_utils as bloom_utils
import models.experimental.bloom_old.tt.bloom_model as bloom_model


def pad_input_32(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def run_bloom_model_test(device):
    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hugging_bloom_reference_model.eval()

    config = hugging_bloom_reference_model.config
    state_dict = hugging_bloom_reference_model.state_dict()
    base_address = "transformer"

    tt_bloom_model = bloom_model.TtBloomModel(config, state_dict, base_address, device)
    pt_bloom_model = hugging_bloom_reference_model.transformer

    # Prepare input
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    input_sentance = "summarize: QuillBot's Summarizer wants to change how you read! Instead of reading through loads of documents, you can get a short annotated summary or bullet points with all the key information."
    tokenized = tokenizer(input_sentance, return_tensors="pt")
    input_ids = pad_input_32(tokenized.input_ids, config.pad_token_id)

    pt_out = pt_bloom_model.forward(input_ids)[0]
    print("PT finished")

    tt_out = tt_bloom_model.forward(device, input_ids)[0]
    print("TT finished")

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)
    tt_out_converted = tt_out_converted.squeeze(0)

    print(f"pt_out shape {pt_out.shape}")
    print(f"tt_out_converted shape {tt_out_converted.shape}")

    print_diff_argmax(pt_out, tt_out_converted)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out_converted, 0.65)

    print(comp_allclose(pt_out, tt_out_converted))
    print(pcc_message)

    if does_pass:
        logger.info("bloom_model: Passed!")
    else:
        logger.warning("bloom_model: Failed!")

    assert does_pass


def test_bloom_model(device):
    run_bloom_model_test(device)

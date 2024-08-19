# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
import pytest
from loguru import logger

from transformers import AutoTokenizer, T5Tokenizer, T5Model
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    comp_pcc,
    is_wormhole_b0,
)
from models.experimental.t5.tt.t5_model import TtT5Model

pytestmark = pytest.mark.skipif(is_wormhole_b0(), reason="Skip for Wormhole B0")


def run_test_T5Model_inference(device, use_attention_mask, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())

    # Prepare input
    input_sentance = "Studies have been shown that owning a dog is good for you"
    tokenized = tokenizer(input_sentance, padding="max_length", max_length=32, return_tensors="pt")  # Batch size 1

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask if use_attention_mask else None

    decoder_input_sentence = "Studies show that"
    tokenized = tokenizer(
        decoder_input_sentence, padding="max_length", max_length=32, return_tensors="pt"
    )  # Batch size 1

    decoder_input_ids = tokenized.input_ids
    decoder_attention_mask = tokenized.attention_mask if use_attention_mask else None

    # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    decoder_input_ids = hf_reference_model._shift_right(decoder_input_ids)

    # PyTorch forward pass
    pt_out = hf_reference_model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=decoder_attention_mask,
    )
    pt_out = pt_out.last_hidden_state
    pt_out = pt_out.unsqueeze(0)

    hf_reference_model = T5Model.from_pretrained(model_name, torch_dtype=torch.float16)
    hf_reference_model.eval()

    tt_model = TtT5Model(config, hf_reference_model.state_dict(), device)
    tt_model_outputs = tt_model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=decoder_attention_mask,
    )
    tt_out = tt2torch_tensor(tt_model_outputs[0])
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)
    logger.info(pcc_message)

    pt_decoded_out = tokenizer.decode(pt_out[0][0].softmax(0).argmax(1))
    tt_decoded_out = tokenizer.decode(tt_out[0][0].softmax(0).argmax(1))

    logger.info(f"Pt decoded output: {pt_decoded_out}")
    logger.info(f"Tt decoded output: {tt_decoded_out}")

    if does_pass:
        logger.info(f"test_T5Model_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5Model_inference {model_name} Failed!")

    assert does_pass


def test_T5Model_inference_t5_small(device):
    run_test_T5Model_inference(device, use_attention_mask=True, model_name="t5-small")


def test_T5Model_inference_flan_t5_small(device):
    run_test_T5Model_inference(device, use_attention_mask=True, model_name="google/flan-t5-small")


def test_T5Model_inference_t5_base(device):
    run_test_T5Model_inference(device, use_attention_mask=True, model_name="t5-base")

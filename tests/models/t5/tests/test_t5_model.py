# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from loguru import logger
import pytest
from transformers import AutoTokenizer, T5Model
from models.utility_functions import (
    tt_to_torch_tensor,
)
from models.utility_functions import comp_pcc, comp_allclose
from models.t5.tt.t5_model import TtT5Model


def run_test_T5Model_inference(pcc, device, use_attention_mask, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = hf_reference_model.config

    # Prepare input
    input_sentance = "Studies have been shown that owning a dog is good for you"
    tokenized = tokenizer(
        input_sentance, padding="max_length", max_length=32, return_tensors="pt"
    )

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask if use_attention_mask else None

    decoder_input_sentence = "Studies show that"
    tokenized = tokenizer(
        decoder_input_sentence, padding="max_length", max_length=32, return_tensors="pt"
    )

    decoder_input_ids = tokenized.input_ids
    decoder_attention_mask = tokenized.attention_mask if use_attention_mask else None

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
    tt_out = tt_to_torch_tensor(tt_model_outputs.last_hidden_state)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, pcc)
    logger.info(comp_allclose(pt_out, tt_out))
    logger.info(pcc_message)

    pt_decoded_out = tokenizer.decode(pt_out[0][0].softmax(0).argmax(1))
    tt_decoded_out = tokenizer.decode(tt_out[0][0].softmax(0).argmax(1))

    logger.info(f"Pt decoded output: {pt_decoded_out}")
    logger.info(f"Tt decoded output: {tt_decoded_out}")

    if does_pass:
        logger.info(f"test_T5Model_inference {model_name} Passed!")
    else:
        logger.warning(f"test_T5Model_inference {model_name} Failed!")

    assert does_pass, f"T5Model output does not meet PCC requirement {pcc}."


@pytest.mark.parametrize(
    "pcc, model_name",
    ((0.99, "t5-small"),),
)
def test_T5Model_inference_t5_small(pcc, model_name, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    run_test_T5Model_inference(
        pcc, device, use_attention_mask=True, model_name=model_name
    )
    tt_lib.device.CloseDevice(device)


@pytest.mark.parametrize(
    "pcc, model_name",
    ((0.99, "google/flan-t5-small"),),
)
def test_T5Model_inference_flan_t5_small(pcc, model_name, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    run_test_T5Model_inference(
        pcc, device, use_attention_mask=True, model_name=model_name
    )


@pytest.mark.parametrize(
    "pcc, model_name",
    ((0.98, "t5-base"),),
)
def test_T5Model_inference_t5_base(pcc, model_name, reset_seeds):
    device = tt_lib.device.CreateDevice(0)
    tt_lib.device.SetDefaultDevice(device)
    run_test_T5Model_inference(
        pcc, device, use_attention_mask=True, model_name=model_name
    )
    tt_lib.device.CloseDevice(device)

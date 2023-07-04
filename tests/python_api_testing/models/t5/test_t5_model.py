from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import copy
import torch
import json
from torch import nn
import tt_lib
from loguru import logger

from transformers import AutoTokenizer, T5Tokenizer, T5Model
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from python_api_testing.models.t5.t5_model import TtT5Model


def run_test_T5Model_inference(device, use_attention_mask, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=32)
    hf_reference_model = T5Model.from_pretrained(model_name)
    hf_reference_model.eval()

    config = json.loads(hf_reference_model.config.to_json_string())

    # Prepare input
    input_sentance = "Studies have been shown that owning a dog is good for you"
    tokenized = tokenizer(
        input_sentance, padding="max_length", max_length=32, return_tensors="pt"
    )  # Batch size 1

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
        logger.info("test_T5Model_inference Passed!")
    else:
        logger.warning("test_T5Model_inference Failed!")

    assert does_pass


def test_T5Model_inference():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5Model_inference(device, use_attention_mask=True, model_name="t5-small")
    tt_lib.device.CloseDevice(device)


def test_T5Model_inference_flan():
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    run_test_T5Model_inference(
        device, use_attention_mask=True, model_name="google/flan-t5-small"
    )
    tt_lib.device.CloseDevice(device)

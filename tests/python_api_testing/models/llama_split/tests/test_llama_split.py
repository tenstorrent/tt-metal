import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import math
import time
import torch
from torch import nn
import tt_lib
from loguru import logger
import pytest
from python_api_testing.models.llama.llama_utils import tt2torch_tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from typing import List, Optional, Tuple, Union
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama_split.llama_split_utils import prepare_llama_input
from python_api_testing.models.llama_split.tt.llama import (
    llama_first_half,
    llama_second_half,
)
from utility_functions_new import comp_pcc
from transformers.generation.configuration_utils import GenerationConfig


def call_tt_llama_forward_func(
    configuration,
    state_dict,
    base_url,
    max_position_embeddings,
    prompt,
    tokenizer,
    input_ids,
    attention_mask,
    position_ids,
    first_decoder_start,
    second_decoder_start,
    num_consecutive_decoders,
    is_causallm,
):
    logger.debug(f"The first half started")
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    first_out = run_test_llama_split_inference(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=first_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=input_ids,
        att_mask=attention_mask,
        position_ids=position_ids,
        half=1,
    )
    tt_lib.device.CloseDevice(device)
    logger.debug(f"The first half ended")

    # The second call -------------------------------------------------------
    logger.debug(f"The second half started")
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # send input tensor from host to tt device
    tt_input = first_out

    tt_out = run_test_llama_split_inference(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=second_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=tt_input,
        att_mask=attention_mask,
        position_ids=position_ids,
        half=2,
        is_causallm=is_causallm,
    )
    logger.debug(f"The second half ended")

    # squeeze the output
    tt_out = tt_out.squeeze(1)
    return tt_out


def run_test_llama_split_inference(
    device,
    state_dict,
    base_url,
    max_position_embeddings,
    configuration,
    num_decoders_start,
    num_decoders,
    x_inputs=None,
    att_mask=None,
    position_ids=None,
    half=1,
    is_causallm=True,
):
    if half == 1:
        logger.debug("First pass throught TT model")
        first_model_create_start = time.time()
        tt_llama_model = llama_first_half(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        first_model_create_end = time.time()
        logger.debug(
            f"First half - duration of model creation: {first_model_create_end-first_model_create_start}"
        )
        start = time.time()
        tt_out = tt_llama_model(
            input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids
        )
        end = time.time()
        logger.debug(f"First half - inference duration: {end-start}")
    else:
        logger.debug("Second pass throught TT model")
        second_model_create_start = time.time()
        tt_llama_model = llama_second_half(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
            is_causallm,
        )
        second_model_create_end = time.time()
        logger.debug(
            f"Second half - duration of model creation: {second_model_create_end-second_model_create_start}"
        )

        start = time.time()
        tt_out = tt_llama_model(
            input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids
        )
        end = time.time()
        logger.debug(f"Second half - inference duration: {end-start}")

    # returned type from the model is tuple
    tt_output = tt2torch_tensor(tt_out[0])
    return tt_output


# parameters --------------------------------------------------
_tokenizer_name = "huggyllama/llama-7b"
_llama_model_name = "huggyllama/llama-7b"
# base url from the model state dictionary
_base_url = "model.layers"
_batch = 1
_seq_len = 32
_max_position_embeddings = 2048
# is causallm llama model (it has additional linear layer at the end)
_is_causallm = False
_on_weka = False

# how many decoders to use
# number of decoders to be stacked started from the selected id in the original llama model
# e.g. stack 16 consecutive decoders
_num_consecutive_decoders = 16

# decoder id from which decoder stacking starts (the first half of the model)
# e.g. start from 0 add use 3 decoders (0, 1, and 2)
_first_decoder_start = 0

# decoder id from which decoder stacking starts (the second half of the model)
# e.g. start from 16 add use 3 decoders (16, 17, and 18)
_second_decoder_start = 16
# parameters --------------------------------------------------


@pytest.mark.parametrize(
    "pcc",
    ((0.98),),
)
def test_llama_split_inference(pcc):
    # set parameters ================================================================
    model_version = _llama_model_name
    tokenizer_version = _tokenizer_name
    base_url = _base_url
    batch = _batch
    seq_len = _seq_len
    max_position_embeddings = _max_position_embeddings
    is_causallm = _is_causallm
    first_decoder_start = _first_decoder_start
    second_decoder_start = _second_decoder_start
    num_consecutive_decoders = _num_consecutive_decoders
    on_weka = _on_weka

    # load llama pytorch model ======================================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_version)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_version)
    hugging_face_reference_model.eval()

    # get configurations
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # prepare real prompt ===========================================================
    prompt = "I believe the meaning of life is"

    is_input_padded = True
    input_ids, attention_mask, position_ids = prepare_llama_input(
        prompt, tokenizer, configuration, is_input_padded
    )

    if not is_causallm:
        hugging_face_reference_model = hugging_face_reference_model.get_decoder()
        pytorch_out = hugging_face_reference_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pytorch_out = pytorch_out.last_hidden_state
    else:
        pytorch_out = hugging_face_reference_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pytorch_out = pytorch_out.logits

    logger.debug(f"Pytorch output shape: {pytorch_out.shape}")

    # execute Tenstorrent model (no linear layer at the end) ========================
    tt_out = call_tt_llama_forward_func(
        configuration,
        state_dict,
        base_url,
        max_position_embeddings,
        prompt,
        tokenizer,
        input_ids,
        attention_mask,
        position_ids,
        first_decoder_start,
        second_decoder_start,
        num_consecutive_decoders,
        is_causallm,
    )

    # check outputs ================================================================
    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {pcc_value}")

    if does_pass:
        logger.info("Llama Model Passed!")
    else:
        logger.warning("Llama Model Failed!")
        assert does_pass, f"PCC value ({pcc_value}) is lower than {pcc}."

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
import pytest
from torch import nn
import tt_lib
from loguru import logger
from python_api_testing.models.llama.llama_utils import (
    tt2torch_tensor,
    gen_position_ids,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from typing import List, Optional, Tuple, Union
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama_split.llama_split_utils import (
    pad_input_32_left,
    prepare_llama_input,
    get_next_llama_output_token,
)
from python_api_testing.models.llama_split.tt.llama import (
    llama_first_half,
    llama_second_half,
)
from llama_split_utils import get_logits_processor
from utility_functions_new import (
    profiler,
    enable_compile_cache,
    disable_compile_cache,
    comp_pcc,
)


def run_llama_split_inference(
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
    is_causallm=False,
):
    if half == 1:
        logger.debug("First pass throught TT model")
        tt_llama_model = llama_first_half(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(
            input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids
        )
    else:
        logger.debug("Second pass throught TT model")
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
        tt_out = tt_llama_model(
            input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids
        )

    # returned type from the model is tuple
    tt_output = tt2torch_tensor(tt_out[0])
    return tt_output


def call_tt_llama_forward_func(
    configuration,
    state_dict,
    base_url,
    max_position_embeddings,
    logits_processor,
    tokenizer,
    input_ids,
    attention_mask,
    first_decoder_start,
    second_decoder_start,
    num_consecutive_decoders,
    is_causallm,
):
    input_ids_padded = input_ids
    attention_mask_padded = attention_mask
    position_ids_padded = gen_position_ids(input_ids_padded)

    logger.debug(f"The first call started")
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)
    host = tt_lib.device.GetHost()

    first_out = run_llama_split_inference(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=first_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=input_ids_padded,
        att_mask=attention_mask_padded,
        position_ids=position_ids_padded,
        half=1,
    )
    tt_lib.device.CloseDevice(device)
    logger.debug(f"The first call ended")

    # The second call -------------------------------------------------------
    logger.debug(f"The second call started")
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    tt_lib.device.SetDefaultDevice(device)

    # send input tensor from host to tt device
    tt_input = first_out

    tt_out = run_llama_split_inference(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        num_decoders_start=second_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_inputs=tt_input,
        att_mask=attention_mask_padded,
        position_ids=position_ids_padded,
        half=2,
        is_causallm=is_causallm,
    )
    logger.debug(f"The second call ended")

    # squeeze output
    tt_out = tt_out.squeeze(1)

    tt_lib.device.CloseDevice(device)
    device = None

    return tt_out


# parameters --------------------------------------------------
_tokenizer_name = "huggyllama/llama-7b"
_llama_model_name = "huggyllama/llama-7b"
# base url from the model state dictionary
_base_url = "model.layers"
_max_position_embeddings = 2048
_is_causallm = False

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

# promp = """Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis.
# They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.
# Mention the large language model based product mentioned in the paragraph above:"""
promp = "I believe the meaning of life is to"


@pytest.mark.parametrize(
    "PERF_CNT, prompt, pcc",
    ((1, promp, 0.9),),
)
def test_llama_pcc(PERF_CNT, prompt, pcc):
    # set parameters =================================================================
    tokenizer_name = _tokenizer_name
    llama_model_name = _llama_model_name
    is_causallm = _is_causallm
    base_url = _base_url
    max_position_embeddings = _max_position_embeddings

    # how many decoders to use
    first_decoder_start = _first_decoder_start
    second_decoder_start = _second_decoder_start
    num_consecutive_decoders = _num_consecutive_decoders

    # load llama pytorch model ================================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(
        llama_model_name
    )

    hugging_face_reference_model.eval()
    # get configurations
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    profiler.enable()

    # generate real input =====================================================
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    logits_processor = get_logits_processor(
        input_ids, hugging_face_reference_model.config
    )

    is_input_padded = True
    input_ids_padded, attention_mask_padded, position_ids_padded = prepare_llama_input(
        prompt, tokenizer, configuration, is_input_padded
    )

    # PyTorch output ===========================================================
    hugging_face_reference_model = hugging_face_reference_model.get_decoder()
    profiler.start("\nExec time of reference model")
    pytorch_out = hugging_face_reference_model(
        input_ids=input_ids_padded,
        attention_mask=attention_mask_padded,
        position_ids=position_ids_padded,
    )
    profiler.end("\nExec time of reference model")
    pytorch_out = pytorch_out.last_hidden_state

    # TT output: call forward() function several times ========================
    profiler.start("\nExecution time of tt_llama first run")
    tt_out = call_tt_llama_forward_func(
        configuration,
        state_dict,
        base_url,
        max_position_embeddings,
        logits_processor,
        tokenizer,
        input_ids_padded,
        attention_mask_padded,
        first_decoder_start,
        second_decoder_start,
        num_consecutive_decoders,
        is_causallm,
    )
    profiler.end("\nExecution time of tt_llama first run")

    enable_compile_cache()

    logger.info(f"\nRunning the tt_llama model for {PERF_CNT} iterations . . . ")

    for i in range(PERF_CNT):
        profiler.start("\nAverage execution time of tt_llama model")
        tt_out = call_tt_llama_forward_func(
            configuration,
            state_dict,
            base_url,
            max_position_embeddings,
            logits_processor,
            tokenizer,
            input_ids_padded,
            attention_mask_padded,
            first_decoder_start,
            second_decoder_start,
            num_consecutive_decoders,
            is_causallm,
        )
        profiler.end("\nAverage execution time of tt_llama model")

    profiler.print()

    # check outputs ================================================================
    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"{pcc_value}")

    if does_pass:
        logger.info("Llama Model Passed!")
    else:
        logger.warning("Llama Model Failed!")
        assert does_pass, f"PCC value ({pcc_value}) is lower than {pcc}."

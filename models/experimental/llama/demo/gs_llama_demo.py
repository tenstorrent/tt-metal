# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger

from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.experimental.llama.llama_utils import (
    pad_input_32_left,
    get_next_llama_output_token,
    gen_position_ids,
    get_logits_processor,
)

from models.experimental.llama.tt.llama import llama_first_half, llama_second_half


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
):
    if half == 1:
        logger.debug("First pass through TT model")
        tt_llama_model = llama_first_half(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids)
    else:
        logger.debug("Second pass through TT model")
        tt_llama_model = llama_second_half(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(input_ids=x_inputs, attention_mask=att_mask, position_ids=position_ids)

    # returned type from the model is tuple
    tt_output = tt_to_torch_tensor(tt_out[0])
    return tt_output


def call_tt_llama_forward_func(
    configuration,
    state_dict,
    base_url,
    max_position_embeddings,
    initial_prompt,
    logits_processor,
    tokenizer,
    input_ids,
    attention_mask,
    first_decoder_start,
    second_decoder_start,
    num_consecutive_decoders,
    num_words=2,
):
    text = initial_prompt
    for i in range(num_words):
        # pad input tensors
        input_ids_padded = pad_input_32_left(input_ids, configuration.pad_token_id)
        attention_mask_padded = pad_input_32_left(attention_mask, configuration.pad_token_id)
        position_ids_padded = gen_position_ids(input_ids_padded)

        logger.debug(f"The first call started: loop {i+1}")
        device = ttnn.open_device(0)
        ttnn.SetDefaultDevice(device)

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
        ttnn.close_device(device)
        logger.debug(f"The first call ended: loop {i+1}")

        # The second call -------------------------------------------------------
        logger.debug(f"The second call started: loop {i+1}")
        device = ttnn.open_device(0)
        ttnn.SetDefaultDevice(device)

        # send input tensor from host to tt device
        tt_input = torch_to_tt_tensor_rm(first_out, device)

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
        )
        logger.debug(f"The second call ended: loop {i+1}")

        # squeeze output
        tt_out = tt_out.squeeze(1)

        # Get next token
        next_tokens = get_next_llama_output_token(logits_processor, input_ids_padded, tt_out, i, "Tenstorrent")

        # save output words
        s = tokenizer.decode(next_tokens.item(), skip_special_tokens=True)
        logger.debug(f"TT {i+1}-th generated word: {s}")
        text = text + " " + s

        # update input ids
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.full((1, 1), 1)], dim=-1)
        position_ids = gen_position_ids(input_ids)

        ttnn.close_device(device)
        device = None

    logger.debug(f"All TT generated tokens: {text}")
    return input_ids


# parameters --------------------------------------------------
_tokenizer_name = "huggyllama/llama-7b"
_llama_model_name = "huggyllama/llama-7b"
# base url from the model state dictionary
_base_url = "model.layers"
_max_position_embeddings = 2048

# how many decoders to use
# number of decoders to be stacked started from the selected id in the original llama model
# e.g. stack 16 consecutive decoders
_num_consecutive_decoders = 16

# decoder id from which decoder stacking starts (the first half of the model)
# e.g. start from 0 add use 3 decoders (0, 1, and 2)
_first_decoder_start = 0

# decoder id from which decoder stacking starts (the second half of the model)
# e.g. start from 16 add use 3 decoders (16, 17, and 18)
_second_decoder_start = _num_consecutive_decoders
# parameters --------------------------------------------------

# promp = """Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis.
# They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.
# Mention the large language model based product mentioned in the paragraph above:"""
promp = "I believe the meaning of life is to"


@pytest.mark.parametrize(
    "prompt, num_words",
    ((promp, 1),),
)
def test_gs_demo(prompt, num_words):
    # set parameters =================================================================
    tokenizer_name = _tokenizer_name
    llama_model_name = _llama_model_name

    base_url = _base_url
    max_position_embeddings = _max_position_embeddings

    # how many decoders to use
    first_decoder_start = _first_decoder_start
    second_decoder_start = _second_decoder_start
    num_consecutive_decoders = _num_consecutive_decoders

    # load llama pytorch model ================================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(llama_model_name)

    hugging_face_reference_model.eval()
    # get configurations
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # generate real input =====================================================
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    logger.info(f"Initial prompt: {prompt}")
    logger.info(f"Initial prompt ids: {input_ids}")

    # get position_ids values
    seq_length = input_ids.shape[1]
    position_ids = gen_position_ids(input_ids)

    logits_processor = get_logits_processor(input_ids, hugging_face_reference_model.config)

    # TT output: call forward() function several times ========================
    tt_generated_ids = call_tt_llama_forward_func(
        configuration,
        state_dict,
        base_url,
        max_position_embeddings,
        prompt,
        logits_processor,
        tokenizer,
        input_ids,
        attention_mask,
        first_decoder_start,
        second_decoder_start,
        num_consecutive_decoders,
        num_words,
    )

    # decode output with tokenizer
    tt_generated_text = tokenizer.batch_decode(
        tt_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    logger.info(f"Tenstorrent generated text: {tt_generated_text}")

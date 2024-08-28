# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch import nn
import ttnn
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.llama.llama_utils import (
    prepare_llama_input,
    gen_position_ids,
)
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_allclose_and_pcc,
    comp_pcc,
)
from models.experimental.llama.tt.cpu_stacked_decoders import PytorchLlamaDecoderModelStacked
from models.experimental.llama.tt.llama_stacked_decoders import TtLlamaDecoderModelStacked


def tt_llama_second_half_decoders(
    configuration,
    state_dict,
    base_url,
    max_position_embeddings,
    inputs,
    second_decoder_start,
    num_consecutive_decoders,
):
    position_ids_padded = gen_position_ids(inputs)

    device = ttnn.open_device(0)
    ttnn.SetDefaultDevice(device)
    tt_inputs = torch_to_tt_tensor_rm(inputs, device)

    logger.debug(f"The call of the second half started")

    tt_llama_model = TtLlamaDecoderModelStacked(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        second_decoder_start,
        num_consecutive_decoders,
    )
    second_out = tt_llama_model(x=tt_inputs, y=position_ids_padded)

    # returned type from the model is tuple
    second_out = tt_to_torch_tensor(second_out)
    logger.debug(f"The call of the second half ended")

    ttnn.close_device(device)

    return second_out


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
_second_decoder_start = 16

# parameters --------------------------------------------------

# prompt = """Author-contribution statements and acknowledgements in research papers should state clearly and specifically whether, and to what extent, the authors used AI technologies such as ChatGPT in the preparation of their manuscript and analysis.
# They should also indicate which LLMs were used. This will alert editors and reviewers to scrutinize manuscripts more carefully for potential biases, inaccuracies and improper source crediting. Likewise, scientific journals should be transparent about their use of LLMs, for example when selecting submitted manuscripts.
# Mention the large language model based product mentioned in the paragraph above:"""
prompt = "I believe the meaning of life is to"


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_llama_second_half(pcc, model_location_generator, reset_seeds):
    # set parameters =================================================================
    tokenizer_name = _tokenizer_name
    llama_model_name = _llama_model_name
    base_url = _base_url
    max_position_embeddings = _max_position_embeddings

    input_loc = model_location_generator("llama/LlamaForCausalLM/inputs/first_half_output.pt")

    # how many decoders to use
    second_decoder_start = _second_decoder_start
    num_consecutive_decoders = _num_consecutive_decoders

    decoder_stack_list = [i + 16 for i in range(num_consecutive_decoders)]

    # load llama pytorch model ================================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(llama_model_name)

    hugging_face_reference_model.eval()
    # get configurations
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # generate real input =====================================================
    real_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = real_inputs.input_ids

    is_input_padded = True
    input_ids_padded, _, position_ids_padded = prepare_llama_input(prompt, tokenizer, configuration, is_input_padded)

    second_half_input = torch.load(input_loc)
    second_half_input = second_half_input.squeeze(1)
    # TT output: call forward() function several times ========================
    with torch.no_grad():
        # call huggingface model
        pt_llama_second_half = PytorchLlamaDecoderModelStacked(hugging_face_reference_model, decoder_stack_list)
        pt_llama_second_half.eval()
        pytorch_out = pt_llama_second_half(x=second_half_input, y=position_ids_padded)

        tt_out = tt_llama_second_half_decoders(
            configuration,
            state_dict,
            base_url,
            max_position_embeddings,
            second_half_input,
            second_decoder_start,
            num_consecutive_decoders,
        )
        tt_out = tt_out.squeeze(1)

    # check outputs =========================================================================
    _, pcc_output = comp_allclose_and_pcc(pytorch_out, tt_out, pcc)
    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output {pcc_output}")

    if does_pass:
        logger.info("Test for SecondHalf of Llama Model passed!")
    else:
        logger.warning("Test for SecondHalf of Llama Model failed!")
        assert does_pass, f"PCC value is lower than {pcc}"

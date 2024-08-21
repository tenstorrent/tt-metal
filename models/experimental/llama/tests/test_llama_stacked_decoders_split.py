# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import ttnn

from transformers import AutoTokenizer, AutoModelForCausalLM

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_pcc,
    comp_allclose_and_pcc,
)
from models.experimental.llama.llama_utils import gen_position_ids

from models.experimental.llama.tt.tt_stacked_decoders import TtLlamaDecoderModelStacked
from models.experimental.llama.tt.cpu_stacked_decoders import (
    PytorchLlamaDecoderModelStacked,
)


def run_test_llama_decoder_split_inference(
    device,
    state_dict,
    configuration,
    base_url,
    max_position_embeddings,
    num_decoders_start,
    num_decoders,
    x_input=None,
    position_ids=None,
    half=1,
    has_layer_norm=False,
    is_causal=False,
):
    if half == 1:
        logger.info(f"First half of the model execution")
        tt_llama_model = TtLlamaDecoderModelStacked(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(x=x_input, y=position_ids)
    else:
        logger.info(f"Second half of the model execution")
        tt_llama_model = TtLlamaDecoderModelStacked(
            device,
            state_dict,
            base_url,
            max_position_embeddings,
            configuration,
            num_decoders_start,
            num_decoders,
        )
        tt_out = tt_llama_model(
            x=x_input,
            y=position_ids,
            half=2,
            has_layer_norm=has_layer_norm,
            is_causal=is_causal,
        )

    tt_out = tt_to_torch_tensor(tt_out)

    return tt_out


# parameters --------------------------------------------------
_tokenizer_name = "huggyllama/llama-7b"
_llama_model_name = "huggyllama/llama-7b"
# base url from the model state dictionary
_base_url = "model.layers"
_batch = 1
_seq_len = 32
_max_position_embeddings = 2048
# is causallm llama model (it has additional linear layer at the end)
_on_weka = False

# how many decoders to use
# number of decoders to be stacked started from the selected id in the original llama model
# e.g. stack 16 consecutive decoders
_num_consecutive_decoders = 4

# decoder id from which decoder stacking starts (the first half of the model)
# e.g. start from 0 add use 3 decoders (0, 1, and 2)
_first_decoder_start = 0

# decoder id from which decoder stacking starts (the second half of the model)
# e.g. start from 16 add use 3 decoders (16, 17, and 18)
_second_decoder_start = _num_consecutive_decoders

# has_layer_norm - add norm layer after stacked decoders
# is_causal - add linear layer after decoders
# parameters --------------------------------------------------


@pytest.mark.parametrize(
    "pcc, has_layer_norm, is_causal",
    (
        (
            0.96,
            False,
            False,
        ),
    ),
)
def test_llama_decoder_split_inference(pcc, has_layer_norm, is_causal, reset_seeds):
    # set parameters ================================================================
    model_version = _llama_model_name
    tokenizer_version = _tokenizer_name
    base_url = _base_url
    batch = _batch
    seq_len = _seq_len
    max_position_embeddings = _max_position_embeddings
    on_weka = _on_weka
    first_decoder_start = _first_decoder_start
    second_decoder_start = _second_decoder_start
    num_consecutive_decoders = _num_consecutive_decoders

    # Set number
    decoder_stack_list = [i for i in range(2 * num_consecutive_decoders + 1)]

    # Prepare input =============================================================
    llama_input = (torch.rand(batch, seq_len, 4096) * 2) - 1

    # get positions_ids values
    position_ids = gen_position_ids(llama_input)

    # Load hugging face model ========================================================
    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # PyTorch output ==================================================================
    pytorch_LlamaDecoder_model = PytorchLlamaDecoderModelStacked(hugging_face_reference_model, decoder_stack_list)
    pytorch_LlamaDecoder_model.eval()

    # get output
    pytorch_out = pytorch_LlamaDecoder_model(x=llama_input, y=position_ids, is_causal=is_causal)

    # TT hardware execution ============================================================
    # The first call --------------------------
    device = ttnn.open_device(0)
    ttnn.SetDefaultDevice(device)

    # prepare input for TT hardware
    tt_llama_input = llama_input.unsqueeze(1)
    tt_llama_input = torch_to_tt_tensor_rm(tt_llama_input, device)

    first_out = run_test_llama_decoder_split_inference(
        device,
        state_dict=state_dict,
        configuration=configuration,
        base_url=base_url,
        max_position_embeddings=max_position_embeddings,
        num_decoders_start=first_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_input=tt_llama_input,
        position_ids=position_ids,
        half=1,
    )

    ttnn.close_device(device)

    # The second call -------------------------------------------------------
    device = ttnn.open_device(0)
    ttnn.SetDefaultDevice(device)

    # send input tensor from host to tt device
    tt_input = torch_to_tt_tensor_rm(first_out, device)

    tt_out = run_test_llama_decoder_split_inference(
        device,
        state_dict=state_dict,
        configuration=configuration,
        base_url=base_url,
        max_position_embeddings=max_position_embeddings,
        num_decoders_start=second_decoder_start,
        num_decoders=num_consecutive_decoders,
        x_input=tt_input,
        position_ids=position_ids,
        half=2,
        has_layer_norm=has_layer_norm,
        is_causal=is_causal,
    )

    ttnn.close_device(device)

    # squeeze output
    tt_out = tt_out.squeeze(1)

    logger.debug(f"Pytorch output shape: {pytorch_out.shape}")
    logger.debug(f"Tenstorrent output shape: {tt_out.shape}")

    # check outputs ---------------------------------------------------------
    _, pcc_output = comp_allclose_and_pcc(pytorch_out, tt_out, pcc)
    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output {pcc_output}")

    if does_pass:
        logger.info("Stacked Decoders test Passed!")
    else:
        logger.warning("Stacked Decoders test Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"

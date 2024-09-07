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


from models.experimental.llama.tt.llama_stacked_decoders import TtLlamaDecoderModelStacked
from models.experimental.llama.tt.cpu_stacked_decoders import (
    PytorchLlamaDecoderModelStacked,
)


def run_test_llama_decoder_inference(
    device,
    llama_input,
    model_version,
    tokenizer_version,
    base_url,
    batch,
    seq_len,
    max_position_embeddings,
    num_decoders,
    on_weka,
    pcc,
):
    # stack decoders
    start = 0
    decoder_stack_list = [i for i in range(num_decoders + 1)]

    # get positions_ids values
    position_ids = gen_position_ids(llama_input)

    # Load Pytorch model ===================================================================
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_version)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_version, torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # PyTorch output =========================================================================
    pytorch_LlamaDecoder_model = PytorchLlamaDecoderModelStacked(hugging_face_reference_model, decoder_stack_list)
    logger.info(f"inputs: {llama_input}")
    logger.info(f"shape: {llama_input.shape}")
    logger.info(f"positions: {position_ids}")
    logger.info(f"shape: {position_ids.shape}")
    pytorch_LlamaDecoder_model.eval()
    pytorch_out = pytorch_LlamaDecoder_model(x=llama_input, y=position_ids)

    # TT hardware execution =================================================================
    tt_llama_input = llama_input.unsqueeze(1)
    tt_llama_input = torch_to_tt_tensor_rm(tt_llama_input, device)

    tt_LlamaDecoder_model = TtLlamaDecoderModelStacked(
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        configuration,
        start,
        num_decoders,
    )

    tt_out = tt_LlamaDecoder_model(x=tt_llama_input, y=position_ids)

    # transform to PyTorch tensor
    tt_out = tt_to_torch_tensor(tt_out)
    tt_out = tt_out.squeeze(1)

    # check outputs =========================================================================
    _, pcc_output = comp_allclose_and_pcc(pytorch_out, tt_out, pcc)
    does_pass, pcc_value = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output {pcc_output}")

    if does_pass:
        logger.info("Test for stacked decoders passed!")
    else:
        logger.warning("Test for stacked decoders failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


# parameters --------------------------------------------------
_tokenizer_name = "huggyllama/llama-7b"
_llama_model_name = "huggyllama/llama-7b"
# base url from the model state dictionary
_base_url = "model.layers"
_batch = 1
_seq_len = 32
_max_position_embeddings = 2048
_on_weka = False
_num_decoders = 4
# num_decoders - number of consecutive decoders
# parameters --------------------------------------------------


@pytest.mark.parametrize(
    "pcc",
    ((0.98),),
)
def test_llama_decoder_inference(pcc, reset_seeds):
    # set parameters ================================================================
    model_version = _llama_model_name
    tokenizer_version = _tokenizer_name
    base_url = _base_url
    batch = _batch
    seq_len = _seq_len
    max_position_embeddings = _max_position_embeddings
    on_weka = _on_weka
    num_decoders = _num_decoders

    # Prepare input ========================================================================
    llama_input = (torch.rand(batch, seq_len, 4096) * 2) - 1

    # Initialize the device
    device = ttnn.open_device(0)
    ttnn.SetDefaultDevice(device)

    run_test_llama_decoder_inference(
        device,
        llama_input,
        model_version,
        tokenizer_version,
        base_url,
        batch,
        seq_len,
        max_position_embeddings,
        num_decoders,
        on_weka,
        pcc,
    )
    ttnn.close_device(device)

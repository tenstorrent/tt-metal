# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
import ttnn
from loguru import logger

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_pcc,
    comp_allclose_and_pcc,
)
from models.experimental.llama.tt.llama_decoder import TtLlamaDecoderLayer


class PytorchLlamaDecoderModel(torch.nn.Module):
    def __init__(self, hf_reference_model, decoder_id):
        super().__init__()
        self.decoder = hf_reference_model.model.layers[decoder_id]

        # Disable dropout
        self.decoder.eval()

    def forward(self, x, y):
        result = self.decoder(hidden_states=x, position_ids=y)[0]
        return result


def run_test_LlamaDecoder_inference(
    device, model_version, tokenizer_version, batch, seq_len, decoder_idx, on_weka, pcc
):
    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input ========================================================================
    torch.manual_seed(0)
    llama_input = (torch.rand(batch, seq_len, 4096) * 2) - 1
    base_url = "model.layers"
    decoder_id = decoder_idx
    # max_position_embeddings parameter should be in the config file, but the used pretrained model doesn't consist this parameter
    max_position_embeddings = 2048

    # get positions_ids values
    past_key_values_length = 0
    seq_length = llama_input.shape[1]

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=None,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # PyTorch output =======================================================================
    pytorch_LlamaDecoder_model = PytorchLlamaDecoderModel(hugging_face_reference_model, decoder_id)
    pytorch_LlamaDecoder_model.eval()
    pytorch_out = pytorch_LlamaDecoder_model(x=llama_input, y=position_ids)

    # TT hardware execution =================================================================
    tt_llama_input = llama_input.unsqueeze(1)
    tt_llama_input = torch_to_tt_tensor_rm(tt_llama_input, device)

    # get TT Attention module
    tt_LlamaDecoder_model = TtLlamaDecoderLayer(
        device,
        state_dict,
        base_url,
        decoder_id,
        max_position_embeddings,
        configuration,
    )
    tt_out = tt_LlamaDecoder_model(hidden_states=tt_llama_input, position_ids=position_ids)
    # transform to PyTorch tensor
    # take only hidden_states tensor if tuple is obtained
    tt_out = tt_to_torch_tensor(tt_out[0])
    tt_out = tt_out.squeeze(1)

    # check outputs ----------------------------------------------------------------------
    _, pcc_output = comp_allclose_and_pcc(pytorch_out, tt_out, pcc)
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)

    logger.info(f"Output {pcc_output}")

    if does_pass:
        logger.info("Llama Decoder output Passed!")
    else:
        logger.warning("Llama Decoder output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, decoder_id, on_weka, pcc",
    (
        (
            "huggyllama/llama-7b",
            "huggyllama/llama-7b",
            1,
            128,
            5,
            False,
            0.98,
        ),
    ),
)
def test_LlamaDecoder_inference(model_version, tokenizer_version, batch, seq_len, decoder_id, on_weka, pcc):
    # Initialize the device
    device = ttnn.open_device(0)
    ttnn.SetDefaultDevice(device)

    run_test_LlamaDecoder_inference(
        device,
        model_version,
        tokenizer_version,
        batch,
        seq_len,
        decoder_id,
        on_weka,
        pcc,
    )
    ttnn.close_device(device)

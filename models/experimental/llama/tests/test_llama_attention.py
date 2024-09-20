# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn
from loguru import logger
import ttnn

from transformers import AutoTokenizer, AutoModelForCausalLM
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
    comp_pcc,
    comp_allclose_and_pcc,
)
from models.llama.tt.llama_attention import TtLlamaAttention


class PytorchLlamaAttentionModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attention = hf_reference_model.model.layers[layer_num].self_attn

        # Disable dropout
        self.attention.eval()

    def forward(self, x, y):
        result = self.attention(hidden_states=x, position_ids=y)[0]
        return result


def run_test_LlamaAttention_inference(device, model_version, tokenizer_version, batch, seq_len, on_weka, pcc):
    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hugging_face_reference_model.eval()

    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare inputs ========================================================================
    torch.manual_seed(0)
    # hidden states tensor: batch size is equal to 32, sequence length: 32
    attention_input = (torch.rand(batch, seq_len, 4096) * 2) - 1
    layer_num = 0
    base_url = "model.layers"
    # max_position_embeddings parameter should be in the config file, but the used pretrained model doesn't consist this parameter
    max_position_embeddings = 2048

    # get positions_ids values
    seq_length = attention_input.shape[1]
    past_key_values_length = 0

    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=None,
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    # PyTorch output =======================================================================
    pytorch_LlamaAttention_model = PytorchLlamaAttentionModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaAttention_model(x=attention_input, y=position_ids)

    # TT hardware execution =================================================================
    tt_attention_input = attention_input.unsqueeze(1)
    tt_attention_input = torch_to_tt_tensor_rm(tt_attention_input, device)

    # get TT Attention module
    tt_LlamaAttention_model = TtLlamaAttention(
        device,
        base_url,
        state_dict,
        layer_num,
        configuration.hidden_size,
        configuration.num_attention_heads,
        max_position_embeddings,
    )

    tt_out, attn_weights, past_key_value = tt_LlamaAttention_model(
        hidden_states=tt_attention_input, position_ids=position_ids
    )
    tt_out = tt_to_torch_tensor(tt_out).squeeze(1)

    # check outputs ----------------------------------------------------------------------
    _, pcc_output = comp_allclose_and_pcc(pytorch_out, tt_out, pcc)
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)

    logger.info(f"Output {pcc_output}")

    if does_pass:
        logger.info("Llama Attention output Passed!")
    else:
        logger.warning("Llama Attention output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, on_weka, pcc",
    (
        pytest.param(
            "huggyllama/llama-7b",
            "huggyllama/llama-7b",
            1,
            128,
            False,
            0.98,
        ),
    ),
)
def test_LlamaAttention_inference(model_version, tokenizer_version, batch, seq_len, on_weka, pcc):
    device = ttnn.open_device(0)
    ttnn.SetDefaultDevice(device)

    run_test_LlamaAttention_inference(device, model_version, tokenizer_version, batch, seq_len, on_weka, pcc)
    ttnn.close_device(device)

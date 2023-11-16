# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from torch import nn

from loguru import logger

from transformers import AutoTokenizer, AutoModelForCausalLM


from models.experimental.llama_old.llama_utils import *
from models.utility_functions import (
    comp_allclose,
    comp_pcc,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.experimental.llama_old.tt.llama_layer_norm import TtLlamaRMSNorm


class PytorchLlamaRMSNormModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.layer_norm = hf_reference_model.model.layers[layer_num].input_layernorm

        # Disable dropout
        self.layer_norm.eval()

    def forward(self, x):
        result = self.layer_norm(x)
        return result


def run_test_LlamaLayerNorm_inference(device, model_version, tokenizer_version, batch, seq_len, on_weka, pcc):
    model_name = model_version
    tokenizer_name = tokenizer_version

    # https://huggingface.co/baffo32/decapoda-research-llama-7B-hf
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    llama_layer_norm_input = (torch.rand(batch, 1, seq_len, 4096) * 2) - 1
    layer_num = 0

    # PyTorch output ---------------------------------------------------------------------
    pytorch_LlamaRMSNorm_model = PytorchLlamaRMSNormModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaRMSNorm_model(llama_layer_norm_input)
    logger.info(f"PyTorch output shape: {pytorch_out.shape}")

    # TT hardware execution --------------------------------------------------------------
    layer_position = "input_layernorm"
    base_url = "model.layers"
    tt_LlamaRMSNorm_model = TtLlamaRMSNorm(
        device,
        state_dict,
        base_url,
        layer_num,
        layer_position,
        configuration.hidden_size,
    )

    tt_layer_norm_input = torch_to_tt_tensor_rm(llama_layer_norm_input, device)

    # call model for input
    tt_out = tt_LlamaRMSNorm_model(tt_layer_norm_input)
    tt_out = tt_to_torch_tensor(tt_out)
    logger.info(f"TT output shape: {tt_out.shape}")

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Llama LayerNorm output Passed!")
    else:
        logger.warning("Llama LayerNorm output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, on_weka, pcc",
    (
        (
            "baffo32/decapoda-research-llama-7B-hf",
            "hf-internal-testing/llama-tokenizer",
            1,
            2048,
            False,
            0.98,
        ),
    ),
)
def test_LlamaLayerNorm_inference(model_version, tokenizer_version, batch, seq_len, on_weka, pcc, device):
    run_test_LlamaLayerNorm_inference(device, model_version, tokenizer_version, batch, seq_len, on_weka, pcc)

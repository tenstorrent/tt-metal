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
from models.experimental.llama.tt.llama_layer_norm import TtLlamaRMSNorm


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

    # check outputs ----------------------------------------------------------------------
    _, pcc_output = comp_allclose_and_pcc(pytorch_out, tt_out, pcc)
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)

    logger.info(f"Output {pcc_output}")
    logger.info(f"PCC OUTPUT {output_pcc}")

    if does_pass:
        logger.info("Llama LayerNorm output Passed!")
    else:
        logger.warning("Llama LayerNorm output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, on_weka, pcc",
    (
        (
            "huggyllama/llama-7b",
            "huggyllama/llama-7b",
            1,
            2048,
            False,
            0.98,
        ),
    ),
)
def test_LlamaLayerNorm_inference(model_version, tokenizer_version, batch, seq_len, on_weka, pcc):
    # Initialize the device
    device = ttnn.open_device(0)
    ttnn.SetDefaultDevice(device)

    run_test_LlamaLayerNorm_inference(device, model_version, tokenizer_version, batch, seq_len, on_weka, pcc)
    ttnn.close_device(device)

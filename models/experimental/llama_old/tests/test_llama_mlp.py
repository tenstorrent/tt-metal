# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from torch import nn


from transformers import AutoTokenizer, AutoModelForCausalLM


from models.experimental.llama_old.llama_utils import *
from models.utility_functions import (
    comp_allclose,
    comp_pcc,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.experimental.llama_old.tt.llama_mlp import TtLlamaMLP


class PytorchLlamaMLPModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.mlp = hf_reference_model.model.layers[layer_num].mlp

        # Disable dropout
        self.mlp.eval()

    def forward(self, x):
        result = self.mlp(x)
        return result


def run_test_LlamaMLP_inference(device, model_version, tokenizer_version, batch, seq_len, on_weka, pcc):
    model_name = model_version
    tokenizer_name = tokenizer_version

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    llama_mlp_input = (torch.rand(batch, 1, seq_len, 4096) * 2) - 1
    layer_num = 0
    base_url = "model.layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaMLP_model = PytorchLlamaMLPModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaMLP_model(llama_mlp_input)  # .unsqueeze(1)

    # TT hardware execution -------------------------------------------------------------
    tt_LlamaMLP_model = TtLlamaMLP(
        device,
        state_dict,
        base_url,
        layer_num,
        configuration.hidden_size,
        configuration.intermediate_size,
        configuration.hidden_act,
    )

    tt_mlp_input = torch_to_tt_tensor_rm(llama_mlp_input, device)

    tt_out = tt_LlamaMLP_model(tt_mlp_input)
    tt_out = tt_to_torch_tensor(tt_out)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Llama MLP output Passed!")
    else:
        logger.warning("Llama MLP output Failed!")
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
def test_LlamaMLP_inference(device, model_version, tokenizer_version, batch, seq_len, on_weka, pcc):
    run_test_LlamaMLP_inference(device, model_version, tokenizer_version, batch, seq_len, on_weka, pcc)

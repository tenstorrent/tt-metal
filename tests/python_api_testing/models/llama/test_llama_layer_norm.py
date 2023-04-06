import sys
from pathlib import Path
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
from abc import abstractmethod
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from libs import tt_lib as ttl

from transformers import T5Tokenizer, T5Model, AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict

from python_api_testing.fused_ops.linear import Linear as TtLinear
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, nearest_32, print_diff_argmax, tt2torch, tt2torch_rm
from python_api_testing.models.llama.llama_utils import *
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm


class PytorchLlamaRMSNormModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.layer_norm = hf_reference_model.model.layers[layer_num].input_layernorm

        # Disable dropout
        self.layer_norm.eval()

    def forward(self, x):
        result = self.layer_norm(x)
        return result


def run_test_LlamaLayerNorm_inference(device, host, model_version, tokenizer_version, batch, seq_len, on_weka, pcc):

    model_name = model_version
    tokenizer_name = tokenizer_version

    # https://huggingface.co/decapoda-research/llama-7b-hf
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
    layer_position = 'input_layernorm'
    base_url = 'model.layers'
    tt_LlamaRMSNorm_model = TtLlamaRMSNorm(device, state_dict, base_url, layer_num, layer_position, configuration.hidden_size)

    tt_layer_norm_input = torch2tt_tensor(llama_layer_norm_input, device)

    # call model for input
    tt_out = tt_LlamaRMSNorm_model(tt_layer_norm_input).to(host)
    tt_out = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))

    # check outputs ----------------------------------------------------------------------
    print(comp_allclose(pytorch_out, tt_out))
    print(comp_pcc(pytorch_out, tt_out))

    passing_pcc, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)

    assert passing_pcc, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, on_weka, pcc",
    (
        ("decapoda-research/llama-7b-hf", "hf-internal-testing/llama-tokenizer", 4, 2048, False, 0.98),
    ),
)
def test_LlamaLayerNorm_inference(model_version, tokenizer_version, batch, seq_len, on_weka, pcc):
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_test_LlamaLayerNorm_inference(
        device,
        host,
        model_version,
        tokenizer_version,
        batch,
        seq_len,
        on_weka,
        pcc
    )
    ttl.device.CloseDevice(device)

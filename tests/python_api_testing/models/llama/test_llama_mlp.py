import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import pytest
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from libs import tt_lib as ttl

from transformers import T5Tokenizer, T5Model, AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict

from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, nearest_32, print_diff_argmax, tt2torch, tt2torch_rm
from fused_ops.linear import Linear as TtLinear
from python_api_testing.models.llama.llama_utils import *
from sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.llama.llama_mlp import TtLlamaMLP


class PytorchLlamaMLPModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.mlp = hf_reference_model.model.layers[layer_num].mlp

        # Disable dropout
        self.mlp.eval()

    def forward(self, x):
        result = self.mlp(x)
        return result


def run_test_LlamaMLP_inference(device, host, model_version, tokenizer_version, batch, seq_len, on_weka, pcc):

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
    pytorch_out = pytorch_LlamaMLP_model(llama_mlp_input) # .unsqueeze(1)

    # TT hardware execution -------------------------------------------------------------
    tt_LlamaMLP_model = TtLlamaMLP(
        device,
        state_dict,
        base_url,
        layer_num,
        configuration.hidden_size,
        configuration.intermediate_size,
        configuration.hidden_act
    )

    tt_mlp_input = torch2tt_tensor(llama_mlp_input, device)

    tt_out = tt_LlamaMLP_model(tt_mlp_input).to(host)
    tt_out = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))

    # check outputs ----------------------------------------------------------------------
    print(comp_allclose(pytorch_out, tt_out))
    print(comp_pcc(pytorch_out, tt_out))

    passing_pcc, output_pcc = comp_pcc(pytorch_out, tt_out, 0.98)

    assert passing_pcc, "PCC value is lower than 0.98"


@pytest.mark.parametrize(
    "model_version, tokenizer_version, batch, seq_len, on_weka, pcc",
    (
        ("decapoda-research/llama-7b-hf", "hf-internal-testing/llama-tokenizer", 4, 2048, False, 0.98),
    ),
)
def test_LlamaMLP_inference(model_version, tokenizer_version, batch, seq_len, on_weka, pcc):
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_test_LlamaMLP_inference(
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

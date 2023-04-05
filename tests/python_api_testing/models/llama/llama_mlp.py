import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

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


class TtLlamaMLP(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.out_gate_proj = torch2tt_tensor(self.state_dict[f"{base_url}.{layer_num}.mlp.gate_proj.weight"], ttl.device.GetHost())
        self.out_down_proj = torch2tt_tensor(self.state_dict[f"{base_url}.{layer_num}.mlp.down_proj.weight"], ttl.device.GetHost())
        self.out_up_proj = torch2tt_tensor(self.state_dict[f"{base_url}.{layer_num}.mlp.up_proj.weight"], ttl.device.GetHost())

        self.gate_proj = TtLinear(in_features=self.hidden_size, out_features=self.intermediate_size, weight=self.out_gate_proj.data(), bias=None, device=self.device)
        self.down_proj = TtLinear(in_features=self.intermediate_size, out_features=self.hidden_size, weight=self.out_down_proj.data(), bias=None, device=self.device)
        self.up_proj = TtLinear(in_features=self.hidden_size, out_features=self.intermediate_size, weight=self.out_up_proj.data(), bias=None, device=self.device)

        if hidden_act == "silu": # $$ silu
            self.act_fn = ttl.tensor.sigmoid

    def forward(self, x):
        # gate proj
        gate = self.gate_proj(x)
        # apply silu activation function
        activation = self.act_fn(gate)
        gate = ttl.tensor.mul(gate, activation)
        # up proj
        up = self.up_proj(x)
        # product
        prod = ttl.tensor.mul(gate, up)
        # down
        hidden_states = self.down_proj(prod)
        # return TT Tensor
        return hidden_states


class PytorchLlamaMLPModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.mlp = hf_reference_model.model.layers[layer_num].mlp

        # Disable dropout
        self.mlp.eval()

    def forward(self, x):
        result = self.mlp(x)
        return result


def run_LlamaMLP_inference():
    tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    llama_mlp_input = (torch.rand(4, 1, 2048, 4096) * 2) - 1
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


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_LlamaMLP_inference()
    ttl.device.CloseDevice(device)

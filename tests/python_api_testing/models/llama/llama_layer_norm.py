import sys
from pathlib import Path
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

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


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class TtLlamaRMSNorm(nn.Module):
    def __init__(self, device, state_dict, base_url, layer_num, layer_position, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()

        self.device = device
        self.variance_epsilon = eps
        # get weights
        self.state_dict = state_dict
        # check if it is final norm layer
        if layer_num is not None:
            pytorch_weights = self.state_dict[f"{base_url}.{layer_num}.{layer_position}.weight"]
        else:
            pytorch_weights = self.state_dict[f"model.norm.weight"]

        pytorch_weights = pytorch_weights.repeat(1, 1, 32, 1)
        self.weight = torch2tt_tensor(pytorch_weights, self.device)

    def forward(self, hidden_states):
        # handle variance in PyTorch
        torch_hidden_states = tt2torch_tensor(hidden_states)
        variance = torch_hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        variance = variance.repeat(1, 1, 1, 32)
        tt_variance = torch2tt_tensor(variance, self.device)

        # Pytorch implementation for: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # handle constant variance_epsilon
        tt_variance_epsilon_const = ttl.tensor.Tensor(
            [self.variance_epsilon] + [0.0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.TILE,
            self.device
        )

        # Product 2: torch.rsqrt(variance + self.variance_epsilon)
        op_add = ttl.tensor.bcast(tt_variance, tt_variance_epsilon_const, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H)
        term_2 = ttl.tensor.recip(ttl.tensor.sqrt(op_add))

        # Product 1 * Product 2: hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = ttl.tensor.bcast(hidden_states, term_2, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.W)

        # weight * hidden_states
        result = ttl.tensor.bcast(hidden_states, self.weight, ttl.tensor.BcastOpMath.MUL, ttl.tensor.BcastOpDim.H)
        return result


class PytorchLlamaRMSNormModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.layer_norm = hf_reference_model.model.layers[layer_num].input_layernorm

        # Disable dropout
        self.layer_norm.eval()

    def forward(self, x):
        result = self.layer_norm(x)
        return result


def run_LlamaLayerNorm_inference():

    # https://huggingface.co/decapoda-research/llama-7b-hf
    tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    hugging_face_reference_model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", torch_dtype=torch.float32)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    llama_layer_norm_input = (torch.rand(4, 1, 2048, 4096) * 2) - 1
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
    tt_out1 = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))

    # check outputs ----------------------------------------------------------------------
    print(comp_allclose(pytorch_out, tt_out1))
    print(comp_pcc(pytorch_out, tt_out1))

    pcc_test = comp_pcc(pytorch_out, tt_out1, 0.98)

    assert pcc_test, "PCC value is lower than 0.98"


if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_LlamaLayerNorm_inference()
    ttl.device.CloseDevice(device)

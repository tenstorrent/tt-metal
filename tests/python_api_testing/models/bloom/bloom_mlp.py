from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from abc import abstractmethod
import torch
import math
from torch.nn import functional as F
from transformers import BloomForCausalLM

from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, nearest_32, print_diff_argmax, tt2torch, tt2torch_rm


from libs import tt_lib as ttm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
import numpy as np
import bloom_utils as bloom_utils

from fused_ops.linear import Linear as TtLinear

import dropout_add as dropout_add
import bloom_gelu_forward as bloom_gelu_forward

class TtBloomMLP(torch.nn.Module):
    def __init__(self, hugging_bloom_reference_model, hidden_dropout, hidden_size, training, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.training = training

        state_dict = hugging_bloom_reference_model.state_dict()

        tt_weight_mlp_h4h = tilize_to_list(pad_weight(state_dict[f"transformer.h.0.mlp.dense_h_to_4h.weight"]))
        tt_bias_mlp_h4h = tilize_to_list(pad_weight(state_dict[f"transformer.h.0.mlp.dense_h_to_4h.bias"]))
        self.dense_h_to_4h = TtLinear(hidden_size, 4 * hidden_size, tt_weight_mlp_h4h, tt_bias_mlp_h4h, device)

        self.gelu_impl = bloom_gelu_forward.tt_bloom_gelu_forward

        tt_weight_mlp_4hh = tilize_to_list(pad_weight(state_dict[f"transformer.h.0.mlp.dense_4h_to_h.weight"]))
        tt_bias_mlp_4hh = tilize_to_list(pad_weight(state_dict[f"transformer.h.0.mlp.dense_4h_to_h.bias"]))

        self.dense_4h_to_h = TtLinear(4*hidden_size, hidden_size, tt_weight_mlp_4hh, tt_bias_mlp_4hh, device)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, device) -> torch.Tensor:

        tt_hs = bloom_utils.torch2tt_tensor(hidden_states, device)


        tt_h4h = self.dense_h_to_4h(tt_hs)

        tt_hidden_states = self.gelu_impl(tt_h4h, device)

        tt_intermediate_output = self.dense_4h_to_h(tt_hidden_states)

        tt_res_temp = tilize_to_list(residual)

        tt_res = bloom_utils.torch2tt_tensor(residual, device)

        res = bloom_utils.tt2torch_tensor(tt_res)

        intermediate_output = bloom_utils.tt2torch_tensor(tt_intermediate_output)

        output = dropout_add.tt_dropout_add(intermediate_output, res, self.hidden_dropout, self.training, device)

        return output


class BloomMLP(torch.nn.Module):

    def __init__(self, state_dict, hidden_dropout, hidden_size, training):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.training = training



        self.dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = bloom_gelu_forward.bloom_gelu_forward
        self.dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)

        weight_mlp_h_to_4h = torch.nn.Parameter(torch.tensor(state_dict[f"transformer.h.0.mlp.dense_h_to_4h.weight"]))
        bias_mlp_h_to_4h = torch.nn.Parameter(torch.tensor(state_dict[f"transformer.h.0.mlp.dense_h_to_4h.bias"]))
        weight_mlp_4hh = torch.nn.Parameter(torch.tensor(state_dict[f"transformer.h.0.mlp.dense_4h_to_h.weight"]))
        bias_mlp_4hh = torch.nn.Parameter(torch.tensor(state_dict[f"transformer.h.0.mlp.dense_4h_to_h.bias"]))

        self.dense_4h_to_h.weight = weight_mlp_4hh
        self.dense_4h_to_h.bias = bias_mlp_4hh
        self.dense_h_to_4h.weight = weight_mlp_h_to_4h
        self.dense_h_to_4h.bias = bias_mlp_h_to_4h

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        intermediate_output = self.dense_4h_to_h(hidden_states)

        output = dropout_add.dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output


def run_bloom_mlp_inference(device):

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    state_dict = hugging_bloom_reference_model.state_dict()
    # Prepare input
    torch.manual_seed(0)

    test_in = torch.rand(1, 1, 4096, 1024)
    res = torch.rand(1, 1, 4096, 1024)

    tt_mlp = TtBloomMLP(hugging_bloom_reference_model, 0.0, 1024, False, device)

    tt_out =  tt_mlp.forward(test_in, res, device)

    pt_mlp = BloomMLP(hugging_bloom_reference_model.state_dict(), 0.0, 1024, False)

    pt_out = pt_mlp.forward(test_in, res)

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print(comp_allclose(pt_out, tt_out_converted))
    print(comp_pcc(pt_out, tt_out_converted))


if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_mlp_inference(device)
    ttm.device.CloseDevice(device)

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
from libs import tt_lib as ttm
from loguru import logger

from transformers import T5Model
from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.t5.t5_utils import tt2torch_tensor, torch2tt_tensor, read_model_config, print_corr_coef


# class T5LayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
#         # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
#         # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
#         # half-precision inputs is done in fp32

#         variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)

#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         t = torch.rsqrt(variance + self.variance_epsilon)

#         # convert into half-precision if necessary
#         if self.weight.dtype in [torch.float16, torch.bfloat16]:
#             hidden_states = hidden_states.to(self.weight.dtype)
#         return self.weight * hidden_states


class TtT5LayerNorm(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.variance_epsilon = config["layer_norm_epsilon"]
        self.device = device

        # get weights
        pytorch_weights = state_dict[f"{base_address}.weight"]
        pytorch_weights = pytorch_weights.repeat(1, 1, 32, 1)

        self.weight = torch2tt_tensor(pytorch_weights, device)

    def forward(self, hidden_states):
        # handle variance
        torch_hidden_states = tt2torch_tensor(hidden_states)
        variance = torch_hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        variance = variance.repeat(1, 1, 1, 32)
        tt_variance = torch2tt_tensor(variance, self.device)

        # Pytorch implementation: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # hadle constant variance_epsilon
        tt_variance_epsilon_const = ttm.tensor.Tensor(
            [self.variance_epsilon] + [0.0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.TILE,
            self.device
        )

        # Product 2: torch.rsqrt(variance + self.variance_epsilon)
        op_add = ttm.tensor.bcast(tt_variance, tt_variance_epsilon_const, ttm.tensor.BcastOpMath.ADD, ttm.tensor.BcastOpDim.H)
        term_2 = ttm.tensor.recip(ttm.tensor.sqrt(op_add))

        # Product 1 * Product 2: hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = ttm.tensor.bcast(hidden_states, term_2, ttm.tensor.BcastOpMath.MUL, ttm.tensor.BcastOpDim.W)

        # weight * hidden_states
        result = ttm.tensor.bcast(hidden_states, self.weight, ttm.tensor.BcastOpMath.MUL, ttm.tensor.BcastOpDim.H)
        return result


def test_T5LayerNorm_inference(device):
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    model_json_config = "tests/python_api_testing/models/t5/t5-small.json"
    config = read_model_config(model_json_config)

    # Module to test
    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[0].layer[1].layer_norm
        base_address = f"decoder.block.0.layer.1.layer_norm"
    else:
        hf_reference_module = hf_reference_model.encoder.block[0].layer[1].layer_norm
        base_address = f"encoder.block.0.layer.1.layer_norm"

    # Prepare input
    torch.manual_seed(0)
    t5_layer_norm_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(t5_layer_norm_input)[0].unsqueeze(1)
    tt_T5LayerNorm_model = TtT5LayerNorm(config, hf_reference_model.state_dict(), base_address, device)

    # TT hardware execution
    tt_layer_norm_input = torch2tt_tensor(t5_layer_norm_input, device)

    tt_out = tt_T5LayerNorm_model(tt_layer_norm_input)
    tt_out = tt2torch_tensor(tt_out)

    print(pt_out[0, 0, 1:10, 1:10])
    print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("test_T5LayerNorm_inference Passed!")
    else:
        logger.warning("test_T5LayerNorm_inference Failed!")


if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    test_T5LayerNorm_inference(device)
    ttm.device.CloseDevice(device)

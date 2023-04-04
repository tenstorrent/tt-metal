from abc import abstractmethod
import torch
from transformers import BloomForQuestionAnswering
import math
from torch.nn import functional as F

from pymetal import ttmetal as ttm
from python_api_testing.models.bert.embeddings import PytorchEmbeddings
from python_api_testing.models.bert.mha import TtMultiHeadAttentionModel
from python_api_testing.models.bert.ffn import TtFeedForwardModel
from python_api_testing.models.bert.bert_encoder import TtBertEncoder
from python_api_testing.fused_ops.linear import Linear as ttLinear
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, print_diff_argmax
from utility_functions import enable_binary_cache, enable_compile_cache, get_compile_cache_enabled, get_binary_cache_enabled
import numpy as np

def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def tt_dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> ttm.tensor.Tensor:


    tt_res = tilize_to_list(pad_activation(residual))
    tt_res = ttm.tensor.Tensor(tt_res, [1,1,64,64], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

    out = F.dropout(x, p=prob, training=training)
    tt_out = tilize_to_list(pad_activation(out))
    tt_out = ttm.tensor.Tensor(tt_out, [1,1,64,64], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

    total = ttm.tensor.add(tt_res, tt_out)

    return total

def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

def tt_bloom_gelu_forward(x, d1, d2, d3, d4):
    z = x

    k1 = torch.full((d1, d2, d3, d4), 0.5)
    k1 = tilize_to_list(k1)
    k1_dev = ttm.tensor.Tensor(k1, [d1, d2, d3, d4], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    k2 = torch.full((d1, d2, d3, d4), 0.044715)
    k2 = tilize_to_list(k2)
    k2_dev = ttm.tensor.Tensor(k2, [d1, d2, d3, d4], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    k3 = torch.full((d1, d2, d3, d4), 0.79788456)
    k3 = tilize_to_list(k3)
    k3_dev = ttm.tensor.Tensor(k3, [d1, d2, d3, d4], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    #0.5*x
    factor1 = ttm.tensor.mul(k1_dev, z) # exp(z)
    #x*x
    pow2 = ttm.tensor.mul(z, z)
    #(x + 0.044715 * torch.pow(x, 3)))
    #torch.pow(x, 3))
    pow3 = ttm.tensor.mul(pow2, z)
    factor3 = ttm.tensor.mul(k2_dev, pow3)
    #(x + 0.044715 * torch.pow(x, 3)))
    factor3 = ttm.tensor.add(factor3, z)

    sumtanh = ttm.tensor.mul(k3_dev, factor3)

    tanh = ttm.tensor.tanh(sumtanh)

    k4 = torch.full((d1, d2, d3, d4), 1)
    k4 = tilize_to_list(k4)
    k4_dev = ttm.tensor.Tensor(k4, [d1, d2, d3, d4], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    total = ttm.tensor.add(k4_dev, tanh)

    output = ttm.tensor.mul(factor1, total)

    return output



class ttBloomMLP(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        hidden_size = 64

        self.pretraining_tp = 1
        self.slow_but_exact = False

        tt_weight_mlp_h4h = tilize_to_list(pad_weight(sd[f"transformer.h.0.mlp.dense_h_to_4h.bias"]))
        tt_bias_mlp_h4h = tilize_to_list(pad_weight(sd[f"transformer.h.0.mlp.dense_h_to_4h.bias"]))

        self.aux_dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size)
        self.aux_dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)

        self.dense_h_to_4h = ttLinear(hidden_size, 4 * hidden_size, tt_weight_mlp_h4h, tt_bias_mlp_h4h, device)

        self.gelu_impl = tt_bloom_gelu_forward

        tt_weight_mlp_4hh = tilize_to_list(pad_weight(sd[f"transformer.h.0.mlp.dense_4h_to_h.bias"]))
        tt_bias_mlp_4hh = tilize_to_list(pad_weight(sd[f"transformer.h.0.mlp.dense_4h_to_h.bias"]))

        self.dense_4h_to_h = ttLinear(4*hidden_size, hidden_size, tt_weight_mlp_4hh, tt_bias_mlp_4hh, device)

        self.hidden_dropout = 0.0
        self.training = False

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:

        tt_hidden_states_input = tilize_to_list(pad_activation(hidden_states))
        tt_hs = ttm.tensor.Tensor(tt_hidden_states_input, hidden_states.shape, ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

        tt_h4h = self.dense_h_to_4h(tt_hs)

        tt_hidden_states = self.gelu_impl(tt_h4h, 1, 1, 64, 256)

        tt_intermediate_output = self.dense_4h_to_h(tt_hidden_states)

        tt_res_temp = tilize_to_list(residual)

        tt_res = ttm.tensor.Tensor(tt_res_temp, tt_intermediate_output.shape(), ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

        res = tt_res.to(host).data()

        tt_got_back_res = torch.Tensor(res).reshape((1,1,64,64))
        tt_got_back_res = untilize(tt_got_back_res)

        intermediate_output =tt_intermediate_output.to(host).data()

        tt_got_back_intermediate_output = torch.Tensor(intermediate_output).reshape((1,1,64,64))
        tt_got_back_intermediate_output = untilize(tt_got_back_intermediate_output)

        output = tt_dropout_add(tt_got_back_intermediate_output, tt_got_back_res, self.hidden_dropout, self.training)

        return output



class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp

class BloomGelu(torch.nn.Module):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)

class BloomMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 64

        self.pretraining_tp = False
        self.slow_but_exact = False
        self.dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = 0.0

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        intermediate_output = self.dense_4h_to_h(hidden_states)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output


def run_bloom_mlp_inference():

    hugging_bloom_reference_model = BloomForQuestionAnswering.from_pretrained("bigscience/bloom-560m", torchscript=False)

    # Prepare input
    torch.manual_seed(0)
    test_in = torch.rand(1, 1, 64, 64)
    res = torch.rand(1, 1, 64, 64)
    ttmlp = ttBloomMLP(hugging_bloom_reference_model.state_dict())

    tt_out =  ttmlp.forward(test_in, res).to(host)

    pmlp = BloomMLP()

    pytorch_out = pmlp.forward(test_in, res)
    print(pytorch_out)

    tt_out =  tt_dropout_add(test_in, res, 0.3, False).to(host)
    tt_out = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))
    print(tt_out)

    assert np.allclose(pytorch_out.detach().numpy(), tt_out.numpy(), 1e-5, 0.17)
    print('Test PASSED: bloom_mlp')


if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_bloom_mlp_inference()
    ttm.device.CloseDevice(device)

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

def merge_heads(x: torch.Tensor) -> torch.Tensor:

    num_heads = 32

    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    head_dim = 1024 // num_heads

    x = x.view(batch_size, num_heads, seq_length, head_dim)

    x = x.permute(0, 2, 1, 3)

    return x.reshape(1, batch_size, seq_length, num_heads * head_dim)

def tt_merge_heads(x: torch.Tensor) -> torch.Tensor:

    num_heads = 32
    head_dim = 1024 // num_heads

    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    tt_x = tilize_to_list(pad_activation(x))
    tt_x = ttm.tensor.Tensor(tt_x, [1, batch_size_and_num_heads, seq_length, head_dim], ttm.tensor.DataType.BFLOAT16,  ttm.tensor.Layout.TILE, device)

    reshaped = ttm.tensor.reshape(tt_x, batch_size, num_heads, seq_length, head_dim)
    p_reshaped = torch.Tensor(reshaped.to(host).data()).reshape(reshaped.shape())
    p_reshaped = torch.Tensor(x).reshape(batch_size, num_heads, seq_length, head_dim)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    p_permuted = p_reshaped.permute(0, 2, 1, 3)

    permuted = ttm.tensor.Tensor(tilize_to_list(p_permuted), [batch_size, num_heads, seq_length, head_dim], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)

    third = num_heads*head_dim

    reshaped_2 = ttm.tensor.reshape(permuted, 1, batch_size, seq_length, num_heads*head_dim)

    return reshaped_2

def run_merge_heads_inference():
    # Prepare input
    torch.manual_seed(0)
    test_in = torch.rand(4096, 128, 32)

    pytorch_out = merge_heads(test_in)

    tt_out =  tt_merge_heads(test_in).to(host)
    tt_out = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))

    assert np.allclose(pytorch_out.detach().numpy(), tt_out.numpy(), 1e-5, 0.17)
    print('Test PASSED: merge_heads')

if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_merge_heads_inference()
    ttm.device.CloseDevice(device)

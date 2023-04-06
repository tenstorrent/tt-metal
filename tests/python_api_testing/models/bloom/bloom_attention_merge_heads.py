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

from libs import tt_lib as ttm
from utility_functions import comp_allclose, comp_pcc
import numpy as np
import python_api_testing.models.bloom.bloom_utils as bloom_utils

def merge_heads(x: torch.Tensor, num_heads, hidden_size, num_attention_heads) -> torch.Tensor:

    head_dim = hidden_size // num_attention_heads

    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    x = x.view(batch_size, num_heads, seq_length, head_dim)

    x = x.permute(0, 2, 1, 3)

    return x.reshape(1, batch_size, seq_length, num_heads * head_dim)

def tt_merge_heads(x: torch.Tensor, num_heads, hidden_size, num_attention_heads, device) -> torch.Tensor:

    head_dim = hidden_size // num_attention_heads

    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    tt_x = bloom_utils.torch2tt_tensor(x, device)
    tt_reshaped = ttm.tensor.reshape(tt_x, batch_size, num_heads, seq_length, head_dim)

    p_reshaped = bloom_utils.tt2torch_tensor(tt_reshaped)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    p_permuted = p_reshaped.permute(0, 2, 1, 3)

    tt_permuted = bloom_utils.torch2tt_tensor(p_permuted, device)

    reshaped_2 = ttm.tensor.reshape(tt_permuted, 1, batch_size, seq_length, num_heads*head_dim)

    return reshaped_2

def run_merge_heads_inference(device, num_heads, hidden_size, num_attention_heads):
    # Prepare input
    torch.manual_seed(0)
    test_in = torch.rand(4096, 128, 32)

    pt_out = merge_heads(test_in, num_heads, hidden_size, num_attention_heads)

    tt_out = tt_merge_heads(test_in, num_heads, hidden_size, num_attention_heads, device)

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print(comp_allclose(pt_out, tt_out_converted))
    print(comp_pcc(pt_out, tt_out_converted))

if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_merge_heads_inference(device, 32, 1024, 32)
    ttm.device.CloseDevice(device)

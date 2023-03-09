import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

import ttmetal
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close

RSUM = ttmetal.tensor.ReduceOpMath.SUM
RW = ttmetal.tensor.ReduceOpDim.W
RH = ttmetal.tensor.ReduceOpDim.H
RHW = ttmetal.tensor.ReduceOpDim.HW

# Initialize the device
device = ttmetal.device.CreateDevice(ttmetal.device.Arch.GRAYSKULL, 0)
ttmetal.device.InitializeDevice(device)
host = ttmetal.device.GetHost()
ttmetal.device.StartDebugPrintServer(device)

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

if __name__ == "__main__":
    N = 1
    C = 1
    H = 32
    W = 32
    shape = [N,C,H,W]
    torch.manual_seed(123)
    x = (torch.randn((N,C,H,W))+0.01).to(torch.bfloat16).to(torch.float32)
    xt = ttmetal.tensor.Tensor(tilize_to_list(x), [N, C, H, W], ttmetal.tensor.DataType.BFLOAT16, ttmetal.tensor.Layout.TILE, device)
    tt_res = ttmetal.tensor.gelu(xt)
    tt_host_rm = tt_res.to(host).data()

    ref_gelu = new_gelu(x)
    pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(shape)
    pyt_got_back_rm = untilize(pyt_got_back_rm)
    print_diff_argmax(pyt_got_back_rm, ref_gelu)

ttmetal.device.CloseDevice(device)

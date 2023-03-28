import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from libs import tt_lib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close

RSUM = tt_lib.tensor.ReduceOpMath.SUM
RW = tt_lib.tensor.ReduceOpDim.W
RH = tt_lib.tensor.ReduceOpDim.H
RHW = tt_lib.tensor.ReduceOpDim.HW

# Initialize the device
device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
tt_lib.device.InitializeDevice(device)
host = tt_lib.device.GetHost()
tt_lib.device.StartDebugPrintServer(device)

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

    for op in ["gelu", "exp", "sqrt", "sigmoid", "recip", "tanh", "log", "relu"]:
        x = (torch.randn((N,C,H,W))+0.01).to(torch.bfloat16).to(torch.float32)
        if op == "recip":
            x = x+5.0
        if op == "sqrt":
            x = x*x
        if op == "log":
            x = (x+10.0)*100.0
        xt = tt_lib.tensor.Tensor(tilize_to_list(x), [N, C, H, W], tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.TILE, device)
        if op == "relu":
            print("---- Testing relu")
            tt_res = tt_lib.tensor.relu(xt)
            ref = x.relu()
        if op == "log":
            print("---- Testing log")
            tt_res = tt_lib.tensor.tanh(xt)
            ref = x.log()
        if op == "tanh":
            print("---- Testing tanh")
            tt_res = tt_lib.tensor.tanh(xt)
            ref = torch.tanh(x)
        if op == "recip":
            print("---- Testing recip")
            tt_res = tt_lib.tensor.recip(xt)
            ref = 1.0/x
        if op == "sigmoid":
            print("---- Testing sigmoid")
            tt_res = tt_lib.tensor.sigmoid(xt)
            ref = torch.sigmoid(x)
        if op == "gelu":
            print("---- Testing gelu")
            tt_res = tt_lib.tensor.gelu(xt)
            ref = new_gelu(x)
        elif op == "exp":
            print("---- Testing exp")
            tt_res = tt_lib.tensor.exp(xt)
            ref = x.exp()
        elif op == "sqrt":
            print("---- Testing sqrt")
            tt_res = tt_lib.tensor.sqrt(xt)
            ref = x.sqrt()
        tt_host_rm = tt_res.to(host).data()
        pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(shape)
        pyt_got_back_rm = untilize(pyt_got_back_rm)
        print_diff_argmax(pyt_got_back_rm, ref)

tt_lib.device.CloseDevice(device)

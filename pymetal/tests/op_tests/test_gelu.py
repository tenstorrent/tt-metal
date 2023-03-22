import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from pymetal import ttlib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close

RSUM = ttlib.tensor.ReduceOpMath.SUM
RW = ttlib.tensor.ReduceOpDim.W
RH = ttlib.tensor.ReduceOpDim.H
RHW = ttlib.tensor.ReduceOpDim.HW

# Initialize the device
device = ttlib.device.CreateDevice(ttlib.device.Arch.GRAYSKULL, 0)
ttlib.device.InitializeDevice(device)
host = ttlib.device.GetHost()
ttlib.device.StartDebugPrintServer(device)

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
        xt = ttlib.tensor.Tensor(tilize_to_list(x), [N, C, H, W], ttlib.tensor.DataType.BFLOAT16, ttlib.tensor.Layout.TILE, device)
        if op == "relu":
            print("---- Testing relu")
            tt_res = ttlib.tensor.relu(xt)
            ref = x.relu()
        if op == "log":
            print("---- Testing log")
            tt_res = ttlib.tensor.tanh(xt)
            ref = x.log()
        if op == "tanh":
            print("---- Testing tanh")
            tt_res = ttlib.tensor.tanh(xt)
            ref = torch.tanh(x)
        if op == "recip":
            print("---- Testing recip")
            tt_res = ttlib.tensor.recip(xt)
            ref = 1.0/x
        if op == "sigmoid":
            print("---- Testing sigmoid")
            tt_res = ttlib.tensor.sigmoid(xt)
            ref = torch.sigmoid(x)
        if op == "gelu":
            print("---- Testing gelu")
            tt_res = ttlib.tensor.gelu(xt)
            ref = new_gelu(x)
        elif op == "exp":
            print("---- Testing exp")
            tt_res = ttlib.tensor.exp(xt)
            ref = x.exp()
        elif op == "sqrt":
            print("---- Testing sqrt")
            tt_res = ttlib.tensor.sqrt(xt)
            ref = x.sqrt()
        tt_host_rm = tt_res.to(host).data()
        pyt_got_back_rm = torch.Tensor(tt_host_rm).reshape(shape)
        pyt_got_back_rm = untilize(pyt_got_back_rm)
        print_diff_argmax(pyt_got_back_rm, ref)

ttlib.device.CloseDevice(device)

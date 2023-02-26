import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from gpai import gpai
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight


def Layernorm(gamma, beta, epsilon: float, H, W, device):
    # gamma, beta, epsilon should be tt::tensors of size 32*32 with one value
    # n_minus_one should be a tensor of size HW-ish (dims TBD)

    gamma = gpai.tensor.Tensor(
        gamma,
        [1, 1, 32, W],
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device
    )

    beta = gpai.tensor.Tensor(
        beta,
        [1, 1, 32, W],
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device
    )

    epsilon = gpai.tensor.Tensor(
        [epsilon] + [0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device
    )

    var_scaler = gpai.tensor.Tensor(
        [1 / (H * W)] + [0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device
    )

    #gpai.tensor.DataFormat.FLOAT32
    RSUM = gpai.tensor.ReduceOpMath.SUM
    RW = gpai.tensor.ReduceOpDim.W
    RH = gpai.tensor.ReduceOpDim.H
    BCW = gpai.tensor.BcastOpDim.W
    BCH = gpai.tensor.BcastOpDim.H
    BCHW = gpai.tensor.BcastOpDim.HW
    BCMUL = gpai.tensor.BcastOpMath.MUL
    BCSUB = gpai.tensor.BcastOpMath.SUB
    BCADD = gpai.tensor.BcastOpMath.ADD

    # unbiased_var = [(x-m)^2]/(n-1)
    # m = E[x]
    # var = E[(x-m)^2]
    # (x - E[x])/sqrt(var+epsilon)*gamma+beta
    # TODO(AP): need proper constants for reduce, probably change api to remove constant, use 1/H, 1/W
    def layernorm_(x):
        H,W = x.shape()[2], x.shape()[3]
        # first compute the mean (m)
        redW = gpai.tensor.reduce(x, RSUM, RW, 1.0/W) # -> NCH1
        mean = gpai.tensor.reduce(redW, RSUM, RH, 1.0/H) # -> NC11 (HW reduce doesn't behave well with small scaler)
        x_minus_mean = gpai.tensor.bcast(x, mean, BCSUB, BCHW)

        var = gpai.tensor.mul(x_minus_mean, x_minus_mean)
        var_redW = gpai.tensor.reduce(var, RSUM, RW, 1.0) # sum[(x-m)^2]
        var_redHW = gpai.tensor.reduce(var_redW, RSUM, RH, 1.0) # sum[(x-m)^2]
        var_div_n1 = gpai.tensor.bcast(var_redHW, var_scaler, BCMUL, BCHW) # *= 1/(everything not batch)
        var_plus_eps = gpai.tensor.bcast(var_div_n1, epsilon, BCADD, BCHW)
        var_sqrt = gpai.tensor.sqrt(var_plus_eps)
        inv_sqrt = gpai.tensor.recip(var_sqrt)

        x_div_sqrt = gpai.tensor.bcast(x_minus_mean, inv_sqrt, BCMUL, BCHW)
        x_gamma = gpai.tensor.bcast(x_div_sqrt, gamma, BCMUL, BCH)
        x_beta = gpai.tensor.bcast(x_gamma, beta, BCADD, BCH)

        return x_beta

    return layernorm_


def ref_layernorm(x, eps, gamma, beta, H, W):
    lnorm = torch.nn.LayerNorm((1,1,H,W), eps)
    lnorm.weight = torch.nn.Parameter(torch.full(x.shape, gamma))
    lnorm.bias = torch.nn.Parameter(torch.full(x.shape, beta))
    return lnorm(x)

if __name__ == "__main__":
    # Initialize the device
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()

    H = 64
    W = 96
    epsf = 1e-4
    betaf = 0.345
    gammaf = 0.123
    torch.manual_seed(123)
    x = torch.randn((1,1,H,W))
    ref_lnorm = ref_layernorm(x, epsf, gammaf, betaf, H, W)

    gamma = pad_weight(torch.full((1,1,H,W), gammaf))
    beta = pad_weight(torch.full((1,1,H,W), betaf))
    eps = pad_weight(torch.full((1,1,1,1), epsf))
    var_scaler = pad_weight(torch.full((1,1,1,1), 1.0/(H*W) )) # inverse n for biased variance

    t0 = gpai.tensor.Tensor(tilize_to_list(x), [1, 1, H, W], gpai.tensor.DataFormat.FLOAT32, gpai.tensor.Layout.TILE, device)
    ttgamma = tilize_to_list(gamma)
    ttbeta = tilize_to_list(beta)
    func = Layernorm(ttgamma, ttbeta, eps, var_scaler, H, W)

    t1 = func(t0)
    t2_data = t1.to(host).data()

    tt_got_back = torch.Tensor(t2_data).reshape((1,1,H,W))
    tt_got_back = untilize(tt_got_back)

    print("Layernorm max absdiff=")
    print_diff_argmax(tt_got_back, ref_lnorm)

    gpai.device.CloseDevice(device)

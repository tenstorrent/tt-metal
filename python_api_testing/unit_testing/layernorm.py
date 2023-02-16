import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

import ll_buda_bindings.ll_buda_bindings._C as _C
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight

# Initialize the device
device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
_C.device.InitializeDevice(device)
host = _C.device.GetHost()


def tt_layernorm():
    # gamma, beta, epsilon should be tt::tensors of size 32*32 with one value
    # n_minus_one should be a tensor of size HW-ish (dims TBD)
    def tt_layernorm_(x, gamma, beta, epsilon, var_scaler):
        #_C.tensor.DataFormat.FLOAT32
        RSUM = _C.tensor.ReduceOpMath.SUM
        RW = _C.tensor.ReduceOpDim.W
        RH = _C.tensor.ReduceOpDim.H
        BCW = _C.tensor.BcastOpDim.W
        BCHW = _C.tensor.BcastOpDim.HW
        BCMUL = _C.tensor.BcastOpMath.MUL
        BCSUB = _C.tensor.BcastOpMath.SUB
        BCADD = _C.tensor.BcastOpMath.ADD

        # unbiased_var = [(x-m)^2]/(n-1)
        # m = E[x]
        # var = E[(x-m)^2]
        # (x - E[x])/sqrt(var+epsilon)*gamma+beta
        # TODO(AP): need proper constants for reduce, probably change api to remove constant, use 1/H, 1/W
        H,W = x.shape()[2], x.shape()[3]
        # first compute the mean (m)
        redW = _C.tensor.reduce(x, RSUM, RW, 1.0/W) # -> NCH1
        mean = _C.tensor.reduce(redW, RSUM, RH, 1.0/H) # -> NC11 (HW reduce doesn't behave well with small scaler)
        x_minus_mean = _C.tensor.bcast(x, mean, BCSUB, BCHW)
        var = _C.tensor.mul(x_minus_mean, x_minus_mean)

        var_redW = _C.tensor.reduce(var, RSUM, RW, 1.0) # sum[(x-m)^2]
        var_redHW = _C.tensor.reduce(var_redW, RSUM, RH, 1.0) # sum[(x-m)^2]
        var_div_n1 = _C.tensor.bcast(var_redHW, var_scaler, BCMUL, BCHW) # *= 1/n

        var_plus_eps = _C.tensor.bcast(var_div_n1, epsilon, BCADD, BCHW)
        var_sqrt = _C.tensor.sqrt(var_plus_eps)
        inv_sqrt = _C.tensor.recip(var_sqrt)
        x_div_sqrt = _C.tensor.bcast(x_minus_mean, inv_sqrt, BCMUL, BCHW)
        x_gamma = _C.tensor.mul(x_div_sqrt, gamma)
        x_beta = _C.tensor.add(x_gamma, beta)

        return x_beta

    return tt_layernorm_

def ref_layernorm(x, eps, gamma, beta, H, W):
    lnorm = torch.nn.LayerNorm((1,1,H,W), eps)
    lnorm.weight = torch.nn.Parameter(torch.full(x.shape, gamma))
    lnorm.bias = torch.nn.Parameter(torch.full(x.shape, beta))
    return lnorm(x)

if __name__ == "__main__":
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

    t0 = _C.tensor.Tensor(tilize_to_list(x), [1, 1, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    ttgamma = _C.tensor.Tensor(tilize_to_list(gamma), [1, 1, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    ttbeta = _C.tensor.Tensor(tilize_to_list(beta), [1, 1, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    tteps = _C.tensor.Tensor(tilize_to_list(eps), [1, 1, 32, 32], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    ttvar_scaler = _C.tensor.Tensor(tilize_to_list(var_scaler), [1, 1, 32, 32], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    func = tt_layernorm()

    t1 = func(t0, ttgamma, ttbeta, tteps, ttvar_scaler)
    t2_data = t1.to(host).data()

    tt_got_back = torch.Tensor(t2_data).reshape((1,1,H,W))
    tt_got_back = untilize(tt_got_back)

    print("Layernorm max absdiff=")
    print_diff_argmax(tt_got_back, ref_lnorm)

_C.device.CloseDevice(device)



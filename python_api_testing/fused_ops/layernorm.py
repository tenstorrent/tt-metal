import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

from pymetal import ttmetal as ttm
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, ttP, set_FR, get_FR
from python_api_testing.models.utility_functions import tt2torch as t2t, tt2torch_rm as t2trm
from python_api_testing.models.utility_functions import roundup32, float_to_bits
from python_api_testing.models.utility_functions import enable_binary_cache, enable_compile_cache
from python_api_testing.models.utility_functions import print_diff_tt_pyt, is_close

# This ref implementation is only here for debugging
def ref_ln(x, gamma, beta = None, epsilon = 1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + epsilon).sqrt()
    invstd = 1.0/std
    y1 = (x - mean) * invstd
    y = y1.clone()
    if gamma is not None:
        y *= gamma
    if beta is not None:
        y += beta
    return y, mean, var, std, invstd, y1

# TODO(AP): refactor to support any num_dims
# H,W correspond to normalized_shape in pytorch Layernorm spec
def Layernorm(gamma, beta, epsilon: float, H, W, device, num_dims = 2):
    num_dims_ = num_dims
    assert(num_dims == 1)

    # gamma, beta, epsilon should be tt::tensors of size 32*W
    # with a single populated top row
    # H, W need to be from the "true" shape (unpadded)
    assert(gamma is None or len(gamma) == W*32) # single H-tile
    assert(beta is None or len(beta) == W*32) # single H-tile

    H_ = H
    W_ = W
    padded_h = roundup32(H)
    if num_dims == 1:
        padded_h = 32
    padded_w = roundup32(W)
    gamma_ = ttm.tensor.Tensor(
        gamma,
        [1, 1, padded_h, padded_w],
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device
    )


    beta_ = None
    if beta is not None:
        beta_ = ttm.tensor.Tensor(
            beta,
            [1, 1, padded_h, padded_w],
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.TILE,
            device
        )

    epsilon_ = ttm.tensor.Tensor(
        [epsilon] + [0.0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.TILE,
        device
    )

    if num_dims == 2:
        var_scaler_ = ttm.tensor.Tensor(
            [1 / (H * W)] + [0.0 for _ in range(32 * 32 - 1)],
            [1, 1, 32, 32],
            ttm.tensor.DataType.BFLOAT16,
            ttm.tensor.Layout.TILE,
            device
        )
    else:
        # For num_dims==1 var_scaler_ is implemented using dynamic mask
        assert(num_dims == 1)

    #ttm.tensor.DataType.BFLOAT16
    RSUM = ttm.tensor.ReduceOpMath.SUM
    RW = ttm.tensor.ReduceOpDim.W
    RH = ttm.tensor.ReduceOpDim.H
    BCW = ttm.tensor.BcastOpDim.W
    BCH = ttm.tensor.BcastOpDim.H
    BCHW = ttm.tensor.BcastOpDim.HW
    BCMUL = ttm.tensor.BcastOpMath.MUL
    BCSUB = ttm.tensor.BcastOpMath.SUB
    BCADD = ttm.tensor.BcastOpMath.ADD

    # 1D variant
    # TODO(AP): merge with 2d? refactor.
    def layernorm_1d_(x, overrideH = None, refx = None, refgamma = None):

        N = x.shape()[0]
        C = x.shape()[1]
        H = x.shape()[2]
        W = x.shape()[3]

        H_ = 1
        if overrideH is not None:
            H_ = overrideH

        # first compute the mean (m)
        means = ttm.tensor.reduce(x, RSUM, RW, 1.0/W) # -> NCH1
        x_minus_mean = ttm.tensor.bcast(x, means, BCSUB, BCW) # need to blank out the H for non-multiple of 32
        if False and refx is not None:
            ry, rmean, rvar, rstd, rinvstd, ry1 = ref_ln(refx, refgamma)
            #print_diff_tt_pyt(x_minus_mean, refx-rmean)

        var = ttm.tensor.mul(x_minus_mean, x_minus_mean) # (x-m)^2
        var_redW = ttm.tensor.reduce(var, RSUM, RW, 1.0) # sum[(x-m)^2]

        constant = float_to_bits(1/W)
        scaler = (constant >> 16) & 0xFFFF
        var_scaler_ = ttm.tensor.fill_rm(1, 1, roundup32(H), 32, H_, 1, epsilon_, scaler, 0)
        var_scaler_ = ttm.tensor.tilize(var_scaler_)

        var_div_n1 = ttm.tensor.bcast(var_redW, var_scaler_, BCMUL, BCW)
        var_plus_eps = ttm.tensor.bcast(var_div_n1, epsilon_, BCADD, BCHW)

        var_sqrt = ttm.tensor.sqrt(var_plus_eps)
        inv_sqrt = ttm.tensor.recip(var_sqrt)
        if False and refx is not None:
            qq = t2t(inv_sqrt)[0,0,0:9,0]
            if not is_close(qq, rinvstd[0,:,0], 0.03):
                rerun = is_close(qq, rinvstd[0,:,0], 0.03)
                print(qq)
                print(rinvstd[0,:,0])

        x_div_sqrt = ttm.tensor.bcast(x_minus_mean, inv_sqrt, BCMUL, BCW)

        if False and refx is not None:
            qq1 = t2t(x_div_sqrt)[0,0,0:9,:]
            print_diff_argmax(qq1, ry1, "without gamma")

        x_gamma = ttm.tensor.bcast(x_div_sqrt, gamma_, BCMUL, BCH)
        if beta_ is not None:
            x_beta = ttm.tensor.bcast(x_gamma, beta_, BCADD, BCH)
            return x_beta
        else:
            return x_gamma

    def layernorm_2d_(x):

        N = x.shape()[0]
        C = x.shape()[1]
        H = x.shape()[2]
        W = x.shape()[3]

        # first compute the mean (m)
        redW = ttm.tensor.reduce(x, RSUM, RW, 1.0/W) # -> NCH1
        mean = ttm.tensor.reduce(redW, RSUM, RH, 1.0) # -> NC11 (HW reduce doesn't behave well with small scaler)
        x_minus_mean0 = ttm.tensor.bcast(x, mean, BCSUB, BCHW) # need to blank out the H for non-multiple of 32
        hmasku = ttm.tensor.fill_ones_rm(N, C, H, 32, 1, 1, x) # generate a H-mask with mask[h, w] = 1.0 where h,w < 1
        hmaskt = ttm.tensor.tilize(hmasku) # tilize the mask
        x_minus_mean = ttm.tensor.bcast(x_minus_mean0, hmaskt, BCMUL, BCW) # zero out (x-m) for h>=H_, h<H

        var = ttm.tensor.mul(x_minus_mean, x_minus_mean) # (x-m)^2
        var_redW = ttm.tensor.reduce(var, RSUM, RW, 1.0) # sum[(x-m)^2]
        var_redHW = ttm.tensor.reduce(var_redW, RSUM, RH, 1.0) # sum[(x-m)^2]
        var_div_n1 = ttm.tensor.bcast(var_redHW, var_scaler_, BCMUL, BCHW) # *= 1/(everything not batch)
        var_plus_eps = ttm.tensor.bcast(var_div_n1, epsilon_, BCADD, BCHW)

        var_sqrt = ttm.tensor.sqrt(var_plus_eps)
        inv_sqrt = ttm.tensor.recip(var_sqrt)

        x_div_sqrt = ttm.tensor.bcast(x_minus_mean, inv_sqrt, BCMUL, BCHW)
        x_gamma = ttm.tensor.mul(x_div_sqrt, gamma_, BCMUL, BCH)
        if beta_ is not None:
            x_beta = ttm.tensor.add(x_gamma, beta_, BCADD, BCH)
            return x_beta
        else:
            return x_gamma

    # unbiased_var = [(x-m)^2]/(n-1)
    # m = E[x]
    # var = E[(x-m)^2]
    # result = (x - E[x])/sqrt(var+epsilon)*gamma+beta
    def layernorm_(x, overrideH = None, refx = None, refgamma = None):
        if num_dims_ == 1:
            return layernorm_1d_(x, overrideH, refx, refgamma)

        assert(num_dims_ == 2) # Only 1d and 2d are supported at the moment
        return layernorm_2d_(x)

    return layernorm_


def ref_layernorm(x, eps, gamma, beta, H, W):
    lnorm = torch.nn.LayerNorm((W,), eps)
    lnorm.weight = torch.nn.Parameter(torch.full((W,), gamma))
    lnorm.bias = torch.nn.Parameter(torch.full((W,), beta))
    return lnorm(x)

if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()

    enable_compile_cache()
    enable_binary_cache()

    H = 64
    W = 96
    epsf = 1e-4
    betaf = 0.345
    gammaf = 0.123
    torch.manual_seed(123)
    x = torch.randn((1,1,H,W))
    ref_lnorm = ref_layernorm(x, epsf, gammaf, betaf, H, W)

    gamma = pad_weight(torch.full((1,1,1,W), gammaf))
    beta = pad_weight(torch.full((1,1,1,W), betaf))

    t0 = ttm.tensor.Tensor(tilize_to_list(x), [1, 1, H, W], ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    ttgamma = tilize_to_list(gamma)
    ttbeta = tilize_to_list(beta)
    func = Layernorm(ttgamma, ttbeta, epsf, 1, W, device, num_dims=1)

    t1 = func(t0, overrideH=H)
    t2_data = t1.to(host).data()

    tt_got_back = torch.Tensor(t2_data).reshape((1,1,H,W))
    tt_got_back = untilize(tt_got_back)

    print("Layernorm max absdiff=")
    print_diff_argmax(tt_got_back, ref_lnorm)

    ttm.device.CloseDevice(device)

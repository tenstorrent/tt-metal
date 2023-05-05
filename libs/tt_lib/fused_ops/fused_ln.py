import math
from pathlib import Path
import sys
import time
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../tests")

import torch

from libs import tt_lib
#from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close
from libs.tt_lib.utils import (
    _nearest_32 as nearest_32,
    pad_activation,
    pad_weight,
    tilize,
    tilize_to_list,
    untilize,
    print_diff_argmax,
    tt2torch,
    tt2torch_rm,
    roundup,
    roundup32,
    float_to_bits,
    divup,
    channels_last,
    convert_weights_2d_matrix,
    is_close
)

tensor = tt_lib.tensor
device = tt_lib.device

# This ref implementation is only here for debugging
def ref_ln(x, gamma, beta = None, epsilon = 1e-5):
    # prints a tile slice with these tile coord range
    torch.set_printoptions(precision=2, threshold=1000, sci_mode=False, edgeitems=8, linewidth=480)
    sth = 16
    stw = 16
    st = stw
    ph0, ph1, pw0, pw1 = 0, 2, 0, 2
    #ph0, ph1, pw0, pw1 = 0, 1, 0, 1

    print("x.shape=", x.shape)
    #print("eps=", epsilon)
    #print(f"slice={ph0}:{ph1}, {pw0}:{pw1}")
    #print("Ref x=\n", x[0, 0, ph0*32 : ph1*32 : sth, pw0*32 : pw1*32 : stw])
    mean = x.mean(dim=-1, keepdim=True)
    #print("Ref Ex=\n", mean[0, 0, ph0*32 : ph1*32 : sth, 0*32 : 1*32 : stw])
    xmm = x - mean
    #print("Ref xmm=\n", xmm[0, 0, ph0*32 : ph1*32 : st, pw0*32 : pw1*32 : st])
    xmm2 = xmm**2
    #print("Ref xmm2=\n", xmm2[0, 0, ph0*32 : ph1*32 : sth, pw0*32 : pw1*32 : stw])
    exmm2 = xmm2.mean(dim=-1, keepdim=True)
    #print("Ref exmm2=\n", exmm2[0, 0, ph0*32 : ph1*32 : sth, 0*32 : 1*32 : stw])

    std = (exmm2 + epsilon).sqrt()
    #print("Ref sqrt_exmm2=\n", std[0, 0, ph0*32 : ph1*32 : st, 0*32 : 1*32 : st])

    invstd = 1.0/std
    #print("Ref 1/sqrt_exmm2=\n", invstd[0, 0, ph0*32 : ph1*32 : st, 0*32 : 1*32 : st])
    y1 = xmm * invstd
    #print("Ref y*1+0=\n", y1[0, 0, ph0*32 : ph1*32 : st, pw0*32 : pw1*32 : st])
    y = y1.clone()
    if gamma is not None:
        y *= gamma
    if beta is not None:
        y += beta
    return y, mean, exmm2, std, invstd, y1

def ref_layernorm(x, eps, gamma, beta, H, W):
    lnorm = torch.nn.LayerNorm((W,), eps)
    lnorm.weight = torch.nn.Parameter(torch.full((W,), gamma))
    lnorm.bias = torch.nn.Parameter(torch.full((W,), beta))
    return lnorm(x)

if __name__ == "__main__":
    # Initialize the device
    dev = device.CreateDevice(device.Arch.GRAYSKULL, 0)
    device.InitializeDevice(dev)
    #device.StartDebugPrintServer(dev)
    host = device.GetHost()

    #N, C, H, W = 1, 1, 4*32, 9*32
    epsf = 1e-2
    torch.manual_seed(123)
    test_dims = ((1, 1, 4*32, 8*32), (1, 1, 3*32, 8*32), (1, 1, 32, 12*32), (1, 1, 8*32, 32*32), )
    #test_dims = ((1,1,32,1024),)
    #test_dims = ((1,9,384,1024),)
    test_dims = ((1,1,384,1024),)
    for nchw in test_dims:
        for i in range(0, 1): # 0: no gamma/beta, 1: gamma, 2: gamma+beta
            i = 2 # force gamma+beta path
            for nrepeat in range(0, 1):
                (N, C, H, W) = nchw
                print("NCHW=", nchw)
                if i == 0:
                    gammaf = 1.0
                    betaf = 0.0
                if i >= 1:
                    gammaf = 0.65
                    gamma = pad_weight(torch.full((1,1,1,W), gammaf))
                    ttgamma = tensor.Tensor(tilize_to_list(gamma), [N, C, 32, W], tensor.DataType.BFLOAT16, tensor.Layout.TILE, dev)
                if i >= 2:
                    betaf = 1.345
                    beta = pad_weight(torch.full((1,1,1,W), betaf))
                    ttbeta = tensor.Tensor(tilize_to_list(beta), [N, C, 32, W], tensor.DataType.BFLOAT16, tensor.Layout.TILE, dev)

                x = torch.randn((N,C,H,W))
                #ref_lnorm = ref_layernorm(x, epsf, gammaf, betaf, H, W)
                ref_lnorm, _, _, _, _, _ = ref_ln(x, gammaf, betaf, epsf)

                ttx = tensor.Tensor(tilize_to_list(x), [N, C, H, W], tensor.DataType.BFLOAT16, tensor.Layout.TILE, dev)

                if i == 0:
                    print("=== Running LN_NOGB")
                    tty = tensor.layernorm(ttx, epsf)
                elif i == 1:
                    print("=== Running LN_G")
                    tty = tensor.layernorm_gamma(ttx, epsf, ttgamma)
                elif i == 2:
                    print("=== Running LN_GB")
                    tty = tensor.layernorm_gamma_beta(ttx, epsf, ttgamma, ttbeta)
                else:
                    assert(False)
                print("=== Done Running LN")
                t2_data = tty.to(host).data()

                tt_got_back = torch.Tensor(t2_data).reshape((N,C,H,W))
                tt_got_back = untilize(tt_got_back)

                time.sleep(0.3) # sleep to avoid print intermixing with kernel prints

                if not is_close(tt_got_back, ref_lnorm):
                    print("****  Mismatch!")

    device.CloseDevice(dev)

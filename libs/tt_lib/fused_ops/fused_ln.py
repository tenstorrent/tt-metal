import math
from pathlib import Path
import sys
import time
import os
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../tests")

DEBUG_PRINTS = True
DEBUG_PRINTS = False
PROFILE = not DEBUG_PRINTS

import torch

from libs import tt_lib as ttl

# from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close
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
    is_close,
)

tensor = ttl.tensor
device = ttl.device


# This ref implementation is only here for debugging
def ref_ln(x, gamma, beta=None, epsilon=1e-5, b=None):
    # prints a tile slice with these tile coord range
    torch.set_printoptions(
        precision=2, threshold=1000, sci_mode=False, edgeitems=8, linewidth=480
    )
    sth = 16
    stw = 16
    ph0, ph1, pw0, pw1 = 1, 2, 0, 1
    # ph0, ph1, pw0, pw1 = 0, 1, 0, 1

    print("x.shape=", x.shape)
    # print("eps=", epsilon)
    print(f"slice={ph0}:{ph1}, {pw0}:{pw1}")
    print("Ref x=\n", x[0, 0, ph0 * 32 : ph1 * 32 : sth, pw0 * 32 : pw1 * 32 : stw])
    print(
        "Ref a=\n", (x - b)[0, 0, ph0 * 32 : ph1 * 32 : sth, pw0 * 32 : pw1 * 32 : stw]
    )
    print("Ref b=\n", b[0, 0, ph0 * 32 : ph1 * 32 : sth, pw0 * 32 : pw1 * 32 : stw])
    mean = x.mean(dim=-1, keepdim=True)
    # print("Ref Ex=\n", mean[0, 0, ph0*32 : ph1*32 : sth, 0*32 : 1*32 : stw])
    xmm = x - mean
    # print("Ref xmm=\n", xmm[0, 0, ph0*32 : ph1*32 : st, pw0*32 : pw1*32 : st])
    xmm2 = xmm**2
    # print("Ref xmm2=\n", xmm2[0, 0, ph0*32 : ph1*32 : sth, pw0*32 : pw1*32 : stw])
    exmm2 = xmm2.mean(dim=-1, keepdim=True)
    # print("Ref exmm2=\n", exmm2[0, 0, ph0*32 : ph1*32 : sth, 0*32 : 1*32 : stw])

    std = (exmm2 + epsilon).sqrt()
    # print("Ref sqrt_exmm2=\n", std[0, 0, ph0*32 : ph1*32 : st, 0*32 : 1*32 : st])

    invstd = 1.0 / std
    # print("Ref 1/sqrt_exmm2=\n", invstd[0, 0, ph0*32 : ph1*32 : st, 0*32 : 1*32 : st])
    y1 = xmm * invstd
    # print("Ref y*1+0=\n", y1[0, 0, ph0*32 : ph1*32 : st, pw0*32 : pw1*32 : st])
    y = y1.clone()
    if gamma is not None:
        print("yshape=", y.shape)
        print("gshape=", gamma.shape)
        y *= gamma
    if beta is not None:
        y += beta
    # y = gamma.repeat(x.shape[0], x.shape[1], x.shape[2], x.shape[3]//gamma.shape[3]) # Debug gamma
    return y, mean, exmm2, std, invstd, y1


def ref_layernorm(x, eps, gamma, beta, H, W):
    lnorm = torch.nn.LayerNorm((W,), eps)
    lnorm.weight = torch.nn.Parameter(torch.full((W,), gamma))
    lnorm.bias = torch.nn.Parameter(torch.full((W,), beta))
    return lnorm(x)


if __name__ == "__main__":
    out_dram = False
    in_dram = False
    # Initialize the device
    if PROFILE:
        os.environ["TT_PROFILE"] = "1"
    os.environ["TT_BLOCK_SIZE"] = "2"
    os.environ["TT_FORCE_NUMCORES"] = "108"
    # os.environ["TT_FORCE_NUMCORES"] = "1"
    dev = device.CreateDevice(device.Arch.GRAYSKULL, 0)
    device.InitializeDevice(dev, ttl.device.MemoryAllocator.L1_BANKING)
    memcfg = ttl.tensor.MemoryConfig(
        buffer_type=(
            ttl.tensor.BufferType.DRAM if in_dram else ttl.tensor.BufferType.L1
        )
    )
    if DEBUG_PRINTS:
        device.StartDebugPrintServerOnCores(
            dev, [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]]
        )
    host = device.GetHost()

    # N, C, H, W = 1, 1, 4*32, 9*32
    epsf = 1e-2
    torch.manual_seed(123)

    # TODO: Clean up
    # test_dims = (
    #     (1, 1, 4 * 32, 8 * 32),
    #     (1, 1, 3 * 32, 8 * 32),
    #     (1, 1, 32, 12 * 32),
    #     (1, 1, 8 * 32, 32 * 32),
    # )
    # # test_dims = ((1,9,384,1024),)
    # test_dims = ((1, 1, 32 * 6, 1024),)
    # test_dims = ((1, 1, 384, 1024),)
    # test_dims = ((1, 10, 256, 1024),)
    # # test_dims = ((1,6,32,1024),)
    # test_dims = ((1, 1, 32, 1024),)
    test_dims = ((1, 9, 384, 1024),)
    for nchw in test_dims:
        for i in range(0, 4):  # 0: no gamma/beta, 1: gamma, 2: gamma+beta
            # i = 0  # force ln(x)*1+0 path
            # i = 1  # force ln(x)*g+0 path
            # i = 2  # force ln(a+b)*gamma+beta path
            # i = 3  # force ln(x)*gamma+beta path
            for nrepeat in range(0, 1):
                (N, C, H, W) = nchw
                print("NCHW=", nchw)
                if i >= 0:
                    gamma = torch.ones(1, 1, 1, W)
                    beta = torch.zeros(1, 1, 1, W)
                if i >= 1:
                    gamma = torch.rand(1, 1, 1, W) * 2 - 1
                    # gamma = torch.arange(0.0, float(W)-0.01, 1.0).reshape(1,1,1,W) # debug gamma
                    # gammah32 = tilize_to_list(pad_weight(gamma.repeat(1,1,32,1))) # debug gamma
                    print(gamma)
                    # gamma[:,:,:,320:] = 0.0
                    gammah32 = tilize_to_list(pad_weight(gamma))
                    ttgamma = tensor.Tensor(
                        gammah32,
                        [1, 1, 32, W],
                        tensor.DataType.BFLOAT16,
                        tensor.Layout.TILE,
                        dev,
                        memcfg,
                    )
                if i >= 2:
                    beta = torch.rand(1, 1, 1, W) * 2.0 - 1.1
                    betah32 = tilize_to_list(pad_weight(beta))
                    ttbeta = tensor.Tensor(
                        betah32,
                        [1, 1, 32, W],
                        tensor.DataType.BFLOAT16,
                        tensor.Layout.TILE,
                        dev,
                        memcfg,
                    )

                x = torch.rand((N, C, H, W)) * 2 - 0.95
                y = torch.rand((N, C, H, W)) * 2 - 0.8
                if i < 3:
                    y *= 0.0  # zero out the y to exclude x+y from reference calculation
                # ref_lnorm = ref_layernorm(x, epsf, gammaf, betaf, H, W)
                ref_lnorm, _, _, _, _, _ = ref_ln(x + y, gamma, beta, epsf, y)

                ttx = tensor.Tensor(
                    tilize_to_list(x),
                    [N, C, H, W],
                    tensor.DataType.BFLOAT16,
                    tensor.Layout.TILE,
                    dev,
                    memcfg,
                )
                tty = tensor.Tensor(
                    tilize_to_list(y),
                    [N, C, H, W],
                    tensor.DataType.BFLOAT16,
                    tensor.Layout.TILE,
                    dev,
                    memcfg,
                )

                if i == 0:
                    logger.info("Running LN_NOGB")
                    ttz = tensor.layernorm(ttx, epsf, out_dram)
                elif i == 1:
                    logger.info("Running LN_G")
                    ttz = tensor.layernorm_gamma(ttx, epsf, ttgamma, out_dram)
                elif i == 2:
                    logger.info("Running LN_GB")
                    ttz = tensor.layernorm_gamma_beta(
                        ttx, epsf, ttgamma, ttbeta, out_dram
                    )
                elif i == 3:
                    logger.info("Running add_LN_GB")
                    ttz = tensor.add_layernorm_gamma_beta(
                        ttx, tty, epsf, ttgamma, ttbeta, out_dram
                    )
                else:
                    assert False
                logger.info("Done Running LN")
                t2_data = ttz.to(host).data()

                tt_got_back = torch.Tensor(t2_data).reshape((N, C, H, W))
                tt_got_back = untilize(tt_got_back)
                # print("tt_result[ht=0,wt=0]=", tt_got_back[:,:,0:32:1,0:32:1])

                time.sleep(0.3)  # sleep to avoid print intermixing with kernel prints

                # if not is_close(tt_got_back[:,0,0:128,:], ref_lnorm[:,0,0:128,:]):
                if not is_close(tt_got_back, ref_lnorm):
                    assert False
                    print("****  Mismatch!")

    device.CloseDevice(dev)

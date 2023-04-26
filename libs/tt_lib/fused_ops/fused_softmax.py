import math
from pathlib import Path
import sys
import time
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../tests")

import torch

from libs import tt_lib
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close

tensor = tt_lib.tensor
device = tt_lib.device


def ref_stable_softmax(x):
    torch.set_printoptions(precision=2, threshold=1000, sci_mode=False, edgeitems=8, linewidth=180)
    #print("Ref x=\n", x[0, 0, 0:32:8, 0:64:8])
    z = x #- torch.max(x, dim=3, keepdim=True)[0]
    numerator = torch.exp(z)
    #print(x.shape)
    H = x.shape[-2]
    #print("H=", H)
    pw0 = 0 # prints a tile slice with these tile coord range
    pw1 = 3
    ph0 = 0
    ph1 = 2
    #print("Ref exps=\n", numerator[0, 0, ph0*32 : ph1*32 : 8, pw0*32 : pw1*32 : 8])
    denominator = torch.sum(numerator, 3, keepdim=True)
    #print("denom shape=", denominator.shape)
    #print("Ref sumexp=\n", torch.reshape(denominator, (-1,H))[:, ph0*32:ph1*32])

    denom1 = torch.reciprocal(denominator)
    #print("ref 1/sumexp=\n", denom1[0, 0, 0:32:8, 0:64:8])
    softmax = numerator*denom1
    #print("softmaxManual=\n", softmax[0, 0, 0:32:8, 0:64:8])
    softmax = torch.nn.Softmax(3)(x)
    #print("softmaxTorch=\n", softmax[0, 0, 0:32:8, 0:64:8])

    return softmax

if __name__ == "__main__":
    dev = device.CreateDevice(device.Arch.GRAYSKULL, 0)
    device.InitializeDevice(dev)
    #device.StartDebugPrintServer(dev)
    host = device.GetHost()
    #N, C, H, W = 1, 7, 5*32, 17*32
    N, C, H, W = 1, 1, 2048, 4*8*32 # W must be a multiple of 8*32
    torch.manual_seed(123)

    for j in range(0, 1):
        x = torch.randn((N,C,H,W)) + 0.01

        ref_sm = ref_stable_softmax(x)

        x_t = tilize_to_list(x)
        t0 = tensor.Tensor(x_t, [N, C, H, W], tensor.DataType.BFLOAT16, tensor.Layout.TILE, dev)

        t1_fused = tensor.softmax(t0)
        t2_data_fused = t1_fused.to(host).data()
        tt_got_back_fused = torch.Tensor(t2_data_fused).reshape((N,C,H,W))
        tt_unt = untilize(tt_got_back_fused)

        time.sleep(0.33) # so prints don't overlap with kernel prints
        print("Max diff=")
        print_diff_argmax(tt_unt, ref_sm)

    device.CloseDevice(dev)

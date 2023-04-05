import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import torch

import ll_buda_bindings.ll_buda_bindings._C as _C
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax


def softmax(x, stable=False):
    RMAX = _C.tensor.ReduceOpMath.MAX
    RSUM = _C.tensor.ReduceOpMath.SUM
    RW = _C.tensor.ReduceOpDim.W
    BCW = _C.tensor.BcastOpDim.W
    BCMUL = _C.tensor.BcastOpMath.MUL
    BCSUB = _C.tensor.BcastOpMath.MUL

    if stable:
        sumsW = _C.tensor.reduce(x, RMAX, RW, 1.0)
        z = _C.tensor.bcast(x, sumsW, BCSUB, BCW) # x-max(x)
    else:
        z = x
    numerator = _C.tensor.exp(z) # exp(z)
    denom1 = _C.tensor.reduce(numerator, RSUM, RW, 1.0) # torch.sum(x, 3)
    denom = _C.tensor.recip(denom1)
    output = _C.tensor.bcast(numerator, denom, BCMUL, BCW)

    return output

def ref_stable_softmax(x):
    """
    z = x - torch.max(x, dim=3, keepdim=True)[0]
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, 3)
    denom1 = torch.reciprocal(denominator)
    softmax = numerator*denom1
    """
    softmax = torch.nn.Softmax(3)(x)

    return softmax

if __name__ == "__main__":
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    _C.device.StartDebugPrintServer(device)
    host = _C.device.GetHost()
    H, W = 32, 32
    torch.manual_seed(123)

    x = torch.randn((1,1,H,W))
    ref_sm = ref_stable_softmax(x)

    x_t = tilize_to_list(x)
    t0 = _C.tensor.Tensor(x_t, [1, 1, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    test_unfused = False
    if test_unfused:
        func = softmax
        t1 = func(t0)
        t2_data = t1.to(host).data()
        tt_got_back = torch.Tensor(t2_data).reshape((1,1,H,W))
        tt_got_back = untilize(tt_got_back)

        print("Max diff=")
        print_diff_argmax(tt_got_back, ref_sm)

    test_fused = True
    if test_fused:
        print("skip_hlkc = ", _C.device.GetSkipHlkc())
        _C.device.SetSkipHlkc(False)
        _C.device.SetSkipHlkc(True)
        print("skip_hlkc = ", _C.device.GetSkipHlkc())
        t1_fused = _C.tensor.softmax(t0)
        t2_data_fused = t1_fused.to(host).data()
        tt_got_back_fused = torch.Tensor(t2_data_fused).reshape((1,1,H,W))
        tt_unt = untilize(tt_got_back_fused)
        print("Max diff=")
        print_diff_argmax(tt_unt, ref_sm)


    _C.device.CloseDevice(device)

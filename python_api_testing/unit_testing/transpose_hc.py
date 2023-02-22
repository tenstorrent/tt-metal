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

if __name__ == "__main__":
    N = 3
    C = 32*2
    H = 32*4
    W = 32*3
    x = torch.randn((N,C,H,W))
    xp = pad_weight(x)

    xt = _C.tensor.Tensor(tilize_to_list(xp), [N, C, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    xtt = _C.tensor.transpose_hc(xt)
    assert(xtt.shape() == [N,H,C,W])

    xtt_data = xtt.to(host).data()
    tt_got_back = torch.Tensor(xtt_data).reshape((N,H,C,W))
    tt_got_back = untilize(tt_got_back)

    print("reshape() max absdiff=")
    transposed_ref = x.permute(0, 2, 1, 3)
    print_diff_argmax(tt_got_back, transposed_ref)

_C.device.CloseDevice(device)



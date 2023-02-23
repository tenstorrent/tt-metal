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

def tile_major_reshape():
    N = 3
    C = 5
    H = 64
    W = 96
    x = torch.randn((N,C,H,W))
    xp = pad_weight(x)

    xtt = _C.tensor.Tensor(tilize_to_list(xp), [N, C, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)
    _C.tensor.reshape(xtt, 5, 3, 96, 64)
    assert(xtt.shape() == [5,3,96,64])
    _C.tensor.reshape(xtt, 3, 5, 64, 96)
    assert(xtt.shape() == [3,5,64,96])
    _C.tensor.reshape(xtt, -1, 5, 64, 96)
    assert(xtt.shape() == [3,5,64,96])
    _C.tensor.reshape(xtt, 3, -1, 64, 96)
    assert(xtt.shape() == [3,5,64,96])
    _C.tensor.reshape(xtt, 3, 5, -1, 96)
    assert(xtt.shape() == [3,5,64,96])
    _C.tensor.reshape(xtt, 3, 5, 64, -1)
    assert(xtt.shape() == [3,5,64,96])
    _C.tensor.reshape(xtt, 3, 5, 32, -1)
    assert(xtt.shape() == [3,5,32,96*2])

    xtt_data = xtt.to(host).data()
    tt_got_back = torch.Tensor(xtt_data).reshape((N,C,H,W))
    tt_got_back = untilize(tt_got_back)

    print("reshape() max absdiff=")
    print_diff_argmax(tt_got_back, x)

def row_major_reshape():
    N = 1
    C = 1
    H = 128
    W = 128
    x = torch.arange(N*C*H*W).reshape(N, C, H, W).float()
    xp = pad_weight(x).view(-1).tolist()

    xtt = _C.tensor.Tensor(xp, [N, C, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.ROW_MAJOR, device)
    reshaped = _C.tensor.reshape(xtt, 1, 128, 2, 64).to(host).data()
    reshaped_tensor = torch.Tensor(reshaped).reshape(1, 128, 2, 64)
    return

if __name__ == "__main__":
    # tile_major_reshape()
    row_major_reshape()

_C.device.CloseDevice(device)

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
_C.device.StartDebugPrintServer(device)

if __name__ == "__main__":
    torch.manual_seed(123)
    N = 1
    C = 128 # 2
    H = 2 # 128
    W = 64
    x = torch.randn((N,C,H,W))

    xt = _C.tensor.Tensor(x.reshape(-1).tolist(), [N, C, H, W], _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.ROW_MAJOR, device)

    # test that reading back from row major is about the same (+/- BF16 conversion)
    xt_data = xt.to(host).data()
    tt_got_back_rm = torch.Tensor(xt_data).reshape((N,C,H,W))

    print("row_major read back max absdiff=")
    print_diff_argmax(tt_got_back_rm, x)

    # apply  h-padding
    xtp = _C.tensor.transpose_hc_rm(xt)
    assert(xtp.shape() == [N,H,C,W])
    xtp_data = xtp.to(host).data()
    tt_got_back = torch.Tensor(xtp_data).reshape((N,H,C,W))

    print("pad_h_rm() max absdiff=")
    hc_ref = x.permute(0, 2, 1, 3)
    print_diff_argmax(tt_got_back, hc_ref)

_C.device.CloseDevice(device)

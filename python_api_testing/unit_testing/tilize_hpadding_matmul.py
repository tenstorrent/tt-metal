from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

from pymetal import ttmetal as ttm
from pymetal.ttmetal.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, pad_activation
from python_api_testing.models.utility_functions import print_diff_argmax, is_close
import torch

def run_tilize_matmul_test(M, K, N):
    a_shape = [1,1,M,K]
    a_shape_padded = [1,1,_nearest_32(M),K]
    b_shape = [1,1,K,N]
    output_shape = [1,1,_nearest_32(M),N]
    A = torch.randn(a_shape)
    A_padded = pad_activation(A)
    B = torch.randn(b_shape) - 0.95

    a = ttm.tensor.Tensor(
        A.flatten().tolist(),
        a_shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.ROW_MAJOR,
        device
    )
    a_t = ttm.tensor.tilize_with_zero_padding(a)
    print("Shape of A_t - " + str(a_t.shape()))
    b_t = ttm.tensor.Tensor(tilize_to_list(B), b_shape, ttm.tensor.DataType.BFLOAT16, ttm.tensor.Layout.TILE, device)
    print("Shape of B_t - " + str(b_t.shape()))
    t2 = ttm.tensor.bmm(a_t, b_t)
    assert(t2.shape() == output_shape)
    tt_host_rm = t2.to(host).data()
    pyt_got_back = torch.Tensor(tt_host_rm).reshape(output_shape)
    #TODO: add support to remove padding in untilize
    pyt_got_back_rm = untilize(pyt_got_back)

    ref_bmm = torch.matmul(A_padded.reshape(a_shape_padded[1:]), B.reshape(b_shape[1:]))
    ref_bmm = ref_bmm.reshape(output_shape)
    maxmag = ref_bmm.abs().max().item() # % of max magnitude since that determines cancellations
    match = is_close(pyt_got_back_rm, ref_bmm, 0.07, 0.07, maxmag, 0.01)
    print("Match=", match.item())

if __name__ == "__main__":
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_tilize_matmul_test(4, 32*9, 32)
    ttm.device.CloseDevice(device)

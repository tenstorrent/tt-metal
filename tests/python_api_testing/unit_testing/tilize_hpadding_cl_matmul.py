from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from pymetal import ttlib as ttl
from pymetal.ttlib.utils import tilize_to_list, tilize, untilize, channels_last, _nearest_32, pad_activation
from python_api_testing.models.utility_functions import print_diff_argmax, is_close
import torch

def run_tilize_matmul_test (M, K, N):

    a_shape_logical = [1,K,1,M]
    a_shape = [1,1,M,K]
    a_shape_padded = [1,1,_nearest_32(M),K]
    b_shape = [1,1,K,N]
    output_shape = [1,1,_nearest_32(M),N]

    A_pre_cl = torch.randn(a_shape_logical)
    A = channels_last(A_pre_cl)
    A_padded = pad_activation(A)
    B = torch.randn(b_shape) - 0.95

    a = ttl.tensor.Tensor(
        torch.flatten(A).tolist(),
        a_shape_logical,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.CHANNELS_LAST,
        device
    )
    a_t = ttl.tensor.tilize_with_zero_padding(a)
    b_t = ttl.tensor.Tensor(tilize_to_list(B), b_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)

    t2 = ttl.tensor.matmul(a_t, b_t)
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
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_tilize_matmul_test(4, 32*9, 32)
    ttl.device.CloseDevice(device)

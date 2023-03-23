from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from pymetal import ttlib as ttl
from models.utility_functions import tilize
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close
import torch

def run_tilize_matmul_test(M, K, N):
    a_shape = [1,1,M,K]
    b_shape = [1,1,K,N]

    A = torch.randn(a_shape)
    B = torch.randn(b_shape) - 0.95

    a = ttl.tensor.Tensor(
        A.flatten().tolist(),
        a_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device
    )
    a_t = ttl.tensor.tilize(a)
    b_t = ttl.tensor.Tensor(tilize_to_list(B), b_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    t2 = ttl.tensor.matmul(a_t, b_t)
    assert(t2.shape() == [1, 1, M, N])
    tt_host_rm = t2.to(host).data()
    pyt_got_back = torch.Tensor(tt_host_rm).reshape((1,1,M,N))
    pyt_got_back_rm = untilize(pyt_got_back)

    ref_bmm = torch.matmul(A.reshape(1,M,K), B.reshape(1, K, N))
    ref_bmm = ref_bmm.reshape(1, 1, M, N)
    maxmag = ref_bmm.abs().max().item() # % of max magnitude since that determines cancellations
    match = is_close(pyt_got_back_rm, ref_bmm, 0.07, 0.07, maxmag, 0.01)
    print("Match=", match.item())

if __name__ == "__main__":
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_tilize_matmul_test(32, 32, 32)
    ttl.device.CloseDevice(device)

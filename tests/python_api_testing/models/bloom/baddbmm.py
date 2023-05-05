from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from abc import abstractmethod
import torch
import math
from torch.nn import functional as F

from libs import tt_lib as ttm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
import numpy as np
import python_api_testing.models.bloom.bloom_utils as bloom_utils
from libs import tt_lib as ttl


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0, out=None) -> ttm.tensor.Tensor:

    tt_batch1 = bloom_utils.torch2tt_tensor(batch1, device)
    tt_batch2 = bloom_utils.torch2tt_tensor(batch2, device)

    input_shape = input.shape

    if input_shape[1] == 1:

        input= input.squeeze(0)

        res1 = ttm.tensor.bmm(tt_batch1, tt_batch2)

        print('res1')
        print(res1.shape())
        res1_shape = res1.shape()

        input= input.repeat(1, 1, res1_shape[2], 1)
        tt_input = bloom_utils.torch2tt_tensor(input, device)
        print('input shape')
        print(tt_input.shape())
        res2 = ttm.tensor.add(tt_input, res1)
        #res2 = ttl.tensor.bcast(tt_input, res1, ttl.tensor.BcastOpMath.ADD, ttl.tensor.BcastOpDim.H)
    else:
        tt_input = bloom_utils.torch2tt_tensor(input, device)
        res1 = ttm.tensor.bmm(tt_batch1, tt_batch2)

        print(f"tt_input shape {tt_input.shape()}")
        print(f"res1 shape {res1.shape()}")

        res2 = ttm.tensor.add(tt_input, res1)

    return res2

def run_baddbmm_inference():
    torch.manual_seed(0)
    #input = torch.rand(32, 64, 64)
    #batch1 = torch.rand(32, 64, 32)
    #batch2 = torch.rand(32, 32, 64)

    input = torch.rand(32, 1, 64)
    batch1 = torch.rand(32, 64, 32)
    batch2 = torch.rand(32, 32, 64)

    pt_out = torch.baddbmm(input, batch1, batch2)
    pt_out_size = list(pt_out.shape)

    while len(pt_out_size) < 4:
        pt_out_size.insert(0, 1)

    pt_out = torch.reshape(pt_out, pt_out_size)

    tt_out = tt_baddbmm(device, input, batch1, batch2)

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print(comp_allclose(pt_out, tt_out_converted))
    print(comp_pcc(pt_out, tt_out_converted))


if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_baddbmm_inference()
    ttm.device.CloseDevice(device)

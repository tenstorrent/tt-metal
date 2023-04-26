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


def tt_baddbmm(device, input, batch1, batch2, beta=1.0, alpha=1.0, out=None) -> ttm.tensor.Tensor:

    tt_batch1 = bloom_utils.torch2tt_tensor(batch1, device)
    tt_batch2 = bloom_utils.torch2tt_tensor(batch2, device)
    tt_input = bloom_utils.torch2tt_tensor(input, device)

    tt_beta = bloom_utils.tt_const_tensor(beta, tt_input.shape(), device)
    res1 = ttm.tensor.mul(tt_beta, tt_input)
    res2 = ttm.tensor.bmm(tt_batch1, tt_batch2)

    tt_alpha = bloom_utils.tt_const_tensor(alpha, res2.shape(), device)

    res3 = ttm.tensor.bmm(tt_alpha, res2)
    res4 = ttm.tensor.add(res1, res3)

    return res4

def run_baddbmm_inference():
    torch.manual_seed(0)
    input = torch.rand(32, 64, 64)
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

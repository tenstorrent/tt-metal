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
import bloom_utils as bloom_utils

def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def tt_dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool, device) -> ttm.tensor.Tensor:

    tt_res = bloom_utils.torch2tt_tensor(residual, device)
    out = F.dropout(x, p=prob, training=training)
    tt_out = bloom_utils.torch2tt_tensor(out, device)
    total = ttm.tensor.add(tt_res, tt_out)

    return total

def run_dropout_add_inference(device):
    # Prepare input
    torch.manual_seed(0)
    test_in = torch.rand(1, 1, 64, 64)
    res = torch.rand(1, 1, 64, 64)

    pt_out = dropout_add(test_in, res, 0.3, False)

    tt_out =  tt_dropout_add(test_in, res, 0.3, False)

    tt_out_converted = bloom_utils.tt2torch_tensor(tt_out)

    print(comp_allclose(pt_out, tt_out_converted))
    print(comp_pcc(pt_out, tt_out_converted))

if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_dropout_add_inference(device)
    ttm.device.CloseDevice(device)

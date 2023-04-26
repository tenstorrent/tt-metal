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

def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def tt_dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool, device) -> ttm.tensor.Tensor:

    pad_res_shape = bloom_utils.calculate_shape(residual.shape)

    tt_res_pad = bloom_utils.create_padded_tensor(residual.shape, residual, pad_res_shape, 0, device, input_tensor_start=[0,0,0,0])

    out = F.dropout(x, p=prob, training=training)

    pad_out_shape = bloom_utils.calculate_shape(out.shape)

    tt_out_pad = bloom_utils.create_padded_tensor(out.shape, out, pad_out_shape, 0, device, input_tensor_start=[0,0,0,0])

    total = ttm.tensor.add(tt_res_pad, tt_out_pad)

    return total

def run_dropout_add_inference(device):
    # Prepare input
    torch.manual_seed(0)
    test_in = torch.rand(1, 1, 61, 61)
    res = torch.rand(1, 1, 61, 61)


    pt_out = dropout_add(test_in, res, 0.3, False)

    tt_out =  tt_dropout_add(test_in, res, 0.3, False, device)

    tt_out = bloom_utils.create_unpadded_tensor(tt_out, pt_out.shape, input_tensor_start=[0,0,0,0])

    input_tensor_shape = pt_out.size()

    tt_out_converted = torch.Tensor(tt_out.data()).reshape(*input_tensor_shape)

    print(comp_allclose(pt_out, tt_out_converted))
    print(comp_pcc(pt_out, tt_out_converted))

if __name__ == "__main__":
    # Initialize the device
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_dropout_add_inference(device)
    ttm.device.CloseDevice(device)

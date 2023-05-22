import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import (
    pad_activation,
    pad_weight,
    tilize,
    untilize,
    tilize_to_list,
    print_diff_argmax,
    pad_weight,
    is_close,
    comp_pcc,
)
import torch

@pytest.mark.parametrize(
    "Hat, Wat, Wbt",
    (
        (1, 32, 1),
    ),
)
def mm_bias_test(N, Hat, Wat, Wbt):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    TILE_HEIGHT = TILE_WIDTH = 32

    Ha = Hat * TILE_HEIGHT
    Wa = Wat * TILE_WIDTH
    Wb = Wbt * TILE_WIDTH
    host = ttl.device.GetHost()
    a_shape = [N, 1, Ha, Wa]
    b_shape = [N, 1, Wa, Wb]
    bias_shape_pytorch = [1, 1, 1, Wb]
    bias_shape = [1, 1, 32, Wb]

    torch.manual_seed(0)
    a = torch.rand(a_shape, dtype=torch.bfloat16).float()
    b = torch.rand(b_shape, dtype=torch.bfloat16).float()-0.5
    bias = torch.rand(bias_shape_pytorch, dtype=torch.bfloat16).float()*0.0

    layout_a = ttl.tensor.Layout.TILE
    def tt_a():
        return tilize_to_list(a)

    tta = ttl.tensor.Tensor( tt_a(), a_shape, ttl.tensor.DataType.BFLOAT16, layout_a, device)
    ttb = ttl.tensor.Tensor( tilize_to_list(b), b_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    ttbias = ttl.tensor.Tensor( tilize_to_list(pad_weight(bias)), bias_shape, ttl.tensor.DataType.BFLOAT16, ttl.tensor.Layout.TILE, device)
    out = ttl.tensor.matmul_bias(tta, ttb, ttbias)
    out_shape = [1,1,Ha,Wb]
    out = out.to(host)
    out_pytorch = untilize(torch.tensor(out.data()).reshape(out_shape))
    ttl.device.CloseDevice(device)

    golden_pytorch = torch.matmul(a,b)+bias
    assert(out_pytorch.shape == golden_pytorch.shape)
    print_diff_argmax(out_pytorch, golden_pytorch)
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc

if __name__ == "__main__":
    #
    # Bert dims: [2,1,128,1024]*[2,1,1024,1024]
    # Bias shape: [1,1,32,1024]
    # Output shape: [2, 1, 128, 1024]
    #
    mm_bias_test(1, 1024//32, 128//32, 1024//32)

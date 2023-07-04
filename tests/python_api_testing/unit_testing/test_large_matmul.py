import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

import tt_lib as ttl
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
    "Hat, Wat, Wbt, tilize_act, untilize_out",
    (
        (1, 9, 1, True, True),
        (1, 9, 1, True, False),
        (1, 9, 1, False, True),
        (1, 9, 1, False, False),
        (2, 9, 9, True, True),
        (2, 9, 9, True, False),
        (2, 9, 9, False, True),
        (2, 9, 9, False, False),
        (1, 32, 1, True, True),
        (1, 32, 1, True, False),
        (1, 32, 1, False, True),
        (1, 32, 1, False, False),
    ),
)
def test_run_large_matmul_test(Hat, Wat, Wbt, tilize_act, untilize_out):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    TILE_HEIGHT = TILE_WIDTH = 32

    Ha = Hat * TILE_HEIGHT
    Wa = Wat * TILE_WIDTH
    Wb = Wbt * TILE_WIDTH
    host = ttl.device.GetHost()
    a_shape = [1, 1, Ha, Wa]
    b_shape = [1, 1, Wa, Wb]
    torch.manual_seed(0)
    a = torch.randn(a_shape, dtype=torch.bfloat16).float()
    b = torch.randn(b_shape, dtype=torch.bfloat16).float()

    layout_a = ttl.tensor.Layout.ROW_MAJOR if tilize_act else ttl.tensor.Layout.TILE
    def tt_a():
        if layout_a == ttl.tensor.Layout.ROW_MAJOR:
            return a.flatten().tolist()
        else:
            return tilize_to_list(a)


    tta = ttl.tensor.Tensor(
        tt_a(),
        a_shape,
        ttl.tensor.DataType.BFLOAT16,
        layout_a,
        device)

    ttb = ttl.tensor.Tensor(
        tilize_to_list(b),
        b_shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device)

    out = ttl.tensor.large_bmm(tta, ttb, tilize_act, untilize_out)
    out_shape = [1,1,Ha,Wb]
    out = out.to(host)
    if not untilize_out:
        # untilize
        out = out.to(ttl.tensor.Layout.ROW_MAJOR)
    out_pytorch = torch.tensor(out.data()).reshape(out_shape)
    ttl.device.CloseDevice(device)

    golden_pytorch = torch.matmul(a,b)
    assert(out_pytorch.shape == golden_pytorch.shape)
    passing_pcc, output_pcc = comp_pcc(golden_pytorch, out_pytorch, 0.99)
    print("Passing=", passing_pcc)
    print("Output pcc=", output_pcc)
    assert passing_pcc

if __name__ == "__main__":
    #test_run_large_matmul_test(2, 9, 9, False, True)
    #test_run_large_matmul_test(2, 9, 9, False, True)
    # Incorrect output
    #test_run_conv_as_large_matmul(256, 128, 28, 28, 3, 3, 2, 2, 1, 1)
    #test_run_large_matmul_test(7, 36, 8, True, True)
    # Hang
    # test_run_conv_as_large_matmul(32, 1024, 5, 5, 1, 1, 1, 1, 0, 0)
    #test_run_large_matmul_test(1, 32, 1, True, False)
    # Crash
    #test_run_conv_as_large_matmul(256, 256, 14, 14, 3, 3, 1, 1, 1, 1)
    test_run_large_matmul_test(7, 72, 8, True, True)

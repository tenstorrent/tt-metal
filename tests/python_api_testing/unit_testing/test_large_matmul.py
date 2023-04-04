import pytest
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from libs import tt_lib as ttl
from python_api_testing.models.utility_functions import pad_activation, pad_weight, tilize, untilize, tilize_to_list, print_diff_argmax, pad_weight, is_close
import torch


@pytest.mark.parametrize(
    "Hat, Wat, Wbt, tilize_a, untilize_out",
    (
        (16, 4, 4, False, False),
        (16, 4, 4, False, True),
        (16, 4, 4, True, False),
        (16, 4, 4, True, True),
        (8, 4, 4, False, False),
        (8, 4, 4, False, True),
        (8, 4, 4, True, False),
        (8, 4, 4, True, True),
        (8, 4, 4, True, True),
    ),
)
def test_run_large_matmul_test(Hat, Wat, Wbt, tilize_a, untilize_out):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    TILE_HEIGHT = TILE_WIDTH = 32

    Ha = Hat * TILE_HEIGHT
    Wa = Wat * TILE_WIDTH
    Wb = Wbt * TILE_WIDTH
    torch.manual_seed(0)
    host = ttl.device.GetHost()
    a_shape = [1, 1, Ha, Wa]
    b_shape = [1, 1, Wa, Wb]

    a = torch.randn(a_shape, dtype=torch.bfloat16).float()
    b = torch.eye(*b_shape[2:]).reshape(b_shape)

    layout_a = ttl.tensor.Layout.ROW_MAJOR if tilize_a else ttl.tensor.Layout.TILE

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

    out = ttl.tensor.large_bmm(tta, ttb, tilize_a, untilize_out)
    out_shape = [1,1,Ha,Wb]
    out_pytorch = torch.tensor(out.to(host).data()).reshape(out_shape)
    if not untilize_out:
        out_pytorch = untilize(out_pytorch)
    ttl.device.CloseDevice(device)

    assert (out_pytorch == a).all(), "Output should be identical to pytorch"

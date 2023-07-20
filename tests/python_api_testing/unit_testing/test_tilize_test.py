import pytest
from pathlib import Path
import sys
import torch

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/..")

import numpy as np

import tt_lib as ttl
from tests.python_api_testing.models.utility_functions import tilize


@pytest.mark.parametrize(
    "nb, nc, nh, nw",
    (
        (5, 2, 4, 8),
        (5, 2, 4, 7),
    ),
)
def test_run_tilize_test(nb, nc, nh, nw):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)

    nt = nb * nc * nh * nw
    shape = [nb, nc, 32 * nh, 32 * nw]

    inp = np.random.rand(*shape)

    a = ttl.tensor.Tensor(
        inp.flatten().tolist(),
        shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
        device,
    )
    b = ttl.tensor.tilize(a)
    c = torch.frombuffer(b.cpu().data(), dtype=torch.bfloat16).to(torch.float32).reshape(shape).numpy()

    tilized_inp = tilize(inp.reshape(*shape))

    del b

    ttl.device.CloseDevice(device)
    assert (
        abs(tilized_inp - c) < 0.02
    ).all(), "Max abs difference for tilize can be 0.02 due to bfloat conversions"

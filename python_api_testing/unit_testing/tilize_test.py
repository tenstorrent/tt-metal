from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

from pymetal import ttmetal as ttm
from models.utility_functions import tilize

def run_tilize_test(nb, nc, nh, nw):
    nt = nb * nc * nh * nw
    shape = [nb, nc, 32 * nh, 32 * nw]

    inp = np.random.rand(*shape)

    a = ttm.tensor.Tensor(
        inp.flatten().tolist(),
        shape,
        ttm.tensor.DataType.BFLOAT16,
        ttm.tensor.Layout.ROW_MAJOR,
        device
    )
    b = ttm.tensor.tilize(a)
    c = np.array(b.to(host).data(), dtype=float).reshape(*shape)

    tilized_inp = tilize(inp.reshape(*shape))

    assert (abs(tilized_inp - c) < 0.02).all(), "Max abs difference for tilize can be 0.02 due to bfloat conversions"

if __name__ == "__main__":
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    host = ttm.device.GetHost()
    run_tilize_test(5, 2, 4, 8) # fast power of 2 width path
    run_tilize_test(5, 2, 4, 7) # slow non-power of 2 width path
    ttm.device.CloseDevice(device)

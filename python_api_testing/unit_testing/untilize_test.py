from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

from gpai import gpai
from models.utility_functions import untilize

def run_untilize_test(nb, nc, nh, nw):
    nt = nb * nc * nh * nw
    shape = [nb, nc, 32 * nh, 32 * nw]

    inp = np.random.rand(*shape)

    a = gpai.tensor.Tensor(
        inp.flatten().tolist(),
        shape,
        gpai.tensor.DataFormat.FLOAT32,
        gpai.tensor.Layout.TILE,
        device
    )
    b = gpai.tensor.untilize(a)
    c = np.array(b.to(host).data(), dtype=float).reshape(*shape)

    untilized_inp = untilize(inp.reshape(*shape))
    assert (abs(untilized_inp - c) < 0.02).all(),  "Max abs difference for untilize can be 0.02 due to bfloat conversions"

if __name__ == "__main__":
    device = gpai.device.CreateDevice(gpai.device.Arch.GRAYSKULL, 0)
    gpai.device.InitializeDevice(device)
    host = gpai.device.GetHost()
    run_untilize_test(5, 2, 4, 8) # fast power of 2 width path
    run_untilize_test(5, 2, 4, 7) # slow non-power of 2 width path

    gpai.device.CloseDevice(device)

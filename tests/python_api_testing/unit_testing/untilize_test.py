from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import numpy as np

from pymetal import ttlib as ttl
from models.utility_functions import untilize

def run_untilize_test(nb, nc, nh, nw):
    nt = nb * nc * nh * nw
    shape = [nb, nc, 32 * nh, 32 * nw]

    inp = np.random.rand(*shape)

    a = ttl.tensor.Tensor(
        inp.flatten().tolist(),
        shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device
    )
    b = ttl.tensor.untilize(a)
    c = np.array(b.to(host).data(), dtype=float).reshape(*shape)

    untilized_inp = untilize(inp.reshape(*shape))
    assert (abs(untilized_inp - c) < 0.02).all(),  "Max abs difference for untilize can be 0.02 due to bfloat conversions"

if __name__ == "__main__":
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_untilize_test(5, 2, 4, 8) # fast power of 2 width path
    run_untilize_test(5, 2, 4, 7) # slow non-power of 2 width path

    ttl.device.CloseDevice(device)

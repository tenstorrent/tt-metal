from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")

import numpy as np

import ll_buda_bindings.ll_buda_bindings._C as _C
from models.utility_functions import untilize

def run_untilize_test():
    nb = 5
    nc = 2
    nh = 4
    nw = 4
    nt = nb * nc * nh * nw
    shape = [nb, nc, 32 * nh, 32 * nw]

    inp = np.random.rand(*shape)

    a = _C.tensor.Tensor(
        inp.flatten().tolist(),
        shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device
    )
    b = _C.tensor.untilize(a)
    c = np.array(b.to(host).data(), dtype=float).reshape(*shape)

    untilized_inp = untilize(inp.reshape(*shape))
    assert (abs(untilized_inp - c) < 0.02).all(),  "Max abs difference for untilize can be 0.02 due to bfloat conversions"

if __name__ == "__main__":
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()
    run_untilize_test()
    _C.device.CloseDevice(device)

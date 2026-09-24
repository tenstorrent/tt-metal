import sys

import torch
import ttnn

sys.path.insert(0, "/localdev/mnedeljkovic/tt-metal/tests/ttnn/unit_tests/operations/gated_delta_net_backward")
from test_gated_delta_net_backward_debug import device_algorithm, pcc

from eval.golden_tests.gated_delta_net_backward.helpers import make_reference_inputs
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward


def rel_rms(got, exp):
    d = exp.double().pow(2).mean().sqrt()
    if d == 0:
        return float(got.double().pow(2).mean().sqrt())
    return float((got.double() - exp.double()).pow(2).mean().sqrt() / d)


def test_loose(device):
    shape, chunk = (1, 128, 2, 64, 64), 32
    ref = make_reference_inputs(shape, state_mode="with_h0_and_dht", g_scale=8.0)
    exp, _, _ = device_algorithm(
        ref["q"],
        ref["k"],
        ref["v"],
        ref["g"],
        ref["beta"],
        ref["do"],
        dht=ref["dht"],
        initial_state=ref["h0"],
        chunk_size=chunk,
    )

    def dev(t):
        if t is None:
            return None
        return ttnn.from_torch(
            t.to(torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    got = gated_delta_net_backward(
        dev(ref["q"]),
        dev(ref["k"]),
        dev(ref["v"]),
        dev(ref["g"]),
        dev(ref["beta"]),
        dev(ref["do"]),
        dht=dev(ref["dht"]),
        initial_state=dev(ref["h0"]),
        chunk_size=chunk,
    )
    for name, a, e in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), got, exp):
        if e is None or a is None:
            continue
        h = ttnn.to_torch(a).double()
        print(f"LOOSE {name}: pcc={pcc(h, e):.6f} relrms={rel_rms(h, e):.5f} maxabs={float((h-e).abs().max()):.3g}")

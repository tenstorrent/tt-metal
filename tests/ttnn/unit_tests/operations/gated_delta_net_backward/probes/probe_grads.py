import os
import sys

import torch
import ttnn
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

sys.path.insert(0, "/localdev/mnedeljkovic/tt-metal/tests/ttnn/unit_tests/operations/gated_delta_net_backward")
from test_gated_delta_net_backward_debug import device_algorithm, make_inputs, pcc

DT = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16}
TDT = {"fp32": torch.float32, "bf16": torch.bfloat16}


def test_grads(device):
    dtype = os.environ.get("GDN_DT", "fp32")
    cases = [((1, 32, 1, 32, 32), 32), ((1, 128, 2, 64, 64), 32), ((1, 128, 2, 64, 128), 64)]
    for shape, chunk in cases:
        ref = make_inputs(shape)
        exp, _, _ = device_algorithm(ref["q"], ref["k"], ref["v"], ref["g"], ref["beta"], ref["do"], chunk_size=chunk)

        def dev(t):
            return ttnn.from_torch(
                t.to(TDT[dtype]),
                dtype=DT[dtype],
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
            chunk_size=chunk,
        )
        line = [dtype, str(shape), f"c{chunk}"]
        for name, a, e in zip(("dq", "dk", "dv", "dg", "dbeta"), got, exp):
            h = ttnn.to_torch(a).double()
            line.append(f"{name}={pcc(h, e):.5f}")
        print("GRADPCC " + " ".join(line))

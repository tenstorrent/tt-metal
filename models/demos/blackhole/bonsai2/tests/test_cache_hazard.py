# Two ternary linears of the same shape, different weights: does a program-cache hit run the second with the first's
# buffer addresses?
import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3, reference


def test_cache_hazard(device):
    device.enable_program_cache()
    K, N = 1024, 64
    rng = np.random.default_rng(0)
    x = torch.randn(K).bfloat16()
    act = ttnn.from_torch(
        x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    outs = []
    for i in range(2):
        t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
        s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
        lin = TernaryDecodeLinear3.from_host(device, t, s, inline_scales=True)
        y = ttnn.to_torch(lin(act)).float()[0, 0] if len(lin(act).shape) == 4 else None
        y = ttnn.to_torch(lin(act)).float().reshape(-1, N)[0]
        ref = reference(x, t, s)
        rel = ((y - ref).norm() / ref.norm()).item()
        print(f"CACHE call {i} rel {rel:.4f}")
        outs.append(rel)
    assert max(outs) < 0.02, outs

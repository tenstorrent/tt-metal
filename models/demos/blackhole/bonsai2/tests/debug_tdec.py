import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear import TernaryDecodeLinear, block_diag_activation

device = ttnn.open_device(device_id=0)
try:
    K, N = 1024, 64
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    x = torch.randn(K).bfloat16()
    a = block_diag_activation(x)
    lin = TernaryDecodeLinear.from_host(device, t, s, cores=1)
    act = ttnn.from_torch(x.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    P = a.float() @ torch.from_numpy(t).float()  # [32, N]: row kb = partial sums of block kb
    S = torch.zeros(32, N)
    S[: K // 128] = torch.from_numpy(s.astype(np.float32)).T
    exp = {1: P, 2: P * S, 3: P * S}
    for d in (1, 2, 3, 0):
        y = ttnn.to_torch(lin(act, debug=d)).float()
        if d == 0:
            ref = (P * S).sum(0)
            print("final", (y[0] - ref).norm().item() / ref.norm().item(), y[0, :4].tolist(), ref[:4].tolist())
            continue
        e = exp[d]
        rel = ((y - e).norm() / e.norm()).item()
        print("debug", d, "rel", rel, "rows used", (e.abs().sum(1) > 0).nonzero().flatten().tolist())
        print(
            "  got row0",
            y[0, :4].tolist(),
            "exp",
            e[0, :4].tolist(),
            " got row1",
            y[1, :4].tolist(),
            "exp",
            e[1, :4].tolist(),
        )
finally:
    ttnn.close_device(device)

import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3

device = ttnn.open_device(device_id=0)
try:
    K, N = 512, 32
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    x = torch.randn(K).bfloat16()
    act = ttnn.from_torch(x.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    P = x.float()[:128] @ torch.from_numpy(t[:128]).float()
    S = torch.from_numpy(s[:, 0].astype(np.float32))
    lin = TernaryDecodeLinear3.from_host(device, t, s, cores=1, group16=True)
    o = ttnn.to_torch(lin(act, dbg=True)).float()  # tiles P, S, acc
    p, sc, acc = o[:, 0:32], o[:, 32:64], o[:, 64:96]
    nz = lambda m: [r for r in range(32) if m[r].abs().sum() > 0]
    print("P rows", nz(p), "rel", ((p[0] - P).norm() / P.norm()).item())
    print(
        "S rows",
        nz(sc),
        "S row0 rel",
        ((sc[0] - S).norm() / S.norm()).item(),
        "S row2 vs kb1",
        ((sc[2] - torch.from_numpy(s[:, 1].astype(np.float32))).norm()).item(),
    )
    print("acc rows", nz(acc), "acc rel", ((acc[0] - P * S).norm() / (P * S).norm()).item())
    print("acc row0[12:20]", acc[0, 12:20].tolist())
    print("exp      [12:20]", (P * S)[12:20].tolist())
finally:
    ttnn.close_device(device)

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
    for fp32 in (False, True):
        lin = TernaryDecodeLinear3.from_host(device, t, s, cores=1, scale_fp32=fp32)
        o = ttnn.to_torch(lin(act, dbg=True)).float()  # [32, 96]: tiles P, S, acc
        p, sc, acc = o[:, 0:32], o[:, 32:64], o[:, 64:96]
        nzrows = lambda m: [r for r in range(32) if m[r].abs().sum() > 0]
        print(
            f"fp32={fp32} P rows nz {nzrows(p)} rel {((p[0]-P).norm()/P.norm()).item():.4f}  S rows nz {nzrows(sc)} S row0 rel {((sc[0]-S).norm()/S.norm()).item():.4f}  acc rows nz {nzrows(acc)} acc rel {((acc[0]-P*S).norm()/(P*S).norm()).item():.4f}"
        )
        print("   S row0[14:20]", sc[0, 14:20].tolist(), "exp", S[14:20].tolist())
        print("   P row0[14:20]", p[0, 14:20].tolist(), "exp", P[14:20].tolist())
finally:
    ttnn.close_device(device)

# NormRotation (RMSNorm + rotation in one op): the norm against torch fp32 (the interleaved ttnn.rms_norm of a
# [1, 1, K] activation is 4.4% off torch here), the rotation against DecodeRotation of the same normalised input.
import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation, NormRotation


def test_norm_rot(device):
    torch.manual_seed(0)
    K = 5120
    signs = np.sign(np.random.default_rng(0).standard_normal(K)).astype(np.float32)
    w = (1.0 + 0.3 * torch.randn(K)).bfloat16().float()
    x = (torch.randn(1, 1, K) * 3).bfloat16()
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xf = x.float().reshape(-1)
    t = xf / torch.sqrt((xf * xf).mean() + 1e-6) * w
    tn = ttnn.from_torch(t.reshape(1, 1, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ref_r = ttnn.to_torch(DecodeRotation(device, signs)(tn)).float()
    xn, xr = NormRotation(device, signs, w, 1e-6)(xt)
    xn = ttnn.to_torch(xn).float().reshape(-1, K)[0]
    xr = ttnn.to_torch(xr).float()
    rn = ((xn - t).norm() / t.norm()).item()
    rr = ((xr - ref_r).norm() / ref_r.norm()).item()
    q = xn / t
    print(
        f"NROT norm rel {rn:.5f} rotated rel {rr:.5f} ratio mean {q.mean():.4f} std {q.std():.4f} first {q[:3].tolist()} block means {[round(q[1024*b:1024*b+1024].mean().item(), 4) for b in range(5)]}"
    )
    assert rn < 0.01 and rr < 0.01

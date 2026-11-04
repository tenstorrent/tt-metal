# DecodeRotation(g, u=u) (silu(g) * u rotated in one op) against ttnn.mul + DecodeRotation.
import numpy as np
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation


def test_rot_glu(device):
    torch.manual_seed(0)
    K = 17408
    signs = np.sign(np.random.default_rng(0).standard_normal(K)).astype(np.float32)
    rot = DecodeRotation(device, signs)
    g = (torch.randn(1, 1, K) * 2).bfloat16()
    u = torch.randn(1, 1, K).bfloat16()
    gt = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ut = ttnn.from_torch(u, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ref = ttnn.to_torch(rot(ttnn.mul(gt, ut, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]))).float()
    out = ttnn.to_torch(rot(gt, u=ut)).float()
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"ROTGLU rel {rel:.5f}")
    assert rel < 0.01

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import fwht_blocks
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation


@pytest.mark.parametrize("K", [5120, 17408])
def test_rotate(device, K):
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=K).astype(np.float32)
    x = torch.randn(32, K).bfloat16()
    rot = DecodeRotation(device, signs)
    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    out = ttnn.to_torch(rot(xt)).float()
    ref = torch.from_numpy(fwht_blocks(x[0].float().numpy().astype(np.float64) * signs, 1024).astype(np.float32))
    for c in (0, 7):
        rel = ((out[c] - ref).norm() / ref.norm()).item()
        print(f"K={K} copy {c} rel {rel:.6f}")
        assert rel < 0.01


def test_rotate_copy(device):
    K = 5120
    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=K).astype(np.float32)
    x = torch.randn(32, K).bfloat16()
    rot = DecodeRotation(device, signs)
    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    out = ttnn.to_torch(rot(xt, debug_copy=True)).float()
    ref = x[0].float() * torch.from_numpy(signs)
    print("copy-mode got", out[0, :6].tolist(), "exp", ref[:6].tolist())
    print("copy-mode rel", ((out[0] - ref).norm() / ref.norm()).item())

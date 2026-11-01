# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_1C_CONV_HM equivalence: the head-major decode conv ([1, C/128, 128] taps and state) must give
the same per-head q/k/v as the row form ([1, 1, C], then slice + reshape). Elementwise ops on the same
values in a different tile packing, so the check is exact equality per head.

    pytest models/demos/blackhole/bonsai2/tp1/test_conv_hm.py -s
"""
import pytest
import torch

import ttnn

NK, NV, D, K = 16, 48, 128, 4
C = (2 * NK + NV) * D


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_conv_hm(mesh_device):
    torch.manual_seed(0)
    taps = [torch.randn(1, 1, C) for _ in range(K)]
    states = [torch.randn(1, 1, C) for _ in range(K)]

    def dev(t, shape):
        return ttnn.from_torch(
            t.reshape(shape).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )

    def conv(shape):
        st = [dev(s, shape) for s in states]
        tp = [dev(t, shape) for t in taps]
        acc = ttnn.multiply(st[0], tp[0])
        for j in range(1, K):
            acc = ttnn.mac(st[j], tp[j], acc)
        return ttnn.silu(acc)

    row = conv((1, 1, C))
    kd = NK * D
    ref = [
        ttnn.reshape(ttnn.slice(row, (0, 0, 0), (1, 1, kd)), (1, NK, D)),
        ttnn.reshape(ttnn.slice(row, (0, 0, kd), (1, 1, 2 * kd)), (1, NK, D)),
        ttnn.reshape(ttnn.slice(row, (0, 0, 2 * kd), (1, 1, C)), (1, NV, D)),
    ]
    hm = conv((1, C // D, D))
    got = [
        ttnn.slice(hm, (0, 0, 0), (1, NK, D)),
        ttnn.slice(hm, (0, NK, 0), (1, 2 * NK, D)),
        ttnn.slice(hm, (0, 2 * NK, 0), (1, 2 * NK + NV, D)),
    ]
    for name, r, g in zip("qkv", ref, got):
        r, g = ttnn.to_torch(r), ttnn.to_torch(g)
        assert r.shape == g.shape, (name, r.shape, g.shape)
        bad = (r != g).flatten(0, -2).any(-1).sum().item()
        print(f"{name}: shape {tuple(g.shape)}, heads differing {bad}, max|diff| {(r - g).abs().max().item():.3e}")
        assert bad == 0

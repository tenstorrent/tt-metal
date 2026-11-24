# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill ring write of a long chunk (tt/fused/pf_ring.py): the ring equals the op chain's (masked sum of the
bf16 blocks, typecast to bfp8, fill_cache), for chunk lengths n at and off tile and block boundaries."""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_ring import WINDOW, PfRing

H, D = 8, 128


def _chain(dev, k, ring, n):
    """prefill_model._write_ring, long chunk: every block, masks from the slot sources."""
    t = k.shape[2]
    q = torch.arange(WINDOW)
    src = q + WINDOW * ((n - 1 - q) // WINDOW)
    zero = ttnn.from_torch(torch.zeros(1, H, WINDOW, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    new = None
    for blk in range(-(-t // WINDOW)):
        lo, hi = blk * WINDOW, min(t, (blk + 1) * WINDOW)
        part = ttnn.slice(k, [0, 0, lo, 0], [1, H, hi, D])
        if hi - lo < WINDOW:
            part = ttnn.concat([part, ttnn.slice(zero, [0, 0, 0, 0], [1, H, WINDOW - (hi - lo), D])], dim=2)
        m = ((src >= lo) & (src < lo + WINDOW)).float().view(1, 1, WINDOW, 1)
        m = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        term = ttnn.multiply(part, m)
        new = term if new is None else ttnn.add(new, term)
    ttnn.fill_cache(ring, ttnn.typecast(new, ttnn.bfloat8_b), 0, update_idx=0)


@pytest.mark.parametrize("t", [3104, 8448])
def test_pf_ring(device, t):
    dev = device
    g = torch.Generator().manual_seed(t)
    b8, tile = ttnn.bfloat8_b, ttnn.TILE_LAYOUT
    mk = lambda: torch.randn(1, H, t, D, generator=g) * 2.0 ** torch.randint(
        -6, 7, (1, H, t, 1), generator=g
    )  # noqa: E731
    ks = [ttnn.from_torch(mk(), dtype=ttnn.bfloat16, layout=tile, device=dev) for _ in range(2)]
    cast = [ttnn.typecast(k, b8) for k in ks]
    ring = lambda v: ttnn.from_torch(torch.full((1, H, WINDOW, D), v), dtype=b8, layout=tile, device=dev)  # noqa: E731
    nreal = ttnn.from_torch(torch.zeros(1, 16, dtype=torch.int32), dtype=ttnn.uint32, device=dev)
    mod = PfRing(dev)
    ns = [1025, 1040, 1056, 2047, 2048, 2049, t - 31, t - 17, t] + ([4096, 7424 + 13] if t > 4096 else [])
    for n in ns:
        want, got = [ring(3.0), ring(3.0)], [ring(5.0), ring(5.0)]
        for k, w in zip(ks, want):
            _chain(dev, k, w, n)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[n] + [0] * 15], dtype=torch.int32), dtype=ttnn.uint32), nreal
        )
        mod(list(zip(cast, got)), nreal)
        for i, (w, x) in enumerate(zip(want, got)):
            a, b = ttnn.to_torch(w).float(), ttnn.to_torch(x).float()
            assert torch.equal(a, b), (t, n, i, int((a != b).sum()))

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The host-side arithmetic of fused/collective/collective.py: no device."""

import torch

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl


def test_payload_mask():
    # 64 bytes of Float32 are the first 16 elements of a tile's first face: row 0, columns 0-15.
    mask = cl.payload_mask(64, ttnn.float32, 1)
    assert int(mask.sum()) == 16 and bool(mask[0, :16].all())
    # A tile and a half: the second tile's first two faces are its rows 0-15.
    mask = cl.payload_mask(4096 + 2048, ttnn.float32, 2)
    assert bool(mask[:32].all()) and bool(mask[32:48].all()) and not bool(mask[48:].any())
    # BFloat16: a tile and a quarter is one more face, rows 0-15 x columns 0-15.
    mask = cl.payload_mask(2048 + 512, ttnn.bfloat16, 2)
    assert bool(mask[:32].all()) and bool(mask[32:48, :16].all()) and int(mask.sum()) == 1024 + 256
    # Striped: every part has its own first bytes.
    mask = cl.payload_mask(64, ttnn.float32, 1, parts=3)
    assert mask.shape == (96, 32) and all(bool(mask[32 * k, :16].all()) for k in range(3)) and int(mask.sum()) == 48


def test_rings_and_cores():
    assert cl.axis_rings((8, 4), 1, [2]) == [[(2, 0), (2, 1), (2, 2), (2, 3)]]
    rings = cl.axis_rings((8, 4), 0)
    assert len(rings) == 4 and rings[3][:2] == [(0, 3), (1, 3)] and all(len(ring) == 8 for ring in rings)
    assert cl.row_cores((5, 0), 3) == [(5, 0), (6, 0), (7, 0)]


def test_to_bf16():
    # Ties: half away from zero keeps the larger magnitude, half to even the even mantissa.
    x = torch.tensor([-6.203125, 122.25, 13.90625, 1.0, -1.0, 0.0])
    assert cl.to_bf16(x, "away").tolist() == [-6.21875, 122.5, 13.9375, 1.0, -1.0, 0.0]
    assert cl.to_bf16(x, "even").tolist() == [-6.1875, 122.0, 13.875, 1.0, -1.0, 0.0]
    # Off a tie the two agree, and BFloat16 values pass through.
    g = torch.Generator().manual_seed(0)
    y = torch.randn(4096, generator=g) * 37.0
    away, even = cl.to_bf16(y, "away"), cl.to_bf16(y, "even")
    ties = (y.view(torch.int32) & 0xFFFF) == 0x8000
    assert bool((away[~ties] == even[~ties]).all()) and bool((cl.to_bf16(even, "away") == even).all())

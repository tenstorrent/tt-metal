# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproducers for TTNN op bugs with tiny tiles (TILE_HEIGHT < 32, e.g. 16x32).

The pi0.5 action-expert decode path wants a 16-row activation tile (one 32-tile holds two
suffix rows), but several TTNN ops misbehave at a 16x32 tile. Each test below runs the SAME op at
a 32x32 tile (passes) and a 16x32 tile (fails on the current build) so the regression is isolated
to the tiny-tile geometry.

Only bfloat16 is used -- blocked dtypes (bfloat8_b/bfloat4_b) are not supported at a tiny tile.

Run:  pytest models/experimental/pi0_5/tests/test_tiny_tile_ttnn_bugs.py
"""
from __future__ import annotations

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

_DEVICE_PARAMS = {"l1_small_size": 24576}
_PCC = 0.99


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1, t2 = a.flatten().float(), b.flatten().float()
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - torch.mean(t1)) * (t2 - torch.mean(t2)))
    return (cov / (s1 * s2)).item()


def _to_dev(t: torch.Tensor, dev, tile_h: int, mem=None):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=mem or ttnn.L1_MEMORY_CONFIG,
        tile=ttnn.Tile((tile_h, 32)),
    )


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("tile_h", [32, 16], ids=["tile32", "tile16"])
def test_sdpa_tiny_tile_numerics(mesh_device, tile_h):
    """ttnn.transformer.scaled_dot_product_attention is numerically wrong with 16-row q/k/v tiles.

    Same inputs at a 32-row tile give PCC ~0.9999; at a 16-row tile PCC collapses to ~0.5. The
    pi0.5 decode block works around this by retiling q/k/v up to a 32 tile around SDPA
    (denoise_block._sdpa_retile_32).
    """
    torch.manual_seed(0)
    nh, sq, skv, hd = 8, 32, 1056, 256
    scale = 1.0 / (hd**0.5)
    q = torch.randn(1, nh, sq, hd) * 0.1
    k = torch.randn(1, 1, skv, hd) * 0.1
    v = torch.randn(1, 1, skv, hd) * 0.1
    kb, vb = k.expand(1, nh, skv, hd), v.expand(1, nh, skv, hd)
    ref = torch.softmax((q @ kb.transpose(-1, -2)) * scale, dim=-1) @ vb

    qt, kt, vt = _to_dev(q, mesh_device, tile_h), _to_dev(k, mesh_device, tile_h), _to_dev(v, mesh_device, tile_h)
    out = ttnn.transformer.scaled_dot_product_attention(qt, kt, vt, is_causal=False, scale=scale)
    pcc = _pcc(ref, ttnn.to_torch(out))
    print(f"\n[SDPA tile_h={tile_h}] PCC={pcc:.5f}")
    assert pcc >= _PCC, f"SDPA PCC {pcc:.5f} < {_PCC} at tile_h={tile_h} (tiny-tile SDPA numerics bug)"


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("tile_h", [32, 16], ids=["tile32", "tile16"])
def test_pad_tiny_tile_corrupts_data(mesh_device, tile_h):
    """ttnn.pad corrupts the tensor when it pads the sequence of a 16x32-tile tensor.

    Padding [1,1,1040,256] -> [1,1,1056,256] (16 rows) preserves the original values at a 32-row
    tile (PCC ~1.0) but scrambles them at a 16-row tile (PCC ~0.02).
    """
    torch.manual_seed(0)
    x = torch.randn(1, 1, 1040, 256) * 0.1
    xt = _to_dev(x, mesh_device, tile_h)
    xp = ttnn.pad(xt, [(0, 0), (0, 0), (0, 16), (0, 0)], value=0.0)
    got = ttnn.to_torch(xp)[:, :, :1040, :]
    pcc = _pcc(x, got)
    print(f"\n[pad tile_h={tile_h}] preserved-PCC={pcc:.5f}")
    assert pcc >= _PCC, f"pad PCC {pcc:.5f} < {_PCC} at tile_h={tile_h} (tiny-tile pad corruption bug)"


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("tile_h", [32, 16], ids=["tile32", "tile16"])
def test_addcmul_tiny_tile_promotes_tile(mesh_device, tile_h):
    """ttnn.addcmul silently promotes a 16x32 input tile to a 32x32 output tile.

    The math is correct (PCC ~1.0), but the output tile geometry does not match the 16x32 inputs.
    This breaks the tiny-tile residual stream (the promoted 32-row tile makes a downstream
    sharded op see a non-32-aligned M and crash). The pi0.5 block avoids addcmul on the tiny-tile
    residual, using explicit multiply + add instead (denoise_block._gated_residual).
    """
    torch.manual_seed(0)
    res = torch.randn(1, 16, 1024) * 0.5
    gate = torch.randn(1, 1, 1024) * 0.1
    x = torch.randn(1, 16, 1024) * 0.1
    rt, gt, xt = _to_dev(res, mesh_device, tile_h), _to_dev(gate, mesh_device, tile_h), _to_dev(x, mesh_device, tile_h)
    out = ttnn.addcmul(rt, gt, xt, value=1.0)
    out_tile = tuple(out.get_tile().tile_shape)
    print(
        f"\n[addcmul tile_h={tile_h}] in_tile=({tile_h}, 32) out_tile={out_tile} PCC={_pcc(res + gate * x, ttnn.to_torch(out)):.5f}"
    )
    assert out_tile == (tile_h, 32), (
        f"addcmul promoted the output tile to {out_tile} (expected ({tile_h}, 32)) at tile_h={tile_h} "
        "(tiny-tile addcmul tile-promotion bug)"
    )

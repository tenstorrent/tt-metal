# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit test of the fused partial-RoPE kernel (tt/attn_fused.py:rope_inplace) vs a torch adjacent-pair rotation."""
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import attn_fused as AF


def rot_ref(x, c, s):  # x [..., 512]; c,s [..., 64] (pair-repeated)
    r = x.clone().float()
    t = x[..., 448:].float()
    sw = torch.zeros_like(t)
    sw[..., 0::2] = -t[..., 1::2]
    sw[..., 1::2] = t[..., 0::2]
    r[..., 448:] = t * c + sw * s
    return r


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="p")], indirect=True)
@pytest.mark.parametrize("T", [4, 16])
@torch.no_grad()
def test_rope_inplace(mesh_device, T):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t.to(torch.bfloat16).contiguous(),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    down = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    ang = torch.rand(T, 32) * 6.0
    cos, sin = ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)
    # heads layout
    x = torch.randn(1, T, 9, 512).to(torch.bfloat16).float()
    xp = torch.zeros(1, T, 32, 512)
    xp[:, :, :9] = x
    C = torch.ones(1, T, 1, 512)
    S = torch.zeros(1, T, 1, 512)
    C[..., 448:], S[..., 448:] = cos.reshape(1, T, 1, 64), sin.reshape(1, T, 1, 64)
    for sign in (1, -1):
        tx = up(xp)
        AF.rope_inplace(tx, up(C), up(S * sign), T)
        got = down(tx)[:, :, :9]
        want = rot_ref(x, cos.reshape(1, T, 1, 64), sign * sin.reshape(1, T, 1, 64))
        err = (got - want).abs().max().item()
        print(f"heads T={T} sign={sign}: max abs err {err:.4f} (ref absmax {want.abs().max():.2f})", flush=True)
        assert err < 0.05
    # rows layout (all users share row 0 of the tables)
    xr = torch.randn(1, 1, T, 512).to(torch.bfloat16).float()
    Cr = torch.ones(1, 1, T, 512)
    Sr = torch.zeros(1, 1, T, 512)
    Cr[..., 448:], Sr[..., 448:] = cos[0], sin[0]
    tx = up(xr)
    AF.rope_inplace(tx, up(Cr), up(Sr), 1, rows_layout=True)
    got = down(tx)[:, :, :T]
    want = rot_ref(xr, cos[0], sin[0])
    err = (got - want).abs().max().item()
    print(f"rows T={T}: max abs err {err:.4f}", flush=True)
    assert err < 0.05

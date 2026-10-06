# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Engram device forward at T=256 (one call, DSV41_PFA_ENGRAM_BATCH) vs eight calls at T=32 on the same inputs (the old column-split loop): PCC / max diff."""
import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_engram_batch(mesh_device):
    md = mesh_device
    chain = DSV41DecodeChain(md, users_per_row=4, max_comp=256)
    sh = _Shards()
    e = DSV41DeviceEngram(md, 1, sh, mesh_config=chain.mesh_config, ccl=chain.ccl)
    T, G = 32, 8
    g = torch.Generator().manual_seed(1)
    rep = ttnn.ReplicateTensorToMesh(md)
    x = torch.randn(T * G, 1, 4, 5120, generator=g)
    rows = torch.randn(1, 1, T * G, e.kin, generator=g).to(torch.bfloat16)
    xd = ttnn.from_torch(x, device=md, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
    rd = ttnn.from_torch(rows, device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
    outs = []
    for j in range(G):
        xj = ttnn.slice(xd, [j * T, 0, 0, 0], [(j + 1) * T, 1, 4, 5120])
        rj = ttnn.slice(rd, [0, 0, j * T, 0], [1, 1, (j + 1) * T, e.kin])
        e.forward_v2(xj, rj)  # eager warm
        outs.append(e.forward_v2(xj, rj))
    ref = ttnn.concat(outs, dim=0)
    e.forward_v2(xd, rd)
    bat = e.forward_v2(xd, rd)
    r = ttnn.to_torch(ttnn.get_device_tensors(ref)[0]).float()
    b = ttnn.to_torch(ttnn.get_device_tensors(bat)[0]).float()
    print(
        f"ENGRAM_BATCH pcc {pcc(b, r):.6f} max abs diff {(b - r).abs().max():.3e} (ref absmax {r.abs().max():.2f})",
        flush=True,
    )
    assert pcc(b, r) > 0.9999

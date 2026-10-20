# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: TtRouter.joint (router_topk's joint rows) vs TtRouter(dense=False) (moe_grouped_topk, fp32 selection) on the
same x: per-token id sets and weights. Random gate weight, the bias in the real layers' range (7.4 .. 7.8)."""

import torch
from loguru import logger

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S_ = spec()
pytestmark = device_timeout(S_)


@mesh_parametrize
def test_router_joint(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.tt.router import TtRouter

    E, H, K, S = 288, 4096, 8, 640
    g = torch.Generator().manual_seed(0)
    w = torch.randn(E, H, generator=g) * 0.02
    b = 7.4 + 0.4 * torch.rand(E, generator=g)
    r = TtRouter(mesh_device, w, b, max_rows=S, top_k=K, route_scale=2.5)
    x = torch.randn(S, H, generator=g)
    xd = ttnn.from_torch(
        x.view(1, 1, S, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    _, idx, wts = r(xd, dense=False)
    ia = ttnn.to_torch(ttnn.get_device_tensors(idx)[0]).reshape(S, K).long()
    wa = ttnn.to_torch(ttnn.get_device_tensors(wts)[0]).reshape(S, K).float()
    rows = r.joint(xd)
    jr = ttnn.to_torch(ttnn.get_device_tensors(rows)[0]).reshape(S, 2 * K).to(torch.int32) & 0xFFFF
    ib = jr[:, :K].long()
    wb = (jr[:, K:].to(torch.int32) << 16).view(torch.float32)
    same = sum(set(ia[t].tolist()) == set(ib[t].tolist()) for t in range(S))
    da = {(t, int(e)): float(v) for t in range(S) for e, v in zip(ia[t], wa[t])}
    diffs = [abs(da[(t, int(e))] - float(v)) for t in range(S) for e, v in zip(ib[t], wb[t]) if (t, int(e)) in da]
    # (the exact reference: fp32 on the host)
    xf = x.to(torch.bfloat16).float()
    sc = torch.sigmoid(xf @ w.T)
    ch = sc + (b - b.mean())
    ti = ch.topk(K, dim=-1).indices
    tw = sc.gather(-1, ti)
    tw = tw / tw.sum(-1, keepdim=True) * 2.5
    ref = {(t, int(e)): float(v) for t in range(S) for e, v in zip(ti[t], tw[t])}
    same_ref = sum(set(ti[t].tolist()) == set(ib[t].tolist()) for t in range(S))
    rd = [abs(ref[(t, int(e))] - float(v)) for t in range(S) for e, v in zip(ib[t], wb[t]) if (t, int(e)) in ref]
    logger.info(
        f"ROUTER joint vs moe_grouped_topk: same id sets {same}/{S}, weight |diff| max {max(diffs):.2e} mean "
        f"{sum(diffs) / len(diffs):.2e}; joint vs host fp32: same sets {same_ref}/{S}, |diff| max {max(rd):.2e}"
    )
    assert same >= S - 2 and max(diffs) < 2e-2

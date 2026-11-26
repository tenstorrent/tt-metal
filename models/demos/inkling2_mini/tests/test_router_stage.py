# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Router weights staged into the router cores' L1 (tt/mk/router_stage.py: stage_pages_fill.cpp, then the router
reader's local copy) vs the router reading them from DRAM: ids and weights bit-equal, random inputs, two layers'
weights through the same staging shard."""

from types import SimpleNamespace

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.router import FusedRouter
from models.demos.inkling2_mini.tt.mk.router_stage import RouterStage
from models.demos.inkling2_mini.tt.moe import HIDDEN, NUM_EXPERTS, ROUTE_SCALE, ROUTER_SPLIT, TOP_K


def test_router_stage(device):
    torch.manual_seed(0)
    routers = []
    for L in range(2):
        w = torch.randn(HIDDEN, NUM_EXPERTS) * 0.05
        wt = ttnn.from_torch(
            w.reshape(1, ROUTER_SPLIT, HIDDEN // ROUTER_SPLIT, NUM_EXPERTS), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
        )  # fmt: skip
        routers.append(FusedRouter(device, wt, torch.randn(NUM_EXPERTS) * 0.01, TOP_K, ROUTE_SCALE, 1e-20, first_row=2))
    model = SimpleNamespace(
        device=device, layers=[SimpleNamespace(L=L, moe=SimpleNamespace(router=r)) for L, r in enumerate(routers)]
    )
    rs = RouterStage(model)
    try:
        for r in routers:
            x = ttnn.from_torch(
                torch.randn(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            w0, i0 = (ttnn.to_torch(t) for t in r(x))
            ttnn.generic_op([r.w, rs.stage], ttnn.ProgramDescriptor(kernels=rs.fill_kernels(r), semaphores=[], cbs=[]))
            w1, i1 = (ttnn.to_torch(t) for t in r(x, w_stage=rs.stage))
            assert torch.equal(i0.reshape(-1)[:TOP_K], i1.reshape(-1)[:TOP_K])
            assert torch.equal(w0.reshape(-1)[:TOP_K], w1.reshape(-1)[:TOP_K])
            assert w0.reshape(-1)[:TOP_K].float().abs().sum() > 0
    finally:
        rs.uninstall()

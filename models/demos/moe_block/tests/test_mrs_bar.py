# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The bar for the fused head's down + SE reduce-scatter stage: the ported matmul_reduce_scatter op
(ttnn/ttnn/operations/matmul_reduce_scatter) as-is on the SE down shapes, over the TP axis of the LoudBox 2x4, at the
model's fabric payload. Times need TT_METAL_DEVICE_PROFILER=1 (see moe_block_runs/parts.sh)."""

import os

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload
from ttnn.operations.ccl import Topology

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.tests.test_tp_collectives import _calls, gen, ncalls

R, C = 2, 4

# name: (I, H, scatter_dim): h [rows, I] @ W [I, H], reduced over TP; GLM scatters rows to the owners, K3 the width
SHAPES = {"glm": (512, 4096, -2), "k3": (1536, 7168, -1)}


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_mrs_bar(mesh_device, device_params):
    from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

    rows = int(os.environ.get("HXC_ROWS", "640"))
    links = int(os.environ.get("MRS_LINKS", "2"))
    for name in os.environ.get("MRS_CASES", ",".join(SHAPES)).split(","):
        I, H, sd = SHAPES[name]
        g = torch.Generator().manual_seed(5)
        w = (torch.randn(R, C, I, H, generator=g) * I**-0.5).to(torch.bfloat16)
        hs = [torch.randn(R, C, rows, I, generator=gen(5, k)).to(torch.bfloat16) for k in range(ncalls())]  # (per call)

        def dev(t, dtype):
            glob = torch.cat([torch.cat(list(t[r]), dim=1) for r in range(R)], dim=0)
            return ttnn.from_torch(
                glob,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C)),
            )

        hts, wt = [dev(h, ttnn.bfloat16) for h in hs], dev(w, ttnn.bfloat8_b)
        wq = ttnn.to_torch(
            wt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=(0, 1))
        ).float()
        wq = torch.stack([torch.stack(list(wq[r * I : (r + 1) * I].split(H, dim=1))) for r in range(R)])
        run = lambda k: matmul_reduce_scatter(
            hts[k], wt, cluster_axis=1, scatter_dim=sd, topology=Topology.Linear, num_links=links
        )

        def check(k, y):  # (call k against its own h)
            ys = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(y)]
            worst = 1.0
            for r in range(R):
                total = sum(hs[k][r, c].float() @ wq[r, c] for c in range(C))
                for c in range(C):
                    worst = min(worst, pcc(ys[r * C + c].reshape(-1), torch.chunk(total, C, dim=sd)[c].reshape(-1)))
            assert worst > 0.999, (k, worst)
            return worst

        us, ws = _calls(mesh_device, run, check)
        logger.info(
            f"MRS {name} rows {rows} links {links}: {us} us, worst PCC {min(ws):.6f} ({len(ws)} calls, each its own h)"
        )
        for t in hts + [wt]:
            ttnn.deallocate(t)

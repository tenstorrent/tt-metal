# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the fused head's x chain v3 (HeadSE3) planned, built and enqueued for a Galaxy 8 x 4 with the TP axis a
ring of 4 (FABRIC_2D_TORUS_X), on the mock Galaxy (no execution, no numerics). Run with
TT_METAL_MOCK_CLUSTER_DESC_PATH=$TT_METAL_HOME/tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_galaxy.yaml
(skipped otherwise). Checks: the TP ring schedule (own shard both ways, the opposite shard's even / odd row halves one
way each: 1.5 shards on the busiest link), every chip's program builds and enqueues at the Galaxy per-chip shapes."""

import gc
import os

import pytest
import torch

import ttnn
from models.demos.moe_block.tests.test_fused_head_parts import CHAIN

pytestmark = pytest.mark.skipif(
    "blackhole_galaxy" not in os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", ""), reason="mock Galaxy only"
)


def test_head_mock_galaxy():
    from models.demos.moe_block.tt.fused_head import HeadSE3

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_X)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    R, C = 8, 4
    try:
        g = mesh.compute_with_storage_grid_size()
        assert (g.x, g.y) == (12, 10), (g.x, g.y)  # (the Galaxy grid)
        for name in os.environ.get("L2C_CHAIN", ",".join(CHAIN)).split(","):
            H, N, glu, _, _ = CHAIN[name]
            I = N // 2
            head = HeadSE3(mesh, rows=640, H=H, I=I, glu=glu)
            assert head.xg.ring
            for coord, ch in head.xg.chips.items():
                rg = ch["rings"][0]
                p = rg["p"]
                assert rg["fwd"] == [(p, 0), ((p - 1) % C, 1)] and rg["bwd"] == [(p, 0), ((p + 1) % C, 2)], (coord, rg)
            x = ttnn.from_torch(
                torch.zeros(R, C, 640, H // C),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(R, C)),
            )
            w = head.upload([torch.zeros(H, N)] * (R * C), [torch.zeros(I, H)] * (R * C))
            y = head.run(x, w)
            for t in (y, x, w, head.eng.arena):
                ttnn.deallocate(t)
            del head
            gc.collect()  # (the gather's global semaphores too: L1 fragmentation otherwise)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

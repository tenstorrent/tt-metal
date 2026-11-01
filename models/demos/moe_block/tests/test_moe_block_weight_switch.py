# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MoEBlock with a layer's tensors after another layer's (a model's every MoE layer on one block): program-cache hits
with x / head weights / expert weights at new addresses (allocated after the programs were built, as a model's are)
must give the output of the same values at the build's addresses. GLM-5.3 dims, 8x4, random weights; the combine
snapped to 2^-14 so the outputs compare bit for bit."""

import os

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.moe_block.tests.test_moe_block_smoke import R, C, CHUNK, _flat_weights_module, _glm53
from models.demos.moe_block.tt.moe_block import MoEBlock, MoEBlockWeights

MOCK = "blackhole_galaxy" in os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        (
            (R, C),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
                # (the combine's L1 hand-off: a GLM row (2 x 6144 B) and its trailer in one packet)
                "fabric_router_config": create_fabric_router_config(12352),
                "l1_small_size": 5888,  # (the block's persistent words and semaphores)
            },
        )
    ],
    indirect=True,
)
def test_moe_block_weight_switch(mesh_device):
    mesh = mesh_device
    if MOCK:
        pytest.skip("needs a Galaxy")
    mb = ttnn.experimental.moe_block
    fw = _flat_weights_module()
    dims = _glm53()
    E, H, I, Is, N = dims.num_experts, dims.emb_dim, dims.routed_expert_hidden_dim, 2048, R * C
    EPC = E // N
    dtypes = mb.MoEDtypes()
    res = mb.MoEBlockResources(
        mesh,
        dims,
        mb.MoEBlockConfig(chunk_tokens=CHUNK, num_links=2, combine_snap=2.0**-14, persist_l1_small=True),
        dtypes,
    )
    try:
        block = MoEBlock(mesh, res, dtypes=dtypes, accumulators=11, activation=fw.ACTS["silu"])
        kw = block.expert_kw

        def head(seed):
            g = torch.Generator().manual_seed(seed)
            h = lambda *s, sc=0.02: ttnn.from_torch(torch.randn(*s, generator=g) * sc, dtype=ttnn.float32)
            return mb.prepare_fused_head_weights(
                h(Is, H), h(Is, H), h(H, Is), None, h(E, H, sc=0.05), h(E, sc=0.01),
                resources=res, dtypes=dtypes, accumulators=block.accumulators,
            )  # fmt: skip

        def experts(seed):
            g = torch.Generator().manual_seed(100 + seed)
            r = lambda *s: torch.randn(*s, generator=g) * 0.02  # noqa: E731
            one = [(r(H, I), r(H, I), r(I, H)) for _ in range(EPC)]
            return fw.FlatRoutedExpert(
                mesh, [one] * N, m=kw["max_tokens_per_expert"], H=H, I=I,
                gids=[list(range(c * EPC, (c + 1) * EPC)) for c in range(N)], n_global=E, act="silu", pin=1,
                layout=kw["layout"],
            )  # fmt: skip

        def x_of(seed):
            g = torch.Generator().manual_seed(1000 + seed)
            return ttnn.from_torch(
                torch.randn(1, 1, CHUNK, H, generator=g) * 0.5,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
            )

        comp = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(R, C), dims=(2, 3))

        def run(x, h, e):
            y = block(x, MoEBlockWeights(head=h, expert_ids=e.gidx, gate_up=e.w_gu, down=e.w_d))
            out = ttnn.to_torch(y, mesh_composer=comp).float()
            ttnn.deallocate(y)
            return out

        x0, h0, e0 = x_of(0), head(0), experts(0)
        x1, h1, e1 = x_of(1), head(1), experts(1)
        want0, want1 = run(x0, h0, e0), run(x1, h1, e1)  # (the build, then a hit with every tensor moved)
        bad = []

        def check(name, got, want):
            diff = int((got != want).sum())
            print(f"MOE_WEIGHT_SWITCH {name}: {diff} values differ")
            if diff:
                bad.append(name)

        check("back to layer 0's tensors", run(x0, h0, e0), want0)
        # (a model prepares each layer's weights, and gets each layer's x, after the programs were built: the same
        # values at new addresses must give the same output)
        x0b = x_of(0)
        check("a new x", run(x0b, h0, e0), want0)
        h0b = head(0)
        check("new head weights", run(x0, h0b, e0), want0)
        e0b = experts(0)
        check("new expert weights", run(x0, h0, e0b), want0)
        run(x1, h1, e1)
        check("all new, after another layer", run(x0b, h0b, e0b), want0)
        check("that layer again", run(x1, h1, e1), want1)
        assert not bad, f"a program-cache hit with tensors at new addresses differs: {bad}"
        for e in (e0, e1, e0b):
            e.close()
    finally:
        res.release()
